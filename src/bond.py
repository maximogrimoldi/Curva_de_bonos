from __future__ import annotations

import calendar
import warnings
from datetime import date
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd


class Bond:
    """
    Bono de renta fija con soporte para sinking fund, step-up coupon,
    day count e indexación CER/DL.

    Parámetros
    ----------
    ticker          : identificador del bono
    face_value      : valor nominal (típicamente 100)
    settlement_date : fecha de liquidación / valuación
    maturity_date   : fecha de vencimiento
    coupon          : tasa anual fija (float) o step-up como lista [(end_date, rate), ...]
    frequency       : pagos por año (2 = semestral, 4 = trimestral)
    amort           : None → bullet al vencimiento
                      list[(date, pct)] → sinking fund; pct relativo al face_value
    currency        : "USD" | "ARS_CER" | "ARS_DL" | "ARS_UVA"
    day_count       : "30/360" (USD/DL nominales) | "Actual/Actual" (CER/UVA)
    base_cer        : índice CER de emisión (para ajuste nominal, opcional)

    Convenciones
    ------------
    - El cupón se calcula sobre el outstanding ANTES de la amortización del período.
    - Precios en Argentina son dirty. accrued_interest() devuelve el interés corrido.
    - Las fechas de pago se auto-generan hacia atrás desde maturity_date con el
      paso de 12//frequency meses.
    """

    def __init__(
        self,
        ticker: str,
        face_value: float,
        settlement_date: date,
        maturity_date: date,
        coupon: Union[float, List[Tuple[date, float]]],
        frequency: int = 2,
        amort: Optional[List[Tuple[date, float]]] = None,
        currency: str = "USD",
        day_count: str = "30/360",
        base_cer: float = 1.0,
    ) -> None:
        self.ticker          = ticker
        self.face_value      = float(face_value)
        self.settlement_date = settlement_date
        self.maturity_date   = maturity_date
        self.frequency       = frequency
        self.currency        = currency
        self.day_count       = day_count
        self.base_cer        = base_cer

        payment_dates = self._generate_payment_dates()
        self.amortization_schedule = self._build_amort_df(payment_dates, amort)
        self.coupon_schedule       = self._build_coupon_df(coupon)
        self._cash_flows: Optional[pd.DataFrame] = None

    # ── Schedule builders ────────────────────────────────────────────────────

    def _generate_payment_dates(self) -> list:
        """
        Genera fechas de pago retrocediendo desde maturity_date en pasos de
        12//frequency meses. El día se preserva, capado al último día del mes.
        Cubre hasta el año 2000 (suficiente para todos los instrumentos en uso).
        """
        step = 12 // self.frequency
        dates = []
        y, m, d = self.maturity_date.year, self.maturity_date.month, self.maturity_date.day
        while y >= 2000:
            last = calendar.monthrange(y, m)[1]
            dates.append(date(y, m, min(d, last)))
            m -= step
            while m <= 0:
                m += 12
                y -= 1
        return sorted(dates)

    def _build_amort_df(self, payment_dates: list, amort) -> pd.DataFrame:
        """
        Combina fechas de pago regulares con el schedule de amortización.
        amort=None → bullet 100% en maturity_date.
        amort=[(date, pct), ...] → sinking fund.
        """
        if amort is None:
            amort_map = {self.maturity_date: 1.0}
        else:
            amort_map = dict(amort)

        all_dates = sorted(set(payment_dates) | set(amort_map.keys()))
        rows = [{"date": d, "amort_pct": amort_map.get(d, 0.0)} for d in all_dates]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date").reset_index(drop=True)

    def _build_coupon_df(self, coupon) -> pd.DataFrame:
        """
        coupon: float → tasa fija para toda la vida del bono.
        coupon: [(end_date, rate), ...] → step-up; cada entrada cubre desde el
                final del segmento anterior hasta end_date.
        """
        if isinstance(coupon, (int, float)):
            rows = [{"start_date": date(1900, 1, 1),
                     "end_date":   self.maturity_date,
                     "rate":       float(coupon)}]
        else:
            rows = []
            prev = date(1900, 1, 1)
            for end_dt, rate in sorted(coupon):
                rows.append({"start_date": prev, "end_date": end_dt, "rate": float(rate)})
                prev = end_dt
        df = pd.DataFrame(rows)
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
        df["end_date"]   = pd.to_datetime(df["end_date"]).dt.date
        return df.sort_values("start_date").reset_index(drop=True)

    # ── Coupon rate lookup ────────────────────────────────────────────────────

    def _get_coupon_rate(self, payment_date: date) -> float:
        for _, row in self.coupon_schedule.iterrows():
            if row["start_date"] < payment_date <= row["end_date"]:
                return float(row["rate"])
        warnings.warn(
            f"[{self.ticker}] No se encontró tasa de cupón para {payment_date}. Se asume 0.0.",
            stacklevel=2,
        )
        return 0.0

    # ── Cash flow generation ──────────────────────────────────────────────────

    def generate_cash_flow_schedule(self) -> pd.DataFrame:
        """
        Recorre todo el schedule (incluyendo fechas pasadas) para acumular
        correctamente el outstanding antes de guardar los flujos futuros.
        """
        rows = []
        outstanding = self.face_value

        for _, amort_row in self.amortization_schedule.iterrows():
            pmt_date: date = amort_row["date"]
            amort_usd  = float(amort_row["amort_pct"]) * self.face_value
            coupon_usd = outstanding * self._get_coupon_rate(pmt_date) / self.frequency

            if pmt_date > self.settlement_date:
                years = self._year_fraction(self.settlement_date, pmt_date)
                rows.append({
                    "date":         pmt_date,
                    "years":        years,
                    "coupon":       coupon_usd,
                    "amortization": amort_usd,
                    "total_cf":     coupon_usd + amort_usd,
                    "outstanding":  outstanding,
                })

            outstanding = max(outstanding - amort_usd, 0.0)

        if not rows:
            raise ValueError(f"[{self.ticker}] No hay flujos futuros desde {self.settlement_date}.")

        self._cash_flows = pd.DataFrame(rows)
        return self._cash_flows

    # ── Day count ─────────────────────────────────────────────────────────────

    def _year_fraction(self, d1: date, d2: date) -> float:
        if self.day_count == "30/360":
            return self._days_30_360(d1, d2) / 360.0
        else:  # Actual/Actual
            return (d2 - d1).days / 365.25

    @staticmethod
    def _days_30_360(d1: date, d2: date) -> int:
        """Días entre dos fechas bajo convención 30/360 (ISDA)."""
        d1d = min(d1.day, 30)
        d2d = min(d2.day, 30) if d1d == 30 else d2.day
        return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (d2d - d1d)

    # ── Accrued interest ──────────────────────────────────────────────────────

    def accrued_interest(self) -> float:
        """
        Interés corrido (AI) al settlement_date.
        Precio_limpio = Precio_sucio − accrued_interest().
        """
        all_dates = sorted(self.amortization_schedule["date"].tolist())
        past   = [d for d in all_dates if d <= self.settlement_date]
        future = [d for d in all_dates if d >  self.settlement_date]

        if not past or not future:
            return 0.0

        last_coupon = past[-1]
        next_coupon = future[0]
        coupon_rate = self._get_coupon_rate(next_coupon)

        outstanding = self.face_value
        for _, row in self.amortization_schedule.iterrows():
            if row["date"] <= last_coupon:
                outstanding = max(outstanding - float(row["amort_pct"]) * self.face_value, 0.0)

        if self.day_count == "30/360":
            days_accrued = self._days_30_360(last_coupon, self.settlement_date)
            days_period  = self._days_30_360(last_coupon, next_coupon)
        else:
            days_accrued = (self.settlement_date - last_coupon).days
            days_period  = (next_coupon - last_coupon).days

        if days_period == 0:
            return 0.0

        return outstanding * coupon_rate * days_accrued / days_period / self.frequency

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def cash_flows(self) -> pd.DataFrame:
        if self._cash_flows is None:
            self.generate_cash_flow_schedule()
        return self._cash_flows

    @property
    def maturity_years(self) -> float:
        return float(self.cash_flows["years"].max())

    def price_from_discount_func(self, discount_func: Callable[[float], float]) -> float:
        """VP = Σ CF_t · Z(t)"""
        cf = self.cash_flows
        return float(sum(row["total_cf"] * discount_func(row["years"]) for _, row in cf.iterrows()))

    def __repr__(self) -> str:
        try:
            cf = self.cash_flows
            n, mat = len(cf), cf["date"].max()
        except Exception:
            n, mat = "?", "?"
        return f"Bond('{self.ticker}', maturity={mat}, flows={n}, day_count={self.day_count})"

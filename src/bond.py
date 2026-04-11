from __future__ import annotations

import warnings
from datetime import date
from typing import Callable, Optional

import pandas as pd


class Bond:
    """
    Bono con soporte para sinking fund, step-up coupon, day count e indexación CER.

    Convenciones:
    - El cupón se calcula sobre el outstanding ANTES de la amortización del período.
    - day_count: "30/360" (USD Globales, ONs USD/DL) | "Actual/Actual" (CER/UVA).
    - base_cer: índice CER de emisión (para indexar flujos reales a nominales, opcional).
    - Precios de mercado en Argentina son DIRTY (sucio). accrued_interest() devuelve el AI
      para obtener el precio limpio: precio_limpio = precio_sucio - accrued_interest().
    """

    def __init__(
        self,
        ticker: str,
        face_value: float,
        settlement_date: date,
        amortization_schedule: pd.DataFrame,
        coupon_schedule: pd.DataFrame,
        frequency: int = 2,
        currency: str = "USD",
        day_count: str = "30/360",
        base_cer: float = 1.0,
    ) -> None:
        self.ticker = ticker
        self.face_value = float(face_value)
        self.settlement_date = settlement_date
        self.frequency = frequency
        self.currency = currency
        self.day_count = day_count
        self.base_cer = base_cer

        self.amortization_schedule = self._parse_amort(amortization_schedule)
        self.coupon_schedule = self._parse_coupon(coupon_schedule)
        self._cash_flows: Optional[pd.DataFrame] = None

    @staticmethod
    def _parse_amort(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _parse_coupon(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
        return df.sort_values("start_date").reset_index(drop=True)

    def _get_coupon_rate(self, payment_date: date) -> float:
        for _, row in self.coupon_schedule.iterrows():
            if row["start_date"] < payment_date <= row["end_date"]:
                return float(row["rate"])
        warnings.warn(
            f"[{self.ticker}] No se encontró tasa de cupón para {payment_date}. Se asume 0.0.",
            stacklevel=2,
        )
        return 0.0

    def generate_cash_flow_schedule(self) -> pd.DataFrame:
        """
        Recorre todo el schedule (incluyendo fechas pasadas) para acumular
        correctamente el outstanding antes de guardar los flujos futuros.
        """
        rows = []
        outstanding = self.face_value

        for _, amort_row in self.amortization_schedule.iterrows():
            pmt_date: date = amort_row["date"]
            amort_usd = float(amort_row["amort_pct"]) * self.face_value
            coupon_usd = outstanding * self._get_coupon_rate(pmt_date) / self.frequency

            if pmt_date > self.settlement_date:
                years = self._year_fraction(self.settlement_date, pmt_date)
                rows.append({
                    "date":           pmt_date,
                    "years":          years,
                    "coupon":         coupon_usd,
                    "amortization":   amort_usd,
                    "total_cf":       coupon_usd + amort_usd,
                    "outstanding":    outstanding,
                })

            outstanding = max(outstanding - amort_usd, 0.0)

        if not rows:
            raise ValueError(f"[{self.ticker}] No hay flujos futuros desde {self.settlement_date}.")

        self._cash_flows = pd.DataFrame(rows)
        return self._cash_flows

    def _year_fraction(self, d1: date, d2: date) -> float:
        """Fracción de año entre dos fechas según la convención day_count del bono."""
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

        # Outstanding justo después de la amortización del último cupón pagado
        outstanding = self.face_value
        for _, row in self.amortization_schedule.iterrows():
            if row["date"] <= last_coupon:
                outstanding = max(outstanding - float(row["amort_pct"]) * self.face_value, 0.0)

        if self.day_count == "30/360":
            days_accrued = self._days_30_360(last_coupon, self.settlement_date)
            days_period  = self._days_30_360(last_coupon, next_coupon)
        else:  # Actual/Actual
            days_accrued = (self.settlement_date - last_coupon).days
            days_period  = (next_coupon - last_coupon).days

        if days_period == 0:
            return 0.0

        return outstanding * coupon_rate * days_accrued / days_period / self.frequency

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

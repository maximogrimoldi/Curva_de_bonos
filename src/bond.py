from __future__ import annotations

import warnings
from datetime import date
from typing import Callable, Optional

import pandas as pd


class Bond:
    """
    Bono con soporte para sinking fund y cupones step-up.

    El cupón se calcula sobre el outstanding ANTES de la amortización
    de ese período (convención de mercado).
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
    ) -> None:
        self.ticker = ticker
        self.face_value = float(face_value)
        self.settlement_date = settlement_date
        self.frequency = frequency
        self.currency = currency

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
                years = (pmt_date - self.settlement_date).days / 365.25
                rows.append({
                    "date": pmt_date,
                    "years": years,
                    "coupon": coupon_usd,
                    "amortization": amort_usd,
                    "total_cf": coupon_usd + amort_usd,
                    "outstanding": outstanding,
                })

            outstanding = max(outstanding - amort_usd, 0.0)

        if not rows:
            raise ValueError(f"[{self.ticker}] No hay flujos futuros desde {self.settlement_date}.")

        self._cash_flows = pd.DataFrame(rows)
        return self._cash_flows

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
        return f"Bond('{self.ticker}', maturity={mat}, flows={n})"

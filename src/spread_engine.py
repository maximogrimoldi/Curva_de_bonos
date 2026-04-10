from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .bond import Bond
from .nss_curve import NSSCurve


class SpreadEngine:
    """Calcula Z-Spread de ONs respecto a la curva soberana NSS."""

    def __init__(self, curve: NSSCurve) -> None:
        if curve.params is None:
            raise ValueError("La curva NSS no está calibrada. Ejecutar NSSCurve.fit() primero.")
        self.curve = curve

    def _price_with_spread(self, bond: Bond, z: float) -> float:
        """P(z) = Σ CF_t / (1 + s(t) + z)^t"""
        pv = 0.0
        for _, row in bond.cash_flows.iterrows():
            s_t = self.curve.get_spot_rate(row["years"])
            pv += row["total_cf"] * (1 + max(s_t + z, -0.9999)) ** (-row["years"])
        return pv

    def z_spread(self, bond: Bond, dirty_price: float, z_min: float = -0.50, z_max: float = 3.00) -> Dict:
        def price_diff(z: float) -> float:
            return self._price_with_spread(bond, z) - dirty_price

        try:
            if price_diff(z_min) * price_diff(z_max) > 0:
                z_min, z_max = z_min - 0.50, z_max + 0.50
                if price_diff(z_min) * price_diff(z_max) > 0:
                    raise ValueError(f"No se encontró raíz en [{z_min:.2f}, {z_max:.2f}].")
            z_sol = brentq(price_diff, z_min, z_max, xtol=1e-10, maxiter=300)
        except Exception as exc:
            nan = float("nan")
            return {"ticker": bond.ticker, "z_spread_bps": nan, "market_price": dirty_price,
                    "theo_price": nan, "pricing_error": nan, "error_msg": str(exc)}

        theo_price = self._price_with_spread(bond, z_sol)
        return {
            "ticker":        bond.ticker,
            "z_spread_bps":  round(z_sol * 1e4, 2),
            "market_price":  dirty_price,
            "theo_price":    round(theo_price, 6),
            "pricing_error": round(theo_price - dirty_price, 6),
        }

    def spread_table(self, bonds: List[Bond], prices: List[float]) -> pd.DataFrame:
        rows = [self.z_spread(bond, price) for bond, price in zip(bonds, prices)]
        return (pd.DataFrame(rows)
                  [["ticker", "market_price", "z_spread_bps", "pricing_error"]]
                  .sort_values("z_spread_bps")
                  .reset_index(drop=True))

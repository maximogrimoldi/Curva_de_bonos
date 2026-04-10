from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .bond import Bond


class NSSCurve:
    """
    Curva de Tasas Spot Nelson-Siegel-Svensson calibrada a precios de mercado.

    s(t) = β₀ + β₁·φ(t,τ₁) + β₂·[φ(t,τ₁) - e^(-t/τ₁)] + β₃·[φ(t,τ₂) - e^(-t/τ₂)]
    Z(t) = (1 + s(t))^(-t)
    """

    PARAM_NAMES = ["β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"]

    def __init__(self) -> None:
        self.params: Optional[np.ndarray] = None
        self.fit_result = None

    @staticmethod
    def _nelson_siegel_factor(t: np.ndarray, tau: float) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        ratio = t / tau
        small = np.abs(ratio) < 1e-6
        return np.where(
            small,
            1.0 - ratio / 2.0 + ratio**2 / 6.0,
            (1.0 - np.exp(-ratio)) / np.where(small, 1.0, ratio),
        )

    @staticmethod
    def spot_rate(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        beta0, beta1, beta2, beta3, tau1, tau2 = params
        t = np.maximum(np.asarray(t, dtype=float), 1e-8)
        phi1 = NSSCurve._nelson_siegel_factor(t, tau1)
        phi2 = NSSCurve._nelson_siegel_factor(t, tau2)
        return beta0 + beta1 * phi1 + beta2 * (phi1 - np.exp(-t / tau1)) + beta3 * (phi2 - np.exp(-t / tau2))

    @staticmethod
    def discount_factor(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        s = NSSCurve.spot_rate(t, params)
        base = np.maximum(1.0 + s, 1e-6)
        return np.clip(base ** (-np.asarray(t, dtype=float)), 1e-12, 1e6)

    @staticmethod
    def forward_rate(t: np.ndarray, params: np.ndarray, dt: float = 1.0) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        z_t = NSSCurve.discount_factor(t, params)
        z_t_dt = NSSCurve.discount_factor(t + dt, params)
        return (z_t / np.maximum(z_t_dt, 1e-12)) ** (1.0 / dt) - 1.0

    def _theoretical_price(self, bond: Bond, params: np.ndarray) -> float:
        cf = bond.cash_flows
        z = self.discount_factor(cf["years"].values, params)
        return float(np.dot(cf["total_cf"].values, z))

    def _loss(self, params, bonds, market_prices, weights) -> float:
        total = 0.0
        for bond, p_mkt, w in zip(bonds, market_prices, weights):
            try:
                p_nss = self._theoretical_price(bond, params)
                if not np.isfinite(p_nss):
                    return 1e8
                total += w * (p_mkt - p_nss) ** 2
            except Exception:
                return 1e8
        return total

    def fit(
        self,
        bonds: List[Bond],
        market_prices: List[float],
        weights: Optional[List[float]] = None,
        n_restarts: int = 25,
        verbose: bool = True,
    ) -> Dict:
        assert len(bonds) == len(market_prices)

        w_arr = np.ones(len(bonds)) if weights is None else np.array(weights)

        bounds = [
            (0.01, 0.60),   # β₀
            (-0.50, 0.40),  # β₁
            (-0.60, 0.60),  # β₂
            (-0.60, 0.60),  # β₃
            (0.10,  8.00),  # τ₁
            (0.50, 20.00),  # τ₂
        ]

        seeds = [
            [0.14, -0.06,  0.00,  0.00,  1.5,  5.0],
            [0.18, -0.10,  0.02, -0.01,  2.0,  6.0],
            [0.22, -0.12, -0.02,  0.03,  1.0,  4.0],
            [0.12, -0.04,  0.01,  0.01,  2.5,  7.0],
            [0.16, -0.07,  0.03, -0.02,  1.5,  5.5],
            [0.10, -0.01,  0.00,  0.00,  3.0,  8.0],
            [0.13,  0.01, -0.01,  0.01,  2.0,  6.0],
            [0.15, -0.08,  0.02,  0.01,  0.5,  3.0],
            [0.12, -0.05,  0.01, -0.01,  4.0, 12.0],
        ]

        rng = np.random.default_rng(seed=42)
        while len(seeds) < n_restarts:
            seeds.append([
                rng.uniform(0.06, 0.35),
                rng.uniform(-0.35, 0.20),
                rng.uniform(-0.30, 0.30),
                rng.uniform(-0.30, 0.30),
                rng.uniform(0.3,   5.0),
                rng.uniform(0.8,  15.0),
            ])

        best_loss = np.inf
        best_result = None

        if verbose:
            print(f"\n  Calibrando NSS con {len(bonds)} bonos | {n_restarts} restarts...", end="", flush=True)

        for x0 in seeds[:n_restarts]:
            try:
                res = minimize(
                    fun=self._loss,
                    x0=x0,
                    args=(bonds, market_prices, w_arr),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 10_000, "ftol": 1e-14, "gtol": 1e-9},
                )
                if np.isfinite(res.fun) and res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("La calibración NSS falló en todos los restarts.")

        if verbose:
            print(f" OK  (loss={best_loss:.6f})")

        self.params = best_result.x
        self.fit_result = best_result

        fit_metrics = self._compute_fit_errors(bonds, market_prices)
        if verbose:
            self._print_summary(fit_metrics)

        return fit_metrics

    def _compute_fit_errors(self, bonds: List[Bond], market_prices: List[float]) -> Dict:
        bond_errors = []
        for bond, p_mkt in zip(bonds, market_prices):
            p_nss = self._theoretical_price(bond, self.params)
            err_abs = p_mkt - p_nss
            bond_errors.append({
                "ticker":            bond.ticker,
                "market_price":      p_mkt,
                "theoretical_price": p_nss,
                "error_abs":         err_abs,
                "error_pct":         err_abs / p_mkt * 100,
                "maturity_years":    bond.maturity_years,
            })

        errors_abs = [abs(e["error_abs"]) for e in bond_errors]
        errors_pct = [abs(e["error_pct"]) for e in bond_errors]

        return {
            "params":         dict(zip(self.PARAM_NAMES, self.params)),
            "bond_errors":    bond_errors,
            "rmse_price":     float(np.sqrt(np.mean([e**2 for e in errors_abs]))),
            "mae_pct":        float(np.mean(errors_pct)),
            "max_error_pct":  float(np.max(errors_pct)),
        }

    def _print_summary(self, metrics: Dict) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print("  CALIBRACIÓN NSS — RESULTADOS")
        print(sep)
        for name, val in metrics["params"].items():
            if "τ" in name:
                print(f"    {name}  =  {val:8.4f}")
            else:
                print(f"    {name}  =  {val:8.4f}   ({val*100:+.2f}%)")
        print("-" * 65)
        print(f"  {'Ticker':<7} {'Vto(y)':>6} {'Mkt P':>8} {'NSS P':>8} {'Err$':>7} {'Err%':>7}")
        print("-" * 65)
        for e in metrics["bond_errors"]:
            print(
                f"  {e['ticker']:<7} {e['maturity_years']:>6.1f} "
                f"{e['market_price']:>8.4f} {e['theoretical_price']:>8.4f} "
                f"{e['error_abs']:>+7.4f} {e['error_pct']:>+6.3f}%"
            )
        print("-" * 65)
        print(
            f"  RMSE: ${metrics['rmse_price']:.4f}  |  "
            f"MAE: {metrics['mae_pct']:.3f}%  |  "
            f"Max error: {metrics['max_error_pct']:.3f}%"
        )
        print(sep + "\n")

    def _check_fitted(self) -> None:
        if self.params is None:
            raise RuntimeError("Curva no calibrada. Ejecutar fit() primero.")

    def get_spot_rate(self, t: float) -> float:
        self._check_fitted()
        return float(self.spot_rate(np.array([t]), self.params)[0])

    def get_discount_factor(self, t: float) -> float:
        self._check_fitted()
        return float(self.discount_factor(np.array([t]), self.params)[0])

    def get_forward_rate(self, t: float, dt: float = 1.0) -> float:
        self._check_fitted()
        return float(self.forward_rate(np.array([t]), self.params, dt=dt)[0])

    def curve_dataframe(self, t_max: float = 30.0, n_points: int = 500) -> pd.DataFrame:
        self._check_fitted()
        t = np.linspace(0.01, t_max, n_points)
        spot = self.spot_rate(t, self.params)
        fwd = self.forward_rate(t, self.params, dt=1.0)
        return pd.DataFrame({
            "years":          t,
            "spot_rate_pct":  spot * 100,
            "forward_1y_pct": fwd * 100,
        })

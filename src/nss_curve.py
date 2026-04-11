from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from .bond import Bond


class NSSCurve:
    """
    Curva de Tasas Spot Nelson-Siegel-Svensson calibrada a precios de mercado.

    s(t) = β₀ + β₁·φ(t,τ₁) + β₂·[φ(t,τ₁) - e^(-t/τ₁)] + β₃·[φ(t,τ₂) - e^(-t/τ₂)]
    Z(t) = (1 + s(t))^(-t)
    """

    PARAM_NAMES = ["β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"]

    BOUNDS = [
        (0.01, 0.60),   # β₀
        (-0.50, 0.40),  # β₁
        (-0.60, 0.60),  # β₂
        (-0.60, 0.60),  # β₃
        (0.10,  8.00),  # τ₁
        (0.50, 20.00),  # τ₂
    ]

    PARAM_INTERP = [
        "Nivel LP (tasa larga)",
        "Pendiente (corto − largo)",
        "Curvatura 1 (hump corto)",
        "Curvatura 2 (hump largo)",
        "Escala hump 1 (años)",
        "Escala hump 2 (años)",
    ]

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

    @staticmethod
    def _ytm(bond: Bond, price: float) -> float:
        """Calcula TIR (yield to maturity) de un bono dado su precio sucio."""
        cf = bond.cash_flows
        times = cf["years"].values
        flows = cf["total_cf"].values

        def pv(y: float) -> float:
            return float(np.sum(flows / (1.0 + y) ** times)) - price

        try:
            return brentq(pv, -0.50, 5.0, xtol=1e-10, maxiter=300)
        except Exception:
            return float("nan")

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
        verbose: bool = False,
    ) -> Dict:
        assert len(bonds) == len(market_prices)

        w_arr = np.ones(len(bonds)) if weights is None else np.array(weights)

        seeds = [
            [0.08,  0.10, -0.03,  0.01,  2.0,  6.0],
            [0.08,  0.12, -0.05,  0.02,  1.5,  4.0],
            [0.09,  0.09,  0.00,  0.00,  2.5,  7.0],
            [0.07,  0.11, -0.02,  0.01,  1.0,  3.0],
            [0.04,  0.06, -0.02,  0.01,  1.5,  5.0],
            [0.04,  0.07, -0.01,  0.00,  2.0,  6.0],
            [0.03,  0.05, -0.01,  0.005, 1.5,  5.0],
            [0.03,  0.04,  0.00,  0.00,  2.0,  6.0],
            [0.10,  0.05, -0.04,  0.02,  2.0,  5.0],
            [0.06,  0.08, -0.03,  0.01,  1.5,  4.0],
            [0.12,  0.08, -0.05,  0.02,  3.0,  8.0],
            [0.05,  0.03,  0.01, -0.01,  2.0,  7.0],
            [0.14, -0.06,  0.00,  0.00,  1.5,  5.0],
            [0.12, -0.04,  0.01,  0.01,  2.5,  7.0],
            [0.10, -0.01,  0.00,  0.00,  3.0,  8.0],
            [0.13,  0.01, -0.01,  0.01,  2.0,  6.0],
            [0.12, -0.05,  0.01, -0.01,  4.0, 12.0],
        ]

        rng = np.random.default_rng(seed=42)
        while len(seeds) < n_restarts:
            seeds.append([
                rng.uniform(0.03, 0.35),
                rng.uniform(-0.20, 0.25),
                rng.uniform(-0.20, 0.20),
                rng.uniform(-0.20, 0.20),
                rng.uniform(0.5,   5.0),
                rng.uniform(1.0,  12.0),
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
                    bounds=self.BOUNDS,
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

        # ── Inferencia estadística de los parámetros ──────────────────
        self.std_errors: np.ndarray = np.full(6, np.nan)
        self.t_stats:    np.ndarray = np.full(6, np.nan)
        self.ic_low:     np.ndarray = np.full(6, np.nan)
        self.ic_high:    np.ndarray = np.full(6, np.nan)
        try:
            hess_inv = best_result.hess_inv @ np.eye(6)
            cov = hess_inv * 2
            diag = np.diag(cov)
            if np.all(diag > 0):
                se = np.sqrt(diag)
                self.std_errors = se
                self.t_stats    = self.params / se
                self.ic_low     = self.params - 1.96 * se
                self.ic_high    = self.params + 1.96 * se
            else:
                # algunos negativos: calcular solo los válidos
                for i, d in enumerate(diag):
                    if d > 0:
                        se_i = float(np.sqrt(d))
                        self.std_errors[i] = se_i
                        self.t_stats[i]    = self.params[i] / se_i
                        self.ic_low[i]     = self.params[i] - 1.96 * se_i
                        self.ic_high[i]    = self.params[i] + 1.96 * se_i
        except Exception:
            pass  # atributos quedan en nan

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
                "bond":              bond,
            })

        errors_abs = np.array([abs(e["error_abs"]) for e in bond_errors])
        errors_pct = np.array([abs(e["error_pct"]) for e in bond_errors])
        prices_mkt = np.array([e["market_price"] for e in bond_errors])
        prices_nss = np.array([e["theoretical_price"] for e in bond_errors])

        ss_res = float(np.sum((prices_mkt - prices_nss) ** 2))
        ss_tot = float(np.sum((prices_mkt - np.mean(prices_mkt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        return {
            "params":          dict(zip(self.PARAM_NAMES, self.params)),
            "bond_errors":     bond_errors,
            "rmse_price":      float(np.sqrt(np.mean(errors_abs ** 2))),
            "mae_pct":         float(np.mean(errors_pct)),
            "max_error_pct":   float(np.max(errors_pct)),
            "max_error_abs":   float(np.max(errors_abs)),
            "r2":              r2,
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

    def print_extended_analysis(self, metrics: Dict, tir_reference: Dict[str, float]) -> None:
        """Imprime análisis extendido: parámetros, métricas, TIR errors y sensibilidad τ₁."""
        sep = "=" * 65
        dash = "-" * 65

        # ── Tabla de parámetros ───────────────────────────────────────
        print(f"\n  ANÁLISIS DE PARÁMETROS NSS:")
        print(f"  {'Param':<5} {'Valor':>8} {'Interpretación':<32} {'Rango':>14} {'% rango':>8}")
        print("  " + dash)
        for i, (name, val) in enumerate(metrics["params"].items()):
            lo, hi = self.BOUNDS[i]
            pct_range = (val - lo) / (hi - lo) * 100
            interp = self.PARAM_INTERP[i]
            if "τ" in name:
                val_str = f"{val:8.4f}"
                interp_full = f"{interp}: {val:.4f}"
            else:
                val_str = f"{val:8.4f}"
                interp_full = f"{interp}: {val*100:+.2f}%"
            rango_str = f"[{lo:.2f}, {hi:.2f}]"
            print(f"  {name:<5} {val_str} {interp_full:<32} {rango_str:>14} {pct_range:>7.1f}%")

        # ── Métricas de ajuste ────────────────────────────────────────
        print(f"\n  MÉTRICAS DE AJUSTE:")
        print(
            f"  R²: {metrics['r2']:.6f}  |  "
            f"RMSE: ${metrics['rmse_price']:.4f}  |  "
            f"MAE: {metrics['mae_pct']:.3f}%  |  "
            f"Max error: ${metrics['max_error_abs']:.4f}"
        )

        # ── Error en TIR por bono ─────────────────────────────────────
        print(f"\n  ERROR EN TIR POR BONO:")
        print(f"  {'Ticker':<7} {'TIR Mkt':>10} {'TIR NSS':>10} {'Error (bps)':>12}")
        print("  " + dash)
        for e in metrics["bond_errors"]:
            ticker = e["ticker"]
            tir_ref = tir_reference.get(ticker)
            tir_nss = self._ytm(e["bond"], e["theoretical_price"])
            if tir_ref is None or not np.isfinite(tir_nss):
                print(f"  {ticker:<7} {'N/D':>10} {'N/D':>10} {'N/D':>12}")
                continue
            err_bps = (tir_nss - tir_ref) * 1e4
            print(
                f"  {ticker:<7} {tir_ref*100:>9.2f}% {tir_nss*100:>9.2f}% {err_bps:>+11.1f} bps"
            )

        # ── Sensibilidad τ₁ ──────────────────────────────────────────
        print(f"\n  SENSIBILIDAD τ₁ (±20%):")
        t_nodes = [1, 3, 5, 10]
        tau1_base = self.params[4]
        tau1_minus = tau1_base * 0.80
        tau1_plus  = tau1_base * 1.20

        print(
            f"  {'t':>6} {'Spot base':>10} {'τ₁ −20%':>10} {'τ₁ +20%':>10} "
            f"{'Δbps (−)':>10} {'Δbps (+)':>10}"
        )
        print("  " + dash)
        for t in t_nodes:
            s_base = self.get_spot_rate(t) * 100

            p_minus = self.params.copy(); p_minus[4] = tau1_minus
            s_minus = float(self.spot_rate(np.array([t]), p_minus)[0]) * 100

            p_plus = self.params.copy(); p_plus[4] = tau1_plus
            s_plus = float(self.spot_rate(np.array([t]), p_plus)[0]) * 100

            dbps_minus = (s_minus - s_base) * 100
            dbps_plus  = (s_plus  - s_base) * 100

            print(
                f"  {t:>5}a {s_base:>9.2f}% {s_minus:>9.2f}% {s_plus:>9.2f}% "
                f"{dbps_minus:>+9.1f}  {dbps_plus:>+9.1f}"
            )
        print()

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

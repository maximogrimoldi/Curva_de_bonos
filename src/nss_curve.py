from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from .bond import Bond


class NSSCurve:
    """
    Curva de Tasas Spot — Nelson-Siegel (NS, 4 params) o Nelson-Siegel-Svensson (NSS, 6 params).

    NS:  s(t) = β₀ + β₁·φ(t,τ) + β₂·[φ(t,τ) − e^(−t/τ)]
    NSS: s(t) = β₀ + β₁·φ(t,τ₁) + β₂·[φ(t,τ₁) − e^(−t/τ₁)] + β₃·[φ(t,τ₂) − e^(−t/τ₂)]

    donde  φ(t,τ) = (1 − e^(−t/τ)) / (t/τ)

    Parámetros
    ----------
    use_nss : True activa el modelo NSS (6 params). Recomendado cuando la curva
              tiene dos humps distintos (ej: CER con tramo corto negativo y hump largo).
    forward_penalty_lambda : regularización que penaliza tramos decrecientes en la
              tasa forward, suavizando artefactos de "pico" post-calibración.
    tau_fixed : NS solamente — fija τ vía bounds muy estrechos.
    """

    # PARAM_NAMES se define como propiedad de instancia según use_nss
    _PARAM_NAMES_NS  = ["β₀", "β₁", "β₂", "τ"]
    _PARAM_NAMES_NSS = ["β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"]

    _PARAM_INTERP_NS  = [
        "Nivel LP (tasa larga)",
        "Pendiente (corto − largo)",
        "Curvatura (hump)",
        "Escala hump (años)",
    ]
    _PARAM_INTERP_NSS = [
        "Nivel LP (tasa larga)",
        "Pendiente (corto − largo)",
        "Curvatura hump 1",
        "Curvatura hump 2",
        "Escala hump 1 (años)",
        "Escala hump 2 (años)",
    ]

    _BOUNDS_NS = [
        (0.001, 0.30),  # β₀
        (-0.30, 0.30),  # β₁
        (-0.30, 0.30),  # β₂
        (1.00,  3.00),  # τ
    ]
    _BOUNDS_NSS = [
        (0.001, 0.30),  # β₀
        (-0.30, 0.30),  # β₁
        (-0.50, 0.50),  # β₂  — mayor amplitud para el primer hump
        (-0.30, 0.30),  # β₃
        (0.10,  3.00),  # τ₁  — escala corta
        (1.50, 10.00),  # τ₂  — escala larga (siempre > τ₁ por diseño de bounds)
    ]

    # Alias para compatibilidad con código externo que lee NSSCurve.BOUNDS / PARAM_NAMES
    BOUNDS      = _BOUNDS_NS
    PARAM_NAMES = _PARAM_NAMES_NS
    PARAM_INTERP = _PARAM_INTERP_NS

    # Pesos por tipo de instrumento en WLS (fit_from_yields)
    _DEFAULT_TYPE_WEIGHTS: Dict[str, float] = {
        "sovereign":  1.5,
        "dual":       0.7,
        "bopreal":    0.7,
        "synthetic":  1.0,
    }

    def __init__(
        self,
        tau_fixed:  Optional[float] = None,
        tau_min:    Optional[float] = None,
        tau_max:    Optional[float] = None,
        beta0_max:  Optional[float] = None,
        beta0_min:  Optional[float] = None,
        beta1_max:  Optional[float] = None,
        beta1_min:  Optional[float] = None,
        beta2_min:  Optional[float] = None,
        beta2_max:  Optional[float] = None,
        use_nss:    bool = False,
        tau1_min:   Optional[float] = None,
        tau1_max:   Optional[float] = None,
        tau2_min:   Optional[float] = None,
        tau2_max:   Optional[float] = None,
        forward_penalty_lambda: float = 0.0,
    ) -> None:
        self.use_nss    = use_nss
        self.model      = "NSS" if use_nss else "NS"
        self.tau_fixed  = tau_fixed
        self.tau_min    = tau_min
        self.tau_max    = tau_max
        self.beta0_max  = beta0_max
        self.beta0_min  = beta0_min
        self.beta1_max  = beta1_max
        self.beta1_min  = beta1_min
        self.beta2_min  = beta2_min
        self.beta2_max  = beta2_max
        self.tau1_min   = tau1_min
        self.tau1_max   = tau1_max
        self.tau2_min   = tau2_min
        self.tau2_max   = tau2_max
        self.forward_penalty_lambda = forward_penalty_lambda
        self.params:     Optional[np.ndarray] = None
        self.fit_result = None

        n = self.n_params
        self.std_errors: np.ndarray = np.full(n, np.nan)
        self.t_stats:    np.ndarray = np.full(n, np.nan)
        self.ic_low:     np.ndarray = np.full(n, np.nan)
        self.ic_high:    np.ndarray = np.full(n, np.nan)

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return 6 if self.use_nss else 4

    @property
    def param_names(self) -> List[str]:
        return self._PARAM_NAMES_NSS if self.use_nss else self._PARAM_NAMES_NS

    @property
    def param_interp(self) -> List[str]:
        return self._PARAM_INTERP_NSS if self.use_nss else self._PARAM_INTERP_NS

    @property
    def bounds(self) -> list:
        b = list(self._BOUNDS_NSS if self.use_nss else self._BOUNDS_NS)
        # β₀ overrides (índice 0)
        if self.beta0_min is not None: b[0] = (self.beta0_min, b[0][1])
        if self.beta0_max is not None: b[0] = (b[0][0], self.beta0_max)
        # β₁ overrides (índice 1)
        if self.beta1_min is not None: b[1] = (self.beta1_min, b[1][1])
        if self.beta1_max is not None: b[1] = (b[1][0], self.beta1_max)
        # β₂ overrides (índice 2)
        if self.beta2_min is not None: b[2] = (self.beta2_min, b[2][1])
        if self.beta2_max is not None: b[2] = (b[2][0], self.beta2_max)
        if self.use_nss:
            # τ₁ overrides (índice 4)
            if self.tau1_min is not None: b[4] = (self.tau1_min, b[4][1])
            if self.tau1_max is not None: b[4] = (b[4][0], self.tau1_max)
            # τ₂ overrides (índice 5)
            if self.tau2_min is not None: b[5] = (self.tau2_min, b[5][1])
            if self.tau2_max is not None: b[5] = (b[5][0], self.tau2_max)
        else:
            # τ overrides (índice 3) — solo NS
            if self.tau_min is not None: b[3] = (self.tau_min, b[3][1])
            if self.tau_max is not None: b[3] = (b[3][0], self.tau_max)
            if self.tau_fixed is not None:
                eps  = self.tau_fixed * 1e-6
                b[3] = (self.tau_fixed - eps, self.tau_fixed + eps)
        return b

    # ── Matemática del modelo ─────────────────────────────────────────────────

    @staticmethod
    def _nelson_siegel_factor(t: np.ndarray, tau: float) -> np.ndarray:
        t     = np.asarray(t, dtype=float)
        ratio = t / tau
        small = np.abs(ratio) < 1e-6
        return np.where(
            small,
            1.0 - ratio / 2.0 + ratio ** 2 / 6.0,
            (1.0 - np.exp(-ratio)) / np.where(small, 1.0, ratio),
        )

    @staticmethod
    def spot_rate(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        NS (4 params):  s(t) = β₀ + β₁·φ(τ) + β₂·[φ(τ) − e^(−t/τ)]
        NSS (6 params): s(t) += β₃·[φ(τ₂) − e^(−t/τ₂)]
        Dispatch automático por longitud de params.
        """
        t = np.maximum(np.asarray(t, dtype=float), 1e-8)
        if len(params) == 4:
            beta0, beta1, beta2, tau = params
            phi = NSSCurve._nelson_siegel_factor(t, tau)
            return beta0 + beta1 * phi + beta2 * (phi - np.exp(-t / tau))
        else:  # NSS — 6 parámetros
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            phi1 = NSSCurve._nelson_siegel_factor(t, tau1)
            phi2 = NSSCurve._nelson_siegel_factor(t, tau2)
            return (beta0
                    + beta1 * phi1
                    + beta2 * (phi1 - np.exp(-t / tau1))
                    + beta3 * (phi2 - np.exp(-t / tau2)))

    @staticmethod
    def discount_factor(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        s    = NSSCurve.spot_rate(t, params)
        base = np.maximum(1.0 + s, 1e-6)
        return np.clip(base ** (-np.asarray(t, dtype=float)), 1e-12, 1e6)

    @staticmethod
    def forward_rate(t: np.ndarray, params: np.ndarray, dt: float = 1.0) -> np.ndarray:
        t      = np.asarray(t, dtype=float)
        z_t    = NSSCurve.discount_factor(t, params)
        z_t_dt = NSSCurve.discount_factor(t + dt, params)
        return (z_t / np.maximum(z_t_dt, 1e-12)) ** (1.0 / dt) - 1.0

    @staticmethod
    def _ytm(bond: Bond, price: float) -> float:
        cf    = bond.cash_flows
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
        z  = self.discount_factor(cf["years"].values, params)
        return float(np.dot(cf["total_cf"].values, z))

    def _loss(self, params, bonds, market_prices, weights) -> float:
        total = 0.0
        for bond, p_mkt, w in zip(bonds, market_prices, weights):
            try:
                p_ns = self._theoretical_price(bond, params)
                if not np.isfinite(p_ns):
                    return 1e8
                total += w * (p_mkt - p_ns) ** 2
            except Exception:
                return 1e8
        return total

    @staticmethod
    def _loss_yields(
        params: np.ndarray,
        maturities: np.ndarray,
        yields: np.ndarray,
        weights: np.ndarray,
        forward_penalty_lambda: float = 0.0,
    ) -> float:
        """
        WLS en espacio de yields: Σ wᵢ·(ŷᵢ − yᵢ)²

        + forward_penalty_lambda · Σ max(−Δfwd, 0)²

        El término de penalidad penaliza caídas en la tasa forward,
        suavizando artefactos de "pico" post-calibración sin forzar
        monotonía estricta.
        """
        spot = NSSCurve.spot_rate(maturities, params)
        if not np.all(np.isfinite(spot)):
            return 1e8
        residuals = yields - spot
        base = float(np.dot(weights, residuals ** 2))

        if forward_penalty_lambda <= 0.0:
            return base

        # Grilla de evaluación de la forward (no depende de los nodos)
        t_fwd = np.linspace(0.10, float(maturities.max()) * 1.15, 120)
        z_t    = NSSCurve.discount_factor(t_fwd,        params)
        z_t1   = NSSCurve.discount_factor(t_fwd + 0.25, params)
        fwd    = (z_t / np.maximum(z_t1, 1e-12)) ** (1.0 / 0.25) - 1.0
        if not np.all(np.isfinite(fwd)):
            return 1e8
        neg_changes = np.minimum(np.diff(fwd), 0.0)
        penalty = forward_penalty_lambda * float(np.sum(neg_changes ** 2))
        return base + penalty

    # ── Calibración ───────────────────────────────────────────────────────────

    def _make_seeds(self, n_restarts: int, rng: np.random.Generator) -> list:
        if self.use_nss:
            return self._make_seeds_nss(n_restarts, rng)
        return self._make_seeds_ns(n_restarts, rng)

    def _make_seeds_ns(self, n_restarts: int, rng: np.random.Generator) -> list:
        tau0 = self.tau_fixed if self.tau_fixed is not None else 2.0
        seeds = [
            [0.03, 0.05, -0.01, tau0],
            [0.04, 0.04, -0.02, tau0],
            [0.05, 0.03,  0.00, tau0],
            [0.03, 0.06, -0.01, tau0 * 0.8],
            [0.04, 0.05, -0.02, tau0 * 1.2],
            [0.06, 0.04, -0.01, tau0],
            [0.02, 0.07,  0.00, tau0],
            [0.05, 0.05, -0.03, tau0 * 1.5],
            [0.03, 0.03,  0.01, tau0 * 0.6],
            [0.07, 0.03, -0.02, tau0],
            # seeds para curvas invertidas (devaluación esperada)
            [0.05, 0.15, -0.02, tau0],
            [0.08, 0.12, -0.03, tau0],
            [0.06, 0.10, -0.01, tau0 * 0.8],
            [0.10, 0.08, -0.02, tau0],
            [0.04, 0.18,  0.00, tau0 * 1.2],
            # seeds para curvas CER (β₁ fuertemente negativo)
            [0.07, -0.20,  0.05, tau0],
            [0.08, -0.18,  0.03, tau0 * 0.8],
            [0.06, -0.22,  0.04, tau0 * 1.2],
            [0.09, -0.17,  0.02, tau0],
            [0.07, -0.20,  0.06, tau0 * 0.6],
        ]
        tau_lo, tau_hi = self.bounds[3]
        while len(seeds) < n_restarts:
            seeds.append([
                rng.uniform(0.001, 0.25),
                rng.uniform(-0.20, 0.25),
                rng.uniform(-0.20, 0.20),
                rng.uniform(tau_lo, min(tau_hi, tau0 * 2)),
            ])
        return seeds

    def _make_seeds_nss(self, n_restarts: int, rng: np.random.Generator) -> list:
        """Seeds para NSS (6 params): [β₀, β₁, β₂, β₃, τ₁, τ₂]."""
        bnds = self.bounds
        t1_lo, t1_hi = bnds[4]
        t2_lo, t2_hi = bnds[5]
        t1_mid = (t1_lo + t1_hi) / 2
        t2_mid = (t2_lo + t2_hi) / 2
        seeds = [
            # Curva real ARS: tramo corto negativo, panza media, largo positivo
            [0.08, -0.20,  0.35, -0.05, t1_mid * 0.6, t2_mid],
            [0.09, -0.18,  0.30, -0.03, t1_mid * 0.5, t2_mid * 0.8],
            [0.07, -0.22,  0.40, -0.06, t1_mid * 0.8, t2_mid * 1.2],
            [0.08, -0.20,  0.28,  0.04, t1_mid * 0.7, t2_mid],
            [0.09, -0.17,  0.45, -0.07, t1_mid * 0.4, t2_mid * 0.9],
            [0.10, -0.20,  0.25,  0.02, t1_mid * 0.9, t2_mid * 1.1],
            [0.07, -0.23,  0.38, -0.04, t1_mid * 0.6, t2_mid * 1.3],
            [0.08, -0.19,  0.32,  0.06, t1_mid,        t2_mid],
            [0.09, -0.21,  0.42, -0.08, t1_mid * 0.5, t2_mid * 0.7],
            [0.08, -0.20,  0.20, -0.10, t1_mid * 0.8, t2_mid * 1.4],
            # Seeds más genéricas
            [0.05, -0.15,  0.20,  0.00, t1_lo * 1.5,  t2_lo * 1.2],
            [0.10, -0.25,  0.50, -0.10, t1_mid,       t2_hi * 0.8],
            [0.07, -0.18,  0.15,  0.10, t1_hi * 0.8,  t2_mid],
            [0.08, -0.20,  0.35, -0.05, t1_lo * 2.0,  t2_hi * 0.6],
            [0.09, -0.22,  0.30,  0.05, t1_mid * 0.3, t2_mid * 1.5],
        ]
        while len(seeds) < n_restarts:
            seeds.append([
                rng.uniform(bnds[0][0], bnds[0][1]),
                rng.uniform(bnds[1][0], bnds[1][1]),
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.25, 0.25),
                rng.uniform(t1_lo, t1_hi),
                rng.uniform(t2_lo, t2_hi),
            ])
        return seeds

    def _run_optimizer(
        self,
        loss_fn,
        loss_args: tuple,
        n_restarts: int,
        verbose: bool,
        label: str,
    ):
        rng   = np.random.default_rng(seed=42)
        seeds = self._make_seeds(n_restarts, rng)
        bnds  = self.bounds

        best_loss, best_result = np.inf, None

        if verbose:
            print(f"\n  Calibrando NS {label} ({n_restarts} restarts)...",
                  end="", flush=True)

        for x0 in seeds[:n_restarts]:
            try:
                res = minimize(
                    fun=loss_fn,
                    x0=x0,
                    args=loss_args,
                    method="L-BFGS-B",
                    bounds=bnds,
                    options={"maxiter": 10_000, "ftol": 1e-14, "gtol": 1e-9},
                )
                if np.isfinite(res.fun) and res.fun < best_loss:
                    best_loss, best_result = res.fun, res
            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("Calibración NS falló en todos los restarts.")

        if verbose:
            print(f" OK  (loss={best_loss:.6e})")

        return best_result

    def _store_result(self, result) -> None:
        self.params     = result.x
        self.fit_result = result
        n = self.n_params
        self.std_errors = np.full(n, np.nan)
        self.t_stats    = np.full(n, np.nan)
        self.ic_low     = np.full(n, np.nan)
        self.ic_high    = np.full(n, np.nan)
        try:
            hess_inv = result.hess_inv @ np.eye(n)
            cov  = hess_inv * 2
            for i, d in enumerate(np.diag(cov)):
                if d > 0:
                    se_i = float(np.sqrt(d))
                    self.std_errors[i] = se_i
                    self.t_stats[i]    = self.params[i] / se_i
                    self.ic_low[i]     = self.params[i] - 1.96 * se_i
                    self.ic_high[i]    = self.params[i] + 1.96 * se_i
        except Exception:
            pass

    def fit(
        self,
        bonds: List[Bond],
        market_prices: List[float],
        weights: Optional[List[float]] = None,
        n_restarts: int = 25,
        verbose: bool = False,
    ) -> Dict:
        """Calibra NS minimizando WLS sobre precios teóricos."""
        assert len(bonds) == len(market_prices)
        w_arr = np.ones(len(bonds)) if weights is None else np.array(weights)

        result = self._run_optimizer(
            loss_fn   = self._loss,
            loss_args = (bonds, market_prices, w_arr),
            n_restarts= n_restarts,
            verbose   = verbose,
            label     = f"sobre precios ({len(bonds)} bonos)",
        )
        self._store_result(result)

        metrics = self._compute_fit_errors(bonds, market_prices)
        if verbose:
            self._print_summary(metrics)
        return metrics

    def fit_from_yields(
        self,
        data: "pd.DataFrame",
        weight_map: Optional[Dict[str, float]] = None,
        spread_adjustments: Optional[Dict[str, float]] = None,
        n_restarts: int = 25,
        verbose: bool = False,
    ) -> Dict:
        """
        Calibra NS con WLS sobre yields observadas.

        Parámetros
        ----------
        data : DataFrame con columnas [ticker, maturity_years, yield, type].
               yield en decimales (ej: 0.08 = 8%).
        weight_map : {type → peso}. Si None usa _DEFAULT_TYPE_WEIGHTS.
        spread_adjustments : {ticker → bps}. Se resta de yield antes de
               calibrar (útil para limpiar spread BCRA de BOPREALs).
        """
        required = {"ticker", "maturity_years", "yield", "type"}
        missing  = required - set(data.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en data: {missing}")

        wmap = {**self._DEFAULT_TYPE_WEIGHTS, **(weight_map or {})}
        adj  = spread_adjustments or {}

        df = data.copy().reset_index(drop=True)
        df["yield_adj"] = df.apply(
            lambda r: r["yield"] - adj.get(r["ticker"], 0.0) / 1e4, axis=1,
        )
        df["weight"] = df["type"].map(lambda t: wmap.get(t, 1.0))

        maturities = df["maturity_years"].values.astype(float)
        yields_adj = df["yield_adj"].values.astype(float)
        weights    = df["weight"].values.astype(float)

        result = self._run_optimizer(
            loss_fn   = self._loss_yields,
            loss_args = (maturities, yields_adj, weights, self.forward_penalty_lambda),
            n_restarts= n_restarts,
            verbose   = verbose,
            label     = f"{'NSS' if self.use_nss else 'NS'} WLS yields ({len(df)} nodos)",
        )
        self._store_result(result)
        return self._compute_fit_errors_yields(df, adj)

    # ── Métricas de ajuste ────────────────────────────────────────────────────

    def _compute_fit_errors(self, bonds: List[Bond], market_prices: List[float]) -> Dict:
        bond_errors = []
        for bond, p_mkt in zip(bonds, market_prices):
            p_model = self._theoretical_price(bond, self.params)
            err_abs = p_mkt - p_model
            bond_errors.append({
                "ticker":            bond.ticker,
                "market_price":      p_mkt,
                "theoretical_price": p_model,
                "error_abs":         err_abs,
                "error_pct":         err_abs / p_mkt * 100,
                "maturity_years":    bond.maturity_years,
                "bond":              bond,
            })

        errors_abs = np.array([abs(e["error_abs"]) for e in bond_errors])
        errors_pct = np.array([abs(e["error_pct"]) for e in bond_errors])
        prices_mkt = np.array([e["market_price"] for e in bond_errors])
        prices_mod = np.array([e["theoretical_price"] for e in bond_errors])

        ss_res = float(np.sum((prices_mkt - prices_mod) ** 2))
        ss_tot = float(np.sum((prices_mkt - np.mean(prices_mkt)) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        return {
            "params":        dict(zip(self.param_names, self.params)),
            "bond_errors":   bond_errors,
            "rmse_price":    float(np.sqrt(np.mean(errors_abs ** 2))),
            "mae_pct":       float(np.mean(errors_pct)),
            "max_error_pct": float(np.max(errors_pct)),
            "max_error_abs": float(np.max(errors_abs)),
            "r2":            r2,
        }

    def _compute_fit_errors_yields(
        self,
        df: "pd.DataFrame",
        spread_adjustments: Dict[str, float],
    ) -> Dict:
        records = []
        for _, row in df.iterrows():
            t       = float(row["maturity_years"])
            y_obs   = float(row["yield"])
            y_adj   = float(row["yield_adj"])
            y_model = float(self.spot_rate(np.array([t]), self.params)[0])
            records.append({
                "ticker":         row["ticker"],
                "type":           row["type"],
                "weight":         float(row["weight"]),
                "maturity_years": t,
                "yield_obs":      y_obs,
                "yield_adj":      y_adj,
                "yield_model":    y_model,
                "error_bps":      (y_adj - y_model) * 1e4,
                "spread_adj_bps": spread_adjustments.get(row["ticker"], 0.0),
            })

        errors_bps = np.array([abs(r["error_bps"]) for r in records])
        yields_adj = np.array([r["yield_adj"] for r in records])
        yields_mod = np.array([r["yield_model"] for r in records])

        ss_res = float(np.sum((yields_adj - yields_mod) ** 2))
        ss_tot = float(np.sum((yields_adj - np.mean(yields_adj)) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        return {
            "params":        dict(zip(self.param_names, self.params)),
            "node_errors":   records,
            "rmse_bps":      float(np.sqrt(np.mean(errors_bps ** 2))),
            "mae_bps":       float(np.mean(errors_bps)),
            "max_error_bps": float(np.max(errors_bps)),
            "r2":            r2,
            # aliases para compatibilidad con print_stats_table
            "bond_errors":   [],
            "rmse_price":    0.0,
            "mae_pct":       0.0,
        }

    def _print_summary(self, metrics: Dict) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  CALIBRACIÓN NS — RESULTADOS")
        print(sep)
        for name, val in metrics["params"].items():
            if "τ" in name:
                print(f"    {name}  =  {val:8.4f}")
            else:
                print(f"    {name}  =  {val:8.4f}   ({val*100:+.2f}%)")
        print("-" * 65)
        print(f"  {'Ticker':<7} {'Vto(y)':>6} {'Mkt P':>8} {'Mod P':>8} {'Err$':>7} {'Err%':>7}")
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
        """Análisis extendido: parámetros, métricas, error TIR y sensibilidad τ."""
        sep  = "=" * 65
        dash = "-" * 65

        print(f"\n  ANÁLISIS DE PARÁMETROS NS:")
        print(f"  {'Param':<5} {'Valor':>8} {'Interpretación':<32} {'Rango':>14} {'% rango':>8}")
        print("  " + dash)
        for i, (name, val) in enumerate(metrics["params"].items()):
            lo, hi = self.bounds[i]
            pct_range  = (val - lo) / (hi - lo) * 100 if (hi - lo) > 0 else float("nan")
            interp     = self.param_interp[i]
            val_str    = f"{val:8.4f}"
            interp_str = f"{interp}: {val:.4f}" if "τ" in name else f"{interp}: {val*100:+.2f}%"
            rango_str  = f"[{lo:.2f}, {hi:.2f}]"
            print(f"  {name:<5} {val_str} {interp_str:<32} {rango_str:>14} {pct_range:>7.1f}%")

        print(f"\n  MÉTRICAS DE AJUSTE:")
        print(
            f"  R²: {metrics['r2']:.6f}  |  "
            f"RMSE: ${metrics['rmse_price']:.4f}  |  "
            f"MAE: {metrics['mae_pct']:.3f}%  |  "
            f"Max error: ${metrics['max_error_abs']:.4f}"
        )

        print(f"\n  ERROR EN TIR POR BONO:")
        print(f"  {'Ticker':<7} {'TIR Mkt':>10} {'TIR Mod':>10} {'Error (bps)':>12}")
        print("  " + dash)
        for e in metrics["bond_errors"]:
            ticker  = e["ticker"]
            tir_ref = tir_reference.get(ticker)
            tir_mod = self._ytm(e["bond"], e["theoretical_price"])
            if tir_ref is None or not np.isfinite(tir_mod):
                print(f"  {ticker:<7} {'N/D':>10} {'N/D':>10} {'N/D':>12}")
                continue
            err_bps = (tir_mod - tir_ref) * 1e4
            print(
                f"  {ticker:<7} {tir_ref*100:>9.2f}% {tir_mod*100:>9.2f}% {err_bps:>+11.1f} bps"
            )

        tau_base  = self.params[3]
        tau_minus = tau_base * 0.80
        tau_plus  = tau_base * 1.20

        print(f"\n  SENSIBILIDAD τ (±20%):")
        t_nodes = [1, 3, 5, 10]
        print(
            f"  {'t':>6} {'Spot base':>10} {'τ −20%':>10} {'τ +20%':>10} "
            f"{'Δbps (−)':>10} {'Δbps (+)':>10}"
        )
        print("  " + dash)
        for t in t_nodes:
            s_base  = self.get_spot_rate(t) * 100
            p_minus = self.params.copy(); p_minus[3] = tau_minus
            p_plus  = self.params.copy(); p_plus[3]  = tau_plus
            s_minus = float(self.spot_rate(np.array([t]), p_minus)[0]) * 100
            s_plus  = float(self.spot_rate(np.array([t]), p_plus)[0])  * 100
            print(
                f"  {t:>5}a {s_base:>9.2f}% {s_minus:>9.2f}% {s_plus:>9.2f}% "
                f"{(s_minus - s_base)*100:>+9.1f}  {(s_plus - s_base)*100:>+9.1f}"
            )
        print()

    # ── Getters ───────────────────────────────────────────────────────────────

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
        t    = np.linspace(0.01, t_max, n_points)
        spot = self.spot_rate(t, self.params)
        fwd  = self.forward_rate(t, self.params, dt=1.0)
        return pd.DataFrame({
            "years":          t,
            "spot_rate_pct":  spot * 100,
            "forward_1y_pct": fwd  * 100,
        })

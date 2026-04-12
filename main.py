import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src import NSSCurve, SpreadEngine, Visualizer, data_loader
from src.bond import Bond

# ── Parámetro principal ───────────────────────────────────────────────────────
CURVE = "USD"

# ── Configuración general ─────────────────────────────────────────────────────
SETTLEMENT_DATE = date(2026, 4, 10)
N_RESTARTS      = 25
CHART_STYLE     = "dark"

# ── Precios de mercado por tipo de curva ──────────────────────────────────────

# Globales USD — precio como % de face value nominal
MARKET_PRICES_USD = {
    "GD29": 65.49,
    "GD30": 64.64,
    "GD35": 79.90,
    "GD38": 83.10,
    "GD41": 74.17,
    "GD46": 70.99,
}

# BONCER — precio dirty en ARS nominal (CER-ajustado). Universo completo Apr-2026.
# Cupón cero  : X15Y6 (Lecer), TZX26, TZXM7 — flujo único al vencimiento.
# Cupón fijo  : TX26 (2%), TX28 (2.25%), TX31 (2.5%) — bullet.
# Amortizable : DICP (5.83%, 20 cuotas × 5% desde Jun-24), PARP (step-up, bullet).
MARKET_PRICES_CER = {
    "X15Y6":   105.53,
    "TZX26":   372.45,
    "TX26":   1333.00,
    "TZXM7":   202.00,
    "TX28":   1926.00,
    "TX31":   1381.00,
    "DICP":  49490.00,
    "PARP":  33790.00,
}

# Dollar Linked — universo extendido: soberanos + duales + BOPREALs
# Modelo: NS 4 parámetros, WLS sobre yields (fit_from_yields).
# Precios en ARS (face_value DL-ajustado en loader: soberanos=134500, duales≈157-160, BOPREALs propio).
MARKET_PRICES_DL = {
    # Soberanos DL (face_value = 100 USD × ARS/USD 1345 = 134500)
    "TZV27": 130840.0,
    "TZV28": 118300.0,
    "TV30":  134500.0,
    # Bonos Duales — proxy DL tramo corto (face_value ≈ ARS DL-ajustado desde emisión)
    "TTJ26":    154.20,
    "TTD26":    150.75,
    # BOPREALs BCRA — yield ajustada por BOPREAL_SPREAD_BPS antes del fit
    "BPY26":  55310.00,
}

# Spread BCRA a restar (bps) de la yield observada de BOPREALs
# para limpiar el riesgo BCRA y usarlos como proxy soberano DL.
BOPREAL_SPREAD_BPS = {
    "BPY26": 200.0,
}

# Nodo sintético opcional: tasa implícita ROFEX + bono CER para anclar 0.25y.
# Descomentar y ajustar con la tasa actual antes de ejecutar.
SYNTHETIC_NODES_DL: List[Dict] = [
    # {"ticker": "ROFEX_0.25", "maturity_years": 0.25, "yield": 0.18, "type": "synthetic"},
]

# ── Registro de curvas ────────────────────────────────────────────────────────
# Cada entrada: (bonds_loader, market_prices, ons_loader, viz_title, save_prefix)
# USD/CER → fit() sobre precios · DL → fit_from_yields() WLS sobre yields extendido.
CURVE_REGISTRY = {
    "USD": (
        data_loader.load_globales,
        MARKET_PRICES_USD,
        data_loader.load_sample_ons,
        f"Curva Soberana Argentina · Globales USD · {SETTLEMENT_DATE.strftime('%d/%m/%Y')}",
        "dashboard_usd",
    ),
    "CER": (
        data_loader.load_boncer,
        MARKET_PRICES_CER,
        data_loader.load_sample_ons_cer,
        f"Curva Soberana Argentina · CER (Tasa Real ARS) · {SETTLEMENT_DATE.strftime('%d/%m/%Y')}",
        "dashboard_cer",
    ),
    "DL": (
        data_loader.load_dl_extended,
        MARKET_PRICES_DL,
        data_loader.load_sample_ons_dollar_linked,
        f"Curva Soberana Argentina · Dollar Linked NS-WLS · {SETTLEMENT_DATE.strftime('%d/%m/%Y')}",
        "dashboard_dl",
    ),
}

_CURVE_LABELS = {
    "USD": "GLOBALES USD",
    "CER": "CER (TASA REAL ARS)",
    "DL":  "DOLLAR LINKED (NS-WLS)",
}

# Monedas válidas para ONs por curva — evita comparar DL contra CER, etc.
_VALID_ON_CURRENCIES: Dict[str, set] = {
    "USD": {"USD"},
    "CER": {"ARS_CER", "ARS_UVA"},
    "DL":  {"ARS_DL"},
}

# Bounds del optimizador NS por curva
_CURVE_NS_BOUNDS: Dict[str, Dict] = {
    "USD": dict(beta0_min=0.08, beta0_max=0.14, tau_max=10.0),
    # CER: modelo NSS (6 params).
    # X15Y6 y TZX26 se grafican como referencia (peso=0) pero NO calibran la curva.
    # β₀ ∈ [7%,10%]: ancla el largo plazo (DICP ~6.4%, PARP ~3.5% sugieren LP > 7%).
    # β₁ ∈ [-25%,-15%]: fuerza inicio negativo — s(0) = β₀+β₁ ∈ [-18%,-5%].
    # τ₁ ∈ [0.20, 1.50]: escala del tramo corto.
    # τ₂ ∈ [3.00, 8.00]: escala del tramo largo (DICP/PARP hump).
    # forward_penalty_lambda = 0.08: suaviza picos de forward.
    "CER": dict(
        use_nss=True,
        beta0_min=0.07,  beta0_max=0.10,
        beta1_min=-0.25, beta1_max=-0.15,
        tau1_min=0.20,   tau1_max=1.50,
        tau2_min=3.00,   tau2_max=8.00,
        forward_penalty_lambda=0.08,
    ),
}


def _mod_duration(bond: Bond, price: float) -> float:
    """Duración modificada aproximada via YTM de los flujos del bono."""
    ytm = NSSCurve._ytm(bond, price)
    if not np.isfinite(ytm) or abs(1.0 + ytm) < 1e-6:
        return bond.maturity_years
    cf    = bond.cash_flows
    times = cf["years"].values
    flows = cf["total_cf"].values
    pvs   = flows / (1.0 + ytm) ** times
    total = pvs.sum()
    if total <= 0:
        return bond.maturity_years
    mac = float((times * pvs).sum() / total)
    return mac / (1.0 + ytm)


_TYPE_ABBREV = {"sovereign": "S", "dual": "D", "bopreal": "B", "synthetic": "X"}


def print_stats_table(
    nss: NSSCurve,
    on_results: List[Dict],
    curve_key: str,
    settlement: date,
    fit_metrics: Dict,
    spread_adjustments: Optional[Dict[str, float]] = None,
) -> None:
    sep      = "=" * 66
    label    = _CURVE_LABELS.get(curve_key, curve_key)
    date_str = settlement.strftime("%d/%m/%Y")
    adj      = spread_adjustments or {}
    nodes    = fit_metrics.get("node_errors", [])

    print(f"\n{sep}")
    print(f"  CURVA SPOT NS — {label} — {date_str}")
    print(sep)

    # ── Nodos utilizados (con tipo abreviado) ─────────────────────────
    node_strs = "  ".join(
        f"{r['ticker']}({r['maturity_years']:.2f}y,{_TYPE_ABBREV.get(r['type'], r['type'][0].upper())})"
        for r in nodes
    )
    print(f"\n  NODOS UTILIZADOS: {node_strs}")

    # ── Parámetros NS ─────────────────────────────────────────────────
    print(f"\n  PARÁMETROS NS:")
    for i, name in enumerate(nss.param_names):
        val    = nss.params[i]
        se     = nss.std_errors[i]
        se_str = f"{se:.4f}" if np.isfinite(se) else "N/A"
        if "τ" in name:
            print(f"    {name:<5} = {val:8.4f}   SE={se_str}")
        else:
            print(f"    {name:<5} = {val:8.4f} ({val*100:+.2f}%)   SE={se_str}")

    # ── Métricas ──────────────────────────────────────────────────────
    r2 = fit_metrics["r2"]
    is_yield_fit = fit_metrics.get("rmse_price", 0.0) == 0.0 and fit_metrics.get("rmse_bps") is not None
    print(f"\n  MÉTRICAS ({'espacio yields' if is_yield_fit else 'espacio precios'}):")
    if is_yield_fit:
        print(f"  RMSE: {fit_metrics['rmse_bps']:.1f} bps  |  MAE: {fit_metrics['mae_bps']:.1f} bps  |  R²: {r2:.6f}")
    else:
        print(f"  RMSE: ${fit_metrics['rmse_price']:.4f}  |  MAE%: {fit_metrics['mae_pct']:.3f}%  |  R²: {r2:.6f}")

    # ── Tabla de errores por nodo (formato unificado) ─────────────────
    print(f"\n  {'Ticker':<8} {'Tipo':<10} {'Vto(y)':>6} "
          f"{'Yield obs':>10} {'Yield adj':>10} {'Mod':>9} {'Error':>8} {'WAdj':>5}")
    print("  " + "-" * 70)
    for r in nodes:
        sp     = adj.get(r["ticker"], 0.0)
        sp_str = f"−{sp:.0f}" if sp else "—"
        y_obs  = r["yield_obs"]
        y_adj  = r.get("yield_adj", y_obs)
        y_mod  = r["yield_model"]
        err    = r.get("error_bps", (y_adj - y_mod) * 1e4)
        print(
            f"  {r['ticker']:<8} {r['type']:<10} {r['maturity_years']:>6.2f} "
            f"{y_obs*100:>9.2f}% {y_adj*100:>9.2f}% "
            f"{y_mod*100:>8.2f}% {err:>+7.1f}bps "
            f"{sp_str:>5}"
        )

    # ── Z-Spreads ─────────────────────────────────────────────────────
    valid_ons = [r for r in on_results if np.isfinite(r.get("z_spread_bps", float("nan")))]
    if valid_ons:
        print(f"\n  Z-SPREADS ONs:")
        for r in valid_ons:
            print(f"  {r['ticker']:<8}  {r['z_spread_bps']:+.0f} bps")

    print(f"\n{sep}")


def _run_curve_section(
    curve_key: str,
    bonds_dict: Dict[str, Bond],
    market_prices: Dict[str, float],
    ons_loader: Callable,
    settlement: date,
    viz_title: str,
    save_prefix: str,
    save_charts: bool,
    use_yield_fit: bool = False,
    # NS bounds
    tau_min:      Optional[float] = None,
    tau_max:      Optional[float] = None,
    beta0_max:    Optional[float] = None,
    beta0_min:    Optional[float] = None,
    beta1_max:    Optional[float] = None,
    beta1_min:    Optional[float] = None,
    beta2_min:    Optional[float] = None,
    beta2_max:    Optional[float] = None,
    # NSS mode
    use_nss:      bool = False,
    tau1_min:     Optional[float] = None,
    tau1_max:     Optional[float] = None,
    tau2_min:     Optional[float] = None,
    tau2_max:     Optional[float] = None,
    forward_penalty_lambda: float = 0.0,
    per_bond_weights: Optional[Dict[str, float]] = None,
) -> None:
    # 1. Generar flujos
    bonds, prices = [], []
    for ticker, price in market_prices.items():
        if ticker not in bonds_dict:
            continue
        bond = bonds_dict[ticker]
        bond.generate_cash_flow_schedule()
        bonds.append(bond)
        prices.append(price)

    # 2. Calibrar curva NS / NSS
    nss = NSSCurve(
        use_nss=use_nss,
        tau_min=tau_min,   tau_max=tau_max,
        beta0_min=beta0_min, beta0_max=beta0_max,
        beta1_min=beta1_min, beta1_max=beta1_max,
        beta2_min=beta2_min, beta2_max=beta2_max,
        tau1_min=tau1_min, tau1_max=tau1_max,
        tau2_min=tau2_min, tau2_max=tau2_max,
        forward_penalty_lambda=forward_penalty_lambda,
    )
    if use_yield_fit:
        # Yield-based WLS: escala uniforme (todos en %), evita dominancia por nivel de precio
        # per_bond_weights permite asignar pesos distintos por ticker (ej: más peso a bonos líquidos)
        if per_bond_weights:
            instrument_types = {b.ticker: f"_bond_{b.ticker}" for b in bonds}
            weight_map = {f"_bond_{t}": w for t, w in per_bond_weights.items()}
        else:
            instrument_types = {b.ticker: "sovereign" for b in bonds}
            weight_map = None
        yield_df = data_loader.build_dl_yield_dataframe(
            bonds            = {b.ticker: b for b in bonds},
            prices           = dict(zip([b.ticker for b in bonds], prices)),
            instrument_types = instrument_types,
        )
        fit_metrics = nss.fit_from_yields(data=yield_df, weight_map=weight_map, n_restarts=N_RESTARTS)
        # Remap internal _bond_<ticker> types back to "sovereign" for display
        for r in fit_metrics.get("node_errors", []):
            if r.get("type", "").startswith("_bond_"):
                r["type"] = "sovereign"
    else:
        # WLS con pesos 1/Duración modificada — bonos cortos no distorsionan el tramo largo
        durations   = [_mod_duration(b, p) for b, p in zip(bonds, prices)]
        wls_weights = [1.0 / max(d, 0.25) for d in durations]
        fit_metrics = nss.fit(
            bonds=bonds, market_prices=prices, weights=wls_weights, n_restarts=N_RESTARTS
        )
        # Adjuntar node_errors en formato yield para visualización consistente con DL/CER
        fit_metrics["node_errors"] = [
            {
                "ticker":         bond.ticker,
                "type":           "sovereign",
                "maturity_years": bond.maturity_years,
                "yield_obs":      NSSCurve._ytm(bond, price),
                "yield_adj":      NSSCurve._ytm(bond, price),
                "yield_model":    nss.get_spot_rate(bond.maturity_years),
                "error_bps":      (NSSCurve._ytm(bond, price) - nss.get_spot_rate(bond.maturity_years)) * 1e4,
            }
            for bond, price in zip(bonds, prices)
        ]

    # 3. Z-Spreads de ONs — solo si la moneda del bono coincide con el benchmark
    engine           = SpreadEngine(nss)
    ons_data         = ons_loader(settlement=settlement)
    valid_currencies = _VALID_ON_CURRENCIES.get(curve_key, set())

    on_results = []
    for ticker, (bond, price) in ons_data.items():
        bond.generate_cash_flow_schedule()
        if bond.currency not in valid_currencies:
            print(f"  [SKIP] {bond.ticker} ({bond.currency}) — no corresponde a curva {curve_key}")
            continue
        if bond.currency == "USD":
            # ONs amortizables USD: precio cotizado como % del valor técnico residual
            residual = sum(
                float(row["amort_pct"])
                for _, row in bond.amortization_schedule.iterrows()
                if row["date"] > bond.settlement_date
            )
            effective_price = price * residual if residual > 0 else price
        else:
            effective_price = price
        on_results.append(engine.z_spread(bond, effective_price))

    # 4. Gráfico — mismo visualizador que DL para consistencia visual
    _t_max  = {"USD": 22.0, "CER": 15.0}.get(curve_key, 10.0)
    _y_lim  = (-15.0, 10.0) if curve_key == "CER" else None
    viz = Visualizer(style=CHART_STYLE)
    save_path = f"{save_prefix}.png" if save_charts else None
    viz.plot_dl_curve(
        nss_curve          = nss,
        fit_metrics        = fit_metrics,
        on_results         = on_results,
        spread_adjustments = None,
        title              = viz_title,
        save_path          = save_path,
        t_max              = _t_max,
        y_lim              = _y_lim,
    )

    # 5. Tabla estadística
    print_stats_table(
        nss        = nss,
        on_results = on_results,
        curve_key  = curve_key,
        settlement = settlement,
        fit_metrics= fit_metrics,
    )


def _run_dl_wls_section(
    settlement: date,
    bonds_loader: Callable,
    market_prices: Dict[str, float],
    ons_loader: Callable,
    viz_title: str,
    save_prefix: str,
    save_charts: bool,
    bopreal_spread_bps: Dict[str, float],
    synthetic_nodes: List[Dict],
) -> None:
    """Flujo DL: universo extendido (soberanos+duales+BOPREALs), NS-WLS sobre yields."""

    # 1. Construir bonos y DataFrame de yields
    bonds_ext = bonds_loader(settlement=settlement)
    yield_df  = data_loader.build_dl_yield_dataframe(
        bonds           = bonds_ext,
        prices          = market_prices,
        synthetic_nodes = synthetic_nodes or [],
    )

    # 2. Calibrar NS con WLS sobre yields ajustadas
    nss = NSSCurve()
    fit_metrics = nss.fit_from_yields(
        data                = yield_df,
        spread_adjustments  = bopreal_spread_bps,
        n_restarts          = N_RESTARTS,
    )

    # 3. Z-Spreads de ONs — solo moneda ARS_DL
    engine   = SpreadEngine(nss)
    ons_data = ons_loader(settlement=settlement)

    on_results = []
    for ticker, (bond, price) in ons_data.items():
        bond.generate_cash_flow_schedule()
        if bond.currency != "ARS_DL":
            print(f"  [SKIP] {bond.ticker} ({bond.currency}) — no corresponde a curva DL")
            continue
        on_results.append(engine.z_spread(bond, price))

    # 4. Gráfico DL
    viz = Visualizer(style=CHART_STYLE)
    save_path = f"{save_prefix}.png" if save_charts else None
    viz.plot_dl_curve(
        nss_curve          = nss,
        fit_metrics        = fit_metrics,
        on_results         = on_results,
        spread_adjustments = bopreal_spread_bps,
        title              = viz_title,
        save_path          = save_path,
    )

    # 5. Tabla estadística en consola
    print_stats_table(
        nss                = nss,
        on_results         = on_results,
        curve_key          = "DL",
        settlement         = settlement,
        fit_metrics        = fit_metrics,
        spread_adjustments = bopreal_spread_bps,
    )


def main(curve: str = "ALL", save_charts: bool = False) -> None:
    import matplotlib.pyplot as plt

    curve = curve.upper()
    keys  = list(CURVE_REGISTRY.keys())
    if curve == "ALL":
        curves_to_run = keys
    elif curve in keys:
        curves_to_run = [curve]
    else:
        raise ValueError(f"CURVE inválido: '{curve}'. Opciones: {keys} | ALL")

    for key in curves_to_run:
        bonds_loader, market_prices, ons_loader, viz_title, save_prefix = CURVE_REGISTRY[key]
        if key == "DL":
            _run_dl_wls_section(
                settlement         = SETTLEMENT_DATE,
                bonds_loader       = bonds_loader,
                market_prices      = market_prices,
                ons_loader         = ons_loader,
                viz_title          = viz_title,
                save_prefix        = save_prefix,
                save_charts        = save_charts,
                bopreal_spread_bps = BOPREAL_SPREAD_BPS,
                synthetic_nodes    = SYNTHETIC_NODES_DL,
            )
        else:
            _bounds = _CURVE_NS_BOUNDS.get(key, {})
            _per_bond_w = {
                # Peso 0 → graficados como referencia pero excluidos del fit NSS.
                # Sus yields muy negativas distorsionan la forma de la curva calibrada.
                "X15Y6": 0.0, "TZX26": 0.0,
                # tramo corto activo: peso 3x para anclar la zona 0.5–1y
                "TX26":  3.0, "TZXM7": 3.0,
                # BONCER estándar tramo medio
                "TX28":  1.0, "TX31":  1.0,
                # amortizable / step-up: peso 1.5x — definen costo de capital LP
                "DICP":  1.5, "PARP":  1.5,
            } if key == "CER" else None
            _run_curve_section(
                curve_key        = key,
                bonds_dict       = bonds_loader(settlement=SETTLEMENT_DATE),
                market_prices    = market_prices,
                ons_loader       = ons_loader,
                settlement       = SETTLEMENT_DATE,
                viz_title        = viz_title,
                save_prefix      = save_prefix,
                save_charts      = save_charts,
                use_yield_fit    = (key == "CER"),
                per_bond_weights = _per_bond_w,
                **_bounds,
            )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve", default=CURVE, choices=["USD", "CER", "DL", "ALL"],
        help="Curva: USD | CER | DL | ALL  (default: %(default)s)",
    )
    parser.add_argument("--save", action="store_true", help="Guardar gráfico como PNG.")
    args = parser.parse_args()
    main(curve=args.curve, save_charts=args.save)

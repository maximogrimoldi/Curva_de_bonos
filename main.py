import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List

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
# Curva implícita ~18% corto plazo → ~8% largo plazo
MARKET_PRICES_USD = {
    "GD29": 81.54,
    "GD30": 87.59,
    "GD35": 67.34,
    "GD38": 63.59,
    "GD41": 59.01,
    "GD46": 50.39,
}

# BONCER — precio como % del valor técnico (capital ajustado por CER); tasa = real ARS
# Curva implícita ~10% corto plazo → ~4% largo plazo (tasa real)
MARKET_PRICES_CER = {
    "TX26": 96.65,
    "TX28": 90.57,
    "TX30": 88.79,
    "DICP": 90.33,
    "PARP": 80.79,
}

# Dollar Linked — precio como % del valor técnico (capital ajustado por TC oficial)
# Curva implícita ~8% corto plazo → ~3% largo plazo
MARKET_PRICES_DL = {
    "TV26D": 96.61,
    "TV27":  92.30,
    "TV28":  90.56,
    "TV30":  88.72,
}

# ── Registro de curvas ────────────────────────────────────────────────────────
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
        data_loader.load_dollar_linked,
        MARKET_PRICES_DL,
        data_loader.load_sample_ons_dollar_linked,
        f"Curva Soberana Argentina · Dollar Linked · {SETTLEMENT_DATE.strftime('%d/%m/%Y')}",
        "dashboard_dl",
    ),
}

_CURVE_LABELS = {
    "USD": "GLOBALES USD",
    "CER": "CER (TASA REAL ARS)",
    "DL":  "DOLLAR LINKED",
}


def print_stats_table(
    nss: NSSCurve,
    bonds: List[Bond],
    on_results: List[Dict],
    curve_key: str,
    settlement: date,
    fit_metrics: Dict,
) -> None:
    sep = "=" * 66
    label = _CURVE_LABELS.get(curve_key, curve_key)
    date_str = settlement.strftime("%d/%m/%Y")

    print(f"\n{sep}")
    print(f"  CURVA SPOT NSS — {label} — {date_str}")
    print(sep)

    # Bonos usados con su vencimiento
    bond_strs = "  ".join(
        f"{b.ticker} ({b.maturity_years:.1f}y)" for b in bonds
    )
    print(f"\n  BONOS UTILIZADOS: {bond_strs}")

    # ── Tabla de parámetros ───────────────────────────────────────────
    param_names  = NSSCurve.PARAM_NAMES
    param_interp = [
        "Tasa largo plazo",
        "Pendiente (corto−largo)",
        "Curvatura tramo medio",
        "Curvatura tramo largo",
        "Hump 1 aparece en",
        "Hump 2 aparece en",
    ]

    print(f"\n  PARÁMETROS ESTIMADOS:")
    print(
        f"  ┌{'─'*10}┬{'─'*10}┬{'─'*12}┬{'─'*11}┬{'─'*16}┬{'─'*36}┐"
    )
    print(
        f"  │ {'Param':<8} │ {'Valor':>8} │ {'Std Error':>10} │ {'t-stat':>9} │ {'IC 95%':>14} │ {'Interpretación':<34} │"
    )
    print(
        f"  ├{'─'*10}┼{'─'*10}┼{'─'*12}┼{'─'*11}┼{'─'*16}┼{'─'*36}┤"
    )

    for i, name in enumerate(param_names):
        val  = nss.params[i]
        se   = nss.std_errors[i]
        tstat = nss.t_stats[i]
        lo   = nss.ic_low[i]
        hi   = nss.ic_high[i]

        val_str   = f"{val:.4f}"
        se_str    = f"{se:.4f}"   if np.isfinite(se)    else "N/A"
        ts_str    = f"{tstat:.1f}" if np.isfinite(tstat) else "N/A"

        if np.isfinite(lo) and np.isfinite(hi):
            ic_str = f"[{lo:.3f},{hi:.3f}]"
        else:
            ic_str = "N/A"

        if "τ" in name:
            interp_str = f"{param_interp[i]}: {val:.2f} años"
        else:
            interp_str = f"{param_interp[i]}: {val*100:+.2f}%"

        print(
            f"  │ {name:<8} │ {val_str:>8} │ {se_str:>10} │ {ts_str:>9} │ {ic_str:>14} │ {interp_str:<34} │"
        )

    print(
        f"  └{'─'*10}┴{'─'*10}┴{'─'*12}┴{'─'*11}┴{'─'*16}┴{'─'*36}┘"
    )

    # ── Métricas de ajuste ────────────────────────────────────────────
    r2   = fit_metrics["r2"]
    rmse = fit_metrics["rmse_price"]
    mae  = fit_metrics["mae_pct"]
    n    = len(bonds)
    print(f"\n  MÉTRICAS DE AJUSTE:")
    print(f"  RMSE precio: ${rmse:.4f}  |  MAE%: {mae:.3f}%  |  R²: {r2:.6f}  |  Bonos: {n}")

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

    # 2. Calibrar curva NSS (silencioso)
    nss = NSSCurve()
    fit_metrics = nss.fit(bonds=bonds, market_prices=prices, n_restarts=N_RESTARTS)

    # 3. Z-Spreads de ONs
    engine   = SpreadEngine(nss)
    ons_data = ons_loader(settlement=settlement)
    on_bonds  = [v[0] for v in ons_data.values()]
    on_prices = [v[1] for v in ons_data.values()]

    on_results = []
    for bond, price in zip(on_bonds, on_prices):
        on_results.append(engine.z_spread(bond, price))

    # 4. Gráfico (2 paneles)
    viz = Visualizer(style=CHART_STYLE)
    save_path = f"{save_prefix}.png" if save_charts else None
    viz.plot_dashboard(
        nss_curve=nss,
        on_results=on_results,
        title=viz_title,
        save_path=save_path,
    )

    # 5. Tabla estadística
    print_stats_table(
        nss=nss,
        bonds=bonds,
        on_results=on_results,
        curve_key=curve_key,
        settlement=settlement,
        fit_metrics=fit_metrics,
    )


def main(curve: str = "ALL", save_charts: bool = False) -> None:
    import matplotlib.pyplot as plt

    curve = curve.upper()
    if curve == "ALL":
        curves_to_run = list(CURVE_REGISTRY.keys())
    elif curve in CURVE_REGISTRY:
        curves_to_run = [curve]
    else:
        raise ValueError(f"CURVE inválido: '{curve}'. Opciones: {list(CURVE_REGISTRY.keys())} | ALL")

    for key in curves_to_run:
        bonds_loader, market_prices, ons_loader, viz_title, save_prefix = CURVE_REGISTRY[key]
        _run_curve_section(
            curve_key=key,
            bonds_dict=bonds_loader(settlement=SETTLEMENT_DATE),
            market_prices=market_prices,
            ons_loader=ons_loader,
            settlement=SETTLEMENT_DATE,
            viz_title=viz_title,
            save_prefix=save_prefix,
            save_charts=save_charts,
        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve", default="ALL", choices=["USD", "CER", "DL", "ALL"],
        help="Tipo de curva a correr: USD | CER | DL | ALL  (default: %(default)s)",
    )
    parser.add_argument("--save", action="store_true", help="Guardar gráfico como PNG.")
    args = parser.parse_args()
    main(curve=args.curve, save_charts=args.save)

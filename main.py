import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Dict

sys.path.insert(0, str(Path(__file__).parent))

from src import NSSCurve, SpreadEngine, Visualizer, data_loader
from src.bond import Bond

# ── Parámetro principal ───────────────────────────────────────────────────────
# Opciones: "USD" | "CER" | "DL"
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
# Cada entrada: (bonds_loader, market_prices, ons_loader, viz_title, save_prefix)
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


def _run_curve_section(
    section_name: str,
    bonds_dict: Dict[str, Bond],
    market_prices: Dict[str, float],
    ons_loader: Callable,
    settlement: date,
    viz_title: str,
    save_prefix: str,
    save_charts: bool,
) -> None:
    print(f"\n{'='*65}")
    print(f"  {section_name}")
    print(f"  Settlement: {settlement.strftime('%d/%m/%Y')} | Bonos: {list(market_prices.keys())}")
    print(f"{'='*65}")

    # 1. Generar flujos
    bonds, prices = [], []
    for ticker, price in market_prices.items():
        if ticker not in bonds_dict:
            print(f"  WARN: {ticker} no encontrado. Omitiendo.")
            continue
        bond = bonds_dict[ticker]
        bond.generate_cash_flow_schedule()
        bonds.append(bond)
        prices.append(price)

    # 2. Calibrar curva NSS
    nss = NSSCurve()
    nss.fit(bonds=bonds, market_prices=prices, n_restarts=N_RESTARTS, verbose=True)

    # 3. Tabla spot / forward en nodos clave
    print(f"\n  Curva Spot NSS — {section_name}:")
    nodos = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    print(f"  {'t(años)':>8} {'Spot (%)':>10} {'Forward 1Y (%)':>15}")
    print("  " + "-" * 36)
    for t in nodos:
        s = nss.get_spot_rate(t) * 100
        f = nss.get_forward_rate(t, dt=1.0) * 100
        print(f"  {t:>8.1f} {s:>10.4f} {f:>15.4f}")

    # 4. Z-Spreads de ONs
    print(f"\n  Z-Spreads de ONs — {section_name}:")
    engine = SpreadEngine(nss)
    ons_data  = ons_loader(settlement=settlement)
    on_bonds  = [v[0] for v in ons_data.values()]
    on_prices = [v[1] for v in ons_data.values()]

    on_results = []
    for bond, price in zip(on_bonds, on_prices):
        on_results.append(engine.z_spread(bond, price))

    df_spreads = engine.spread_table(on_bonds, on_prices)
    print(df_spreads.to_string(index=False))

    # 5. Dashboard
    viz = Visualizer(style=CHART_STYLE)
    save_path = f"{save_prefix}.png" if save_charts else None
    viz.plot_dashboard(
        nss_curve=nss,
        on_results=on_results,
        title=viz_title,
        save_path=save_path,
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
            section_name=f"CURVA SOBERANA — {key}",
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

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src import NSSCurve, SpreadEngine, Visualizer, data_loader

# ── Configuración ─────────────────────────────────────────────────────────────
SETTLEMENT_DATE = date(2026, 4, 10)

MARKET_PRICES = {
    "GD29": 71.50,
    "GD30": 68.00,
    "GD35": 62.00,
    "GD38": 60.00,
    "GD41": 58.50,
    "GD46": 56.00,
}

N_RESTARTS = 25
CHART_STYLE = "dark"


def main(save_charts: bool = False) -> None:
    print(f"\n  Settlement: {SETTLEMENT_DATE.strftime('%d/%m/%Y')} | Bonos: {list(MARKET_PRICES.keys())}\n")

    # 1. Cargar Globales y generar flujos
    globales_dict = data_loader.load_globales(settlement=SETTLEMENT_DATE)
    bonds, prices = [], []
    for ticker, price in MARKET_PRICES.items():
        if ticker not in globales_dict:
            print(f"  WARN: {ticker} no encontrado. Omitiendo.")
            continue
        bond = globales_dict[ticker]
        bond.generate_cash_flow_schedule()
        bonds.append(bond)
        prices.append(price)

    # 2. Calibrar curva NSS
    nss = NSSCurve()
    nss.fit(bonds=bonds, market_prices=prices, n_restarts=N_RESTARTS, verbose=True)

    # 3. Tasas spot en nodos clave
    print("  Curva Spot NSS:")
    nodos = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    print(f"  {'t(años)':>8} {'Spot (%)':>10} {'Forward 1Y (%)':>15}")
    print("  " + "-" * 36)
    for t in nodos:
        s = nss.get_spot_rate(t) * 100
        f = nss.get_forward_rate(t, dt=1.0) * 100
        print(f"  {t:>8.1f} {s:>10.4f} {f:>15.4f}")

    # 4. Z-Spreads de ONs
    print("\n  Z-Spreads de ONs corporativas:")
    engine = SpreadEngine(nss)
    on_data = data_loader.load_sample_ons(settlement=SETTLEMENT_DATE)
    on_bonds  = [v[0] for v in on_data.values()]
    on_prices = [v[1] for v in on_data.values()]

    on_results = []
    for bond, price in zip(on_bonds, on_prices):
        res = engine.z_spread(bond, price)
        on_results.append(res)

    df_spreads = engine.spread_table(on_bonds, on_prices)
    print(df_spreads.to_string(index=False))

    # 5. Gráfico
    viz = Visualizer(style=CHART_STYLE)
    save_path = "dashboard_curva_soberana.png" if save_charts else None
    viz.plot_dashboard(
        nss_curve=nss,
        on_results=on_results,
        title=f"Curva Soberana Argentina · Globales USD · {SETTLEMENT_DATE.strftime('%d/%m/%Y')}",
        save_path=save_path,
    )

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Guardar gráfico como PNG.")
    args = parser.parse_args()
    main(save_charts=args.save)

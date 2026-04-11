from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .nss_curve import NSSCurve


class Visualizer:

    _DARK_PALETTE = {
        "bg":       "#0D1117",
        "fg":       "#E6EDF3",
        "curve":    "#58A6FF",
        "forward":  "#F0883E",
        "on_pos":   "#3FB950",
        "on_neg":   "#FF7B72",
        "grid":     "#21262D",
        "zero_line":"#8B949E",
        "panel_bg": "#161B22",
    }

    _LIGHT_PALETTE = {
        "bg":       "#FFFFFF",
        "fg":       "#24292F",
        "curve":    "#0969DA",
        "forward":  "#E36209",
        "on_pos":   "#1A7F37",
        "on_neg":   "#CF222E",
        "grid":     "#D0D7DE",
        "zero_line":"#6E7781",
        "panel_bg": "#F6F8FA",
    }

    def __init__(self, style: str = "dark") -> None:
        if style not in ("dark", "light"):
            raise ValueError("style debe ser 'dark' o 'light'.")
        self.pal = self._DARK_PALETTE if style == "dark" else self._LIGHT_PALETTE
        self._apply_style()

    def _apply_style(self) -> None:
        plt.rcParams.update({
            "figure.facecolor":  self.pal["bg"],
            "axes.facecolor":    self.pal["panel_bg"],
            "axes.edgecolor":    self.pal["grid"],
            "axes.labelcolor":   self.pal["fg"],
            "xtick.color":       self.pal["fg"],
            "ytick.color":       self.pal["fg"],
            "text.color":        self.pal["fg"],
            "grid.color":        self.pal["grid"],
            "grid.linewidth":    0.6,
            "legend.facecolor":  self.pal["panel_bg"],
            "legend.edgecolor":  self.pal["grid"],
            "legend.framealpha": 0.9,
            "font.family":       "monospace",
            "axes.titleweight":  "bold",
            "axes.titlesize":    12,
            "axes.labelsize":    10,
        })

    def plot_dashboard(
        self,
        nss_curve: NSSCurve,
        on_results: Optional[List[Dict]] = None,
        title: str = "Curva Soberana Argentina · Globales USD",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        curve_df = nss_curve.curve_dataframe(t_max=20.0, n_points=600)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5), facecolor=self.pal["bg"])
        fig.suptitle(title, fontsize=14, fontweight="bold", color=self.pal["fg"], y=1.01)

        # ── Panel izquierdo: Curva Spot + Forward ─────────────────────
        ax0.plot(curve_df["years"], curve_df["spot_rate_pct"],
                 color=self.pal["curve"], linewidth=2.5, label="Curva Spot NSS")
        ax0.plot(curve_df["years"], curve_df["forward_1y_pct"],
                 color=self.pal["forward"], linewidth=1.6, linestyle="--",
                 alpha=0.85, label="Tasa Forward 1Y")

        b = nss_curve.params
        ax0.text(
            0.02, 0.04,
            f"β₀={b[0]*100:.2f}%  β₁={b[1]*100:.2f}%  β₂={b[2]*100:.2f}%  "
            f"β₃={b[3]*100:.2f}%  τ₁={b[4]:.2f}  τ₂={b[5]:.2f}",
            transform=ax0.transAxes, fontsize=7.5, color=self.pal["zero_line"], style="italic",
        )
        ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax0.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax0.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
        ax0.set_xlabel("Plazo (años)")
        ax0.set_ylabel("Tasa (%)")
        ax0.set_title("Curva Spot NSS")
        ax0.set_xlim(left=0)
        ax0.legend(fontsize=9)
        ax0.grid(True, which="major", alpha=0.7, linewidth=0.8)
        ax0.grid(True, which="minor", alpha=0.3, linewidth=0.4)

        # ── Panel derecho: Z-Spreads ──────────────────────────────────
        ax1.set_title("Z-Spread vs. Curva Soberana")
        if not on_results:
            ax1.text(0.5, 0.5, "Sin ONs cargadas", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=11, color=self.pal["zero_line"])
        else:
            valid = [r for r in on_results if np.isfinite(r.get("z_spread_bps", float("nan")))]
            if valid:
                tickers = [r["ticker"] for r in valid]
                spreads = [r["z_spread_bps"] for r in valid]
                bar_colors = [self.pal["on_pos"] if s >= 0 else self.pal["on_neg"] for s in spreads]
                bars = ax1.bar(tickers, spreads, color=bar_colors,
                               edgecolor=self.pal["fg"], linewidth=0.4, alpha=0.85, width=0.55)
                for bar, spread in zip(bars, spreads):
                    offset = 8 if spread >= 0 else -15
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                             f"{spread:+.0f}", ha="center", va="bottom",
                             fontsize=9, fontweight="bold", color=self.pal["fg"])
                ax1.axhline(y=0, color=self.pal["zero_line"], linewidth=0.8, alpha=0.7)
                ax1.set_xlabel("ON")
                ax1.set_ylabel("Z-Spread (bps)")
                ax1.grid(True, alpha=0.4, axis="y")
                ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f"))

        fig.text(0.99, -0.02,
                 f"Generado {date.today().strftime('%d/%m/%Y')} | NSS Framework · Argentina Fixed Income",
                 ha="right", fontsize=7, color=self.pal["zero_line"], alpha=0.7)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=self.pal["bg"])
            print(f"  Gráfico guardado: {save_path}")

        return fig

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
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

    def plot_dl_curve(
        self,
        nss_curve: NSSCurve,
        fit_metrics: Dict,
        on_results: Optional[List[Dict]] = None,
        spread_adjustments: Optional[Dict[str, float]] = None,
        title: str = "Curva Dollar Linked · NS-WLS",
        save_path: Optional[str] = None,
        t_max: float = 6.0,
        y_lim: Optional[tuple] = None,
    ) -> "plt.Figure":
        """
        Dashboard unificado para cualquier curva NS calibrada con fit_from_yields() o fit().

        Panel izquierdo : nodos observados (por tipo) + NS spot + forward 1Y.
        Panel derecho   : Z-Spreads de ONs corporativas.
        """
        adj = spread_adjustments or {}

        # Paleta por tipo de instrumento
        _type_colors = {
            "sovereign": self.pal["curve"],     # azul
            "dual":      self.pal["forward"],   # naranja
            "bopreal":   self.pal["on_neg"],    # rojo
            "synthetic": self.pal["on_pos"],    # verde
        }
        _type_markers = {
            "sovereign": "o",
            "dual":      "s",
            "bopreal":   "^",
            "synthetic": "D",
        }
        _type_labels = {
            "sovereign": "Soberano",
            "dual":      "Dual (proxy DL)",
            "bopreal":   "BOPREAL (yield obs.)",
            "synthetic": "Sintético ROFEX+CER",
        }

        curve_df = nss_curve.curve_dataframe(t_max=t_max, n_points=400)
        nodes    = fit_metrics.get("node_errors", [])

        # Determinar si hay ONs válidas para mostrar el panel derecho
        valid_ons = [r for r in (on_results or [])
                     if np.isfinite(r.get("z_spread_bps", float("nan")))]
        has_ons = bool(valid_ons)

        if has_ons:
            fig, (ax0, ax1) = plt.subplots(
                1, 2, figsize=(14, 5),
                facecolor=self.pal["bg"],
                constrained_layout=True,
            )
        else:
            fig, ax0 = plt.subplots(
                1, 1, figsize=(9, 5),
                facecolor=self.pal["bg"],
                constrained_layout=True,
            )
            ax1 = None

        fig.suptitle(title, fontsize=14, fontweight="bold",
                     color=self.pal["fg"], y=0.98)

        # ── Panel izquierdo: curva NS + nodos ─────────────────────────
        line_spot, = ax0.plot(
            curve_df["years"], curve_df["spot_rate_pct"],
            color=self.pal["curve"], linewidth=2.5, label="Spot NS", zorder=3,
        )
        line_fwd, = ax0.plot(
            curve_df["years"], curve_df["forward_1y_pct"],
            color=self.pal["forward"], linewidth=1.6, linestyle="--",
            alpha=0.85, label="Forward 1Y", zorder=3,
        )

        # Nodos por tipo
        seen_types: set = set()
        _REF_LABEL = "Referencia (excluido del fit)"
        ref_label_used = False
        for rec in nodes:
            t_obs    = rec["type"]
            t_node   = rec["maturity_years"]
            y_obs    = rec["yield_obs"] * 100
            y_adj    = rec["yield_adj"] * 100
            color    = _type_colors.get(t_obs, self.pal["fg"])
            marker   = _type_markers.get(t_obs, "o")
            is_ref   = rec.get("weight", 1.0) == 0.0

            if is_ref:
                # Punto hueco con borde gris — visible pero no calibrado
                ref_lbl = _REF_LABEL if not ref_label_used else None
                ref_label_used = True
                ax0.scatter(t_node, y_obs, marker=marker, s=55, zorder=4,
                            facecolors="none", edgecolors=self.pal["zero_line"],
                            linewidths=1.4, label=ref_lbl)
            else:
                lbl = _type_labels.get(t_obs, t_obs) if t_obs not in seen_types else None
                seen_types.add(t_obs)

                # Yield observada (relleno sólido)
                ax0.scatter(t_node, y_obs, color=color, marker=marker,
                            s=60, zorder=5, label=lbl)

                # Si tiene spread adjustment, mostrar yield ajustada (hueca)
                sp = adj.get(rec["ticker"], 0.0)
                if sp != 0.0:
                    ax0.scatter(t_node, y_adj, color=color, marker=marker,
                                s=60, zorder=5, facecolors="none",
                                edgecolors=color, linewidths=1.5)
                    ax0.annotate(
                        f"−{sp:.0f}bps",
                        xy=(t_node, y_adj), xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=6.5, color=self.pal["zero_line"],
                    )

            # Etiqueta del ticker (siempre)
            ax0.annotate(
                rec["ticker"],
                xy=(t_node, y_obs), xytext=(5, 5),
                textcoords="offset points",
                fontsize=7, color=self.pal["zero_line"] if is_ref else self.pal["fg"],
            )

        # Leyenda técnica con parámetros NS + métricas de fit
        b     = nss_curve.params
        names = nss_curve.param_names
        rmse  = fit_metrics.get("rmse_bps", float("nan"))
        r2    = fit_metrics.get("r2", float("nan"))
        param_parts = "  ".join(
            f"{n}={v:.2f}" if "τ" in n else f"{n}={v*100:.2f}%"
            for n, v in zip(names, b)
        )
        tech_label = f"{param_parts}\nRMSE {rmse:.1f} bps  |  R²={r2:.4f}"
        dummy = Line2D([0], [0], color="none", label=tech_label)

        ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax0.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax0.xaxis.set_minor_locator(mticker.MultipleLocator(0.25))
        ax0.set_xlabel("Plazo (años)")
        ax0.set_ylabel("Yield (%)")
        ax0.set_title(f"Curva Spot {nss_curve.model}")
        ax0.set_xlim(left=0)
        if y_lim is not None:
            ax0.set_ylim(y_lim[0], y_lim[1])
            ax0.axhline(y=0, color=self.pal["zero_line"],
                        linewidth=1.0, linestyle="-", alpha=0.8, zorder=2)
        # Recolectar todos los artistas con label (curvas + nodos por tipo) y agregar técnico
        _handles, _labels = ax0.get_legend_handles_labels()
        ax0.legend(handles=_handles + [dummy], fontsize=8, loc="upper right")
        ax0.grid(True, which="both", linestyle="--", alpha=0.5)

        # ── Panel derecho: Z-Spreads (solo si hay ONs válidas) ───────
        if has_ons and ax1 is not None:
            ax1.set_title("Z-Spread vs. Curva Soberana")
            tickers    = [r["ticker"] for r in valid_ons]
            spreads    = [r["z_spread_bps"] for r in valid_ons]
            bar_colors = [
                self.pal["on_pos"] if s >= 0 else self.pal["on_neg"]
                for s in spreads
            ]
            bars = ax1.bar(tickers, spreads, color=bar_colors,
                           edgecolor=self.pal["fg"], linewidth=0.4,
                           alpha=0.85, width=0.55)
            for bar, spread in zip(bars, spreads):
                offset = 8 if spread >= 0 else -15
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{spread:+.0f}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=self.pal["fg"],
                )
            ax1.axhline(y=0, color=self.pal["zero_line"],
                        linewidth=0.8, alpha=0.7)
            ax1.set_xlabel("ON")
            ax1.set_ylabel("Z-Spread (bps)")
            ax1.grid(True, which="both", linestyle="--", alpha=0.5, axis="y")
            ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f"))

        fig.text(
            0.99, 0.01,
            f"Generado {date.today().strftime('%d/%m/%Y')} | NS-WLS Framework · Argentina DL",
            ha="right", fontsize=7, color=self.pal["zero_line"], alpha=0.7,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=self.pal["bg"])
            print(f"  Gráfico guardado: {save_path}")

        return fig

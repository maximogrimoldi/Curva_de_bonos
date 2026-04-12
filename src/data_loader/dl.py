"""
Instrumentos Dollar Linked para calibración de curva NS-WLS.

Universo de datos:
  Soberanos     → TV27, TV28, TV30   (peso 1.5)
  Duales        → TDJ26, TDN26       (peso 0.7 — proxy DL tramo corto)
  BOPREALs      → BPY26, BPY27       (peso 0.7 — yield ajustada por spread BCRA)
  Sintético     → nodo manual ROFEX+CER en 0.25y (opcional)

Convenciones:
  - Day Count   : Actual/Actual
  - Precios     : Dirty Price como % del valor técnico
  - Yields      : TIR anual (Actual/Actual)

IMPORTANTE: precios son ilustrativos. Reemplazar con datos reales de mercado.
Verificar specs contra prospectos CNV antes de uso en producción.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE


# ═══════════════════════════════════════════════════════════════════════════════
# Metadatos de instrumento
# ═══════════════════════════════════════════════════════════════════════════════

# Tipo de cada ticker para asignación de pesos en WLS
DL_INSTRUMENT_TYPES: Dict[str, str] = {
    "TZV27": "sovereign",
    "TZV28": "sovereign",
    "TV30":  "sovereign",
    "TTJ26": "dual",
    "TTD26": "dual",
    "BPY26": "bopreal",
}

# Factor ARS/USD oficial utilizado para normalizar face_value de bonos DL
# Todos los soberanos DL tienen face_value = 100 USD × FX_ARS_USD = 134500
FX_ARS_USD: float = 1345.0


# ═══════════════════════════════════════════════════════════════════════════════
# Constructores de bonos
# ═══════════════════════════════════════════════════════════════════════════════

# ── Soberanos DL ─────────────────────────────────────────────────────────────

def _build_tzv27(settlement: date) -> Bond:
    """TZV27 — Soberano DL 2027 | 0.50% | bullet Nov-23-2027
    face_value = 100 USD × ARS/USD 1345 = 134500"""
    return Bond("TZV27", 134500.0, settlement, date(2027, 11, 23),
                coupon=0.0050, currency="ARS_DL", day_count="Actual/Actual")


def _build_tzv28(settlement: date) -> Bond:
    """TZV28 — Soberano DL 2028 | 0.50% | bullet Jun-30-2028
    face_value = 100 USD × ARS/USD 1345 = 134500"""
    return Bond("TZV28", 134500.0, settlement, date(2028, 6, 30),
                coupon=0.0050, currency="ARS_DL", day_count="Actual/Actual")


def _build_tv30(settlement: date) -> Bond:
    """TV30 — Soberano DL 2030 | 0.75% | bullet Nov-30-2030
    face_value = 100 USD × ARS/USD 1345 = 134500"""
    return Bond("TV30", 134500.0, settlement, date(2030, 11, 30),
                coupon=0.0075, currency="ARS_DL", day_count="Actual/Actual")


# ── Bonos Duales (proxy DL tramo corto) ──────────────────────────────────────
# Estructura: capital ajustado por MAX(DL, Badlar). Con DL > Badlar,
# funcionan como bonos DL puros. Se modelan como zero-coupon DL bullet.
# Fuente: prospectos CNV. Verificar fechas exactas de vencimiento.

def _build_ttj26(settlement: date) -> Bond:
    """TTJ26 — Dual Jun-2026. Vto: 30/06/2026. Zero-coupon, bullet.
    face_value ≈ 157 (ARS DL-ajustado; FX base emisión ~870 → 100×1345/870 ≈ 155)"""
    return Bond("TTJ26", 157.0, settlement, date(2026, 6, 30),
                coupon=0.0000, currency="ARS_DL", day_count="Actual/Actual")


def _build_ttd26(settlement: date) -> Bond:
    """TTD26 — Dual Nov-2026. Vto: 28/11/2026. Zero-coupon, bullet.
    face_value ≈ 160 (ARS DL-ajustado)"""
    return Bond("TTD26", 160.0, settlement, date(2026, 11, 28),
                coupon=0.0000, currency="ARS_DL", day_count="Actual/Actual")


# ── BOPREALs (proxy DL con riesgo BCRA, requiere spread_adj) ─────────────────
# Emitidos por BCRA para pago de deuda importadores. Cotizan en ARS (FX oficial).
# Llevan una prima de riesgo BCRA que se limpia antes de usar en el fit.
# Fuente: prospectos BCRA. Verificar estructura de amortización exacta.

def _build_bpy26(settlement: date) -> Bond:
    """BPY26 — BOPREAL Serie 3. Vto: 31/05/2026. Zero-coupon, bullet (ARS DL).
    face_value ≈ 56100 (precio/residual a ~11% yield anual)"""
    return Bond("BPY26", 56100.0, settlement, date(2026, 5, 31),
                coupon=0.0000, currency="ARS_DL", day_count="Actual/Actual")



# ═══════════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_dl_extended(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Universo completo DL: soberanos + duales + BOPREALs."""
    return {
        "TZV27": _build_tzv27(settlement),
        "TZV28": _build_tzv28(settlement),
        "TV30":  _build_tv30(settlement),
        "TTJ26": _build_ttj26(settlement),
        "TTD26": _build_ttd26(settlement),
        "BPY26": _build_bpy26(settlement),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cómputo de yields (YTM)
# ═══════════════════════════════════════════════════════════════════════════════

def _ytm(bond: Bond, price: float) -> float:
    """YTM via Brent's method sobre los flujos del bono."""
    bond.generate_cash_flow_schedule()
    cf    = bond.cash_flows
    times = cf["years"].values
    flows = cf["total_cf"].values

    def pv(y: float) -> float:
        return float(np.sum(flows / (1.0 + y) ** times)) - price

    try:
        return brentq(pv, -0.50, 10.0, xtol=1e-10, maxiter=500)
    except Exception:
        return float("nan")


def build_dl_yield_dataframe(
    bonds: Dict[str, Bond],
    prices: Dict[str, float],
    instrument_types: Optional[Dict[str, str]] = None,
    synthetic_nodes: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """
    Construye el DataFrame de yields para fit_from_yields().

    Parámetros
    ----------
    bonds           : {ticker → Bond}  (e.g. load_dl_extended())
    prices          : {ticker → precio dirty como % face}
    instrument_types: {ticker → type}. Si None usa DL_INSTRUMENT_TYPES.
    synthetic_nodes : lista de dicts con claves
                      [ticker, maturity_years, yield, type].
                      Ejemplo: nodo ROFEX+CER para anclar 0.25y.

    Retorna
    -------
    pd.DataFrame con columnas [ticker, maturity_years, yield, type]
    ordenado por maturity_years.
    """
    types = instrument_types or DL_INSTRUMENT_TYPES
    records = []

    for ticker, bond in bonds.items():
        if ticker not in prices:
            continue
        price = prices[ticker]
        ytm   = _ytm(bond, price)
        if not np.isfinite(ytm):
            continue
        records.append({
            "ticker":         ticker,
            "maturity_years": bond.maturity_years,
            "yield":          ytm,
            "type":           types.get(ticker, "sovereign"),
        })

    if synthetic_nodes:
        for node in synthetic_nodes:
            records.append({
                "ticker":         node["ticker"],
                "maturity_years": float(node["maturity_years"]),
                "yield":          float(node["yield"]),
                "type":           node.get("type", "synthetic"),
            })

    df = pd.DataFrame(records, columns=["ticker", "maturity_years", "yield", "type"])
    return df.sort_values("maturity_years").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ONs corporativas Dollar Linked
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample_ons_dollar_linked(
    settlement: date = SETTLEMENT_DATE,
) -> Dict[str, Tuple[Bond, float]]:
    """
    ONs corporativas Dollar Linked:
    - VSCIO  (3.50%, bullet Ago-15-2028; face_value = 134500 = 100 USD × ARS/USD 1345)
    - MGC30  (5.00%, bullet Oct-10-2029; face_value = 134500)
    Verificar specs (cupón, vencimiento, amortización) contra prospectos CNV.
    """
    vscio = Bond("VSCIO", 134500.0, settlement, date(2028, 8, 15),
                 coupon=0.0350, currency="ARS_DL", day_count="Actual/Actual")

    mgc30 = Bond("MGC30", 134500.0, settlement, date(2029, 10, 10),
                 coupon=0.0500, currency="ARS_DL", day_count="Actual/Actual")

    return {
        "VSCIO": (vscio, 137400.0),
        "MGC30": (mgc30, 125900.0),
    }

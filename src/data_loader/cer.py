"""
Bonos soberanos ajustados por CER (tasa real ARS) y ONs corporativas ajustadas por UVA.

Universo de calibración (10/04/2026):
  Cupón cero  → X15Y6, TZX26, TZXM7  (un solo flujo al vencimiento)
  Cupón fijo  → TX26 (2.0%), TX28 (2.25%), TX31 (2.50%)  — bullet
  Amortizable → DICP (5.83%, 20 cuotas semestrales × 5% desde Jun-2024)
  Step-up     → PARP (1.33% → 2.50% → 3.33%, bullet Dic-2038)

Convenciones:
  - Day Count : Actual/Actual
  - Precios   : Dirty Price en ARS (nominal, indexado CER)
  - Yields    : TIR real anual (por encima de CER)

CER actual (10/04/2026): 312.45
Fuente: Prospectos CNV / BCRA / MECON.
"""

from datetime import date
from typing import Dict, Tuple

from ..bond import Bond
from ._helpers import SETTLEMENT_DATE, sinking

# Coeficiente CER a la fecha de valuación
CER_HOY: float = 312.45


# ── Instrumentos cupón cero ───────────────────────────────────────────────────
# Flujo único: face_value en la fecha de vencimiento.
# face_value ≈ VT = VN × (CER_hoy / CER_emision).

def _build_x15y6(settlement: date) -> Bond:
    """X15Y6 — Lecer 0% · Vto 15/05/2026 · Bullet
    Emitido ~Nov-2025 (CER_em ≈ 300.0). VT = 100 × 312.45/300.0 = 104.15.
    Precio > VT → TIR real negativa (~-13%)."""
    return Bond("X15Y6", 104.15, settlement, date(2026, 5, 15),
                coupon=0.0, currency="ARS_CER", day_count="Actual/Actual")


def _build_tzx26(settlement: date) -> Bond:
    """TZX26 — ZeroCER 0% · Vto 30/06/2026 · Bullet
    Emitido ~Feb-2021 (CER_em ≈ 85.8). VT = 100 × 312.45/85.8 = 364.5.
    Precio > VT → TIR real negativa (~-9%)."""
    return Bond("TZX26", 364.5, settlement, date(2026, 6, 30),
                coupon=0.0, currency="ARS_CER", day_count="Actual/Actual")


def _build_tzxm7(settlement: date) -> Bond:
    """TZXM7 — ZeroCER 0% · Vto 31/03/2027 · Bullet
    Emitido ~mid-2024 (CER_em ≈ 154.7). VT = 100 × 312.45/154.7 ≈ 202.0.
    Precio ≈ VT → TIR real ~0%."""
    return Bond("TZXM7", 202.0, settlement, date(2027, 3, 31),
                coupon=0.0, currency="ARS_CER", day_count="Actual/Actual")


# ── BONCER con cupón fijo — Bullet ────────────────────────────────────────────

def _build_tx26(settlement: date) -> Bond:
    """TX26 — BONCER 2026 · 2.0% real · Bullet · Vto 09/11/2026
    face_value = precio ($1333) ≈ VT (CER_em ≈ 23.5, emitido 2016)."""
    return Bond("TX26", 1333.0, settlement, date(2026, 11, 9),
                coupon=0.0200, currency="ARS_CER", day_count="Actual/Actual")


def _build_tx28(settlement: date) -> Bond:
    """TX28 — BONCER 2028 · 2.25% real · Bullet · Vto 09/11/2028
    face_value = precio ($1926) ≈ VT."""
    return Bond("TX28", 1926.0, settlement, date(2028, 11, 9),
                coupon=0.0225, currency="ARS_CER", day_count="Actual/Actual")


def _build_tx31(settlement: date) -> Bond:
    """TX31 — BONCER 2031 · 2.50% real · Bullet · Vto 30/11/2031
    face_value = precio ($1381)."""
    return Bond("TX31", 1381.0, settlement, date(2031, 11, 30),
                coupon=0.0250, currency="ARS_CER", day_count="Actual/Actual")


# ── Bonos amortizables ────────────────────────────────────────────────────────

def _build_dicp(settlement: date) -> Bond:
    """DICP — Discount CER · 5.83% real · 20 cuotas × 5% semestrales desde Jun-2024
    Fechas: Jun-30 y Dic-31 de cada año (EOM). Residual Apr-2026: 80%.
    face_value = 49490 / 0.80 = 61862.5 (ARS CER-ajustado)."""
    return Bond("DICP", 61862.5, settlement, date(2033, 12, 31),
                coupon=0.0583, amort=sinking(2024, 6, 2033, 12, 0.05, day=31),
                currency="ARS_CER", day_count="Actual/Actual")


def _build_parp(settlement: date) -> Bond:
    """PARP — Par CER · Step-up coupon · Bullet · Vto 31/12/2038
    Schedule (reestructuración 2005):
      1.33% hasta Dic-2009 | 2.50% hasta Dic-2015 | 3.33% desde 2016 en adelante.
    face_value = precio ($33790) — bullet, residual 100%."""
    coupon_schedule = [
        (date(2009, 12, 31), 0.0133),
        (date(2015, 12, 31), 0.0250),
        (date(2038, 12, 31), 0.0333),
    ]
    return Bond("PARP", 33790.0, settlement, date(2038, 12, 31),
                coupon=coupon_schedule,
                currency="ARS_CER", day_count="Actual/Actual")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_boncer(settlement: date = SETTLEMENT_DATE) -> Dict[str, Bond]:
    """Universo completo de bonos CER: 3 cupón-cero + 3 BONCER + DICP + PARP."""
    builders = {
        "X15Y6": _build_x15y6,
        "TZX26": _build_tzx26,
        "TX26":  _build_tx26,
        "TZXM7": _build_tzxm7,
        "TX28":  _build_tx28,
        "TX31":  _build_tx31,
        "DICP":  _build_dicp,
        "PARP":  _build_parp,
    }
    return {ticker: builder(settlement) for ticker, builder in builders.items()}


EXAMPLE_MARKET_PRICES_CER: Dict[str, float] = {
    "X15Y6":  105.53,
    "TZX26":  372.45,
    "TX26":  1333.00,
    "TZXM7":  202.00,
    "TX28":  1926.00,
    "TX31":  1381.00,
    "DICP": 49490.00,
    "PARP": 33790.00,
}


# ── ONs corporativas UVA ──────────────────────────────────────────────────────

def load_sample_ons_cer(settlement: date = SETTLEMENT_DATE) -> Dict[str, Tuple[Bond, float]]:
    """
    ONs corporativas ajustadas por UVA:
    - TLCJO  (Telecom Serie J 2.50%, bullet Jun-30-2026)
    - IRCFO  (IRSA Propiedades 4.00%, 3 cuotas anuales 33%+33%+34%, vto Mar-25-2028)
      Residual Apr-26: 67% (primer 33% pagado Mar-2026).
      face_value = 99800 / 0.67 ≈ 149000 (UVA-ajustado)
    """
    tlcjo = Bond("TLCJO", 115000.0, settlement, date(2026, 6, 30),
                 coupon=0.0250,
                 currency="ARS_UVA", day_count="Actual/Actual")

    ircfo = Bond("IRCFO", 149000.0, settlement, date(2028, 3, 25),
                 coupon=0.0400,
                 amort=[(date(2026, 3, 25), 0.33),
                        (date(2027, 3, 25), 0.33),
                        (date(2028, 3, 25), 0.34)],
                 currency="ARS_UVA", day_count="Actual/Actual")

    return {
        "TLCJO": (tlcjo, 114751.25),
        "IRCFO": (ircfo, 99800.00),
    }

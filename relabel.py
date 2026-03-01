"""
relabel.py v2
=============
Estrategia correcta de labeling:

El problema anterior: price_yes en mercados resueltos ya es 0.0 o 1.0
(precio POST-resolución), no el precio que tenía ANTES de resolver.
Usar ese precio para inferir el label es data leakage.

Solución: cambiar completamente la definición de "bueno":

  "bueno" = mercado con características que lo hacen TRADEABLE
            independientemente de si resolvió YES o NO:
            - Alta liquidez (spread bajo, puedes entrar/salir)
            - Alto volumen (hay contraparte)
            - Precio alejado del 50% (hay información, no es pura incertidumbre)
            - Cierre próximo (resolución cercana = menos tiempo de riesgo)
            - Muchos traders (no manipulable por un whale)

  "malo"  = mercado ilíquido, sin volumen, o precio en zona 50/50

Este label es honesto: no predice el outcome, predice si vale la pena
participar. Un modelo que aprende esto es útil aunque no sea perfecto.

Uso:
    python relabel.py
"""

import json
import math
from pathlib import Path

import config

log = config.get_logger("relabel")


def compute_tradeable_label(r: dict) -> int:
    """
    Label binario: ¿es este mercado bueno para tradear?
    
    Criterios basados en características OBSERVABLES en el momento
    de la descarga, sin usar el outcome real (evita data leakage).
    
    Retorna 1 (bueno) o 0 (malo).
    """
    vol24    = r.get("volume_24h", 0)
    voltot   = r.get("volume_total", 0)
    liq      = r.get("liquidity", 0)
    traders  = r.get("num_traders", 0)
    price    = r.get("price_yes", 0.5)
    days     = r.get("days_to_close", 0)
    change   = r.get("price_change_24h", 0)

    # Filtros duros — si falla cualquiera, es malo
    if liq < 2_000:      return 0   # sin liquidez = atrapado
    if vol24 < 1_000:    return 0   # sin actividad reciente
    if voltot < 10_000:  return 0   # mercado marginal

    # Score ponderado
    score = 0.0

    # 1. Liquidez: más es mejor, cap en 100k
    score += min(liq / 100_000, 1.0) * 30

    # 2. Volumen 24h: actividad reciente
    score += min(vol24 / 30_000, 1.0) * 25

    # 3. Precio alejado del 50%: hay información en el mercado
    #    IMPORTANTE: usamos el precio en el momento de snapshot,
    #    NO para inferir el outcome
    edge = abs(price - 0.5)
    if edge >= 0.30:        score += 20   # precio > 80% o < 20%
    elif edge >= 0.15:      score += 12   # precio > 65% o < 35%
    elif edge >= 0.05:      score += 5    # precio > 55% o < 45%
    # else: precio entre 45-55% → sin edge → 0 puntos

    # 4. Traders únicos: diversidad, resistencia a manipulación
    score += min(traders / 500, 1.0) * 15

    # 5. Timing: ventana óptima 1-21 días
    if 1 <= days <= 7:      score += 10
    elif 7 < days <= 21:    score += 7
    elif days == 0:         score += 3    # cierra hoy/mañana
    elif days <= 60:        score += 2
    # más de 60 días → demasiado tiempo de exposición → 0

    # Threshold: >= 45 puntos = bueno para tradear
    return int(score >= 45)


def relabel_dataset():
    path = Path(config.RAW_DATA_FILE)
    if not path.exists():
        log.error("No existe el dataset")
        return

    log.info(f"Cargando {path}...")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    log.info(f"  {len(records)} registros cargados")

    before_good = sum(1 for r in records if r.get("label_good") == 1)
    log.info(f"  Labels actuales: {before_good} buenos / {len(records)-before_good} malos")

    updated = []
    for r in records:
        r = dict(r)
        r["label_good"]      = compute_tradeable_label(r)
        r["has_real_label"]  = True   # este label es determinista y sin leakage
        r.pop("heuristic_score", None)
        updated.append(r)

    good = sum(1 for r in updated if r["label_good"] == 1)
    bad  = len(updated) - good
    log.info(f"  Labels nuevos: {good} buenos ({good/len(updated):.1%}) / {bad} malos ({bad/len(updated):.1%})")

    if good == 0 or bad == 0:
        log.error("  PROBLEMA: todos los labels son iguales. Revisa los thresholds.")
        return

    if good / len(updated) > 0.85 or good / len(updated) < 0.10:
        log.warning(f"  AVISO: ratio muy desequilibrado ({good/len(updated):.1%} buenos). "
                    f"El modelo puede estar sesgado.")

    # Backup + guardar
    backup = path.with_suffix(".jsonl.bak2")
    import shutil
    shutil.copy(path, backup)
    log.info(f"  Backup en {backup}")

    with open(path, "w", encoding="utf-8") as f:
        for r in updated:
            f.write(json.dumps(r) + "\n")

    log.info(f"Relabel completado. Ejecuta: python main.py train")
    log.info(f"Objetivo: AUC 0.65-0.80 (si sale >0.95 hay otro problema)")


if __name__ == "__main__":
    relabel_dataset()
"""
service.py
==========
Servicio REST que:
- Mantiene el modelo cargado en memoria
- Corre inferencia continua en background (cada 60s)
- Agrupa markets por EVENTO (un evento puede tener múltiples outcomes)
- Solo presenta el outcome dominante por evento — sin duplicados
- Detecta whales y price spikes en tiempo real
"""

import json
import time
import pickle
import logging
import threading
import hashlib
import urllib.parse
import requests as req
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from collector import (fetch_active_markets, extract_record, safe_float,
                        get_event_slug, refresh_event_slug_cache,
                        enrich_with_spreads)

log = config.get_logger("service")

app = FastAPI(title="Polymarket AI Service", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Estado global ─────────────────────────────────────────────────────────────
state = {
    "markets"        : [],
    "alerts"         : deque(maxlen=100),
    "last_update"    : None,
    "model_version"  : None,
    "model_ready"    : False,
    "total_evaluated": 0,
    "uptime_start"   : datetime.now(timezone.utc).isoformat(),
}
state_lock = threading.Lock()


# ── Modelo ────────────────────────────────────────────────────────────────────

class ModelEngine:
    def __init__(self):
        self.model      = None
        self.scaler     = None
        self.model_hash = ""
        self.model_type = None

    def _hash(self, path):
        p = Path(path)
        if not p.exists():
            return ""
        return hashlib.md5(f"{p.stat().st_size}".encode()).hexdigest()[:8]

    def load(self) -> bool:
        xgb_path = Path(config.MODEL_DIR) / "model.xgb.json"
        scaler_p = Path(config.SCALER_FILE)

        if not xgb_path.exists():
            return False

        h = self._hash(str(xgb_path))
        if h == self.model_hash:
            return False

        log.info(f"Cargando modelo: {xgb_path}")
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(str(xgb_path))
        self.model      = m
        self.model_type = "xgb"

        if scaler_p.exists():
            with open(scaler_p, "rb") as f:
                self.scaler = pickle.load(f)

        self.model_hash = h
        log.info(f"Modelo listo hash={h}")
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    @property
    def ready(self):
        return self.model_type is not None


engine = ModelEngine()


# ── URL builder ───────────────────────────────────────────────────────────────

def build_url(market_raw: dict, record: dict = None) -> str:
    events = market_raw.get("events", [])
    if isinstance(events, list) and events:
        slug = events[0].get("slug", "")
        if slug:
            return f"https://polymarket.com/event/{slug}"
    cid  = market_raw.get("conditionId") or market_raw.get("id", "")
    slug = get_event_slug(cid, market_raw)
    if slug:
        return f"https://polymarket.com/event/{slug}"
    if record:
        slug = record.get("slug", "")
        if slug:
            return f"https://polymarket.com/event/{slug}"
    title = market_raw.get("question") or ""
    if title:
        return f"https://polymarket.com/search?q={urllib.parse.quote(title[:80])}"
    return "https://polymarket.com"


# ── Agrupación por evento ─────────────────────────────────────────────────────

def get_event_id(market_raw: dict) -> str:
    """Extrae el ID del evento padre. Si no tiene, usa el conditionId."""
    events = market_raw.get("events", [])
    if isinstance(events, list) and events:
        return str(events[0].get("id", ""))
    return market_raw.get("conditionId") or market_raw.get("id", "")


def score_outcome(rec: dict, raw: dict, all_items: list) -> dict:
    """
    Evalúa un outcome individual y devuelve:
      - ai_score    : 0.0-1.0 — qué tan buena es esta apuesta concreta
      - ai_flag     : "PICK" | "OK" | "SKIP" | "SATURADO"
      - ai_reason   : texto corto explicando el flag
      - value_zone  : True si el precio está en zona de valor (40-85%)

    Lógica:
      No buscamos el outcome con precio más alto (ese ya está "descontado").
      Buscamos el outcome con mejor relación entre:
        1. Convicción del mercado (precio alejado del 50%)
        2. Liquidez disponible (puedes entrar)
        3. Actividad reciente (hay información nueva)
        4. Zona de valor (55-85% = hay upside sin ser wishful thinking)
    """
    price    = rec.get("price_yes", 0.5)
    vol24    = rec.get("volume_24h", 0)
    liq      = rec.get("liquidity", 0)
    traders  = rec.get("num_traders", 0)
    change   = abs(rec.get("price_change_24h", 0))

    # ── 1. Zona de valor ──────────────────────────────────────────────
    # 55-85%: hay convicción Y todavía hay upside razonable
    # <40%: apostando contra el mercado (válido pero diferente tesis)
    # >90%: precio ya descontado, poco upside, riesgo asimétrico
    in_value_zone = 0.40 <= price <= 0.85

    # ── 2. Score compuesto ────────────────────────────────────────────
    score = 0.0

    # Liquidez: puedes entrar y salir
    score += min(liq / 80_000, 1.0) * 30

    # Volumen reciente: hay información nueva circulando
    score += min(vol24 / 25_000, 1.0) * 25

    # Zona de valor del precio
    if 0.55 <= price <= 0.85:
        # Sweet spot: convicción clara con upside razonable
        score += 25
    elif 0.40 <= price <= 0.55:
        # Zona incierta pero interesante si hay mucha liquidez
        score += 10
    elif price > 0.85:
        # Muy caro — ya descontado
        score += 5
    else:
        # < 40%: underdog
        score += 8

    # Movimiento reciente: señal de información nueva
    score += min(change / 0.05, 1.0) * 10

    # Traders únicos: mercado diverso, no manipulable
    score += min(traders / 300, 1.0) * 10

    score = round(score, 1)

    # ── 3. Flag ───────────────────────────────────────────────────────
    if price > 0.92:
        flag   = "SATURADO"
        reason = f"Precio muy alto ({price:.0%}) — poco upside"
    elif price < 0.08:
        flag   = "SKIP"
        reason = f"Probabilidad muy baja ({price:.0%})"
    elif liq < 1_500:
        flag   = "SKIP"
        reason = "Liquidez insuficiente para entrar"
    elif vol24 < 500:
        flag   = "SKIP"
        reason = "Sin actividad reciente"
    elif score >= 55 and in_value_zone:
        flag   = "PICK"
        reason = f"Buena liquidez + zona de valor ({price:.0%})"
    elif score >= 35:
        flag   = "OK"
        reason = f"Condiciones aceptables ({price:.0%})"
    else:
        flag   = "SKIP"
        reason = "Score insuficiente"

    return {
        "ai_score"   : score,
        "ai_flag"    : flag,
        "ai_reason"  : reason,
        "value_zone" : in_value_zone,
    }


def group_by_event(records: list, raws: list, probas: np.ndarray) -> list:
    """
    Agrupa outcomes del mismo evento.
    Para cada evento:
      - Evalúa cada outcome con score_outcome()
      - El AI PICK es el outcome con mejor ai_score en zona de valor
      - El bucket del evento se basa en el mejor pick disponible
      - Todos los outcomes se incluyen en la lista para el dropdown
    """
    groups = defaultdict(list)
    for rec, raw, prob in zip(records, raws, probas):
        eid = get_event_id(raw)
        groups[eid].append((rec, raw, float(prob)))

    result = []
    for eid, items in groups.items():

        # ── Evaluar cada outcome ──────────────────────────────────────
        scored = []
        for rec, raw, prob in items:
            s = score_outcome(rec, raw, items)
            scored.append((rec, raw, prob, s))

        # ── Elegir el AI PICK ─────────────────────────────────────────
        # El mejor outcome en zona de valor con mayor score
        picks = [(r, rw, p, s) for r, rw, p, s in scored if s["ai_flag"] == "PICK"]
        oks   = [(r, rw, p, s) for r, rw, p, s in scored if s["ai_flag"] == "OK"]

        if picks:
            picks.sort(key=lambda x: x[3]["ai_score"], reverse=True)
            pick_rec, pick_raw, pick_prob, pick_score = picks[0]
            has_pick = True
        elif oks:
            oks.sort(key=lambda x: x[3]["ai_score"], reverse=True)
            pick_rec, pick_raw, pick_prob, pick_score = oks[0]
            has_pick = False
        else:
            # Sin picks ni oks: usar el de mayor liquidez
            scored_by_liq = sorted(scored, key=lambda x: x[0].get("liquidity", 0), reverse=True)
            pick_rec, pick_raw, pick_prob, pick_score = scored_by_liq[0]
            has_pick = False

        # ── Título del evento ─────────────────────────────────────────
        event_title = None
        events = pick_raw.get("events", [])
        if isinstance(events, list) and events:
            event_title = events[0].get("title", "")

        # ── Bucket del evento ─────────────────────────────────────────
        # Basado en el score del mejor pick, no solo en el precio
        best_score = pick_score["ai_score"]
        best_flag  = pick_score["ai_flag"]

        if has_pick and best_score >= 55:
            bucket = "SEGURA"
        elif best_score >= 35 and best_flag in ("PICK", "OK"):
            bucket = "CINCUENTA"
        else:
            bucket = "NO_NO"

        display_prob = round(pick_rec.get("price_yes", 0.5), 3)

        # ── Lista de outcomes para el dropdown ────────────────────────
        # Ordenados: PICK primero, luego OK, luego SKIP/SATURADO
        flag_order = {"PICK": 0, "OK": 1, "SATURADO": 2, "SKIP": 3}
        scored_sorted = sorted(
            scored,
            key=lambda x: (flag_order.get(x[3]["ai_flag"], 4), -x[3]["ai_score"])
        )

        outcomes_list = []
        for rec, raw, prob, s in scored_sorted:
            is_pick = (rec["market_id"] == pick_rec["market_id"] and has_pick)
            outcomes_list.append({
                "label"     : rec.get("question", ""),
                "price"     : round(rec.get("price_yes", 0), 3),
                "ai_flag"   : s["ai_flag"],
                "ai_score"  : s["ai_score"],
                "ai_reason" : s["ai_reason"],
                "is_pick"   : is_pick,
                "url"       : build_url(raw, rec),
                "vol24"     : round(rec.get("volume_24h", 0), 0),
                "liquidity" : round(rec.get("liquidity", 0), 0),
            })

        result.append({
            "id"            : pick_rec["market_id"],
            "event_id"      : eid,
            "question"      : event_title or pick_rec["question"],
            "outcome_label" : pick_rec["question"] if has_pick and len(items) > 1 else None,
            "n_outcomes"    : len(items),
            "outcomes"      : outcomes_list,
            "url"           : build_url(pick_raw, pick_rec),
            "prob"          : display_prob,
            "bucket"        : bucket,
            "price_yes"     : display_prob,
            "ai_score"      : best_score,
            "ai_flag"       : best_flag,
            "volume_24h"    : round(sum(x[0].get("volume_24h", 0) for x in items), 0),
            "liquidity"     : round(sum(x[0].get("liquidity",  0) for x in items), 0),
            "days_left"     : pick_rec.get("days_to_close", 0),
            "category"      : pick_rec.get("category_raw", "unknown"),
            "has_whale"     : False,
            "has_spike"     : False,
        })

    return result

def classify(prob: float) -> str:
    if prob >= 0.68:
        return "SEGURA"
    elif prob >= 0.42:
        return "CINCUENTA"
    return "NO_NO"


# ── Whale detector ────────────────────────────────────────────────────────────

class WhaleDetector:
    def __init__(self):
        self.seen   = defaultdict(set)
        self.prices = {}

    def check(self, market_id: str, market_name: str, url: str) -> list:
        alerts = []
        try:
            r = req.get(
                f"{config.CLOB_API}/trades",
                params={"market": market_id, "limit": 20},
                timeout=6, verify=False,
            )
            trades = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
        except Exception:
            return []

        for t in trades:
            tid = t.get("id") or t.get("transactionHash", "")
            if not tid or tid in self.seen[market_id]:
                continue
            self.seen[market_id].add(tid)
            try:
                size     = safe_float(t.get("size") or t.get("amount"))
                price    = safe_float(t.get("price"), 0.5)
                outcome  = t.get("outcome", t.get("side", ""))
                usd_size = size * price
            except Exception:
                continue

            if usd_size >= config.WHALE_THRESHOLD_USD:
                alerts.append({
                    "type"     : "WHALE",
                    "market"   : market_name,
                    "market_id": market_id,
                    "usd_size" : round(usd_size, 2),
                    "outcome"  : outcome,
                    "price"    : price,
                    "url"      : url,
                    "ts"       : datetime.now(timezone.utc).isoformat(),
                })

            key = f"{market_id}_{outcome}"
            lp  = self.prices.get(key)
            if lp and abs(price - lp) >= config.PRICE_SPIKE_THRESHOLD:
                alerts.append({
                    "type"   : "SPIKE",
                    "market" : market_name,
                    "market_id": market_id,
                    "from"   : lp,
                    "to"     : price,
                    "pct"    : round((price - lp) * 100, 2),
                    "outcome": outcome,
                    "url"    : url,
                    "ts"     : datetime.now(timezone.utc).isoformat(),
                })
            self.prices[key] = price

        return alerts


whale_detector = WhaleDetector()


# ── Loop de inferencia ────────────────────────────────────────────────────────

def inference_loop():
    log.info("Inference loop arrancado")
    refresh_event_slug_cache()
    while True:
        try:
            engine.load()

            markets_raw = fetch_active_markets(limit=config.MARKETS_PER_FETCH)
            if not markets_raw:
                time.sleep(config.INFERENCE_INTERVAL_SECONDS)
                continue

            records, valid_raw = [], []
            for m in markets_raw:
                r = extract_record(m)
                if r:
                    records.append(r)
                    valid_raw.append(m)

            if not records:
                time.sleep(config.INFERENCE_INTERVAL_SECONDS)
                continue
            # ── Enriquecer con spread real del CLOB (top 50 por volumen) ───────
            enrich_with_spreads(records, valid_raw, top_n=50)
            # ── Inferencia ────────────────────────────────────────────────
            if engine.ready:
                X      = np.array([[rec[c] for c in config.FEATURE_COLS] for rec in records], dtype=np.float32)
                probas = engine.predict(X)
            else:
                from collector import heuristic_score
                probas = np.array([heuristic_score(r) / 100.0 for r in records])

            # ── Agrupar por evento (elimina duplicados) ───────────────────
            classified = group_by_event(records, valid_raw, probas)

            # ── Whale check sobre los mercados individuales ───────────────
            market_url_map = {}
            for rec, raw in zip(records, valid_raw):
                market_url_map[rec["market_id"]] = build_url(raw, rec)

            for item in classified:
                market_id = item["id"]
                url       = item["url"]
                alerts    = whale_detector.check(market_id, item["question"], url)
                if alerts:
                    item["has_whale"] = any(a["type"] == "WHALE" for a in alerts)
                    item["has_spike"] = any(a["type"] == "SPIKE" for a in alerts)
                    for a in alerts:
                        with state_lock:
                            state["alerts"].appendleft(a)

            # Ordenar por prob desc
            classified.sort(key=lambda x: x["prob"], reverse=True)

            with state_lock:
                state["markets"]          = classified
                state["last_update"]      = datetime.now(timezone.utc).isoformat()
                state["model_ready"]      = engine.ready
                state["model_version"]    = engine.model_hash
                state["total_evaluated"] += len(classified)

            n_s = sum(1 for x in classified if x["bucket"] == "SEGURA")
            n_c = sum(1 for x in classified if x["bucket"] == "CINCUENTA")
            n_n = sum(1 for x in classified if x["bucket"] == "NO_NO")
            log.info(f"Ciclo OK: {len(classified)} eventos (de {len(records)} markets) | S={n_s} C={n_c} N={n_n}")

        except Exception as e:
            log.error(f"Error en inference loop: {e}", exc_info=True)

        time.sleep(config.INFERENCE_INTERVAL_SECONDS)


@app.on_event("startup")
def startup():
    # Cargar alertas previas del disco para no perderlas entre reinicios
    alerts_path = Path(f"{config.LOG_DIR}/alerts.jsonl")
    if alerts_path.exists():
        try:
            lines = alerts_path.read_text(encoding="utf-8").strip().splitlines()
            recent = lines[-100:]          # las últimas 100
            for line in reversed(recent):
                try:
                    state["alerts"].appendleft(json.loads(line))
                except Exception:
                    pass
            log.info(f"Alertas previas cargadas: {len(recent)} entradas")
        except Exception as e:
            log.warning(f"No se pudieron cargar alertas previas: {e}")

    t = threading.Thread(target=inference_loop, daemon=True, name="InferenceLoop")
    t.start()
    log.info("Servicio arrancado")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/status")
def status():
    with state_lock:
        markets = state["markets"]
        return {
            "model_ready"    : state["model_ready"],
            "model_version"  : state["model_version"],
            "last_update"    : state["last_update"],
            "total_evaluated": state["total_evaluated"],
            "uptime_start"   : state["uptime_start"],
            "market_count"   : len(markets),
            "buckets": {
                "SEGURA"   : sum(1 for m in markets if m["bucket"] == "SEGURA"),
                "CINCUENTA": sum(1 for m in markets if m["bucket"] == "CINCUENTA"),
                "NO_NO"    : sum(1 for m in markets if m["bucket"] == "NO_NO"),
            },
        }

@app.get("/markets")
def get_markets(bucket: Optional[str] = None, limit: int = 10000):
    with state_lock:
        markets = list(state["markets"])
    if bucket:
        markets = [m for m in markets if m["bucket"] == bucket.upper()]
    return {"markets": markets[:limit], "total": len(markets)}

@app.get("/alerts")
def get_alerts(limit: int = 50):
    with state_lock:
        alerts = list(state["alerts"])
    return {"alerts": alerts[:limit]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
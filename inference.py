"""
inference.py
============
Motor de inferencia autónomo.
- Carga el modelo entrenado (XGBoost o ONNX)
- Evalúa todos los mercados activos cada N segundos
- Detecta whales y price spikes en tiempo real
- Emite alertas (consola + Telegram opcional)
- Se recarga solo si el modelo cambia en disco
"""

import json
import time
import pickle
import logging
import hashlib
import requests
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, deque

import numpy as np

import config
from collector import (
    fetch_active_markets, extract_record,
    safe_float, parse_price_yes,
    enrich_with_spreads,
)

# ── Logging ───────────────────────────────────────────────────────────────────
log = config.get_logger("inference")


# ── Alertas ───────────────────────────────────────────────────────────────────

def send_telegram(text: str):
    if not config.TELEGRAM_ENABLED:
        return
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        requests.post(
            url,
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=5, verify=False,
        )
    except Exception as e:
        log.warning(f"Telegram error: {e}")


def emit_alert(alert: dict):
    ts = datetime.now().strftime("%H:%M:%S")

    if alert["type"] == "GOOD_MARKET":
        prob = alert["prob_good"]
        bar  = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        line = (
            f"\n🎯 BUENA OPORTUNIDAD [{ts}]\n"
            f"   {alert['question'][:65]}\n"
            f"   Score: {bar} {prob:.0%}\n"
            f"   YES: {alert['price_yes']:.1%} | "
            f"Vol24h: ${alert['volume_24h']:,.0f} | "
            f"Liq: ${alert['liquidity']:,.0f}"
        )
        print(line)
        send_telegram(
            f"🎯 <b>BUENA OPORTUNIDAD</b> {prob:.0%}\n"
            f"{alert['question'][:60]}\n"
            f"YES: {alert['price_yes']:.1%} | Vol: ${alert['volume_24h']:,.0f}"
        )

    elif alert["type"] == "WHALE_TRADE":
        line = (
            f"\n🐋 WHALE [{ts}]\n"
            f"   {alert['market'][:65]}\n"
            f"   {alert['outcome']} @ {alert['price']:.3f} — "
            f"${alert['usd_size']:,.0f} USDC [{alert['side'].upper()}]"
        )
        print(line)
        send_telegram(
            f"🐋 <b>WHALE</b> ${alert['usd_size']:,.0f}\n"
            f"{alert['market'][:60]}\n"
            f"{alert['outcome']} @ {alert['price']:.3f}"
        )

    elif alert["type"] == "PRICE_SPIKE":
        direction = "📈" if alert["move_pct"] > 0 else "📉"
        line = (
            f"\n⚡{direction} PRICE SPIKE [{ts}]\n"
            f"   {alert['market'][:65]}\n"
            f"   {alert['outcome']}: "
            f"{alert['price_before']:.3f} → {alert['price_after']:.3f} "
            f"({alert['move_pct']:+.1f}%) | ${alert['usd_size']:,.0f}"
        )
        print(line)
        send_telegram(
            f"⚡ <b>SPIKE {alert['move_pct']:+.1f}%</b>\n"
            f"{alert['market'][:60]}\n"
            f"{alert['outcome']}: {alert['price_before']:.3f}→{alert['price_after']:.3f}"
        )

    # Guardar en log de alertas
    if alert["type"] != "MODEL_DRIFT":
        with open(f"{config.LOG_DIR}/alerts.jsonl", "a") as f:
            f.write(json.dumps({**alert, "emitted_at": datetime.now(timezone.utc).isoformat()}) + "\n")
    else:
        # Drift: solo log, no se persiste en alerts.jsonl
        log.warning(
            f"⚠️  MODEL DRIFT | baseline={alert['baseline_mean']:.3f} "
            f"reciente={alert['recent_mean']:.3f} | caída={alert['drop']:.3f}"
        )
        send_telegram(
            f"⚠️ <b>MODEL DRIFT detectado</b>\n"
            f"Baseline: {alert['baseline_mean']:.3f} → Reciente: {alert['recent_mean']:.3f}\n"
            f"Caída: {alert['drop']:.3f}"
        )


# ── Carga del modelo ──────────────────────────────────────────────────────────

class ModelLoader:
    """Carga y recarga el modelo cuando cambia en disco."""

    def __init__(self):
        self.model       = None
        self.scaler      = None
        self.model_hash  = ""
        self.model_type  = None   # "onnx" | "xgb"
        self._ort_session = None

    def _file_hash(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        size = p.stat().st_size
        return hashlib.md5(f"{path}:{size}".encode()).hexdigest()[:12]

    def _load_onnx(self, path: str):
        import onnxruntime as ort
        self._ort_session = ort.InferenceSession(path)
        self.model_type   = "onnx"
        log.info(f"Modelo ONNX cargado: {path}")

    def _load_xgb(self, path: str):
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(path)
        self.model      = m
        self.model_type = "xgb"
        log.info(f"Modelo XGBoost cargado: {path}")

    def load_or_reload(self) -> bool:
        """Devuelve True si se (re)cargó el modelo."""

        # Prioridad: xgb.json nativo (el ONNX que genera XGBoost no es ONNX real)
        # Solo usar .onnx si fue generado por onnxmltools (no por xgb.save_model)
        xgb_path    = Path(config.MODEL_DIR) / "model.xgb.json"
        onnx_path   = Path(config.MODEL_DIR) / "model.onnx"   # solo onnxmltools
        scaler_path = Path(config.SCALER_FILE)

        model_path = None
        if xgb_path.exists():
            model_path = str(xgb_path)          # preferir siempre xgb.json
        elif onnx_path.exists():
            model_path = str(onnx_path)

        if not model_path:
            return False

        current_hash = self._file_hash(model_path)
        if current_hash == self.model_hash:
            return False  # sin cambios

        log.info(f"Cargando modelo desde {model_path}...")

        if model_path.endswith(".onnx"):
            try:
                self._load_onnx(model_path)
            except Exception as e:
                log.warning(f"ONNX fallo ({e}) — usando XGBoost nativo")
                self._load_xgb(str(xgb_path))
        else:
            self._load_xgb(model_path)

        # Scaler
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self.model_hash = current_hash
        return True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Devuelve array de probabilidades de clase 'bueno'."""
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.model_type == "onnx":
            input_name = self._ort_session.get_inputs()[0].name
            out = self._ort_session.run(None, {input_name: X.astype(np.float32)})
            # out[1] suele ser el dict de probabilidades
            if len(out) > 1 and isinstance(out[1], list):
                return np.array([d.get(1, 0.0) for d in out[1]])
            return out[0][:, 1] if out[0].ndim == 2 else out[0]

        elif self.model_type == "xgb":
            return self.model.predict_proba(X)[:, 1]

        return np.zeros(len(X))

    @property
    def ready(self) -> bool:
        return self.model_type is not None or self._ort_session is not None


# ── Detector de whales ────────────────────────────────────────────────────────

class WhaleDetector:
    def __init__(self):
        self.seen_trades = defaultdict(set)
        self.price_cache = {}

    def get_recent_trades(self, market_id: str) -> list:
        try:
            r = requests.get(
                f"{config.CLOB_API}/trades",
                params={"market": market_id, "limit": 20},
                timeout=8, verify=False,
            )
            data = r.json()
            if isinstance(data, list):
                return data
            return data.get("data", data.get("trades", []))
        except Exception:
            return []

    def check(self, market: dict) -> list:
        market_id   = market.get("conditionId") or market.get("id", "")
        market_name = market.get("question", market.get("title", ""))
        if not market_id:
            return []

        trades = self.get_recent_trades(market_id)
        alerts = []

        for trade in trades:
            tid = trade.get("id") or trade.get("transactionHash", "")
            if not tid or tid in self.seen_trades[market_id]:
                continue
            self.seen_trades[market_id].add(tid)

            try:
                size     = safe_float(trade.get("size") or trade.get("amount"))
                price    = safe_float(trade.get("price"), 0.5)
                outcome  = trade.get("outcome", trade.get("side", ""))
                side     = trade.get("makerSide", trade.get("side", ""))
                usd_size = size * price
            except Exception:
                continue

            if usd_size >= config.WHALE_THRESHOLD_USD:
                alerts.append({
                    "type"     : "WHALE_TRADE",
                    "market"   : market_name[:70],
                    "market_id": market_id,
                    "usd_size" : usd_size,
                    "side"     : side,
                    "outcome"  : outcome,
                    "price"    : price,
                })

            key        = f"{market_id}_{outcome}"
            last_price = self.price_cache.get(key)
            if last_price and abs(price - last_price) >= config.PRICE_SPIKE_THRESHOLD:
                alerts.append({
                    "type"        : "PRICE_SPIKE",
                    "market"      : market_name[:70],
                    "market_id"   : market_id,
                    "price_before": last_price,
                    "price_after" : price,
                    "move_pct"    : round((price - last_price) * 100, 2),
                    "usd_size"    : usd_size,
                    "outcome"     : outcome,
                })

            self.price_cache[key] = price

        return alerts


# ── Monitoreo de drift del modelo ─────────────────────────────────────────────

class DriftMonitor:
    """
    Detecta model drift comparando la media de scores recientes
    contra una baseline establecida en los primeros ciclos.

    Si la media cae (o sube) más de `threshold` puntos, emite MODEL_DRIFT.
    """

    def __init__(self, warmup_cycles: int = 5, window: int = 10, threshold: float = 0.10):
        self.warmup_cycles  = warmup_cycles
        self.window         = window
        self.threshold      = threshold
        self.history: deque = deque(maxlen=50)
        self.baseline_mean: float | None = None
        self._cycles        = 0

    def update(self, probas: np.ndarray) -> None:
        """Recibe el array de probabilidades del ciclo actual y dispara alerta si hay drift."""
        if len(probas) == 0:
            return

        cycle_mean = float(np.mean(probas))
        self.history.append(cycle_mean)
        self._cycles += 1

        # Fase de calentamiento: acumular historia para baseline
        if self._cycles <= self.warmup_cycles:
            if self._cycles == self.warmup_cycles:
                self.baseline_mean = float(np.mean(list(self.history)))
                log.info(f"DriftMonitor baseline establecido: {self.baseline_mean:.3f}")
            return

        if self.baseline_mean is None or len(self.history) < self.window:
            return

        recent_mean = float(np.mean(list(self.history)[-self.window:]))
        drop = self.baseline_mean - recent_mean

        if abs(drop) > self.threshold:
            emit_alert({
                "type"         : "MODEL_DRIFT",
                "baseline_mean": round(self.baseline_mean, 3),
                "recent_mean"  : round(recent_mean, 3),
                "drop"         : round(drop, 3),
            })
            # Actualizar baseline para no disparar en cada ciclo consecutivo
            self.baseline_mean = recent_mean
            log.info(f"DriftMonitor baseline actualizado a {self.baseline_mean:.3f}")


# ── Loop principal ────────────────────────────────────────────────────────────

def run_inference_loop():
    log.info("=" * 55)
    log.info("  INFERENCE ENGINE arrancado")
    log.info(f"  Intervalo: {config.INFERENCE_INTERVAL_SECONDS}s")
    log.info(f"  Threshold bueno: {config.GOOD_MARKET_PROB_THRESHOLD:.0%}")
    log.info("=" * 55)

    loader  = ModelLoader()
    whale   = WhaleDetector()
    monitor = DriftMonitor(warmup_cycles=5, window=10, threshold=0.10)
    tick    = 0

    # Mercados ya alertados (evitar spam)
    alerted_markets = set()

    while True:
        try:
            # ── Recargar modelo si cambió ──────────────────────────
            reloaded = loader.load_or_reload()
            if reloaded:
                log.info("✅ Modelo recargado")
            elif not loader.ready and tick == 0:
                log.warning("⚠️  Modelo no disponible aún — esperando a que trainer.py lo genere...")

            # ── Descargar mercados activos ──────────────────────────
            markets = fetch_active_markets(limit=config.MARKETS_PER_FETCH)
            tick   += 1

            if not markets:
                log.warning("Sin mercados recibidos de la API")
                time.sleep(config.INFERENCE_INTERVAL_SECONDS)
                continue

            # ── Inferencia ML ──────────────────────────────────────
            if loader.ready:
                records = []
                raw_markets = []
                for m in markets:
                    r = extract_record(m)
                    if r:
                        records.append(r)
                        raw_markets.append(m)

                if records:
                    # Enriquecer con spread real del CLOB (top 50 por volumen)
                    enrich_with_spreads(records, raw_markets, top_n=50)

                    X = np.array(
                        [[rec[col] for col in config.FEATURE_COLS] for rec in records],
                        dtype=np.float32,
                    )
                    probas = loader.predict_proba(X)

                    # Monitoreo de drift
                    monitor.update(probas)

                    for i, (rec, prob) in enumerate(zip(records, probas)):
                        if (prob >= config.GOOD_MARKET_PROB_THRESHOLD and
                                rec["market_id"] not in alerted_markets):
                            emit_alert({
                                "type"      : "GOOD_MARKET",
                                "question"  : rec["question"],
                                "market_id" : rec["market_id"],
                                "prob_good" : float(prob),
                                "price_yes" : rec["price_yes"],
                                "volume_24h": rec["volume_24h"],
                                "liquidity" : rec["liquidity"],
                            })
                            alerted_markets.add(rec["market_id"])

            # ── Whale detection ────────────────────────────────────
            # Solo en los top 20 por volumen (para no saturar la API)
            for m in markets[:20]:
                for alert in whale.check(m):
                    emit_alert(alert)

            # ── Status tick ───────────────────────────────────────
            model_status = f"modelo={'✅' if loader.ready else '⏳'}"
            print(
                f"  · [{tick:04d}] {datetime.now().strftime('%H:%M:%S')} "
                f"— {len(markets)} mercados | {model_status}",
                end="\r",
            )

        except KeyboardInterrupt:
            print("\n\n👋 Inference engine detenido.")
            break
        except Exception as e:
            log.error(f"Error en loop: {e}", exc_info=True)

        time.sleep(config.INFERENCE_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_inference_loop()
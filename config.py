# ============================================================
#  config.py — configuracion central del sistema autonomo
# ============================================================
import logging
import subprocess
import sys
import warnings
from pathlib import Path

# Suprimir warning de SSL (verify=False necesario en algunos entornos Windows)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ── APIs ────────────────────────────────────────────────────
GAMMA_API        = "https://gamma-api.polymarket.com"
CLOB_API         = "https://clob.polymarket.com"
GRAPH_API        = "https://api.thegraph.com/subgraphs/name/polymarket/matic-markets"

# ── Rutas locales ────────────────────────────────────────────
DATA_DIR         = "data"
MODEL_DIR        = "models"
LOG_DIR          = "logs"
RAW_DATA_FILE    = "data/markets_raw.jsonl"
FEATURES_FILE    = "data/features.csv"
MODEL_FILE       = "models/model.onnx"
SCALER_FILE      = "models/scaler.pkl"
LABEL_MAP_FILE   = "models/label_map.json"

# ── Recoleccion ──────────────────────────────────────────────
COLLECT_INTERVAL_SECONDS  = 300
HISTORICAL_TARGET         = 3000      # mercados historicos objetivo en bootstrap
MARKETS_PER_FETCH         = 10000
MIN_VOLUME_TO_STORE       = 500
HISTORY_DAYS              = 90

# ── Entrenamiento ────────────────────────────────────────────
RETRAIN_EVERY_N_RECORDS   = 500
MIN_RECORDS_TO_TRAIN      = 200
GOOD_MARKET_THRESHOLD     = 60
TEST_SIZE                 = 0.2
RANDOM_STATE              = 42

# ── Inferencia ───────────────────────────────────────────────
INFERENCE_INTERVAL_SECONDS = 60
GOOD_MARKET_PROB_THRESHOLD = 0.65

# ── Alertas whale ────────────────────────────────────────────
WHALE_THRESHOLD_USD        = 5_000
PRICE_SPIKE_THRESHOLD      = 0.04

# ── Telegram (opcional) ──────────────────────────────────────
TELEGRAM_ENABLED  = False
TELEGRAM_TOKEN    = "TU_TOKEN_AQUI"
TELEGRAM_CHAT_ID  = "TU_CHAT_ID_AQUI"

# ── Auto-detección de device GPU/CPU ─────────────────────────
def _detect_device() -> str:
    """Devuelve 'cuda' si nvidia-smi está disponible, 'cpu' en caso contrario."""
    try:
        subprocess.check_output(
            ["nvidia-smi"], timeout=3,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        )
        return "cuda"
    except Exception:
        return "cpu"

_DEVICE = _detect_device()

# ── XGBoost hyperparams ──────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 6,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "eval_metric"     : "logloss",
    "random_state"    : 42,
    "device"          : _DEVICE,   # auto-detectado: cuda si hay GPU, cpu si no
    "tree_method"     : "hist",    # válido para cuda y cpu
}

# ── Features ─────────────────────────────────────────────────
FEATURE_COLS = [
    "volume_24h",
    "volume_total",
    "liquidity",
    "num_traders",
    "price_yes",
    "price_distance_from_50",
    "days_to_close",
    "price_change_24h",
    "liquidity_to_volume_ratio",
    "volume_acceleration",
    "category_encoded",
    "spread",          # spread bid-ask del CLOB (0.05 por defecto si no disponible)
]

# ── Logging helper (UTF-8 seguro en Windows) ─────────────────

def _make_utf8_stream():
    """Devuelve un stream stdout compatible con UTF-8 en Windows."""
    if sys.platform == "win32":
        import io
        try:
            return io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except AttributeError:
            pass
    return sys.stdout


def get_logger(name: str) -> logging.Logger:
    """
    Logger con salida UTF-8 segura en Windows cp1252.
    Usar en lugar de logging.basicConfig en cada modulo:

        import config
        log = config.get_logger("collector")
    """
    Path(LOG_DIR).mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [{name.upper()}] %(message)s")

    # Archivo: siempre UTF-8
    fh = logging.FileHandler(f"{LOG_DIR}/{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Consola: UTF-8 forzado
    sh = logging.StreamHandler(_make_utf8_stream())
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
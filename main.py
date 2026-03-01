"""
main.py
=======
Orquestador principal — arranca los 3 procesos en paralelo:
  1. collector.py  → recolecta datos cada 5 min
  2. trainer.py    → reentrena el modelo cuando hay datos nuevos
  3. inference.py  → evalúa mercados y emite alertas continuamente

Uso:
  python main.py              → arranca todo
  python main.py bootstrap    → solo carga histórico y sale
  python main.py train        → entrena ahora y sale
  python main.py inference    → solo inferencia (si ya tienes modelo)
  python main.py status       → muestra estado del sistema
"""

import sys
import time
import subprocess
import threading
import logging
import json
from pathlib import Path
from datetime import datetime

import config

# ── Forzar UTF-8 en Windows (antes de cualquier print/log) ───────────────────
import sys, io
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# ── Logging ───────────────────────────────────────────────────────────────────
Path(config.LOG_DIR).mkdir(exist_ok=True)
log = config.get_logger("main")


# ── Runners en threads ────────────────────────────────────────────────────────

def run_collector():
    from collector import run_loop
    run_loop()

def run_trainer():
    from trainer import run_watch_loop
    run_watch_loop()

def run_inference():
    from inference import run_inference_loop
    run_inference_loop()


def start_all():
    print("""
╔══════════════════════════════════════════════════════════╗
║          POLYMARKET AUTONOMOUS TRADING SYSTEM            ║
║                                                          ║
║   collector  →  trainer  →  inference  →  alerts        ║
╚══════════════════════════════════════════════════════════╝
""")

    threads = [
        threading.Thread(target=run_collector,  name="Collector",  daemon=True),
        threading.Thread(target=run_trainer,    name="Trainer",    daemon=True),
        threading.Thread(target=run_inference,  name="Inference",  daemon=True),
    ]

    for t in threads:
        log.info(f"Arrancando {t.name}...")
        t.start()
        time.sleep(2)  # pequeño delay entre arranques

    log.info("✅ Todos los componentes activos. Ctrl+C para detener.")

    try:
        while True:
            alive = [t.name for t in threads if t.is_alive()]
            dead  = [t.name for t in threads if not t.is_alive()]
            if dead:
                log.warning(f"⚠️  Threads caídos: {dead}")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n👋 Sistema detenido.")


def show_status():
    print("\n📊 ESTADO DEL SISTEMA\n" + "=" * 50)

    # Dataset
    raw = Path(config.RAW_DATA_FILE)
    if raw.exists():
        n = sum(1 for _ in open(raw))
        size_kb = raw.stat().st_size / 1024
        print(f"  Dataset    : {n} registros ({size_kb:.1f} KB)")
    else:
        print("  Dataset    : ❌ no existe aún")

    # Modelo
    xgb_path = Path(config.MODEL_DIR) / "model.xgb.json"
    meta_path = Path(config.LABEL_MAP_FILE)
    if xgb_path.exists():
        mtime = datetime.fromtimestamp(xgb_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  Modelo     : ✅ {xgb_path} (modificado: {mtime})")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  AUC-ROC    : {meta.get('auc', 'N/A')}")
            print(f"  Entrenado  : {meta.get('trained_at', 'N/A')}")
            print(f"  Muestras   : {meta.get('n_train', 'N/A')} train / {meta.get('n_test', 'N/A')} test")
    else:
        print("  Modelo     : ⏳ no entrenado aún")

    # Alertas recientes
    alerts_path = Path(f"{config.LOG_DIR}/alerts.jsonl")
    if alerts_path.exists():
        lines = alerts_path.read_text().strip().split("\n")
        n_alerts = len([l for l in lines if l])
        print(f"  Alertas    : {n_alerts} total")
        # Últimas 3
        last = [json.loads(l) for l in lines[-3:] if l]
        for a in last:
            ts = a.get("emitted_at", "")[:16]
            print(f"    [{ts}] {a['type']} — {a.get('question', a.get('market', ''))[:50]}")
    else:
        print("  Alertas    : ninguna aún")

    print()


# ── Instalación de dependencias ───────────────────────────────────────────────

def check_and_install_deps():
    required = {
        "xgboost"     : "xgboost",
        "sklearn"     : "scikit-learn",
        "pandas"      : "pandas",
        "numpy"       : "numpy",
        "requests"    : "requests",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"📦 Instalando dependencias faltantes: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["-q"])
        print("✅ Dependencias instaladas\n")

    # Opcionales
    optional = {"onnxruntime": "onnxruntime", "skl2onnx": "skl2onnx"}
    for mod, pkg in optional.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"ℹ️  {pkg} no instalado (opcional para export ONNX) — pip install {pkg}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    check_and_install_deps()

    if cmd == "bootstrap":
        from collector import collect_historical_bootstrap
        collect_historical_bootstrap()

    elif cmd == "train":
        from trainer import run_training_once
        metrics = run_training_once()
        if metrics:
            print(f"\n✅ AUC: {metrics['auc']} | {metrics['n_train']+metrics['n_test']} samples")

    elif cmd == "inference":
        from inference import run_inference_loop
        run_inference_loop()

    elif cmd == "collect":
        from collector import run_loop
        run_loop()

    elif cmd == "status":
        show_status()

    elif cmd == "all":
        start_all()

    else:
        print(__doc__)
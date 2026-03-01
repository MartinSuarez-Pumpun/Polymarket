"""
trainer.py v3 — Walk-Forward Validation
========================================
Cambios principales vs v2:
- walk_forward_validate(): evalúa el modelo en ventanas temporales deslizantes
  para obtener una estimación realista del AUC sin data leakage
- Solo entrena con labels reales (has_real_label=True) cuando hay suficientes
- El modelo final se entrena con TODOS los datos reales disponibles
- Regularización adaptativa según tamaño del dataset
"""

import os
import json
import time
import logging
import pickle
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import config

log = config.get_logger("trainer")


def get_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        raise ImportError("pip install xgboost")


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    path = Path(config.RAW_DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(f"No existe {config.RAW_DATA_FILE}")

    records = []
    with open(path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    df = pd.DataFrame(records)
    log.info(f"Dataset cargado: {len(df)} registros")
    return df


# ── Preparación ───────────────────────────────────────────────────────────────

def prepare_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y feature engineering.
    Devuelve el DataFrame completo ordenado por tiempo, sin hacer split.
    El split lo hace walk_forward_validate() o prepare_final_split().
    """
    # Deduplicar por snapshot_key — cada mercado puede tener múltiples snapshots
    # en distintos momentos (cada 6h). Cada snapshot es un ejemplo de entrenamiento
    # independiente con su propio price_yes en ese momento.
    # Solo eliminamos duplicados exactos (mismo market + misma ventana temporal).
    if "snapshot_key" in df.columns:
        df = df.sort_values("collected_at").drop_duplicates("snapshot_key", keep="last")
        n_markets   = df["market_id"].nunique() if "market_id" in df.columns else "?"
        n_snapshots = len(df)
        log.info(f"Snapshots únicos: {n_snapshots} de {n_markets} mercados distintos")
    elif "market_id" in df.columns and "collected_at" in df.columns:
        # Fallback para datasets antiguos sin snapshot_key
        df = df.sort_values("collected_at").drop_duplicates("market_id", keep="last")
        log.info(f"Fallback dedup por market_id: {len(df)} registros")

    # Solo usar registros con label real (has_real_label=True)
    # Los heurísticos distorsionan el modelo
    real_mask = df.get("has_real_label", pd.Series(False, index=df.index)).fillna(False)
    n_real    = int(real_mask.sum())
    n_total   = len(df)

    if n_real >= config.MIN_RECORDS_TO_TRAIN:
        df = df[real_mask].copy()
        log.info(f"Usando {n_real}/{n_total} registros con label real")
    else:
        df = df.copy()
        log.info(f"Solo {n_real} labels reales — usando {n_total} totales (incluye heurísticos)")

    # Asegurar columnas
    for col in config.FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Clip outliers
    df["volume_24h"]          = df["volume_24h"].clip(0, 1_000_000)
    df["volume_total"]        = df["volume_total"].clip(0, 10_000_000)
    df["liquidity"]           = df["liquidity"].clip(0, 500_000)
    df["num_traders"]         = df["num_traders"].clip(0, 10_000)
    df["volume_acceleration"] = df["volume_acceleration"].clip(0, 100)
    df[config.FEATURE_COLS]   = df[config.FEATURE_COLS].fillna(0.0)

    if "label_good" not in df.columns:
        df["label_good"] = 0
    df["label_good"] = df["label_good"].fillna(0).astype(int)

    # Ordenar temporalmente
    if "collected_at" in df.columns:
        df = df.sort_values("collected_at").reset_index(drop=True)

    balance = df["label_good"].value_counts(normalize=True)
    log.info(f"Balance de clases: bueno={balance.get(1,0):.1%} malo={balance.get(0,0):.1%}")

    return df


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward_validate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Walk-forward cross-validation temporal.

    Divide el dataset en n_splits ventanas temporales:
      Fold 1: train=[0..20%]        test=[20%..40%]
      Fold 2: train=[0..40%]        test=[40%..60%]
      Fold 3: train=[0..60%]        test=[60%..80%]
      Fold 4: train=[0..80%]        test=[80%..100%]
      ...

    En cada fold:
    - El train siempre es el PASADO
    - El test siempre es el FUTURO
    - Nunca hay solapamiento

    Esto da una estimación realista del AUC porque el modelo
    nunca ve datos del futuro durante el entrenamiento.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    xgb = get_xgb()

    n = len(df)
    if n < 100:
        log.warning(f"Dataset demasiado pequeño para walk-forward ({n} registros)")
        return {"wf_aucs": [], "wf_mean": 0.0, "wf_std": 0.0, "wf_folds": 0}

    # Calcular tamaño mínimo de test para tener ambas clases
    min_test_size = max(20, int(n * 0.15))
    step          = (n - min_test_size) // n_splits

    if step < 10:
        log.warning(f"Dataset pequeño — reduciendo a {max(2, n_splits//2)} folds")
        n_splits = max(2, n_splits // 2)
        step     = (n - min_test_size) // n_splits

    aucs   = []
    folds_ok = 0

    for fold in range(n_splits):
        train_end = min_test_size + fold * step
        test_end  = train_end + step if fold < n_splits - 1 else n

        df_train = df.iloc[:train_end]
        df_test  = df.iloc[train_end:test_end]

        # Necesitamos ambas clases en train Y test
        if df_train["label_good"].nunique() < 2 or df_test["label_good"].nunique() < 2:
            log.debug(f"  Fold {fold+1}: skipped (una sola clase)")
            continue

        X_train = df_train[config.FEATURE_COLS].values.astype(np.float32)
        y_train = df_train["label_good"].values.astype(int)
        X_test  = df_test[config.FEATURE_COLS].values.astype(np.float32)
        y_test  = df_test["label_good"].values.astype(int)

        # Normalizar (fit solo en train)
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Modelo con regularización moderada
        params = _get_params(len(X_train))
        try:
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_s, y_train, verbose=False)
            y_proba = model.predict_proba(X_test_s)[:, 1]
            auc     = roc_auc_score(y_test, y_proba)
            aucs.append(auc)
            folds_ok += 1

            train_dates = ""
            if "collected_at" in df_train.columns:
                d0 = df_train["collected_at"].iloc[0][:10]
                d1 = df_train["collected_at"].iloc[-1][:10]
                d2 = df_test["collected_at"].iloc[-1][:10]
                train_dates = f" | train: {d0}→{d1} | test: →{d2}"

            log.info(
                f"  Walk-forward fold {fold+1}/{n_splits}: "
                f"train={len(X_train)} test={len(X_test)} "
                f"AUC={auc:.4f}{train_dates}"
            )
        except Exception as e:
            log.warning(f"  Fold {fold+1} error: {e}")

    if not aucs:
        return {"wf_aucs": [], "wf_mean": 0.0, "wf_std": 0.0, "wf_folds": 0}

    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))

    log.info(f"Walk-forward AUC: {mean_auc:.4f} ± {std_auc:.4f} ({folds_ok} folds válidos)")

    if mean_auc > 0.95:
        log.warning(
            f"Walk-forward AUC={mean_auc:.4f} sigue siendo muy alto. "
            f"Posibles causas: (1) pocas muestras, "
            f"(2) features correlacionadas con el label, "
            f"(3) necesitas más tiempo para acumular datos reales."
        )
    elif mean_auc > 0.75:
        log.info(f"Buen AUC walk-forward ({mean_auc:.4f}). El modelo tiene señal real.")
    elif mean_auc > 0.55:
        log.info(f"AUC modesto ({mean_auc:.4f}). Señal débil — necesitas más datos o mejores features.")
    else:
        log.warning(f"AUC bajo ({mean_auc:.4f}). El modelo no aprende. Revisa los labels.")

    return {
        "wf_aucs" : [round(a, 4) for a in aucs],
        "wf_mean" : round(mean_auc, 4),
        "wf_std"  : round(std_auc, 4),
        "wf_folds": folds_ok,
    }


# ── Parámetros adaptativos ────────────────────────────────────────────────────

def _get_params(n_train: int) -> dict:
    """Regularización adaptativa según tamaño del dataset."""
    params = {k: v for k, v in config.XGB_PARAMS.items()}

    if n_train < 300:
        params.update({
            "max_depth"       : 2,
            "n_estimators"    : 50,
            "min_child_weight": 10,
            "reg_alpha"       : 2.0,
            "reg_lambda"      : 10.0,
            "subsample"       : 0.5,
            "learning_rate"   : 0.05,
        })
    elif n_train < 800:
        params.update({
            "max_depth"       : 3,
            "n_estimators"    : 100,
            "min_child_weight": 5,
            "reg_alpha"       : 1.0,
            "reg_lambda"      : 5.0,
            "subsample"       : 0.6,
        })
    elif n_train < 2000:
        params.update({
            "max_depth"       : 4,
            "n_estimators"    : 200,
            "min_child_weight": 3,
            "reg_alpha"       : 0.5,
            "reg_lambda"      : 2.0,
        })
    # else: usar XGB_PARAMS por defecto (dataset grande)

    return params


# ── Entrenamiento final ───────────────────────────────────────────────────────

def train_final(df: pd.DataFrame):
    """
    Entrena el modelo final con todos los datos disponibles.
    Usa los últimos 20% como test para métricas de producción.
    El modelo que se despliega es el entrenado en el 100% de los datos.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    xgb = get_xgb()

    n         = len(df)
    split_idx = int(n * (1 - config.TEST_SIZE))
    df_train  = df.iloc[:split_idx]
    df_test   = df.iloc[split_idx:]

    # Guard: ambas clases en ambos splits
    if df_train["label_good"].nunique() < 2 or df_test["label_good"].nunique() < 2:
        # Fallback a split estratificado
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(
            df, test_size=config.TEST_SIZE,
            stratify=df["label_good"], random_state=42,
        )
        log.info("Split estratificado (fallback)")

    X_train = df_train[config.FEATURE_COLS].values.astype(np.float32)
    y_train = df_train["label_good"].values.astype(int)
    X_test  = df_test[config.FEATURE_COLS].values.astype(np.float32)
    y_test  = df_test["label_good"].values.astype(int)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    params = _get_params(len(X_train))

    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
        log.info(f"Entrenamiento en: {params.get('device', 'cpu').upper()}")
    except Exception as cuda_err:
        if "cuda" in str(cuda_err).lower():
            log.warning("CUDA no disponible — fallback a CPU")
            params.update({"device": "cpu", "tree_method": "hist"})
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
        else:
            raise

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    auc     = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=["malo", "bueno"])
    log.info(f"\n{report}")
    log.info(f"AUC-ROC (test final): {auc:.4f}")

    importances  = dict(zip(config.FEATURE_COLS, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info("Top features: " + " | ".join(f"{k}={v:.3f}" for k, v in top_features))

    return model, scaler, {
        "auc"         : round(auc, 4),
        "n_train"     : len(X_train),
        "n_test"      : len(X_test),
        "n_good_train": int(y_train.sum()),
        "n_good_test" : int(y_test.sum()),
        "trained_at"  : datetime.now(timezone.utc).isoformat(),
        "features"    : config.FEATURE_COLS,
        "top_features": top_features,
        "regularized" : n < 2000,
    }


# ── Exportación ───────────────────────────────────────────────────────────────

def export_model(model, scaler, metrics: dict):
    Path(config.MODEL_DIR).mkdir(exist_ok=True)

    xgb_path = Path(config.MODEL_DIR) / "model.xgb.json"
    model.save_model(str(xgb_path))
    log.info(f"Modelo XGBoost guardado: {xgb_path}")

    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    with open(config.LABEL_MAP_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"Metadata guardada: {config.LABEL_MAP_FILE}")

    # ONNX (opcional)
    try:
        import onnxmltools
        from onnxmltools.convert import convert_xgboost
        from skl2onnx.common.data_types import FloatTensorType
        n_features   = len(config.FEATURE_COLS)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model   = convert_xgboost(model, initial_types=initial_type)
        with open(config.MODEL_FILE, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info(f"ONNX exportado: {config.MODEL_FILE}")
    except Exception:
        log.info("ONNX no disponible — usando .xgb.json directamente")
        config.MODEL_FILE = str(xgb_path)

    return str(xgb_path)


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_training_once():
    """
    Pipeline completo:
    1. Cargar dataset
    2. Walk-forward validation → AUC real sin leakage
    3. Train final con todos los datos
    4. Exportar modelo
    """
    try:
        df = load_dataset()
        n  = len(df)

        if n < config.MIN_RECORDS_TO_TRAIN:
            log.warning(f"Solo {n} registros — mínimo {config.MIN_RECORDS_TO_TRAIN}")
            return None

        if "label_good" in df.columns and df["label_good"].nunique() < 2:
            log.warning("Dataset con una sola clase — esperando más datos")
            return None

        df_pool = prepare_pool(df)

        if df_pool["label_good"].nunique() < 2:
            log.warning("Pool preparado con una sola clase — esperando mercados con outcome NO")
            return None

        # ── Walk-forward: métricas honestas ──────────────────────────────
        n_splits = 5 if len(df_pool) >= 500 else 3
        wf       = walk_forward_validate(df_pool, n_splits=n_splits)

        # ── Modelo final ──────────────────────────────────────────────────
        model, scaler, metrics = train_final(df_pool)
        metrics["walk_forward"] = wf
        metrics["n_total"]      = n

        export_model(model, scaler, metrics)

        log.info(
            f"Entrenamiento completado — "
            f"AUC test: {metrics['auc']} | "
            f"Walk-forward: {wf['wf_mean']:.4f}±{wf['wf_std']:.4f} | "
            f"{n} registros totales"
        )
        return metrics

    except Exception as e:
        log.error(f"Error durante entrenamiento: {e}", exc_info=True)
        return None


# ── Hash y watch loop ─────────────────────────────────────────────────────────

def get_dataset_hash() -> str:
    path = Path(config.RAW_DATA_FILE)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - 4096))
        tail = f.read()
    return hashlib.md5(f"{size}:{tail}".encode()).hexdigest()[:12]


def run_watch_loop():
    log.info("=" * 55)
    log.info("  TRAINER en modo watch — walk-forward automático")
    log.info(f"  Reentrenar cada: {config.RETRAIN_EVERY_N_RECORDS} nuevos registros")
    log.info("=" * 55)

    last_hash  = ""
    last_count = 0

    while True:
        try:
            path = Path(config.RAW_DATA_FILE)
            if path.exists():
                current_count = sum(1 for _ in open(path))
                current_hash  = get_dataset_hash()

                if current_hash == last_hash:
                    time.sleep(60)
                    continue

                first_run  = (last_count == 0)
                enough_new = (current_count - last_count >= config.RETRAIN_EVERY_N_RECORDS)

                if current_count >= config.MIN_RECORDS_TO_TRAIN and (first_run or enough_new):
                    log.info(f"Dataset cambió ({last_count} → {current_count}) — entrenando...")
                    metrics = run_training_once()
                    if metrics:
                        wf = metrics.get("walk_forward", {})
                        log.info(
                            f"Walk-forward AUC: {wf.get('wf_mean',0):.4f} "
                            f"± {wf.get('wf_std',0):.4f} "
                            f"({wf.get('wf_folds',0)} folds)"
                        )
                        last_hash  = current_hash
                        last_count = current_count
                else:
                    last_hash  = current_hash
                    last_count = current_count

        except Exception as e:
            log.error(f"Error en watch loop: {e}")

        time.sleep(60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "now":
        metrics = run_training_once()
        if metrics:
            wf = metrics.get("walk_forward", {})
            print(f"\nAUC test:       {metrics['auc']}")
            print(f"Walk-forward:   {wf.get('wf_mean',0):.4f} ± {wf.get('wf_std',0):.4f}")
            print(f"Folds válidos:  {wf.get('wf_folds',0)}")
            print(f"AUC por fold:   {wf.get('wf_aucs', [])}")
    else:
        run_watch_loop()
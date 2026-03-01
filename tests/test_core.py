"""
tests/test_core.py
==================
Tests básicos (smoke tests) para las funciones críticas del sistema.

Ejecutar con:
    python -m pytest tests/ -v
"""

import json
import sys
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures comunes
# ─────────────────────────────────────────────────────────────────────────────

VALID_MARKET = {
    "conditionId"   : "0xabc123",
    "question"      : "¿Ganará el equipo X el campeonato?",
    "volume24hr"    : 15000.0,
    "volume"        : 250000.0,
    "liquidity"     : 40000.0,
    "uniqueTraders" : 180,
    "outcomePrices" : ["0.72", "0.28"],
    "endDate"       : "2026-06-01T00:00:00Z",
    "category"      : "sports",
    "closed"        : False,
    "priceChange24hr": 0.03,
    "clobTokenIds"  : ["token_yes_001", "token_no_001"],
    "events"        : [{"slug": "campeonato-x", "id": "evt_001", "title": "Campeonato X"}],
}

MINIMAL_MARKET = {
    "id"           : "market_minimal",
    "title"        : "Minimal market",
    "volumeTotalUsd": 1000.0,
    "outcomePrices": [0.5, 0.5],
}

MALFORMED_MARKETS = [
    {},                                # vacío
    {"conditionId": "x"},              # sin volumen
    {"volume": "not_a_number",
     "conditionId": "y"},              # tipo incorrecto
    None,                              # None — no debería explotar
]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: collector.extract_record
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractRecord:

    def test_valid_market_returns_dict(self):
        from collector import extract_record
        rec = extract_record(VALID_MARKET)
        assert rec is not None
        assert isinstance(rec, dict)

    def test_valid_market_has_all_features(self):
        import config
        from collector import extract_record
        rec = extract_record(VALID_MARKET)
        for col in config.FEATURE_COLS:
            assert col in rec, f"Columna faltante: {col}"

    def test_valid_market_spread_default(self):
        from collector import extract_record
        rec = extract_record(VALID_MARKET)
        assert "spread" in rec
        assert rec["spread"] == 0.05  # valor por defecto

    def test_price_yes_parsed_correctly(self):
        from collector import extract_record
        rec = extract_record(VALID_MARKET)
        assert abs(rec["price_yes"] - 0.72) < 1e-4

    def test_category_encoded(self):
        from collector import extract_record
        rec = extract_record(VALID_MARKET)
        assert rec["category_encoded"] == 3  # sports → 3

    def test_minimal_market_ok(self):
        from collector import extract_record
        rec = extract_record(MINIMAL_MARKET)
        assert rec is not None

    def test_below_min_volume_returns_none(self):
        import config
        from collector import extract_record
        low_vol = {**VALID_MARKET, "volume": config.MIN_VOLUME_TO_STORE - 1, "volume24hr": 0}
        rec = extract_record(low_vol)
        assert rec is None

    def test_malformed_market_does_not_raise(self):
        from collector import extract_record
        for bad in MALFORMED_MARKETS:
            try:
                result = extract_record(bad)
                # puede devolver None, pero no debe lanzar excepción
                assert result is None or isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"extract_record lanzó excepción con input {bad!r}: {e}")

    def test_no_nan_in_feature_cols(self):
        import config
        from collector import extract_record
        rec = extract_record(VALID_MARKET)
        for col in config.FEATURE_COLS:
            val = rec.get(col, 0.0)
            assert not math.isnan(val), f"NaN en columna {col}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: collector.heuristic_score
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicScore:

    def test_high_quality_market_scores_high(self):
        from collector import heuristic_score
        features = {
            "liquidity"      : 100_000,
            "volume_24h"     : 50_000,
            "price_yes"      : 0.72,
            "num_traders"    : 500,
            "days_to_close"  : 7,
            "price_change_24h": 0.05,
            "volume_total"   : 1_000_000,
        }
        score = heuristic_score(features)
        assert score >= 60, f"Se esperaba score alto, got {score}"

    def test_low_quality_market_scores_low(self):
        from collector import heuristic_score
        features = {
            "liquidity"      : 500,
            "volume_24h"     : 100,
            "price_yes"      : 0.50,
            "num_traders"    : 5,
            "days_to_close"  : 120,
            "price_change_24h": 0.0,
            "volume_total"   : 1_000,
        }
        score = heuristic_score(features)
        assert score < 30, f"Se esperaba score bajo, got {score}"

    def test_score_is_between_0_and_100(self):
        from collector import heuristic_score
        for price in [0.1, 0.3, 0.5, 0.7, 0.9]:
            features = {
                "liquidity"      : 20_000,
                "volume_24h"     : 10_000,
                "price_yes"      : price,
                "num_traders"    : 100,
                "days_to_close"  : 14,
                "price_change_24h": 0.02,
                "volume_total"   : 100_000,
            }
            score = heuristic_score(features)
            assert 0 <= score <= 100, f"Score fuera de rango: {score}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: collector.fetch_book_spread
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchBookSpread:

    def test_returns_spread_from_book(self):
        from collector import fetch_book_spread
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "bids": [{"price": "0.68"}, {"price": "0.67"}],
            "asks": [{"price": "0.72"}, {"price": "0.73"}],
        }
        with patch("collector.requests.get", return_value=mock_response):
            spread = fetch_book_spread("token_abc")
        assert abs(spread - 0.04) < 1e-4  # 0.72 - 0.68

    def test_returns_default_on_api_error(self):
        from collector import fetch_book_spread
        with patch("collector.requests.get", side_effect=Exception("timeout")):
            spread = fetch_book_spread("token_abc")
        assert spread == 0.05

    def test_returns_default_on_empty_book(self):
        from collector import fetch_book_spread
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"bids": [], "asks": []}
        with patch("collector.requests.get", return_value=mock_response):
            spread = fetch_book_spread("token_abc")
        assert spread == 0.05

    def test_returns_default_on_http_error(self):
        from collector import fetch_book_spread
        mock_response = MagicMock()
        mock_response.ok = False
        with patch("collector.requests.get", return_value=mock_response):
            spread = fetch_book_spread("token_abc")
        assert spread == 0.05

    def test_spread_is_non_negative(self):
        from collector import fetch_book_spread
        # Caso raro: bids > asks (cruce) → debe devolver 0
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "bids": [{"price": "0.75"}],
            "asks": [{"price": "0.70"}],
        }
        with patch("collector.requests.get", return_value=mock_response):
            spread = fetch_book_spread("token_abc")
        assert spread >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: collector.enrich_with_spreads
# ─────────────────────────────────────────────────────────────────────────────

class TestEnrichWithSpreads:

    def _make_records_and_raws(self, n=5):
        records = [
            {"volume_24h": (n - i) * 1000, "spread": 0.05}
            for i in range(n)
        ]
        raws = [
            {"clobTokenIds": [f"token_{i}"]}
            for i in range(n)
        ]
        return records, raws

    def test_enriches_top_n_records_in_place(self):
        from collector import enrich_with_spreads
        records, raws = self._make_records_and_raws(5)
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "bids": [{"price": "0.60"}],
            "asks": [{"price": "0.72"}],   # spread = 0.12 ≠ default 0.05
        }
        with patch("collector.requests.get", return_value=mock_response):
            with patch("collector.time.sleep"):  # no esperar en tests
                enrich_with_spreads(records, raws, top_n=3)
        # Los 3 primeros (mayor volumen) deben tener spread real (0.12)
        enriched = [r for r in records if abs(r["spread"] - 0.12) < 1e-4]
        assert len(enriched) == 3

    def test_empty_inputs_do_not_raise(self):
        from collector import enrich_with_spreads
        enrich_with_spreads([], [], top_n=10)   # no debe explotar


# ─────────────────────────────────────────────────────────────────────────────
# Tests: config._detect_device
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectDevice:

    def test_returns_cuda_when_nvidia_smi_available(self):
        from config import _detect_device
        with patch("config.subprocess.check_output", return_value=b"GPU 0"):
            device = _detect_device()
        assert device == "cuda"

    def test_returns_cpu_when_nvidia_smi_missing(self):
        from config import _detect_device
        with patch("config.subprocess.check_output", side_effect=FileNotFoundError):
            device = _detect_device()
        assert device == "cpu"

    def test_xgb_params_has_device_key(self):
        import config
        assert "device" in config.XGB_PARAMS
        assert config.XGB_PARAMS["device"] in ("cuda", "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: inference.DriftMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftMonitor:

    def _make_monitor(self):
        from inference import DriftMonitor
        return DriftMonitor(warmup_cycles=3, window=3, threshold=0.10)

    def _warmup(self, monitor, score=0.65, n=3):
        """Lleva al monitor al estado post-calentamiento."""
        for _ in range(n):
            monitor.update(np.array([score] * 10))

    def test_baseline_established_after_warmup(self):
        monitor = self._make_monitor()
        self._warmup(monitor, score=0.65)
        assert monitor.baseline_mean is not None
        assert abs(monitor.baseline_mean - 0.65) < 0.01

    def test_no_alert_within_threshold(self):
        from inference import DriftMonitor
        monitor = self._make_monitor()
        self._warmup(monitor, score=0.65)
        # Pequeño movimiento — no dispara alerta
        with patch("inference.emit_alert") as mock_emit:
            for _ in range(5):
                monitor.update(np.array([0.62] * 10))  # -0.03 < threshold=0.10
            mock_emit.assert_not_called()

    def test_alert_on_significant_drop(self):
        from inference import DriftMonitor
        monitor = self._make_monitor()
        self._warmup(monitor, score=0.65)
        with patch("inference.emit_alert") as mock_emit:
            for _ in range(5):
                monitor.update(np.array([0.40] * 10))  # -0.25 > threshold=0.10
            mock_emit.assert_called()
            call_args = mock_emit.call_args[0][0]
            assert call_args["type"] == "MODEL_DRIFT"
            assert call_args["drop"] > 0

    def test_alert_on_significant_rise(self):
        """Una subida anómala también debe detectarse."""
        from inference import DriftMonitor
        monitor = self._make_monitor()
        self._warmup(monitor, score=0.40)
        with patch("inference.emit_alert") as mock_emit:
            for _ in range(5):
                monitor.update(np.array([0.80] * 10))  # +0.40 > threshold
            mock_emit.assert_called()

    def test_empty_array_does_not_raise(self):
        monitor = self._make_monitor()
        self._warmup(monitor)
        monitor.update(np.array([]))   # no debe explotar

    def test_baseline_resets_after_drift(self):
        """Tras detectar drift, el baseline se actualiza para no re-alertar."""
        from inference import DriftMonitor
        monitor = self._make_monitor()
        self._warmup(monitor, score=0.65)
        old_baseline = monitor.baseline_mean
        with patch("inference.emit_alert"):
            for _ in range(5):
                monitor.update(np.array([0.40] * 10))
        assert monitor.baseline_mean != old_baseline


# ─────────────────────────────────────────────────────────────────────────────
# Tests: trainer.prepare_pool
# ─────────────────────────────────────────────────────────────────────────────

class TestPreparePool:

    def _make_df(self, n=50):
        import pandas as pd
        import config
        from datetime import datetime, timezone, timedelta

        rows = []
        for i in range(n):
            row = {col: float(i % 10) / 10 for col in config.FEATURE_COLS}
            row.update({
                "market_id"     : f"market_{i}",
                "collected_at"  : (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i*6)).isoformat(),
                "label_good"    : i % 2,
                "has_real_label": True,
                "resolved"      : False,
                "snapshot_key"  : f"market_{i}_snap",
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def test_prepare_pool_returns_dataframe(self):
        from trainer import prepare_pool
        df = self._make_df()
        result = prepare_pool(df)
        import pandas as pd
        assert isinstance(result, pd.DataFrame)

    def test_prepare_pool_has_label_col(self):
        from trainer import prepare_pool
        df = self._make_df()
        result = prepare_pool(df)
        assert "label_good" in result.columns

    def test_prepare_pool_no_nans_in_features(self):
        import config
        from trainer import prepare_pool
        df = self._make_df()
        result = prepare_pool(df)
        for col in config.FEATURE_COLS:
            assert result[col].isna().sum() == 0, f"NaN en columna {col}"

    def test_prepare_pool_clips_outliers(self):
        from trainer import prepare_pool
        df = self._make_df()
        df["volume_24h"] = 999_999_999  # outlier extremo
        result = prepare_pool(df)
        assert result["volume_24h"].max() <= 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Tests: config.FEATURE_COLS
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_feature_cols_contains_spread(self):
        import config
        assert "spread" in config.FEATURE_COLS

    def test_feature_cols_no_duplicates(self):
        import config
        assert len(config.FEATURE_COLS) == len(set(config.FEATURE_COLS))

    def test_inference_threshold_in_range(self):
        import config
        assert 0.0 < config.GOOD_MARKET_PROB_THRESHOLD < 1.0

"""
collector.py
============
Recolector autónomo de datos de Polymarket.

CAMBIOS PRINCIPALES vs versión anterior:
- save_records() ahora ACTUALIZA registros cuando un mercado resuelve
  (antes solo hacía append, nunca actualizaba labels existentes)
- update_resolved_labels() nueva función que verifica mercados pendientes
  y obtiene sus labels reales cuando la API los marca como cerrados
- collect_once() llama a update_resolved_labels() en cada pasada
- Se preserva el price_yes ORIGINAL (antes del cierre) para que el modelo
  vea la señal predictiva, no el precio post-resolución

FLUJO CORRECTO:
  Día 0:   mercado activo guardado → has_real_label=False, price_yes=0.73
  Día 14:  mercado resuelve YES    → API devuelve closed=True, outcomePrices=["1","0"]
  Día 14+: collector detecta el cambio → label_good=1, price_yes=0.73 (preservado)
  Trainer: entrena con has_real_label=True → modelo aprende señal real
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config

# ── Logging ──────────────────────────────────────────────────────────────────
Path(config.LOG_DIR).mkdir(exist_ok=True)
Path(config.DATA_DIR).mkdir(exist_ok=True)
Path(config.MODEL_DIR).mkdir(exist_ok=True)

log = config.get_logger("collector")

# ── Helpers ───────────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    "politics": 0, "elections": 1, "crypto": 2, "sports": 3,
    "economics": 4, "finance": 4, "pop culture": 5, "science": 6,
    "world": 7, "unknown": 8,
}

def safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default

def safe_int(v, default=0):
    try:
        return int(v) if v is not None else default
    except (ValueError, TypeError):
        return default

def parse_price_yes(market: dict) -> float:
    raw = market.get("outcomePrices")
    if isinstance(raw, list) and raw:
        return safe_float(raw[0], 0.5)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return safe_float(parsed[0], 0.5) if parsed else 0.5
        except Exception:
            return safe_float(raw, 0.5)
    return 0.5

def parse_days_to_close(market: dict) -> float:
    for key in ("endDate", "endDateIso", "end_date"):
        val = market.get(key)
        if val:
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return max((dt - datetime.now(timezone.utc)).days, 0)
            except Exception:
                pass
    return 30.0

def heuristic_score(f: dict) -> float:
    """Score heurístico para generar labels de entrenamiento."""
    score = 0.0
    score += min(f["liquidity"] / 50_000, 1.0) * 25
    score += min(f["volume_24h"] / 20_000, 1.0) * 20
    score += (abs(f["price_yes"] - 0.5) / 0.5) * 15
    score += min(f["num_traders"] / 200, 1.0) * 15
    d = f["days_to_close"]
    if d > 0:
        if d <= 14:
            score += (d / 14) * 10
        else:
            score += max(0, 1 - (d - 14) / 60) * 10
    score += min(abs(f["price_change_24h"]) / 0.08, 1.0) * 10
    score += min(f["volume_total"] / 200_000, 1.0) * 5
    return round(score, 2)

def extract_record(market: dict) -> dict | None:
    """Extrae y normaliza un mercado en un dict de features + metadata."""
    try:
        vol24  = safe_float(market.get("volume24hr") or market.get("volumeNum"))
        voltot = safe_float(market.get("volume") or market.get("volumeTotalUsd"))
        liq    = safe_float(market.get("liquidity") or market.get("liquidityNum"))

        if voltot < config.MIN_VOLUME_TO_STORE:
            return None

        price_yes     = parse_price_yes(market)
        days_to_close = parse_days_to_close(market)

        age_days  = max(90 - days_to_close, 1)
        vol_accel = vol24 / (voltot / age_days + 1e-9)

        category_raw = (market.get("category") or market.get("tag") or "unknown").lower()
        category_enc = CATEGORY_MAP.get(category_raw, 8)
        for k in CATEGORY_MAP:
            if k in category_raw:
                category_enc = CATEGORY_MAP[k]
                break

        f = {
            # metadata
            "collected_at"  : datetime.now(timezone.utc).isoformat(),
            "market_id"     : market.get("conditionId") or market.get("id", ""),
            "question"      : market.get("question", market.get("title", ""))[:120],
            "slug"          : get_event_slug(
                                market.get("conditionId") or market.get("id", ""),
                                market
                              ),
            "category_raw"  : category_raw,
            "resolved"      : market.get("closed", False),
            "resolution"    : market.get("resolutionPrice"),

            # features
            "volume_24h"                : vol24,
            "volume_total"              : voltot,
            "liquidity"                 : liq,
            "num_traders"               : safe_int(market.get("uniqueTraders") or market.get("numTraders")),
            "price_yes"                 : price_yes,
            "price_distance_from_50"    : abs(price_yes - 0.5),
            "days_to_close"             : days_to_close,
            "price_change_24h"          : abs(safe_float(market.get("priceChange24hr") or market.get("lastPriceChange"))),
            "liquidity_to_volume_ratio" : liq / (vol24 + 1e-9),
            "volume_acceleration"       : min(vol_accel, 100.0),
            "category_encoded"          : category_enc,
            "spread"                    : 0.05,  # default; enrich_with_spreads() lo sobreescribe
        }

        # ── Label real basado en outcomePrices ───────────────────────────────
        f["has_real_label"]   = False
        f["resolution_price"] = None

        is_closed = market.get("closed", False) or market.get("active") == False
        if is_closed:
            try:
                raw_prices = market.get("outcomePrices", "[]")
                if isinstance(raw_prices, str):
                    prices = json.loads(raw_prices)
                else:
                    prices = raw_prices

                if prices and len(prices) >= 2:
                    p_yes = safe_float(prices[0])
                    p_no  = safe_float(prices[1])

                    if p_yes >= 0.99:
                        resolved_yes = True
                    elif p_no >= 0.99:
                        resolved_yes = False
                    elif p_yes > p_no:
                        resolved_yes = True
                    elif p_no > p_yes:
                        resolved_yes = False
                    else:
                        ltp = safe_float(market.get("lastTradePrice", "0.5"))
                        resolved_yes = ltp >= 0.5

                    # NOTA: price_yes aquí es el precio POST-resolución (siempre ~1.0 o ~0.0)
                    # Para mercados resueltos descargados del histórico esto es inevitable.
                    # Para mercados que seguimos desde activos, update_resolved_labels()
                    # preservará el price_yes del snapshot original (precio ANTES del cierre).

                    # Edge: el precio ANTES del cierre apuntaba al outcome correcto
                    # Solo aplicable cuando tenemos el precio pre-resolución real.
                    # Para histórico: usamos el precio post como proxy (menos fiable).
                    had_edge = (
                        (resolved_yes and price_yes >= 0.60) or
                        (not resolved_yes and price_yes <= 0.40)
                    )

                    f["label_good"]       = int(had_edge)
                    f["has_real_label"]   = True
                    f["resolution_price"] = 1.0 if resolved_yes else 0.0

            except Exception as _e:
                log.debug(f"Error parseando outcomePrices: {_e}")

        if not f["has_real_label"]:
            # Fallback heurístico solo para mercados activos sin resolución
            f["heuristic_score"] = heuristic_score(f)
            f["label_good"]      = int(f["heuristic_score"] >= config.GOOD_MARKET_THRESHOLD)

        return f
    except Exception as e:
        log.debug(f"Error extrayendo record: {e}")
        return None


# ── Fetchers ──────────────────────────────────────────────────────────────────

# ── CLOB spread enrichment ────────────────────────────────────────────
def fetch_book_spread(token_id: str) -> float:
    """
    Consulta el order book del CLOB para un token YES y devuelve
    el spread bid-ask (best_ask - best_bid).
    Devuelve 0.05 si el book está vacío o hay un error.
    """
    try:
        r = requests.get(
            f"{config.CLOB_API}/book",
            params={"token_id": token_id},
            timeout=5, verify=False,
        )
        if not r.ok:
            return 0.05
        data  = r.json()
        bids  = data.get("bids", [])
        asks  = data.get("asks", [])
        if not bids or not asks:
            return 0.05
        best_bid = max(safe_float(b.get("price", 0)) for b in bids)
        best_ask = min(safe_float(a.get("price", 1)) for a in asks)
        spread   = best_ask - best_bid
        return max(0.0, round(spread, 4))
    except Exception:
        return 0.05


def enrich_with_spreads(records: list, markets_raw: list, top_n: int = 100) -> None:
    """
    Actualiza in-place el campo 'spread' de los top_n records (por volume_24h)
    con el spread bid-ask real del CLOB.
    Los records fuera del top_n conservan el valor por defecto (0.05).
    """
    if not records or not markets_raw:
        return

    # Indices ordenados por volumen desc
    indexed = sorted(
        range(min(len(records), len(markets_raw))),
        key=lambda i: records[i].get("volume_24h", 0),
        reverse=True,
    )

    for idx in indexed[:top_n]:
        raw = markets_raw[idx]
        token_ids = raw.get("clobTokenIds", [])
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except Exception:
                token_ids = []
        if token_ids:
            records[idx]["spread"] = fetch_book_spread(str(token_ids[0]))
            time.sleep(0.05)   # ~200 req/min máximo


def fetch_active_markets(offset=0, limit=100) -> list:
    """Pagina automáticamente hasta obtener todos los mercados activos disponibles."""
    all_markets  = []
    page_size    = 500
    current_off  = offset

    while len(all_markets) < limit:
        try:
            r = requests.get(
                f"{config.GAMMA_API}/markets",
                params={"active": "true", "closed": "false",
                        "limit": page_size, "offset": current_off,
                        "order": "volume24hr", "ascending": "false"},
                timeout=15, verify=False,
            )
            r.raise_for_status()
            data = r.json()
            page = data if isinstance(data, list) else data.get("markets", data.get("data", []))

            if not page:
                break

            all_markets.extend(page)
            current_off += len(page)

            if len(page) < page_size:
                break  # última página, no hay más

            time.sleep(0.3)

        except Exception as e:
            log.warning(f"fetch_active_markets error (offset={current_off}): {e}")
            break

    return all_markets

def fetch_resolved_markets(limit=200) -> list:
    try:
        r = requests.get(
            f"{config.GAMMA_API}/markets",
            params={"active": "false", "closed": "true",
                    "limit": limit, "order": "volume", "ascending": "false"},
            timeout=15, verify=False,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("markets", data.get("data", []))
    except Exception as e:
        log.warning(f"fetch_resolved_markets error: {e}")
        return []


def fetch_resolved_paginated(target=3000) -> list:
    all_markets = []
    offset      = 0
    page_size   = 200
    log.info(f"Iniciando scraping historico paginado (objetivo: {target} mercados)...")

    while len(all_markets) < target:
        try:
            r = requests.get(
                f"{config.GAMMA_API}/markets",
                params={
                    "active"    : "false",
                    "closed"    : "true",
                    "limit"     : page_size,
                    "offset"    : offset,
                    "order"     : "startDate",
                    "ascending" : "false",
                },
                timeout=20, verify=False,
            )
            r.raise_for_status()
            data = r.json()
            page = data if isinstance(data, list) else data.get("markets", data.get("data", []))

            if not page:
                log.info(f"  Sin mas paginas en offset {offset}")
                break

            all_markets.extend(page)
            log.info(f"  Pagina offset={offset}: {len(page)} mercados | total acumulado: {len(all_markets)}")
            offset += page_size
            time.sleep(0.3)

        except Exception as e:
            log.warning(f"  Error en pagina offset={offset}: {e}")
            time.sleep(2)
            break

    log.info(f"Scraping historico completado: {len(all_markets)} mercados obtenidos")
    return all_markets


def fetch_historical_via_graph(days_back=90) -> list:
    since_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    query = """
    {
      fixedProductMarketMakers(
        first: 200
        orderBy: scaledCollateralVolume
        orderDirection: desc
        where: { creationTimestamp_gt: %d }
      ) {
        id
        question { id title }
        collateralVolume
        scaledCollateralVolume
        liquidityParameter
        outcomeTokenPrices
        resolutionTimestamp
        creator
      }
    }
    """ % since_ts

    try:
        r = requests.post(
            config.GRAPH_API,
            json={"query": query},
            timeout=20, verify=False,
        )
        r.raise_for_status()
        return r.json().get("data", {}).get("fixedProductMarketMakers", [])
    except Exception as e:
        log.warning(f"fetch_historical_via_graph error: {e}")
        return []


# ── Storage ───────────────────────────────────────────────────────────────────

def count_records() -> int:
    path = Path(config.RAW_DATA_FILE)
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f)


# ── Snapshot interval: guardar nuevo snapshot cada N horas ───────────────────
SNAPSHOT_INTERVAL_HOURS = 6   # snapshot nuevo del mismo mercado cada 6h

def _snapshot_key(market_id: str, collected_at: str) -> str:
    """Clave única para un snapshot: market_id + ventana de 6h."""
    try:
        dt    = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
        # Redondear a ventana de SNAPSHOT_INTERVAL_HOURS
        bucket = dt.replace(
            hour=(dt.hour // SNAPSHOT_INTERVAL_HOURS) * SNAPSHOT_INTERVAL_HOURS,
            minute=0, second=0, microsecond=0
        )
        return f"{market_id}_{bucket.strftime('%Y%m%d_%H')}"
    except Exception:
        return market_id


def save_records(records: list) -> int:
    """
    Guarda snapshots en JSONL con dos comportamientos:

    1. Mercados ACTIVOS: guarda un snapshot nuevo cada SNAPSHOT_INTERVAL_HOURS.
       Esto permite ver la evolución del precio del mercado en el tiempo.
       El modelo aprenderá: "precio en T → outcome en T+cierre".

    2. Mercados RESUELTOS: actualiza el snapshot más reciente del mercado
       preservando el price_yes original (pre-resolución).

    La clave de deduplicación es snapshot_key = market_id + ventana_temporal.
    """
    if not records:
        return 0

    path = Path(config.RAW_DATA_FILE)

    # Cargar dataset: índice por snapshot_key + índice por market_id (para resoluciones)
    existing_snapshots = {}   # snapshot_key -> record (para dedup de activos)
    latest_by_market   = {}   # market_id    -> record más reciente sin label real

    if path.exists():
        with open(path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    mid = obj.get("market_id", "")
                    cat = obj.get("collected_at", "")
                    if not mid:
                        continue
                    sk = obj.get("snapshot_key") or _snapshot_key(mid, cat)
                    existing_snapshots[sk] = obj
                    # Rastrear el más reciente sin label real (para preservar price_yes al resolver)
                    if not obj.get("has_real_label") and not obj.get("resolved"):
                        prev = latest_by_market.get(mid)
                        if prev is None or cat > prev.get("collected_at", ""):
                            latest_by_market[mid] = obj
                except Exception:
                    pass

    saved   = 0
    updated = 0

    for r in records:
        mid = r.get("market_id", "")
        if not mid:
            continue

        cat = r.get("collected_at", datetime.now(timezone.utc).isoformat())
        sk  = _snapshot_key(mid, cat)
        r["snapshot_key"] = sk

        # ── Caso 1: Mercado acaba de resolver ────────────────────────────
        if r.get("has_real_label") or r.get("resolved"):
            prev_active = latest_by_market.get(mid)
            if prev_active:
                # Preservar el price_yes ORIGINAL (antes del cierre)
                original_price_yes = prev_active.get("price_yes")
                if original_price_yes is not None:
                    r["price_yes_at_resolution"] = r.get("price_yes")
                    r["price_yes"]               = original_price_yes
                    r["price_distance_from_50"]  = abs(original_price_yes - 0.5)
                    # Recalcular label con precio real pre-resolución
                    resolved_yes = r.get("resolution_price", 0.5) >= 0.5
                    had_edge = (
                        (resolved_yes     and original_price_yes >= 0.60) or
                        (not resolved_yes and original_price_yes <= 0.40)
                    )
                    r["label_good"]    = int(had_edge)
                r["first_seen_at"] = prev_active.get("collected_at")
                # Actualizar el snapshot original con el label real
                prev_sk = prev_active.get("snapshot_key") or _snapshot_key(mid, prev_active.get("collected_at",""))
                existing_snapshots[prev_sk] = r
                updated += 1
            else:
                # No tenemos snapshot previo — guardar como nuevo igualmente
                existing_snapshots[sk] = r
                saved += 1

        # ── Caso 2: Mercado activo — guardar snapshot si no existe ────────
        elif sk not in existing_snapshots:
            existing_snapshots[sk] = r
            saved += 1
        # else: snapshot de esta ventana ya existe → no guardar duplicado

    if saved > 0 or updated > 0:
        with open(path, "w") as f:
            for r in existing_snapshots.values():
                f.write(json.dumps(r) + "\n")

    if updated > 0:
        log.info(f"  Labels actualizados: {updated} resueltos con price_yes original preservado")

    return saved + updated


# ── Update de labels para mercados pendientes ─────────────────────────────────

def update_resolved_labels() -> int:
    """
    Busca en el dataset local mercados sin label real,
    los consulta en la API para ver si ya resolvieron,
    y actualiza sus labels preservando el price_yes original.

    Esta es la función clave para la Opción A:
    acumula snapshots activos hoy → espera resolución → etiqueta con precio real.
    """
    path = Path(config.RAW_DATA_FILE)
    if not path.exists():
        return 0

    unlabeled = []
    with open(path, "r") as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("has_real_label") and r.get("market_id"):
                    unlabeled.append(r)
            except Exception:
                pass

    if not unlabeled:
        return 0

    log.info(f"  Verificando {len(unlabeled)} mercados sin label real...")

    updated     = 0
    batch_size  = 50

    for i in range(0, len(unlabeled), batch_size):
        batch = unlabeled[i:i+batch_size]
        for r in batch:
            mid = r.get("market_id", "")
            try:
                resp = requests.get(
                    f"{config.GAMMA_API}/markets/{mid}",
                    timeout=8, verify=False,
                )
                if not resp.ok:
                    continue

                market = resp.json()
                # Si la respuesta es una lista, tomar el primero
                if isinstance(market, list):
                    market = market[0] if market else {}

                is_closed = market.get("closed", False) or market.get("active") == False
                if not is_closed:
                    continue

                new_record = extract_record(market)
                if new_record and new_record.get("has_real_label"):
                    # save_records() preservará el price_yes original
                    n = save_records([new_record])
                    updated += n

            except Exception as e:
                log.debug(f"Error verificando {mid}: {e}")

        time.sleep(0.2)

    if updated > 0:
        log.info(f"  ✅ {updated} labels reales nuevos obtenidos via update_resolved_labels")
    else:
        log.debug("  Sin nuevas resoluciones detectadas")

    return updated


# ── Event slug enrichment ────────────────────────────────────────────────────

_event_slug_cache = {}

def refresh_event_slug_cache():
    global _event_slug_cache
    try:
        r = requests.get(
            f"{config.GAMMA_API}/events",
            params={"limit": 100, "active": "true", "closed": "false"},
            timeout=15, verify=False,
        )
        if not r.ok:
            return
        data = r.json()
        events = data if isinstance(data, list) else data.get("events", data.get("data", []))
        found = 0
        for event in events:
            event_slug = event.get("slug", "")
            if not event_slug:
                continue
            for mkt in event.get("markets", []):
                cid = mkt.get("conditionId") or mkt.get("id", "")
                if cid:
                    _event_slug_cache[cid] = event_slug
                    found += 1
        log.debug(f"Event slug cache: {found} entradas")
    except Exception as e:
        log.debug(f"refresh_event_slug_cache error: {e}")


def get_event_slug(condition_id: str, market: dict) -> str:
    events = market.get("events", [])
    if events and isinstance(events, list):
        slug = events[0].get("slug", "")
        if slug:
            if condition_id:
                _event_slug_cache[condition_id] = slug
            return slug

    if condition_id and condition_id in _event_slug_cache:
        return _event_slug_cache[condition_id]

    return market.get("slug", "")


# ── Pipeline principal ────────────────────────────────────────────────────────

def collect_once() -> int:
    """Una pasada de recolección. Devuelve total de nuevos registros."""
    total_new = 0

    refresh_event_slug_cache()

    # 1. Mercados activos (snapshots para futura etiquetación)
    log.info("Descargando mercados activos...")
    active  = fetch_active_markets(limit=config.MARKETS_PER_FETCH)
    records = [r for m in active if (r := extract_record(m)) is not None]
    enrich_with_spreads(records, active, top_n=100)
    n = save_records(records)
    log.info(f"  Activos: {len(active)} mercados → {n} nuevos guardados")
    total_new += n

    # 2. Mercados resueltos recientes (labels inmediatos)
    log.info("Descargando mercados resueltos...")
    resolved = fetch_resolved_markets(limit=200)
    records_r = [r for m in resolved if (r := extract_record(m)) is not None]
    n2 = save_records(records_r)
    log.info(f"  Resueltos: {len(resolved)} mercados → {n2} nuevos guardados")
    total_new += n2

    # 3. ── CLAVE: actualizar labels de mercados que ya resolvieron ──────────
    # Busca mercados que guardamos como activos y ahora la API los marca como cerrados
    # Preserva el price_yes original para que el modelo vea la señal predictiva real
    log.info("Verificando resoluciones pendientes...")
    update_resolved_labels()

    total = count_records()
    log.info(f"  Total en dataset: {total} registros")
    return total_new


def collect_historical_bootstrap():
    """
    Bootstrap histórico: descarga ~3000 mercados resueltos.
    NOTA: estos tendrán price_yes post-resolución (menos fiable).
    Los mercados activos que guardemos HOY serán los que generen
    labels de verdad cuando resuelvan en días/semanas.
    """
    log.info("=== BOOTSTRAP HISTÓRICO ===")

    resolved = fetch_resolved_paginated(target=config.HISTORICAL_TARGET)
    records  = [r for m in resolved if (r := extract_record(m)) is not None]
    real     = sum(1 for r in records if r.get("has_real_label"))
    n        = save_records(records)
    log.info(f"  Resueltos paginados: {len(resolved)} mercados → {n} guardados ({real} con label real)")

    collect_once()

    total = count_records()
    log.info(f"Bootstrap completo. Total: {total} registros")
    log.info(f"  ⏳ Los labels de VERDAD se acumularán conforme los mercados activos vayan resolviendo")
    log.info(f"  📅 Espera al menos 1-2 semanas para tener suficientes labels reales balanceados")


def _get_last_collect_time():
    """
    Devuelve el datetime del registro más reciente en el dataset.
    Lee solo el final del archivo para ser eficiente con datasets grandes.
    """
    path = Path(config.RAW_DATA_FILE)
    if not path.exists():
        return None
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="ignore")
        last_ts = None
        for line in tail.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                ts  = obj.get("collected_at", "")
                if ts and (last_ts is None or ts > last_ts):
                    last_ts = ts
            except Exception:
                pass
        if last_ts:
            return datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    except Exception:
        pass
    return None


def run_loop():
    """Loop autónomo continuo."""
    log.info("=" * 55)
    log.info("  COLLECTOR arrancado en modo autónomo")
    log.info(f"  Intervalo: {config.COLLECT_INTERVAL_SECONDS}s")
    log.info("=" * 55)

    if count_records() < config.MIN_RECORDS_TO_TRAIN:
        log.info("Dataset insuficiente — iniciando bootstrap histórico...")
        collect_historical_bootstrap()

    # Catch-up: si el proceso estuvo caído, recolectar inmediatamente
    last_ts = _get_last_collect_time()
    if last_ts:
        gap_s = (datetime.now(timezone.utc) - last_ts).total_seconds()
        if gap_s > config.COLLECT_INTERVAL_SECONDS * 2:
            log.info(
                f"Catch-up detectado: último registro hace "
                f"{gap_s/3600:.1f}h — recolectando inmediatamente..."
            )
            try:
                collect_once()
            except Exception as e:
                log.error(f"Error en catch-up inicial: {e}")

    while True:
        try:
            new = collect_once()
            if new > 0:
                log.info(f"  +{new} nuevos registros esta pasada")
        except Exception as e:
            log.error(f"Error en collect_once: {e}")

        time.sleep(config.COLLECT_INTERVAL_SECONDS)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "bootstrap":
        collect_historical_bootstrap()
    else:
        run_loop()
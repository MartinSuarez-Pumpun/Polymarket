import requests
from dataclasses import dataclass, field
from typing import Optional

GAMMA_API = "https://gamma-api.polymarket.com"

# ------------------------------------------------------------------

@dataclass
class MarketFeatures:
    question     : str
    market_id    : str
    volume_24h   : float
    volume_total : float
    liquidity    : float
    num_traders  : int
    price_yes    : float        # entre 0.0 y 1.0
    days_to_close: float
    price_change_24h: float     # cambio de precio en 24h (absoluto)
    category     : str


def safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0) -> int:
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def extract_features(market: dict) -> Optional[MarketFeatures]:
    try:
        # Precio YES: puede venir como string "0.72" o lista ["0.72", "0.28"]
        raw_prices = market.get("outcomePrices", None)
        if isinstance(raw_prices, list) and raw_prices:
            price_yes = safe_float(raw_prices[0], 0.5)
        elif isinstance(raw_prices, str):
            # a veces viene como JSON string '["0.72","0.28"]'
            import json
            try:
                parsed = json.loads(raw_prices)
                price_yes = safe_float(parsed[0], 0.5)
            except Exception:
                price_yes = safe_float(raw_prices, 0.5)
        else:
            price_yes = 0.5

        # Días hasta cierre
        end_date = market.get("endDate") or market.get("endDateIso")
        if end_date:
            from datetime import datetime, timezone
            try:
                close_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                now_dt   = datetime.now(timezone.utc)
                days_to_close = max((close_dt - now_dt).days, 0)
            except Exception:
                days_to_close = 30
        else:
            days_to_close = 30

        return MarketFeatures(
            question      = market.get("question", market.get("title", ""))[:80],
            market_id     = market.get("conditionId") or market.get("id", ""),
            volume_24h    = safe_float(market.get("volume24hr") or market.get("volumeNum")),
            volume_total  = safe_float(market.get("volume") or market.get("volumeTotalUsd")),
            liquidity     = safe_float(market.get("liquidity") or market.get("liquidityNum")),
            num_traders   = safe_int(market.get("uniqueTraders") or market.get("numTraders")),
            price_yes     = price_yes,
            days_to_close = days_to_close,
            price_change_24h = abs(safe_float(market.get("priceChange24hr") or market.get("lastPriceChange"))),
            category      = (market.get("category") or market.get("tag") or "unknown").lower(),
        )
    except Exception as e:
        return None


# ------------------------------------------------------------------

class MarketScorer:
    """
    Modelo heurístico interpretable (sin LLM, sin caja negra).
    Puntuación 0-100 basada en features de calidad del mercado.
    """

    CATEGORY_WEIGHTS = {
        "politics"  : 1.20,
        "elections" : 1.20,
        "economics" : 1.15,
        "sports"    : 1.10,
        "crypto"    : 0.90,   # más manipulable
        "pop culture": 0.85,
        "unknown"   : 1.00,
    }

    def score(self, f: MarketFeatures) -> dict:
        s = {}

        # 1. LIQUIDEZ — mercados líquidos = spread bajo = más fiable       [0-25]
        s["liquidity"]  = min(f.liquidity / 50_000, 1.0) * 25

        # 2. VOLUMEN 24H — actividad reciente indica interés real           [0-20]
        s["volume"]     = min(f.volume_24h / 20_000, 1.0) * 20

        # 3. PRECIO ALEJADO DEL 50% — hay información, no es 50/50         [0-15]
        edge            = abs(f.price_yes - 0.5)
        s["edge"]       = (edge / 0.5) * 15

        # 4. TRADERS ÚNICOS — diversidad, no un solo whale                 [0-15]
        s["traders"]    = min(f.num_traders / 200, 1.0) * 15

        # 5. TIMING — ventana óptima 3-14 días                             [0-10]
        #    muy corto (< 2d) o muy largo (> 60d) penaliza
        if f.days_to_close <= 0:
            s["timing"] = 0
        elif f.days_to_close <= 14:
            s["timing"] = (f.days_to_close / 14) * 10
        else:
            s["timing"] = max(0, 1 - (f.days_to_close - 14) / 60) * 10

        # 6. VOLATILIDAD RECIENTE — movimiento = oportunidad               [0-10]
        s["volatility"] = min(f.price_change_24h / 0.08, 1.0) * 10

        # 7. VOLUMEN TOTAL — historial de interés                          [0-5]
        s["volume_total"] = min(f.volume_total / 200_000, 1.0) * 5

        # Multiplicador por categoría
        cat_mult = self.CATEGORY_WEIGHTS.get(f.category, 1.0)
        # Buscar match parcial si no es exacto
        if f.category not in self.CATEGORY_WEIGHTS:
            for k in self.CATEGORY_WEIGHTS:
                if k in f.category or f.category in k:
                    cat_mult = self.CATEGORY_WEIGHTS[k]
                    break

        raw_total = sum(s.values())
        total     = min(raw_total * cat_mult, 100)

        return {
            "total_score"         : round(total, 1),
            "breakdown"           : {k: round(v, 1) for k, v in s.items()},
            "verdict"             : self._verdict(total),
            "category_multiplier" : cat_mult,
        }

    def _verdict(self, score: float) -> str:
        if score >= 75: return "🟢 EXCELENTE"
        if score >= 55: return "🟡 BUENO"
        if score >= 35: return "🟠 MEDIOCRE"
        return                 "🔴 EVITAR"


# ------------------------------------------------------------------

def get_markets(limit=50) -> list:
    url    = f"{GAMMA_API}/markets"
    params = {
        "active"   : "true",
        "closed"   : "false",
        "limit"    : limit,
        "order"    : "volume24hr",
        "ascending": "false",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, list):
        return data
    return data.get("markets", data.get("data", []))


def analyze_top_markets(limit=50, show_top=20, min_score=0):
    print(f"📡 Descargando top {limit} mercados de Polymarket...")
    markets = get_markets(limit=limit)
    print(f"   → {len(markets)} mercados recibidos\n")

    if not markets:
        print("❌ No se recibieron mercados. Revisa la conexión o que la API esté activa.")
        return

    scorer  = MarketScorer()
    results = []

    for m in markets:
        f = extract_features(m)
        if not f:
            continue
        r = scorer.score(f)
        results.append({
            "name"        : f.question,
            "score"       : r["total_score"],
            "verdict"     : r["verdict"],
            "price_yes"   : f.price_yes,
            "volume_24h"  : f.volume_24h,
            "liquidity"   : f.liquidity,
            "traders"     : f.num_traders,
            "days_left"   : f.days_to_close,
            "category"    : f.category,
            "breakdown"   : r["breakdown"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    filtered = [r for r in results if r["score"] >= min_score]

    print(f"🏆 TOP MERCADOS POLYMARKET  (mostrando top {min(show_top, len(filtered))} de {len(filtered)})")
    print("=" * 72)

    for i, r in enumerate(filtered[:show_top], 1):
        print(f"\n#{i:02d} {r['verdict']}  [{r['score']}/100]  [{r['category']}]")
        print(f"     {r['name']}")
        print(
            f"     YES: {r['price_yes']:.1%} | "
            f"Vol24h: ${r['volume_24h']:>8,.0f} | "
            f"Liq: ${r['liquidity']:>8,.0f} | "
            f"Traders: {r['traders']:>4} | "
            f"Días: {r['days_left']}"
        )
        # breakdown compacto
        bd = r["breakdown"]
        print(
            f"     liq={bd['liquidity']} vol={bd['volume']} edge={bd['edge']} "
            f"traders={bd['traders']} timing={bd['timing']} vola={bd['volatility']}"
        )

    print("\n" + "=" * 72)
    verdicts = [r["verdict"] for r in filtered[:show_top]]
    print(f"  🟢 Excelente: {verdicts.count('🟢 EXCELENTE')}  "
          f"🟡 Bueno: {verdicts.count('🟡 BUENO')}  "
          f"🟠 Mediocre: {verdicts.count('🟠 MEDIOCRE')}  "
          f"🔴 Evitar: {verdicts.count('🔴 EVITAR')}")


if __name__ == "__main__":
    analyze_top_markets(limit=50, show_top=20)
import requests
import time
from datetime import datetime
from collections import defaultdict

# APIs de Polymarket
POLYMARKET_API = "https://clob.polymarket.com"
GAMMA_API      = "https://gamma-api.polymarket.com"

# ---- CONFIG ----
WHALE_THRESHOLD_USD  = 5_000   # alerta si alguien mete más de esto de golpe
PRICE_MOVE_THRESHOLD = 0.04    # alerta si el precio mueve +4% en un trade
CHECK_INTERVAL       = 10      # segundos entre checks
TOP_MARKETS          = 30      # cuántos mercados monitorizar
# ----------------

# --- Telegram (opcional) ---
TELEGRAM_ENABLED = False
TELEGRAM_TOKEN   = "TU_TOKEN_AQUI"
TELEGRAM_CHAT_ID = "TU_CHAT_ID_AQUI"
import requests
import time
from datetime import datetime
from collections import defaultdict

# APIs de Polymarket
POLYMARKET_API = "https://clob.polymarket.com"
GAMMA_API      = "https://gamma-api.polymarket.com"

# ---- CONFIG ----
WHALE_THRESHOLD_USD  = 5_000   # alerta si alguien mete más de esto de golpe
PRICE_MOVE_THRESHOLD = 0.04    # alerta si el precio mueve +4% en un trade
CHECK_INTERVAL       = 10      # segundos entre checks
TOP_MARKETS          = 30      # cuántos mercados monitorizar
# ----------------

# --- Telegram (opcional) ---
TELEGRAM_ENABLED = False
TELEGRAM_TOKEN   = "TU_TOKEN_AQUI"
TELEGRAM_CHAT_ID = "TU_CHAT_ID_AQUI"

def send_telegram(text: str):
    if not TELEGRAM_ENABLED:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
    except Exception as e:
        print(f"  [Telegram error: {e}]")

# ------------------------------------------------------------------

def get_active_markets(limit=50):
    """Obtiene mercados activos ordenados por volumen 24h"""
    url = f"{GAMMA_API}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume24hr",
        "ascending": "false"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        # La API puede devolver lista directa o {"markets": [...]}
        if isinstance(data, list):
            return data
        return data.get("markets", data.get("data", []))
    except Exception as e:
        print(f"  [get_active_markets error: {e}]")
        return []


def get_recent_trades(market_id: str, limit=20):
    """Obtiene trades recientes de un mercado via CLOB API"""
    url = f"{POLYMARKET_API}/trades"
    params = {"market": market_id, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("data", data.get("trades", []))
    except Exception as e:
        return []


# ------------------------------------------------------------------

class WhaleDetector:
    def __init__(self):
        self.seen_trades  = defaultdict(set)   # market_id -> set de trade IDs ya procesados
        self.price_cache  = {}                  # "market_id_outcome" -> último precio

    def check_market(self, market: dict):
        # Intentar obtener el ID del mercado (distintas claves según endpoint)
        market_id   = market.get("conditionId") or market.get("id") or market.get("marketMakerAddress", "")
        market_name = market.get("question", market.get("title", "Mercado desconocido"))

        if not market_id:
            return []

        trades  = get_recent_trades(market_id)
        alerts  = []

        for trade in trades:
            trade_id = trade.get("id") or trade.get("transactionHash", "")
            if not trade_id or trade_id in self.seen_trades[market_id]:
                continue
            self.seen_trades[market_id].add(trade_id)

            try:
                size    = float(trade.get("size", trade.get("amount", 0)))
                price   = float(trade.get("price", 0.5))
                outcome = trade.get("outcome", trade.get("side", ""))
                side    = trade.get("makerSide", trade.get("side", ""))
                usd_size = size * price
            except (ValueError, TypeError):
                continue

            # ── WHALE TRADE ──────────────────────────────────────────
            if usd_size >= WHALE_THRESHOLD_USD:
                alert = {
                    "type"      : "WHALE_TRADE",
                    "market"    : market_name,
                    "market_id" : market_id,
                    "usd_size"  : usd_size,
                    "side"      : side,
                    "outcome"   : outcome,
                    "price"     : price,
                    "timestamp" : datetime.now().isoformat(),
                }
                alerts.append(alert)

            # ── PRICE SPIKE ──────────────────────────────────────────
            cache_key  = f"{market_id}_{outcome}"
            last_price = self.price_cache.get(cache_key)
            if last_price is not None and abs(price - last_price) >= PRICE_MOVE_THRESHOLD:
                alert = {
                    "type"        : "PRICE_SPIKE",
                    "market"      : market_name,
                    "market_id"   : market_id,
                    "price_before": last_price,
                    "price_after" : price,
                    "move_pct"    : round((price - last_price) * 100, 2),
                    "usd_size"    : usd_size,
                    "outcome"     : outcome,
                    "timestamp"   : datetime.now().isoformat(),
                }
                alerts.append(alert)

            self.price_cache[cache_key] = price

        return alerts


def send_alert(alert: dict):
    ts = datetime.now().strftime("%H:%M:%S")

    if alert["type"] == "WHALE_TRADE":
        line1 = f"\n🐋 WHALE DETECTED — {ts}"
        line2 = f"   Mercado : {alert['market'][:65]}"
        line3 = f"   Outcome : {alert['outcome']} @ {alert['price']:.3f}"
        line4 = f"   Tamaño  : ${alert['usd_size']:,.0f} USDC  [{str(alert['side']).upper()}]"
        msg   = "\n".join([line1, line2, line3, line4])
        print(msg)
        send_telegram(f"🐋 <b>WHALE</b> ${alert['usd_size']:,.0f}\n{alert['market'][:60]}\n{alert['outcome']} @ {alert['price']:.3f}")

    elif alert["type"] == "PRICE_SPIKE":
        direction = "📈" if alert["move_pct"] > 0 else "📉"
        line1 = f"\n⚡{direction} PRICE SPIKE — {ts}"
        line2 = f"   Mercado : {alert['market'][:65]}"
        line3 = f"   Outcome : {alert['outcome']}"
        line4 = f"   Precio  : {alert['price_before']:.3f} → {alert['price_after']:.3f}  ({alert['move_pct']:+.1f}%)"
        line5 = f"   Trade   : ${alert['usd_size']:,.0f} USDC"
        msg   = "\n".join([line1, line2, line3, line4, line5])
        print(msg)
        send_telegram(f"⚡ <b>SPIKE {alert['move_pct']:+.1f}%</b>\n{alert['market'][:60]}\n{alert['outcome']}: {alert['price_before']:.3f}→{alert['price_after']:.3f}")


def run_monitor():
    detector = WhaleDetector()
    print(f"🚀 Monitor iniciado — Whale threshold: ${WHALE_THRESHOLD_USD:,} | Price spike: {PRICE_MOVE_THRESHOLD*100:.0f}%")
    print(f"   Chequeando cada {CHECK_INTERVAL}s | Top {TOP_MARKETS} mercados por volumen\n")

    tick = 0
    while True:
        try:
            markets     = get_active_markets(limit=TOP_MARKETS)
            all_alerts  = []

            for market in markets:
                alerts = detector.check_market(market)
                all_alerts.extend(alerts)

            for alert in all_alerts:
                send_alert(alert)

            if not all_alerts:
                tick += 1
                print(
                    f"  · [{tick:04d}] {datetime.now().strftime('%H:%M:%S')} "
                    f"— sin movimientos relevantes ({len(markets)} mercados)",
                    end="\r"
                )

        except KeyboardInterrupt:
            print("\n\n👋 Monitor detenido.")
            break
        except Exception as e:
            print(f"\n⚠️  Error en loop principal: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_monitor()
def send_telegram(text: str):
    if not TELEGRAM_ENABLED:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=5)
    except Exception as e:
        print(f"  [Telegram error: {e}]")

# ------------------------------------------------------------------

def get_active_markets(limit=50):
    """Obtiene mercados activos ordenados por volumen 24h"""
    url = f"{GAMMA_API}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume24hr",
        "ascending": "false"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        # La API puede devolver lista directa o {"markets": [...]}
        if isinstance(data, list):
            return data
        return data.get("markets", data.get("data", []))
    except Exception as e:
        print(f"  [get_active_markets error: {e}]")
        return []


def get_recent_trades(market_id: str, limit=20):
    """Obtiene trades recientes de un mercado via CLOB API"""
    url = f"{POLYMARKET_API}/trades"
    params = {"market": market_id, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("data", data.get("trades", []))
    except Exception as e:
        return []


# ------------------------------------------------------------------

class WhaleDetector:
    def __init__(self):
        self.seen_trades  = defaultdict(set)   # market_id -> set de trade IDs ya procesados
        self.price_cache  = {}                  # "market_id_outcome" -> último precio

    def check_market(self, market: dict):
        # Intentar obtener el ID del mercado (distintas claves según endpoint)
        market_id   = market.get("conditionId") or market.get("id") or market.get("marketMakerAddress", "")
        market_name = market.get("question", market.get("title", "Mercado desconocido"))

        if not market_id:
            return []

        trades  = get_recent_trades(market_id)
        alerts  = []

        for trade in trades:
            trade_id = trade.get("id") or trade.get("transactionHash", "")
            if not trade_id or trade_id in self.seen_trades[market_id]:
                continue
            self.seen_trades[market_id].add(trade_id)

            try:
                size    = float(trade.get("size", trade.get("amount", 0)))
                price   = float(trade.get("price", 0.5))
                outcome = trade.get("outcome", trade.get("side", ""))
                side    = trade.get("makerSide", trade.get("side", ""))
                usd_size = size * price
            except (ValueError, TypeError):
                continue

            # ── WHALE TRADE ──────────────────────────────────────────
            if usd_size >= WHALE_THRESHOLD_USD:
                alert = {
                    "type"      : "WHALE_TRADE",
                    "market"    : market_name,
                    "market_id" : market_id,
                    "usd_size"  : usd_size,
                    "side"      : side,
                    "outcome"   : outcome,
                    "price"     : price,
                    "timestamp" : datetime.now().isoformat(),
                }
                alerts.append(alert)

            # ── PRICE SPIKE ──────────────────────────────────────────
            cache_key  = f"{market_id}_{outcome}"
            last_price = self.price_cache.get(cache_key)
            if last_price is not None and abs(price - last_price) >= PRICE_MOVE_THRESHOLD:
                alert = {
                    "type"        : "PRICE_SPIKE",
                    "market"      : market_name,
                    "market_id"   : market_id,
                    "price_before": last_price,
                    "price_after" : price,
                    "move_pct"    : round((price - last_price) * 100, 2),
                    "usd_size"    : usd_size,
                    "outcome"     : outcome,
                    "timestamp"   : datetime.now().isoformat(),
                }
                alerts.append(alert)

            self.price_cache[cache_key] = price

        return alerts


def send_alert(alert: dict):
    ts = datetime.now().strftime("%H:%M:%S")

    if alert["type"] == "WHALE_TRADE":
        line1 = f"\n🐋 WHALE DETECTED — {ts}"
        line2 = f"   Mercado : {alert['market'][:65]}"
        line3 = f"   Outcome : {alert['outcome']} @ {alert['price']:.3f}"
        line4 = f"   Tamaño  : ${alert['usd_size']:,.0f} USDC  [{str(alert['side']).upper()}]"
        msg   = "\n".join([line1, line2, line3, line4])
        print(msg)
        send_telegram(f"🐋 <b>WHALE</b> ${alert['usd_size']:,.0f}\n{alert['market'][:60]}\n{alert['outcome']} @ {alert['price']:.3f}")

    elif alert["type"] == "PRICE_SPIKE":
        direction = "📈" if alert["move_pct"] > 0 else "📉"
        line1 = f"\n⚡{direction} PRICE SPIKE — {ts}"
        line2 = f"   Mercado : {alert['market'][:65]}"
        line3 = f"   Outcome : {alert['outcome']}"
        line4 = f"   Precio  : {alert['price_before']:.3f} → {alert['price_after']:.3f}  ({alert['move_pct']:+.1f}%)"
        line5 = f"   Trade   : ${alert['usd_size']:,.0f} USDC"
        msg   = "\n".join([line1, line2, line3, line4, line5])
        print(msg)
        send_telegram(f"⚡ <b>SPIKE {alert['move_pct']:+.1f}%</b>\n{alert['market'][:60]}\n{alert['outcome']}: {alert['price_before']:.3f}→{alert['price_after']:.3f}")


def run_monitor():
    detector = WhaleDetector()
    print(f"🚀 Monitor iniciado — Whale threshold: ${WHALE_THRESHOLD_USD:,} | Price spike: {PRICE_MOVE_THRESHOLD*100:.0f}%")
    print(f"   Chequeando cada {CHECK_INTERVAL}s | Top {TOP_MARKETS} mercados por volumen\n")

    tick = 0
    while True:
        try:
            markets     = get_active_markets(limit=TOP_MARKETS)
            all_alerts  = []

            for market in markets:
                alerts = detector.check_market(market)
                all_alerts.extend(alerts)

            for alert in all_alerts:
                send_alert(alert)

            if not all_alerts:
                tick += 1
                print(
                    f"  · [{tick:04d}] {datetime.now().strftime('%H:%M:%S')} "
                    f"— sin movimientos relevantes ({len(markets)} mercados)",
                    end="\r"
                )

        except KeyboardInterrupt:
            print("\n\n👋 Monitor detenido.")
            break
        except Exception as e:
            print(f"\n⚠️  Error en loop principal: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_monitor()
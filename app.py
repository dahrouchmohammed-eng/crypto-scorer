from flask import Flask, request, jsonify
import numpy as np
import urllib.request
import urllib.parse
import urllib.error
import json
import time
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("crypto-scorer")

app = Flask(__name__)

# Verrou process-local pour les accès au fichier cooldown.
# (protège contre la concurrence intra-process / multi-thread ;
#  pour du multi-process gunicorn, voir note dans save_cooldown)
_COOLDOWN_LOCK = threading.Lock()

# Parallélisme des appels réseau par symbole.
MAX_WORKERS = 8

# ─── SOURCES DE DONNÉES ─────────────────────────────────────────────────────────
# Binance (fapi/api .binance.com) est souvent bloqué (HTTP 451) depuis des IP cloud
# comme Railway. On le désactive par défaut pour échouer vite et basculer sur Bybit
# puis Binance Vision Spot. Réactivable via variable d'environnement :
#   BINANCE_ENABLED=true
BINANCE_ENABLED = os.environ.get("BINANCE_ENABLED", "false").lower() == "true"

# Timeouts courts : si une source est bloquée, on ne veut pas attendre longtemps.
HTTP_TIMEOUT  = int(os.environ.get("HTTP_TIMEOUT", "8"))
HTTP_RETRIES  = int(os.environ.get("HTTP_RETRIES", "2"))

# ─── CONFIG V5 ────────────────────────────────────────────────────────────────
# SAFE_MODE=True = conserve une logique prudente tant que Binance Futures reste instable.
SAFE_MODE = True

# Universe dynamique : on part de toutes les paires USDT disponibles via ticker 24h,
# puis on garde uniquement les plus liquides / actives.
MAX_DYNAMIC_UNIVERSE = 90
MAX_FULL_ANALYSIS_CANDIDATES = 18
MIN_QUOTE_VOLUME_USDT = 15_000_000

TIER1_ALWAYS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
EXCLUDED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
EXCLUDED_SYMBOLS = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT"}

# Classes d'actifs pour SL / levier (factorisé : était dupliqué dans score_symbol)
MEMECOINS = {"DOGEUSDT", "WIFUSDT", "PEPEUSDT", "BONKUSDT", "FLOKIUSDT", "BOMEUSDT"}
MAJORS    = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


# ─── COOLDOWN ─────────────────────────────────────────────────────────────────
# Cooldown déclenché APRÈS envoi Telegram via /set_cooldown
# PAS dans /full_analysis — pour ne pas bloquer des candidats non sélectionnés

COOLDOWN_FILE    = "/tmp/cooldown.json"
COOLDOWN_SECONDS = 4 * 3600

def load_cooldown():
    try:
        if os.path.exists(COOLDOWN_FILE):
            with open(COOLDOWN_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("load_cooldown a échoué: %s", e)
    return {}

def save_cooldown(data):
    # Écriture atomique : on écrit dans un fichier temporaire puis os.replace,
    # ce qui évite un fichier JSON tronqué/corrompu en cas d'interruption.
    # NB: pour du multi-process (gunicorn -w N), envisager un store partagé
    # (Redis) car ce verrou est local au process.
    try:
        tmp = COOLDOWN_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, COOLDOWN_FILE)
    except Exception as e:
        logger.warning("save_cooldown a échoué: %s", e)

def is_on_cooldown(symbol):
    with _COOLDOWN_LOCK:
        cooldown = load_cooldown()
    if symbol in cooldown:
        elapsed = time.time() - cooldown[symbol]
        if elapsed < COOLDOWN_SECONDS:
            remaining = round((COOLDOWN_SECONDS - elapsed) / 3600, 1)
            return True, remaining
    return False, 0

def set_cooldown_symbols(symbols):
    with _COOLDOWN_LOCK:
        cooldown = load_cooldown()
        now = time.time()
        for symbol in symbols:
            cooldown[symbol] = now
        cooldown = {k: v for k, v in cooldown.items() if now - v < COOLDOWN_SECONDS}
        save_cooldown(cooldown)

# ─── LOG DES SIGNAUX (traçabilité performance) ──────────────────────────────────
# Chaque signal émis est journalisé en JSONL pour pouvoir, plus tard, mesurer
# combien de signaux par seuil et leur taux de réussite réel — et calibrer les
# seuils sur des chiffres plutôt que sur l'intuition.
SIGNAL_LOG_FILE = os.environ.get("SIGNAL_LOG_FILE", "/tmp/signals.jsonl")
_SIGNAL_LOG_LOCK = threading.Lock()

def log_signal(entry):
    try:
        record = {
            "ts": int(time.time()),
            "symbol": entry.get("symbol"),
            "flag": entry.get("flag"),
            "score": entry.get("score"),
            "confidence": entry.get("confidence"),
            "direction": entry.get("direction"),
            "entry_type": entry.get("entry_type"),
            "trend_strength": entry.get("trend_strength"),
            "late_entry_risk": entry.get("late_entry_risk"),
            "risk_reward": entry.get("risk_reward"),
            "data_source": entry.get("data_source"),
            "source_quality": entry.get("source_quality"),
            "market_regime": entry.get("market_regime"),
        }
        with _SIGNAL_LOG_LOCK:
            with open(SIGNAL_LOG_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("log_signal a échoué: %s", e)

# ─── BINANCE API ──────────────────────────────────────────────────────────────

def fetch_binance(url):
    last_err = None
    for attempt in range(HTTP_RETRIES):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last_err = e
            # 451 = bloqué par localisation. 403 = bloqué/interdit. Inutile de retenter.
            if e.code in (451, 403):
                logger.warning("fetch échec %s: HTTP %s (bloqué) — pas de retry",
                               url.split("?")[0], e.code)
                return None
            if attempt == HTTP_RETRIES - 1:
                logger.warning("fetch échec %s: HTTP %s", url.split("?")[0], e.code)
                return None
        except Exception as e:
            last_err = e
            if attempt == HTTP_RETRIES - 1:
                logger.warning("fetch échec %s: %s", url.split("?")[0], last_err)
                return None
    return None

def normalize_bybit_ticker(t):
    """Convertit un ticker Bybit v5 linear vers le format interne type Binance."""
    try:
        price_change_pct = float(t.get("price24hPcnt", 0)) * 100
    except Exception:
        price_change_pct = 0.0
    return {
        "symbol": t.get("symbol", ""),
        "priceChangePercent": str(round(price_change_pct, 6)),
        "quoteVolume": str(t.get("turnover24h", 0)),
        "volume": str(t.get("volume24h", 0)),
        "highPrice": str(t.get("highPrice24h", 0)),
        "lowPrice": str(t.get("lowPrice24h", 0)),
        "lastPrice": str(t.get("lastPrice", 0)),
    }

def fetch_bybit_linear_tickers(symbols=None):
    """Récupère les tickers Bybit USDT perpetual et les normalise."""
    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    data = fetch_binance(url)
    if not data or data.get("retCode") != 0:
        return None
    items = data.get("result", {}).get("list", [])
    wanted = {normalize_symbol(s) for s in symbols} if symbols else None
    out = []
    for t in items:
        sym = normalize_symbol(t.get("symbol", ""))
        if wanted and sym not in wanted:
            continue
        if not is_tradeable_usdt_symbol(sym):
            continue
        nt = normalize_bybit_ticker(t)
        if nt.get("symbol"):
            out.append(nt)
    return out

def bybit_interval(interval):
    mapping = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D"
    }
    return mapping.get(interval, "60")

def get_bybit_klines(symbol, interval="1h", limit=100):
    url = (
        "https://api.bybit.com/v5/market/kline"
        f"?category=linear&symbol={symbol}&interval={bybit_interval(interval)}&limit={limit}"
    )
    data = fetch_binance(url)
    if not data or data.get("retCode") != 0:
        return None
    rows = data.get("result", {}).get("list", [])
    if not rows:
        return None
    # Bybit renvoie souvent les bougies de la plus récente à la plus ancienne.
    rows = sorted(rows, key=lambda x: int(x[0]))
    out = []
    for r in rows:
        # Format Bybit: [startTime, open, high, low, close, volume, turnover]
        # Format interne compatible Binance: indices 2/3/4/5 utilisés plus bas.
        out.append([r[0], r[1], r[2], r[3], r[4], r[5]])
    return out

def get_klines(symbol, interval="1h", limit=100):
    """
    Retourne les chandeliers + source.
    Par défaut Bybit Futures est tenté en premier car Binance est souvent bloqué
    depuis Railway. Binance Futures n'est tenté que si BINANCE_ENABLED=true.
    Dernier recours : Binance Vision Spot, puis Binance Spot si activé.
    """
    if BINANCE_ENABLED:
        url_futures = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        result = fetch_binance(url_futures)
        if result is not None:
            return result, "FUTURES"

    result = get_bybit_klines(symbol, interval, limit)
    if result is not None:
        return result, "BYBIT_FUTURES"

    spot_endpoints = [
        "https://data-api.binance.vision/api/v3/klines",
    ]
    if BINANCE_ENABLED:
        spot_endpoints.append("https://api.binance.com/api/v3/klines")

    for base in spot_endpoints:
        url_spot = f"{base}?symbol={symbol}&interval={interval}&limit={limit}"
        result = fetch_binance(url_spot)
        if result is not None:
            return result, "SPOT_FALLBACK"

    return None, "UNAVAILABLE"

def get_funding_rate(symbol, data_source="FUTURES"):
    """Retourne (funding_rate, disponible) pour Binance Futures ou Bybit Futures."""
    if data_source == "FUTURES":
        try:
            url  = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            data = fetch_binance(url)
            if data and len(data) > 0:
                return float(data[0].get("fundingRate", 0)), True
        except Exception as e:
            logger.warning("get_funding_rate Binance échec %s: %s", symbol, e)
        return 0.0, False

    if data_source == "BYBIT_FUTURES":
        try:
            url = f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol}&limit=1"
            data = fetch_binance(url)
            if data and data.get("retCode") == 0:
                rows = data.get("result", {}).get("list", [])
                if rows:
                    return float(rows[0].get("fundingRate", 0)), True
        except Exception as e:
            logger.warning("get_funding_rate Bybit échec %s: %s", symbol, e)
        return 0.0, False

    return 0.0, False

def interpret_funding(funding_rate, direction, available=True):
    """
    Interprétation simple du funding futures.
    Funding positif élevé  : longs crowded.
    Funding négatif élevé  : shorts crowded.
    La lecture dépend de la direction du trade.
    Si le funding est indisponible (source spot), on ne déduit rien.
    """
    if not available or funding_rate is None:
        return "unavailable", "neutral", "funding indisponible (source spot)", 0

    if funding_rate is None:
        funding_rate = 0.0

    if funding_rate > 0.0005:
        funding_signal = "longs crowded"
    elif funding_rate < -0.0005:
        funding_signal = "shorts crowded"
    else:
        funding_signal = "neutral"

    confidence_adjustment = 0
    derivatives_bias = "neutral"
    derivatives_note = "funding neutral"

    if funding_signal == "longs crowded":
        if direction == "LONG":
            confidence_adjustment = -5
            derivatives_bias = "caution"
            derivatives_note = "longs crowded, prudence sur LONG"
        elif direction == "SHORT":
            confidence_adjustment = 3
            derivatives_bias = "supports short"
            derivatives_note = "longs crowded, contexte favorable au SHORT"
        else:
            derivatives_note = "longs crowded"

    elif funding_signal == "shorts crowded":
        if direction == "SHORT":
            confidence_adjustment = -5
            derivatives_bias = "caution"
            derivatives_note = "shorts crowded, risque de short squeeze"
        elif direction == "LONG":
            confidence_adjustment = 3
            derivatives_bias = "supports long"
            derivatives_note = "shorts crowded, contexte favorable au LONG"
        else:
            derivatives_note = "shorts crowded"

    return funding_signal, derivatives_bias, derivatives_note, confidence_adjustment

# ─── CALCULS TECHNIQUES ───────────────────────────────────────────────────────

def calculate_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 2:
        return 50  # valeur neutre si pas assez de données
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calculate_ema(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period:
        return float(closes[-1])
    k   = 2 / (period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 6)

def calculate_atr(highs, lows, closes, period=14):
    highs  = np.array(highs,  dtype=float)
    lows   = np.array(lows,   dtype=float)
    closes = np.array(closes, dtype=float)
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i]  - closes[i-1])
        )
        trs.append(tr)
    if not trs:
        return 0
    return round(np.mean(trs[-period:]), 6)

def calculate_relative_volume(volumes, period=20):
    volumes = np.array(volumes, dtype=float)
    if len(volumes) < period + 1:
        return 0
    avg = np.mean(volumes[-period-1:-1])
    if avg == 0:
        return 0
    return round(volumes[-1] / avg, 2)

def detect_market_regime(btc_klines, return_details=False):
    """
    Régime BTC enrichi V4.7/V5.
    Retour simple par défaut pour compatibilité.
    Si return_details=True, retourne (regime, details).
    """
    empty = {
        "market_danger_score": 50,
        "market_danger_level": "UNKNOWN",
        "btc_rsi": 50,
        "btc_atr_pct": 0,
        "btc_variation_2h": 0,
        "btc_variation_4h": 0,
        "btc_variation_12h": 0,
        "btc_note": "données BTC insuffisantes"
    }
    if not btc_klines or len(btc_klines) < 30:
        return ("unknown", empty) if return_details else "unknown"

    closes  = [float(k[4]) for k in btc_klines]
    highs   = [float(k[2]) for k in btc_klines]
    lows    = [float(k[3]) for k in btc_klines]
    ema9    = calculate_ema(closes, 9)
    ema21   = calculate_ema(closes, 21)
    rsi     = calculate_rsi(closes)
    atr     = calculate_atr(highs, lows, closes)
    current = closes[-1]

    atr_pct       = (atr / current) * 100 if current > 0 else 0
    variation_2h  = ((closes[-1] - closes[-3]) / closes[-3] * 100) if len(closes) >= 3 and closes[-3] > 0 else 0
    variation_4h  = ((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 and closes[-5] > 0 else 0
    variation_12h = ((closes[-1] - closes[-13]) / closes[-13] * 100) if len(closes) >= 13 and closes[-13] > 0 else 0

    danger_score = 0
    danger_reasons = []

    if atr_pct > 2.5:
        danger_score += 20
        danger_reasons.append("BTC ATR élevé")
    if atr_pct > 4.0:
        danger_score += 20
        danger_reasons.append("BTC ATR extrême")
    if abs(variation_2h) > 2.0:
        danger_score += 20
        danger_reasons.append("variation BTC 2h forte")
    if abs(variation_4h) > 3.5:
        danger_score += 20
        danger_reasons.append("variation BTC 4h forte")
    if abs(variation_12h) > 6.0:
        danger_score += 15
        danger_reasons.append("extension BTC 12h")
    if rsi > 78 or rsi < 22:
        danger_score += 15
        danger_reasons.append("RSI BTC extrême")

    danger_score = min(100, danger_score)
    if danger_score >= 60:
        danger_level = "HIGH"
    elif danger_score >= 30:
        danger_level = "MEDIUM"
    else:
        danger_level = "LOW"

    if danger_score >= 70:
        regime = "danger"
    elif atr_pct > 4 or abs(variation_2h) > 3 or abs(variation_4h) > 5:
        regime = "volatile"
    elif current > ema9 > ema21 and rsi > 50:
        regime = "bullish"
    elif current < ema9 < ema21 and rsi < 50:
        regime = "bearish"
    else:
        regime = "neutral"

    details = {
        "market_danger_score": round(danger_score, 1),
        "market_danger_level": danger_level,
        "btc_rsi": rsi,
        "btc_atr_pct": round(atr_pct, 2),
        "btc_variation_2h": round(variation_2h, 2),
        "btc_variation_4h": round(variation_4h, 2),
        "btc_variation_12h": round(variation_12h, 2),
        "btc_note": ", ".join(danger_reasons) if danger_reasons else "BTC stable"
    }
    return (regime, details) if return_details else regime


def normalize_symbol(symbol):
    return str(symbol or "").upper().strip()


def source_quality_label(data_source):
    """
    Badge de qualité de source à afficher tel quel dans Telegram.
    Objectif : éviter que GPT interprète ou embellisse la fiabilité de la donnée.
    """
    if data_source == "FUTURES":
        return "🟢 FUTURES VERIFIED — Binance Futures"
    if data_source == "BYBIT_FUTURES":
        return "🟡 BYBIT FUTURES — futures alternatif"
    if data_source == "SPOT_FALLBACK":
        return "🟠 SPOT FALLBACK — données spot, dérivés non vérifiés"
    return "⚪ UNAVAILABLE — source non vérifiée"


def is_tradeable_usdt_symbol(symbol):
    symbol = normalize_symbol(symbol)
    if not symbol.endswith("USDT"):
        return False
    if symbol in EXCLUDED_SYMBOLS:
        return False
    if symbol.endswith(EXCLUDED_SUFFIXES):
        return False
    return True


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def build_batch_ticker_url(base_url, symbols):
    # Binance attend un JSON array dans le paramètre symbols.
    symbols_json = json.dumps(symbols, separators=(",", ":"))
    return f"{base_url}?symbols={urllib.parse.quote(symbols_json)}"


def get_dynamic_tickers(symbols_config=None):
    """
    V5 SAFE PROVIDER : universe contrôlé avec fallback futures multi-source.
    Ordre : Binance Futures batch -> Bybit Futures linear -> Binance Vision Spot -> Binance Spot.
    Objectif : ne plus dépendre d'une seule IP Railway -> Binance.
    """
    symbols_config = symbols_config or []

    seen = set()
    symbols = []
    for sym in symbols_config:
        sym = normalize_symbol(sym)
        if not sym or sym in seen:
            continue
        if not is_tradeable_usdt_symbol(sym):
            continue
        seen.add(sym)
        symbols.append(sym)

    if not symbols:
        return [], "UNAVAILABLE"

    def fetch_binance_batches(base_url):
        # IMPORTANT : certains symbols de la liste futures n'existent pas en Spot Vision.
        # Avant, un seul batch KO tuait tout le scan. Maintenant, on ignore le batch KO
        # et on conserve les autres données valides.
        out = []
        for batch in chunked(symbols, 20):
            url = build_batch_ticker_url(base_url, batch)
            data = fetch_binance(url)

            if data is None:
                logger.warning("Batch KO sur %s avec %s symboles: %s", base_url, len(batch), batch)
                continue

            if isinstance(data, dict):
                data = [data]

            out.extend(data)

        return out if out else None

    providers = []
    if BINANCE_ENABLED:
        providers.append(("FUTURES", lambda: fetch_binance_batches("https://fapi.binance.com/fapi/v1/ticker/24hr")))
    providers.append(("BYBIT_FUTURES", lambda: fetch_bybit_linear_tickers(symbols)))
    providers.append(("SPOT_FALLBACK", lambda: fetch_binance_batches("https://data-api.binance.vision/api/v3/ticker/24hr")))
    if BINANCE_ENABLED:
        providers.append(("SPOT_FALLBACK", lambda: fetch_binance_batches("https://api.binance.com/api/v3/ticker/24hr")))

    tickers = None
    data_source = "UNAVAILABLE"
    for source, fn in providers:
        tickers = fn()
        if tickers:
            data_source = source
            logger.info("Ticker provider OK: %s (%s tickers)", source, len(tickers))
            break
        logger.warning("Ticker provider KO: %s", source)

    if not tickers:
        return [], "UNAVAILABLE"

    filtered = []
    for t in tickers:
        symbol = normalize_symbol(t.get("symbol", ""))
        if not is_tradeable_usdt_symbol(symbol):
            continue
        try:
            volume = float(t.get("quoteVolume", 0))
            last_price = float(t.get("lastPrice", 0))
        except Exception:
            continue
        if last_price <= 0:
            continue
        if volume < MIN_QUOTE_VOLUME_USDT and symbol not in TIER1_ALWAYS:
            continue
        filtered.append(t)

    filtered.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    selected = {normalize_symbol(t.get("symbol")): t for t in filtered[:MAX_DYNAMIC_UNIVERSE]}
    all_by_symbol = {normalize_symbol(t.get("symbol")): t for t in tickers if normalize_symbol(t.get("symbol"))}
    for sym in TIER1_ALWAYS:
        if sym in all_by_symbol:
            selected[sym] = all_by_symbol[sym]

    return list(selected.values()), data_source

def quick_late_entry_risk(price_change_pct, range_pct):
    """Pré-filtre rapide basé ticker 24h avant chargement des klines."""
    risk = 0
    flags = []
    abs_change = abs(price_change_pct)

    if abs_change > 8:
        risk += 10
        flags.append("variation_24h_elevee")
    if abs_change > 12:
        risk += 15
        flags.append("variation_24h_tres_elevee")
    if abs_change > 18:
        risk += 25
        flags.append("variation_24h_extreme")
    if range_pct > 10:
        risk += 10
        flags.append("range_24h_large")
    if range_pct > 16:
        risk += 15
        flags.append("range_24h_extreme")

    return min(100, risk), flags


def detailed_late_entry_risk(direction, entry_type, closes, highs, lows, current, rsi, atr_pct, distance_ema21, position_range, price_change):
    """
    V4.6 : détecte les entrées tardives.
    Important : overextended != short automatique. On peut produire SHORT_WATCH plus bas.
    """
    risk = 0
    flags = []

    if len(closes) >= 2 and closes[-2] > 0:
        change_1h = ((closes[-1] - closes[-2]) / closes[-2]) * 100
    else:
        change_1h = 0
    if len(closes) >= 4 and closes[-4] > 0:
        change_3h = ((closes[-1] - closes[-4]) / closes[-4]) * 100
    else:
        change_3h = 0

    last_changes = []
    for i in range(-4, 0):
        if len(closes) >= abs(i) + 1 and closes[i-1] > 0:
            last_changes.append(((closes[i] - closes[i-1]) / closes[i-1]) * 100)

    if direction == "LONG":
        if price_change > 10: risk += 12; flags.append("pump_24h")
        if price_change > 15: risk += 18; flags.append("pump_24h_extreme")
        if change_1h > 3: risk += 15; flags.append("pump_1h")
        if change_3h > 6: risk += 15; flags.append("pump_3h")
        if len(last_changes) >= 3 and all(c > 1.2 for c in last_changes[-3:]):
            risk += 15; flags.append("3_bougies_vertes")
        if len(last_changes) >= 4 and all(c > 0.8 for c in last_changes[-4:]):
            risk += 10; flags.append("4_bougies_vertes")
        if rsi > 72: risk += 12; flags.append("rsi_surchauffe")
        if rsi > 78: risk += 15; flags.append("rsi_extreme")
        if position_range > 0.88: risk += 15; flags.append("proche_high_24h")
    elif direction == "SHORT":
        if price_change < -10: risk += 12; flags.append("dump_24h")
        if price_change < -15: risk += 18; flags.append("dump_24h_extreme")
        if change_1h < -3: risk += 15; flags.append("dump_1h")
        if change_3h < -6: risk += 15; flags.append("dump_3h")
        if len(last_changes) >= 3 and all(c < -1.2 for c in last_changes[-3:]):
            risk += 15; flags.append("3_bougies_rouges")
        if rsi < 28: risk += 12; flags.append("rsi_survendu")
        if rsi < 22: risk += 15; flags.append("rsi_extreme")
        if position_range < 0.12: risk += 15; flags.append("proche_low_24h")

    if distance_ema21 > 3: risk += 8; flags.append("loin_ema21")
    if distance_ema21 > 6: risk += 12; flags.append("tres_loin_ema21")
    if atr_pct > 3.5: risk += 10; flags.append("atr_eleve")
    if atr_pct > 5: risk += 15; flags.append("atr_extreme")

    # Un breakout propre est autorisé à être un peu plus tendu, mais pas FOMO.
    if entry_type == "BREAKOUT":
        risk = max(0, risk - 8)
    if entry_type == "EARLY":
        risk = max(0, risk - 12)

    if risk >= 70:
        level = "HIGH"
    elif risk >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return min(100, round(risk, 1)), level, flags, round(change_1h, 2), round(change_3h, 2)


def detect_short_watch(direction, closes, highs, lows, volumes, rsi, ema9, ema21, current, relative_vol, late_entry_risk, position_range):
    """
    SHORT WATCH uniquement, pas de vrai signal short reversal automatique.
    Objectif : signaler un momentum long potentiellement épuisé à observer.
    """
    if direction != "LONG" or late_entry_risk < 55 or len(closes) < 8:
        return False, []

    reasons = []
    last_high = highs[-1]
    prev_high = max(highs[-6:-1]) if len(highs) >= 6 else highs[-2]
    upper_wick = (last_high - max(closes[-1], float(closes[-2]))) / current * 100 if current > 0 else 0

    last_3_vol_avg = np.mean(volumes[-3:]) if len(volumes) >= 3 else volumes[-1]
    prev_6_vol_avg = np.mean(volumes[-9:-3]) if len(volumes) >= 9 else last_3_vol_avg
    volume_fading = prev_6_vol_avg > 0 and last_3_vol_avg < prev_6_vol_avg * 0.85

    if upper_wick > 1.0:
        reasons.append("mèche haute")
    if current < ema9:
        reasons.append("perte EMA9")
    if ema9 < ema21:
        reasons.append("EMA9 repasse sous EMA21")
    if last_high <= prev_high and position_range > 0.80:
        reasons.append("rejet proche résistance")
    if volume_fading:
        reasons.append("volume en baisse")
    if rsi > 74:
        reasons.append("RSI surchauffé")

    return len(reasons) >= 3, reasons

def limit_weak_candidates(candidates):
    """
    Sécurité côté Python :
    - ne jamais envoyer plus d'un setup trend_strength = weak à GPT
    - conserve l'ordre de tri déjà appliqué par score décroissant
    """
    final = []
    weak_added = False

    for r in candidates:
        if r.get("trend_strength") == "weak":
            if weak_added:
                continue
            weak_added = True
        final.append(r)

    return final

# ─── SCORING v4.4 ─────────────────────────────────────────────────────────────
# v4   : direction stricte, momentum signé, pénalités pump/distance EMA
# v4.1 : ema_score/rsi_score directionnels, volume seuils relevés,
#         position_range, ema_spread, trend_strength
# v4.2 : market_regime BTC intégré, bonus alignement tendance macro
# v4.3 : RSI extrême gradué, distance EMA graduée, 3 bougies explosives,
#         support/résistance range 24h, BTC volatile/danger
# v4.4 : EMA50 structure long terme, cooldown APRÈS Telegram (/set_cooldown),
#         double porte prescore (momentum + early), sécurisation request.json,
#         protection RSI longueur, cleanup endpoints inutiles
# v4.5A: ajout BREAKOUT LONG/SHORT, entry_type BREAKOUT,
#        breakout_level, score breakout, levier breakout contrôlé
# v4.5B1: funding intelligence simple, derivatives_bias/note,
#         ajustement confiance et levier selon funding
# v4.6 : late_entry_risk détaillé, anti-FOMO avant Telegram, SHORT_WATCH
# v4.7 : market_danger_score BTC enrichi
# v5.0 : universe dynamique + préfiltre tradability

def score_symbol(symbol, ticker_data=None, market_regime="unknown", market_details=None):
    market_details = market_details or {}
    klines, data_source = get_klines(symbol, limit=200)
    if not klines or len(klines) < 55:
        return None

    funding_rate, funding_available = get_funding_rate(symbol, data_source)

    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    rsi          = calculate_rsi(closes)
    ema9         = calculate_ema(closes, 9)
    ema21        = calculate_ema(closes, 21)
    ema50        = calculate_ema(closes, 50)
    atr          = calculate_atr(highs, lows, closes)
    relative_vol = calculate_relative_volume(volumes)
    current      = closes[-1]

    # Variation 24h avec signe
    if ticker_data and "priceChangePercent" in ticker_data:
        price_change = float(ticker_data["priceChangePercent"])
        high_24h     = float(ticker_data.get("highPrice", current))
    else:
        price_change = round(((closes[-1] - closes[-24]) / closes[-24]) * 100, 2) if len(closes) >= 24 else 0
        high_24h     = max(highs[-24:]) if len(highs) >= 24 else max(highs)

    distance_high  = round(((high_24h - current) / high_24h) * 100, 2) if high_24h > 0 else 0
    distance_ema21 = round(abs((current - ema21) / ema21) * 100, 2) if ema21 > 0 else 0
    distance_ema50 = round(abs((current - ema50) / ema50) * 100, 2) if ema50 > 0 else 0

    low_24h        = min(lows[-24:]) if len(lows) >= 24 else min(lows)
    range_24h      = high_24h - low_24h
    position_range = round((current - low_24h) / range_24h, 3) if range_24h > 0 else 0.5

    # Low 7 jours approximé sur chandeliers 1h.
    # Si moins de 168 bougies sont disponibles, on utilise tout l'historique chargé.
    lookback_7d = min(len(lows), 168)
    low_7d = min(lows[-lookback_7d:]) if lookback_7d > 0 else low_24h
    distance_low_7d_pct = round(((current - low_7d) / low_7d) * 100, 2) if low_7d > 0 else 999

    above_ema50 = current > ema50
    ema50_trend = "bullish" if above_ema50 else "bearish"

    ema_spread = round(abs((ema9 - ema21) / ema21) * 100, 3) if ema21 > 0 else 0
    if ema_spread > 2:
        trend_strength = "strong"
    elif ema_spread > 0.8:
        trend_strength = "moderate"
    else:
        trend_strength = "weak"

    atr_pct = (atr / current) * 100 if current > 0 else 0

    # ── BREAKOUT v4.5A ───────────────────────────────────────────────────────
    # Cassure du range court terme H1.
    # On compare la bougie actuelle aux 11 bougies précédentes.
    if len(highs) >= 13 and len(lows) >= 13:
        recent_high = max(highs[-12:-1])
        recent_low  = min(lows[-12:-1])
    else:
        recent_high = max(highs[:-1]) if len(highs) > 1 else current
        recent_low  = min(lows[:-1]) if len(lows) > 1 else current

    breakout_long = (
        current > recent_high and
        relative_vol >= 1.3 and
        50 <= rsi <= 68 and
        distance_ema21 <= 4.0 and
        position_range <= 0.88 and
        atr_pct <= 3.5
    )

    breakout_short = (
        current < recent_low and
        relative_vol >= 1.3 and
        32 <= rsi <= 50 and
        distance_ema21 <= 4.0 and
        position_range >= 0.12 and
        atr_pct <= 3.5
    )

    breakout_level = None
    if breakout_long:
        breakout_level = round(recent_high, 8)
    elif breakout_short:
        breakout_level = round(recent_low, 8)

    # ── DIRECTION STRICTE ────────────────────────────────────────────────────
    if current > ema9 > ema21 and 45 <= rsi <= 72:
        direction = "LONG"
    elif current < ema9 < ema21 and 28 <= rsi <= 55:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # ── EARLY ENTRY v4.4 ─────────────────────────────────────────────────────
    # Détecte les setups avant le vrai mouvement
    early_long = (
        current > ema21 and
        ema9 >= ema21 * 0.998 and
        48 <= rsi <= 62 and
        distance_ema21 <= 2.5 and
        relative_vol >= 1.0 and
        0.55 <= position_range <= 0.82 and
        atr_pct <= 2.5
    )
    early_short = (
        current < ema21 and
        ema9 <= ema21 * 1.002 and
        38 <= rsi <= 52 and
        distance_ema21 <= 2.5 and
        relative_vol >= 1.0 and
        0.18 <= position_range <= 0.45 and
        atr_pct <= 2.5
    )

    # ── ENTRY TYPE — priorité : BREAKOUT > EARLY > MOMENTUM ─────────────────
    # BREAKOUT = cassure confirmée d'un niveau court terme.
    # EARLY    = anticipation avant mouvement fort.
    # MOMENTUM = continuation déjà engagée.
    if breakout_long or breakout_short:
        entry_type = "BREAKOUT"
        # Override direction si nécessaire
        if direction == "NEUTRAL":
            direction = "LONG" if breakout_long else "SHORT"
    elif early_long or early_short:
        entry_type = "EARLY"
        # Override direction si NEUTRAL avec early détecté
        if direction == "NEUTRAL":
            direction = "LONG" if early_long else "SHORT"
    elif direction != "NEUTRAL":
        entry_type = "MOMENTUM"
    else:
        entry_type = "NEUTRAL"

    # ── LATE ENTRY RISK v4.6 ────────────────────────────────────────────────
    late_entry_risk, late_entry_level, late_entry_flags, momentum_1h, momentum_3h = detailed_late_entry_risk(
        direction, entry_type, closes, highs, lows, current, rsi, atr_pct,
        distance_ema21, position_range, price_change
    )

    # SHORT WATCH = observation de retournement potentiel, pas signal SHORT automatique.
    short_watch, short_watch_reasons = detect_short_watch(
        direction, closes, highs, lows, volumes, rsi, ema9, ema21,
        current, relative_vol, late_entry_risk, position_range
    )

    # ── FUNDING INTELLIGENCE v4.5B1 ─────────────────────────────────────────
    funding_signal, derivatives_bias, derivatives_note, funding_conf_adj = interpret_funding(
        funding_rate,
        direction,
        funding_available
    )

    # ── SHORT FORBIDDEN GUARD v5.0-safe ─────────────────────────────────────
    # Interdit les shorts quand le prix est déjà trop proche du bas de range
    # et que le funding indique déjà des shorts crowded.
    # Logique: éviter de shorter un actif déjà écrasé avec risque de squeeze/rebond.
    short_forbidden = (
        direction == "SHORT" and
        position_range < 0.30 and
        distance_low_7d_pct <= 3.0 and
        funding_signal == "shorts crowded"
    )
    short_forbidden_reasons = []
    if short_forbidden:
        short_forbidden_reasons = [
            "position_range<0.30",
            "prix_proche_low_7d<=3%",
            "shorts_crowded"
        ]

    # ── SCORE MOMENTUM ───────────────────────────────────────────────────────
    if direction == "LONG":
        if entry_type == "EARLY":
            mom_score = 40  # anticipation prudente
        elif entry_type == "BREAKOUT":
            mom_score = 70  # cassure confirmée : entre EARLY et MOMENTUM
        else:
            mom_score = min(100, (price_change / 10) * 100) if price_change >= 2 else 0
    elif direction == "SHORT":
        if entry_type == "EARLY":
            mom_score = 40
        elif entry_type == "BREAKOUT":
            mom_score = 70
        else:
            mom_score = min(100, (abs(price_change) / 10) * 100) if price_change <= -2 else 0
    else:
        mom_score = 0

    # ── SCORE EMA9/21 ────────────────────────────────────────────────────────
    if direction == "LONG":
        ema_score = 80 if ema9 > ema21 * 1.002 else (60 if ema9 > ema21 else 20)
    elif direction == "SHORT":
        ema_score = 80 if ema9 < ema21 * 0.998 else (60 if ema9 < ema21 else 20)
    else:
        ema_score = 40

    # ── SCORE RSI ────────────────────────────────────────────────────────────
    if direction == "LONG":
        if 45 <= rsi <= 65:             rsi_score = 100
        elif 35 <= rsi < 45 or 65 < rsi <= 72: rsi_score = 60
        elif rsi < 35:                  rsi_score = 20
        else:                           rsi_score = 10
    elif direction == "SHORT":
        if 35 <= rsi <= 55:             rsi_score = 100
        elif 28 <= rsi < 35 or 55 < rsi <= 65: rsi_score = 60
        elif rsi > 65:                  rsi_score = 20
        else:                           rsi_score = 10
    else:
        rsi_score = 30

    # ── SCORE VOLUME ─────────────────────────────────────────────────────────
    if relative_vol >= 1.8:    vol_score = 100
    elif relative_vol >= 1.4:  vol_score = 80
    elif relative_vol >= 1.1:  vol_score = 50
    else:                      vol_score = 10

    # ── SCORE VOLATILITE ─────────────────────────────────────────────────────
    if 0.5 <= atr_pct <= 3.0:  vola_score = 100
    elif atr_pct < 0.5:        vola_score = 20
    else:                      vola_score = 50

    # ── SCORE POSITION RANGE ─────────────────────────────────────────────────
    if direction == "LONG":
        if 0.4 <= position_range <= 0.85:   range_score = 100
        elif 0.2 <= position_range < 0.4:   range_score = 50
        elif position_range > 0.85:         range_score = 30
        else:                               range_score = 10
    elif direction == "SHORT":
        if 0.15 <= position_range <= 0.6:   range_score = 100
        elif 0.6 < position_range <= 0.8:   range_score = 50
        elif position_range < 0.15:         range_score = 30
        else:                               range_score = 10
    else:
        range_score = 30

    # ── SCORE GLOBAL ─────────────────────────────────────────────────────────
    global_score = round(
        mom_score   * 0.22 +
        ema_score   * 0.20 +
        rsi_score   * 0.18 +
        vol_score   * 0.18 +
        vola_score  * 0.12 +
        range_score * 0.10, 1
    )

    # ── PENALITES ────────────────────────────────────────────────────────────

    # 1. RSI extrême gradué
    if direction == "LONG":
        if rsi > 68: global_score -= 10
        if rsi > 72: global_score -= 10
    if direction == "SHORT":
        if rsi < 32: global_score -= 10
        if rsi < 28: global_score -= 10

    # 2. Distance EMA21 graduée
    if distance_ema21 > 3:  global_score -= 5
    if distance_ema21 > 6:  global_score -= 10
    if distance_ema21 > 10: global_score -= 10

    # 3. EMA50 — pénalité si direction contre structure long terme
    if direction == "LONG"  and not above_ema50: global_score -= 10
    if direction == "SHORT" and above_ema50:     global_score -= 10

    # 4. Pump/dump trop tardif
    if direction == "LONG"  and price_change > 12: global_score -= 15
    if direction == "SHORT" and price_change < -12: global_score -= 15

    # 5. 3 bougies consécutives explosives (anti-FOMO)
    last_3_changes = []
    for i in range(-3, 0):
        if closes[i-1] > 0:
            chg = (closes[i] - closes[i-1]) / closes[i-1] * 100
            last_3_changes.append(chg)
    if len(last_3_changes) == 3:
        if all(c > 2.5 for c in last_3_changes) and direction == "LONG":
            global_score -= 20
        if all(c < -2.5 for c in last_3_changes) and direction == "SHORT":
            global_score -= 20

    # 6. Support/résistance range 24h
    if direction == "LONG"  and position_range > 0.88: global_score -= 15
    if direction == "SHORT" and position_range < 0.12: global_score -= 15

    # 7. Late Entry Risk Engine v4.6 — pénalité centrale anti-FOMO
    if late_entry_risk >= 70:
        global_score -= 30
    elif late_entry_risk >= 55:
        global_score -= 22
    elif late_entry_risk >= 40:
        global_score -= 12

    # Si momentum déjà consommé et pas EARLY, on évite de transformer un pump en signal.
    if late_entry_risk >= 65 and entry_type == "MOMENTUM":
        global_score = min(global_score, 51)

    # 8. Direction NEUTRAL → plafonné à 45
    if direction == "NEUTRAL":
        global_score = min(global_score, 45)

    # 9. Market regime BTC
    if market_regime == "danger":                           global_score -= 25
    if market_regime == "volatile":                         global_score -= 15
    if market_regime == "bearish" and direction == "LONG":  global_score -= 10
    if market_regime == "bullish" and direction == "SHORT": global_score -= 10
    if market_regime == "neutral":                          global_score -= 5
    if market_regime == "bearish" and direction == "SHORT": global_score += 8
    if market_regime == "bullish" and direction == "LONG":  global_score += 8

    market_danger_score = float(market_details.get("market_danger_score", 0) or 0)
    market_danger_level = market_details.get("market_danger_level", "UNKNOWN")
    if market_danger_level == "HIGH":
        global_score -= 12
    elif market_danger_level == "MEDIUM":
        global_score -= 5

    # 10. Pénalité légère early entry (moins fiable que momentum)
    if entry_type == "EARLY": global_score -= 5

    # 11. SPOT FALLBACK : accepté temporairement, mais moins fiable pour futures.
    if data_source == "SPOT_FALLBACK":
        global_score -= 8

    # 12. SHORT interdit : pas de signal SHORT proche du plus bas 7j avec shorts crowded.
    if short_forbidden:
        global_score = min(global_score, 45)

    global_score = round(max(0, global_score), 1)

    # ── FLAG ─────────────────────────────────────────────────────────────────
    if global_score >= 58:
        flag = "CANDIDAT"
    elif global_score >= 52:
        flag = "WATCHLIST"
    else:
        flag = "REJET"

    if short_watch:
        # On ne short pas automatiquement : on transforme uniquement en surveillance.
        flag = "SHORT_WATCH"

    if short_forbidden:
        flag = "REJET"

    # Rétrogradation anti-entrée-tardive : un CANDIDAT dont le risque d'entrée
    # tardive est élevé passe en surveillance plutôt que d'être tradé directement.
    # (en plus de la pénalité de score déjà appliquée plus haut)
    if flag == "CANDIDAT" and late_entry_risk >= 55:
        flag = "WATCHLIST"

    # ── CALCULS ENTRY / SL / TP / RR — Python calcule, GPT formate ───────────

    # Entry zone
    if direction == "LONG":
        if entry_type == "EARLY":
            entry_low  = round(current - 0.40 * atr, 8)
            entry_high = round(current, 8)
        else:
            entry_low  = round(current - 0.25 * atr, 8)
            entry_high = round(current + 0.15 * atr, 8)
    elif direction == "SHORT":
        if entry_type == "EARLY":
            entry_low  = round(current, 8)
            entry_high = round(current + 0.40 * atr, 8)
        else:
            entry_low  = round(current - 0.15 * atr, 8)
            entry_high = round(current + 0.25 * atr, 8)
    else:
        entry_low  = round(current, 8)
        entry_high = round(current, 8)

    entry_avg = round((entry_low + entry_high) / 2, 8)

    # Stop Loss
    # Important : SL calculé depuis entry_avg pour garantir un R/R cohérent avec TP2.
    # Avec TP2 = entry_avg ± 2×ATR, un SL à 1×ATR donne R/R ≈ 2.
    if symbol in MEMECOINS:    sl_max_pct = 2.0
    elif symbol in MAJORS:     sl_max_pct = 4.0
    else:                      sl_max_pct = 3.0

    if direction == "LONG":
        sl_raw = round(entry_avg - 1.0 * atr, 8)
        sl_distance = abs(current - sl_raw) / current * 100 if current > 0 else 0
        if sl_distance > sl_max_pct:
            sl_raw = round(current * (1 - sl_max_pct / 100), 8)
        stop_loss = sl_raw
    elif direction == "SHORT":
        sl_raw = round(entry_avg + 1.0 * atr, 8)
        sl_distance = abs(sl_raw - current) / current * 100 if current > 0 else 0
        if sl_distance > sl_max_pct:
            sl_raw = round(current * (1 + sl_max_pct / 100), 8)
        stop_loss = sl_raw
    else:
        stop_loss = round(current, 8)

    # Take Profits
    if direction == "LONG":
        tp1 = round(entry_avg + 1 * atr, 8)
        tp2 = round(entry_avg + 2 * atr, 8)
        tp3 = round(entry_avg + 3 * atr, 8)
        tp4 = round(entry_avg + 4 * atr, 8)
    elif direction == "SHORT":
        tp1 = round(entry_avg - 1 * atr, 8)
        tp2 = round(entry_avg - 2 * atr, 8)
        tp3 = round(entry_avg - 3 * atr, 8)
        tp4 = round(entry_avg - 4 * atr, 8)
    else:
        tp1 = tp2 = tp3 = tp4 = round(current, 8)

    # Risk / Reward
    risk        = abs(entry_avg - stop_loss)
    reward_tp2  = abs(tp2 - entry_avg)
    risk_reward = round(reward_tp2 / risk, 2) if risk > 0 else 0
    rr_valid    = risk_reward >= 2.0

    # Levier maximum
    if symbol in MEMECOINS:    base_leverage = 5
    elif symbol in MAJORS:     base_leverage = 10
    else:                      base_leverage = 7

    leverage_caps = [base_leverage]
    if trend_strength == "weak":    leverage_caps.append(3)
    if market_regime == "neutral":  leverage_caps.append(5)
    if market_regime == "volatile": leverage_caps.append(3)
    if market_regime == "danger":   leverage_caps.append(2)
    if market_danger_level == "HIGH": leverage_caps.append(2)
    if entry_type == "EARLY":       leverage_caps.append(3)
    if flag == "WATCHLIST":         leverage_caps.append(3)
    if flag == "SHORT_WATCH":       leverage_caps.append(1)
    if late_entry_risk >= 55:        leverage_caps.append(3)
    if data_source == "SPOT_FALLBACK": leverage_caps.append(5)
    # Sécurité macro : réduire fortement le levier si le trade est contre le régime BTC
    if (market_regime == "bearish" and direction == "LONG") or \
       (market_regime == "bullish" and direction == "SHORT"):
        leverage_caps.append(3)
    max_leverage = min(leverage_caps)

    # Confiance
    confidence = global_score
    if trend_strength == "strong":                          confidence += 5
    if trend_strength == "weak":                            confidence -= 5
    if market_regime == "neutral":                          confidence -= 5
    if market_regime == "volatile":                         confidence -= 10
    if (market_regime == "bearish" and direction == "LONG") or \
       (market_regime == "bullish" and direction == "SHORT"): confidence -= 10
    confidence += funding_conf_adj
    if entry_type == "EARLY":                               confidence -= 5
    if distance_ema21 > 6:                                  confidence -= 5
    if (direction == "LONG"  and not (0.4 <= position_range <= 0.85)) or \
       (direction == "SHORT" and not (0.15 <= position_range <= 0.6)): confidence -= 5

    # Plafonds confiance
    if trend_strength == "weak": confidence = min(confidence, 68)
    if flag == "WATCHLIST":      confidence = min(confidence, 60)
    if flag == "SHORT_WATCH":    confidence = min(confidence, 55)
    if data_source == "SPOT_FALLBACK": confidence = min(confidence, 72)
    if late_entry_risk >= 55:     confidence = min(confidence, 62)
    if market_danger_level == "HIGH": confidence = min(confidence, 60)
    confidence = round(max(40, min(88, confidence)), 1)

    # Rétrogradation tendance faible : un CANDIDAT en tendance faible ET à confiance
    # modérée passe en surveillance. On garde le volume de signaux mais on évite les
    # setups les plus mous. (confidence n'est connue qu'ici, donc on rétrograde après.)
    if flag == "CANDIDAT" and trend_strength == "weak" and confidence < 65:
        flag = "WATCHLIST"
        confidence = min(confidence, 60)

    # Sécurité levier après calcul de la confiance finale :
    # - confiance < 60%  -> 3x maximum
    # - confiance < 65%  -> 5x maximum
    # Cela évite les signaux 7x avec conviction moyenne/faible.
    if confidence < 60:
        max_leverage = min(max_leverage, 3)
    elif confidence < 65:
        max_leverage = min(max_leverage, 5)

    # Sécurité BREAKOUT :
    # un breakout reste plus fiable qu'un early, mais peut être une fausse cassure.
    # Si la confiance est inférieure à 70%, on plafonne le levier à 5x.
    if entry_type == "BREAKOUT" and confidence < 70:
        max_leverage = min(max_leverage, 5)

    if data_source == "SPOT_FALLBACK":
        max_leverage = min(max_leverage, 5)
    if late_entry_risk >= 55:
        max_leverage = min(max_leverage, 3)
    if market_danger_level == "HIGH":
        max_leverage = min(max_leverage, 2)

    # Sécurité funding :
    # si le funding signale un crowded trade contre notre direction,
    # on évite les leviers agressifs.
    if derivatives_bias == "caution":
        max_leverage = min(max_leverage, 5)

    # Durée estimée calculée par Python
    if trend_strength == "strong":
        duration_label = "24 à 72h"
    elif trend_strength == "moderate":
        duration_label = "6 à 24h"
    else:
        duration_label = "2 à 6h"

    if data_source == "SPOT_FALLBACK" or late_entry_risk >= 55 or market_danger_level == "HIGH":
        duration_label = "30 min à 4h"

    return {
        "symbol":          symbol,
        "score":           global_score,
        "flag":            flag,
        "direction":       direction,
        "entry_type":      entry_type,
        "breakout_level":  breakout_level,
        "trend_strength":  trend_strength,
        "market_regime":   market_regime,
        "rsi":             rsi,
        "ema_spread":      ema_spread,
        "ema50_trend":     ema50_trend,
        "atr":             atr,
        "atr_pct":         round(atr_pct, 2),
        "volume_relatif":  relative_vol,
        "distance_ema21":  distance_ema21,
        "position_range":  position_range,
        "low_7d":          round(low_7d, 8),
        "distance_low_7d_pct": distance_low_7d_pct,
        "short_forbidden": short_forbidden,
        "short_forbidden_reasons": short_forbidden_reasons,
        "momentum_24h":    price_change,
        "funding_rate":    funding_rate,
        "funding_signal":  funding_signal,
        "derivatives_bias": derivatives_bias,
        "derivatives_note": derivatives_note,
        "late_entry_risk": late_entry_risk,
        "late_entry_level": late_entry_level,
        "late_entry_flags": late_entry_flags,
        "momentum_1h":     momentum_1h,
        "momentum_3h":     momentum_3h,
        "market_danger_score": market_danger_score,
        "market_danger_level": market_danger_level,
        "short_watch":     short_watch,
        "short_watch_reasons": short_watch_reasons,
        "current_price":   current,
        "data_source":     data_source,
        "source_quality":  source_quality_label(data_source),
        # ── Niveaux calculés par Python ──────────────────────────────────────
        "entry_low":       entry_low,
        "entry_high":      entry_high,
        "entry_avg":       entry_avg,
        "stop_loss":       stop_loss,
        "tp1":             tp1,
        "tp2":             tp2,
        "tp3":             tp3,
        "tp4":             tp4,
        "risk_reward":     risk_reward,
        "rr_valid":        rr_valid,
        "max_leverage":    max_leverage,
        "confidence":      confidence,
        "duration_label":   duration_label,
    }

# ─── ENDPOINT /set_cooldown ───────────────────────────────────────────────────

@app.route("/set_cooldown", methods=["POST"])
def set_cooldown_endpoint():
    try:
        data    = request.get_json(silent=True) or {}
        symbols = data.get("symbols", [])
        if not symbols or not isinstance(symbols, list):
            return jsonify({"error": "symbols must be a non-empty list"}), 400
        set_cooldown_symbols(symbols)
        cooldown = load_cooldown()
        return jsonify({
            "status":   "ok",
            "symbols":  symbols,
            "cooldown": {k: round((COOLDOWN_SECONDS - (time.time() - v)) / 3600, 1)
                         for k, v in cooldown.items()}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── ENDPOINT /cooldown_status ────────────────────────────────────────────────

@app.route("/cooldown_status", methods=["GET"])
def cooldown_status():
    try:
        cooldown = load_cooldown()
        status   = {}
        for symbol, ts in cooldown.items():
            remaining = max(0, COOLDOWN_SECONDS - (time.time() - ts))
            status[symbol] = {
                "remaining_hours": round(remaining / 3600, 1),
                "expires_in_min":  round(remaining / 60)
            }
        return jsonify({"active_cooldowns": status, "count": len(status)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── ENDPOINT /full_analysis ──────────────────────────────────────────────────

@app.route("/full_analysis", methods=["POST"])
def full_analysis():
    try:
        # Liste fixe conservée en filet de sécurité si le scanner dynamique échoue.
        symbols_config = [
            # Tier 1 / majors
            "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT",
            # Large caps / liquid alts
            "ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","LTCUSDT","BCHUSDT","TRXUSDT",
            "UNIUSDT","AAVEUSDT","MKRUSDT","COMPUSDT","CRVUSDT","SUSHIUSDT",
            "ATOMUSDT","NEARUSDT","APTUSDT","SUIUSDT","SEIUSDT","INJUSDT","TIAUSDT",
            "ARBUSDT","OPUSDT","STRKUSDT","JUPUSDT","PYTHUSDT","WLDUSDT",
            # AI / infra / DePIN
            "FETUSDT","RENDERUSDT","RNDRUSDT","GRTUSDT","TAOUSDT","ARKMUSDT","IOUSDT",
            "FILUSDT","ARUSDT","STXUSDT","ICPUSDT","EGLDUSDT",
            # Momentum / narratives fréquentes
            "PENDLEUSDT","ETHFIUSDT","ENAUSDT","ALTUSDT","PIXELUSDT","PORTALUSDT",
            "MANTAUSDT","OMUSDT","ONDOUSDT","JASMYUSDT","LDOUSDT","RUNEUSDT",
            "GMTUSDT","GALAUSDT","SANDUSDT","MANAUSDT","APEUSDT","AXSUSDT",
            # Memes liquides
            "PEPEUSDT","WIFUSDT","BONKUSDT","FLOKIUSDT","BOMEUSDT","SHIBUSDT",
            # Autres paires souvent actives Binance Futures
            "ETCUSDT","DYDXUSDT","BLURUSDT","ORDIUSDT","1000SATSUSDT","ZKUSDT",
            "ZROUSDT","NOTUSDT","PEOPLEUSDT","BIGTIMEUSDT","BEAMXUSDT","CKBUSDT",
            "MINAUSDT","KASUSDT","HIFIUSDT","MAGICUSDT","ACHUSDT","LQTYUSDT",
            "GMXUSDT"
        ]
        # ── V5.0 : UNIVERSE DYNAMIQUE ───────────────────────────────────────
        tickers_data, data_source_batch = get_dynamic_tickers(symbols_config)

        if not tickers_data:
            return jsonify({
                "text": "SKIP",
                "count": 0,
                "market_regime": "unknown",
                "data_source": "UNAVAILABLE",
                "error": "All providers unreachable: Binance Futures, Bybit Futures, Binance Vision Spot, Binance Spot"
            })

        # ── Market regime BTC enrichi v4.7 ──────────────────────────────────
        btc_klines, btc_data_source = get_klines("BTCUSDT", limit=80)
        market_regime, market_details = detect_market_regime(btc_klines, return_details=True)

        # ── PRESCORE v5 : tradability avant momentum brut ───────────────────
        scored = []
        for t in tickers_data:
            symbol = normalize_symbol(t.get("symbol", ""))
            if not is_tradeable_usdt_symbol(symbol):
                continue

            try:
                price_change_pct = float(t.get("priceChangePercent", 0))
                volume           = float(t.get("quoteVolume", 0))
                high             = float(t.get("highPrice", 1))
                low              = float(t.get("lowPrice", 1))
                last_price        = float(t.get("lastPrice", 0))
            except Exception:
                continue

            if last_price <= 0:
                continue
            if volume < MIN_QUOTE_VOLUME_USDT and symbol not in TIER1_ALWAYS:
                continue

            range_pct = ((high - low) / low) * 100 if low > 0 else 0

            # Porte 1 : momentum classique, mais pas uniquement la hausse brute.
            momentum_gate = abs(price_change_pct) >= 2.0 and range_pct >= 1.2
            # Porte 2 : early setup — compression + range actif mais mouvement pas encore trop fort.
            early_gate = abs(price_change_pct) < 2.0 and range_pct >= 1.5
            # Porte 3 : volatility/tradability — marché actif sans variation 24h énorme.
            tradability_gate = abs(price_change_pct) < 6.0 and 2.0 <= range_pct <= 10.0 and volume >= 30_000_000

            if not momentum_gate and not early_gate and not tradability_gate:
                continue

            mom_score    = min(100, (abs(price_change_pct) / 8) * 100)
            vol_score    = min(100, (volume / 80_000_000) * 100)
            range_score  = min(100, (range_pct / 10) * 100)
            liquidity_bonus = 8 if volume >= 100_000_000 else (4 if volume >= 50_000_000 else 0)
            tier1_bonus = 5 if symbol in TIER1_ALWAYS else 0
            early_bonus = 12 if early_gate else 0
            tradability_bonus = 8 if tradability_gate else 0

            late_quick_risk, late_quick_flags = quick_late_entry_risk(price_change_pct, range_pct)

            # Changement clé : le prescore doit favoriser les actifs tradables,
            # pas seulement les actifs déjà en feu.
            prescore_val = round(max(0,
                mom_score   * 0.30 +
                vol_score   * 0.35 +
                range_score * 0.20 +
                liquidity_bonus + tier1_bonus + early_bonus + tradability_bonus -
                late_quick_risk * 0.65
            ), 1)

            gate = "early" if early_gate else ("tradability" if tradability_gate else "momentum")
            scored.append({
                "symbol": symbol,
                "prescore": prescore_val,
                "ticker": t,
                "gate": gate,
                "late_quick_risk": late_quick_risk,
                "late_quick_flags": late_quick_flags,
                "volume": volume
            })

        scored.sort(key=lambda x: x["prescore"], reverse=True)

        # Représentation équilibrée : pas uniquement les top pumpers.
        momentum_top    = [x for x in scored if x["gate"] == "momentum"][:7]
        early_top       = [x for x in scored if x["gate"] == "early"][:6]
        tradability_top = [x for x in scored if x["gate"] == "tradability"][:5]

        # Déduplication en conservant l'ordre.
        seen = set()
        top_candidates = []
        for item in momentum_top + early_top + tradability_top + scored[:MAX_FULL_ANALYSIS_CANDIDATES]:
            if item["symbol"] in seen:
                continue
            seen.add(item["symbol"])
            top_candidates.append(item)
            if len(top_candidates) >= MAX_FULL_ANALYSIS_CANDIDATES:
                break

        # ── Scoring complet avec filtre cooldown (parallélisé) ──────────────
        results = []
        cooldown_skipped = []
        to_score = []
        for item in top_candidates:
            on_cd, remaining = is_on_cooldown(item["symbol"])
            if on_cd:
                cooldown_skipped.append(f"{item['symbol']} ({remaining}h)")
                continue
            to_score.append(item)

        def _score_item(item):
            result = score_symbol(item["symbol"], item["ticker"], market_regime, market_details)
            if result:
                result["prescore"] = item.get("prescore")
                result["prescore_gate"] = item.get("gate")
                result["late_quick_risk"] = item.get("late_quick_risk")
            return result

        # Chaque score_symbol fait 2 appels réseau séquentiels ; les exécuter
        # en parallèle réduit fortement la latence totale du scan.
        if to_score:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(_score_item, item): item for item in to_score}
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                    except Exception as e:
                        sym = futures[fut].get("symbol", "?")
                        logger.warning("score_symbol échec %s: %s", sym, e)
                        continue
                    if result:
                        results.append(result)

        # Score final : priorité au score complet puis faible late_entry_risk.
        results.sort(key=lambda x: (x["score"], -x.get("late_entry_risk", 0)), reverse=True)

        data_sources_used = {data_source_batch, btc_data_source}
        data_sources_used.update([r.get("data_source", "UNAVAILABLE") for r in results])
        data_source_run = "SPOT_FALLBACK" if "SPOT_FALLBACK" in data_sources_used else (
            "BYBIT_FUTURES" if "BYBIT_FUTURES" in data_sources_used else (
                "FUTURES" if "FUTURES" in data_sources_used else "UNAVAILABLE"
            )
        )
        source_quality_run = source_quality_label(data_source_run)

        # CANDIDAT = vrai signal. On bloque les late risk HIGH.
        candidats = [
            r for r in results
            if r["flag"] == "CANDIDAT"
            and r["rr_valid"]
            and r["direction"] != "NEUTRAL"
            and r.get("late_entry_level") != "HIGH"
            and r.get("market_danger_level") != "HIGH"
        ]
        candidats = limit_weak_candidates(candidats)

        if candidats and candidats[0].get("trend_strength") == "weak":
            candidats = candidats[:1]

        watchlist_fallback = False
        short_watch_fallback = False

        if not candidats:
            watchlist = [
                r for r in results
                if r["flag"] == "WATCHLIST"
                and r["rr_valid"]
                and r["direction"] != "NEUTRAL"
                and r.get("late_entry_level") != "HIGH"
            ]
            watchlist = limit_weak_candidates(watchlist)
            if watchlist and watchlist[0].get("trend_strength") == "weak":
                watchlist = watchlist[:1]
            if watchlist:
                candidats = watchlist[:1]
                watchlist_fallback = True

        # SHORT WATCH : seulement si aucun vrai candidat. Observation, pas trade automatique.
        if not candidats:
            short_watch = [r for r in results if r.get("flag") == "SHORT_WATCH"]
            short_watch.sort(key=lambda x: x.get("late_entry_risk", 0), reverse=True)
            if short_watch:
                candidats = short_watch[:1]
                short_watch_fallback = True

        if not candidats:
            return jsonify({
                "text":             "SKIP",
                "count":            0,
                "market_regime":    market_regime,
                "market_danger":    market_details,
                "data_source":      data_source_run,
                "source_quality":   source_quality_run,
                "universe_size":    len(tickers_data),
                "analyzed_count":   len(top_candidates),
                "cooldown_skipped": cooldown_skipped
            })

        # Formatage texte pour GPT — niveaux précalculés par Python.
        fallback_note = ""
        if watchlist_fallback:
            fallback_note = "\n⚠️ MODE WATCHLIST : aucun CANDIDAT disponible. Signal de calibration uniquement.\n"
        if short_watch_fallback:
            fallback_note = "\n🔴 MODE SHORT_WATCH : momentum potentiellement épuisé. Observation uniquement, pas de short automatique.\n"

        lines = [
            f"market_regime_btc: {market_regime}\n"
            f"market_danger_level: {market_details.get('market_danger_level')} | "
            f"market_danger_score: {market_details.get('market_danger_score')} | "
            f"btc_note: {market_details.get('btc_note')}\n"
            f"data_source: {data_source_run}\n"
            f"source_quality: {source_quality_run}\n"
            f"universe_size: {len(tickers_data)} | analyzed_count: {len(top_candidates)}\n"
            f"{fallback_note}"
        ]

        for r in candidats:
            lines.append(
                f"symbol: {r['symbol']} | flag: {r['flag']} | score: {r['score']} | "
                f"prescore: {r.get('prescore')} | gate: {r.get('prescore_gate')} | "
                f"direction: {r['direction']} | entry_type: {r['entry_type']} | "
                f"breakout_level: {r.get('breakout_level')} | "
                f"trend_strength: {r['trend_strength']} | rsi: {r['rsi']} | "
                f"ema_spread: {r['ema_spread']} | ema50_trend: {r['ema50_trend']} | "
                f"volume_relatif: {r['volume_relatif']} | atr_pct: {r['atr_pct']} | "
                f"momentum_24h: {r['momentum_24h']} | momentum_1h: {r.get('momentum_1h')} | momentum_3h: {r.get('momentum_3h')} | "
                f"distance_ema21: {r['distance_ema21']} | position_range: {r['position_range']} | "
                f"low_7d: {r.get('low_7d')} | distance_low_7d_pct: {r.get('distance_low_7d_pct')} | "
                f"short_forbidden: {r.get('short_forbidden')} | short_forbidden_reasons: {','.join(r.get('short_forbidden_reasons', []))} | "
                f"late_entry_risk: {r.get('late_entry_risk')} | late_entry_level: {r.get('late_entry_level')} | "
                f"late_entry_flags: {','.join(r.get('late_entry_flags', []))} | "
                f"market_regime: {r['market_regime']} | market_danger_level: {r.get('market_danger_level')} | "
                f"data_source: {r.get('data_source', data_source_run)} | "
                f"source_quality: {r.get('source_quality', source_quality_label(r.get('data_source', data_source_run)))} | "
                f"funding_rate: {r['funding_rate']} | funding_signal: {r['funding_signal']} | "
                f"derivatives_bias: {r['derivatives_bias']} | derivatives_note: {r['derivatives_note']} | "
                f"short_watch: {r.get('short_watch')} | short_watch_reasons: {','.join(r.get('short_watch_reasons', []))} | "
                f"entry_low: {r['entry_low']} | entry_high: {r['entry_high']} | entry_avg: {r['entry_avg']} | "
                f"stop_loss: {r['stop_loss']} | tp1: {r['tp1']} | tp2: {r['tp2']} | tp3: {r['tp3']} | tp4: {r['tp4']} | "
                f"risk_reward: {r['risk_reward']} | rr_valid: {r['rr_valid']} | "
                f"max_leverage: {r['max_leverage']} | confidence: {r['confidence']} | "
                f"duration_label: {r['duration_label']}"
            )

        for r in candidats:
            log_signal(r)

        return jsonify({
            "text":             "\n".join(lines),
            "count":            len(candidats),
            "market_regime":    market_regime,
            "market_danger":    market_details,
            "data_source":      data_source_run,
            "universe_size":    len(tickers_data),
            "analyzed_count":   len(top_candidates),
            "cooldown_skipped": cooldown_skipped
        })

    except Exception as e:
        return jsonify({"error": str(e), "text": "SKIP", "count": 0}), 500


# ─── PROVIDER TEST ─────────────────────────────────────────────────────────────

@app.route("/provider_test", methods=["GET"])
def provider_test():
    """
    Debug Railway/API providers.
    À appeler dans le navigateur :
    /provider_test

    Objectif :
    - vérifier si Binance Futures passe
    - vérifier si Bybit Futures passe
    - vérifier si Binance Vision Spot passe
    - vérifier si Binance Spot passe
    """
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results = {}

    try:
        bf_url = build_batch_ticker_url(
            "https://fapi.binance.com/fapi/v1/ticker/24hr",
            symbols
        )
        bf = fetch_binance(bf_url)
        results["binance_futures"] = {
            "ok": bf is not None,
            "count": len(bf) if isinstance(bf, list) else (1 if bf else 0)
        }
    except Exception as e:
        results["binance_futures"] = {"ok": False, "error": str(e)}

    try:
        bybit = fetch_bybit_linear_tickers(symbols)
        results["bybit_futures"] = {
            "ok": bybit is not None and len(bybit) > 0,
            "count": len(bybit) if bybit else 0
        }
    except Exception as e:
        results["bybit_futures"] = {"ok": False, "error": str(e)}

    try:
        bv_url = build_batch_ticker_url(
            "https://data-api.binance.vision/api/v3/ticker/24hr",
            symbols
        )
        bv = fetch_binance(bv_url)
        results["binance_vision_spot"] = {
            "ok": bv is not None,
            "count": len(bv) if isinstance(bv, list) else (1 if bv else 0)
        }
    except Exception as e:
        results["binance_vision_spot"] = {"ok": False, "error": str(e)}

    try:
        bs_url = build_batch_ticker_url(
            "https://api.binance.com/api/v3/ticker/24hr",
            symbols
        )
        bs = fetch_binance(bs_url)
        results["binance_spot"] = {
            "ok": bs is not None,
            "count": len(bs) if isinstance(bs, list) else (1 if bs else 0)
        }
    except Exception as e:
        results["binance_spot"] = {"ok": False, "error": str(e)}

    return jsonify({
        "status": "ok",
        "service": "crypto-scorer",
        "version": "5.4-source-quality-risk-prompt",
        "providers": results
    })


# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "crypto-scorer", "version": "5.4-source-quality-risk-prompt"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

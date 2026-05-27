from flask import Flask, request, jsonify
import numpy as np
import urllib.request
import json

app = Flask(__name__)

# ─── BINANCE API ──────────────────────────────────────────────────────────────

def fetch_binance(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if attempt == 2:
                return None
    return None

def get_klines(symbol, interval="1h", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    return fetch_binance(url)

def get_funding_rate(symbol):
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
    data = fetch_binance(url)
    if data and len(data) > 0:
        return float(data[0].get("fundingRate", 0))
    return 0.0

# ─── CALCULS TECHNIQUES ───────────────────────────────────────────────────────

def calculate_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
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
    k = 2 / (period + 1)
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
    return round(np.mean(trs[-period:]), 6)

def calculate_relative_volume(volumes, period=20):
    volumes = np.array(volumes, dtype=float)
    avg = np.mean(volumes[-period-1:-1])
    if avg == 0:
        return 0
    return round(volumes[-1] / avg, 2)

def detect_market_regime(btc_klines):
    if not btc_klines or len(btc_klines) < 30:
        return "unknown"
    closes  = [float(k[4]) for k in btc_klines]
    highs   = [float(k[2]) for k in btc_klines]
    lows    = [float(k[3]) for k in btc_klines]
    ema9    = calculate_ema(closes, 9)
    ema21   = calculate_ema(closes, 21)
    rsi     = calculate_rsi(closes)
    atr     = calculate_atr(highs, lows, closes)
    current = closes[-1]

    # Détection volatile/danger — priorité absolue
    atr_pct = (atr / current) * 100
    variation_2h = abs((closes[-1] - closes[-3]) / closes[-3] * 100) if len(closes) >= 3 else 0
    variation_4h = abs((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else 0

    if atr_pct > 4 or variation_2h > 3 or variation_4h > 5:
        return "volatile"

    if current > ema9 > ema21 and rsi > 50:
        return "bullish"
    elif current < ema9 < ema21 and rsi < 50:
        return "bearish"
    return "neutral"

# ─── SCORING v4.3 ─────────────────────────────────────────────────────────────
# v4   : direction stricte, momentum signé, pénalités pump/distance EMA
# v4.1 : ema_score et rsi_score directionnels, volume seuils relevés,
#         position_range, ema_spread, trend_strength
# v4.2 : market_regime BTC intégré dans score_symbol (+bonus alignement)
# v4.3 : RSI extrême gradué, distance EMA graduée, 3 bougies explosives,
#         support/résistance range 24h, BTC volatile/danger

def score_symbol(symbol, ticker_data=None, market_regime="unknown"):
    klines = get_klines(symbol)
    if not klines or len(klines) < 30:
        return None

    funding_rate = get_funding_rate(symbol)

    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    rsi          = calculate_rsi(closes)
    ema9         = calculate_ema(closes, 9)
    ema21        = calculate_ema(closes, 21)
    atr          = calculate_atr(highs, lows, closes)
    relative_vol = calculate_relative_volume(volumes)
    current      = closes[-1]

    # Variation 24h (avec signe — on ne prend PLUS abs())
    if ticker_data and "priceChangePercent" in ticker_data:
        price_change = float(ticker_data["priceChangePercent"])
        high_24h     = float(ticker_data.get("highPrice", current))
    else:
        price_change = round(((closes[-1] - closes[-24]) / closes[-24]) * 100, 2) if len(closes) >= 24 else 0
        high_24h     = max(highs[-24:]) if len(highs) >= 24 else max(highs)

    distance_high  = round(((high_24h - current) / high_24h) * 100, 2) if high_24h > 0 else 0
    distance_ema21 = round(abs((current - ema21) / ema21) * 100, 2) if ema21 > 0 else 0

    # Position du prix dans le range 24h (0.0 = bas du range, 1.0 = haut)
    low_24h    = min(lows[-24:]) if len(lows) >= 24 else min(lows)
    range_24h  = high_24h - low_24h
    position_range = round((current - low_24h) / range_24h, 3) if range_24h > 0 else 0.5

    # Force de tendance EMA
    ema_spread = round(abs((ema9 - ema21) / ema21) * 100, 3) if ema21 > 0 else 0
    if ema_spread > 2:
        trend_strength = "strong"
    elif ema_spread > 0.8:
        trend_strength = "moderate"
    else:
        trend_strength = "weak"

    # ── DIRECTION STRICTE v4 ──────────────────────────────────────────────────
    # LONG : prix AU-DESSUS de ema9 et ema9 > ema21 + RSI dans zone saine
    # SHORT : prix EN-DESSOUS de ema9 et ema9 < ema21 + RSI dans zone saine
    if current > ema9 > ema21 and 45 <= rsi <= 72:
        direction = "LONG"
    elif current < ema9 < ema21 and 28 <= rsi <= 55:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # ── SCORE MOMENTUM v4 ────────────────────────────────────────────────────
    # LONG : récompense uniquement les hausses (price_change > 0)
    # SHORT : récompense uniquement les baisses (price_change < 0)
    if direction == "LONG":
        if price_change >= 2:
            mom_score = min(100, (price_change / 10) * 100)
        else:
            mom_score = 0
    elif direction == "SHORT":
        if price_change <= -2:
            mom_score = min(100, (abs(price_change) / 10) * 100)
        else:
            mom_score = 0
    else:
        mom_score = 0

    # ── SCORE EMA v4.1 — symétrique LONG/SHORT ───────────────────────────────
    if direction == "LONG":
        if ema9 > ema21 * 1.002:
            ema_score = 80
        elif ema9 > ema21:
            ema_score = 60
        else:
            ema_score = 20
    elif direction == "SHORT":
        if ema9 < ema21 * 0.998:
            ema_score = 80
        elif ema9 < ema21:
            ema_score = 60
        else:
            ema_score = 20
    else:
        ema_score = 40

    # ── SCORE RSI v4.1 — directionnel ────────────────────────────────────────
    if direction == "LONG":
        if 45 <= rsi <= 65:
            rsi_score = 100
        elif 35 <= rsi < 45 or 65 < rsi <= 72:
            rsi_score = 60
        elif rsi < 35:
            rsi_score = 20   # survendu = pas idéal pour LONG momentum
        else:
            rsi_score = 10   # RSI > 72 = surachat
    elif direction == "SHORT":
        if 35 <= rsi <= 55:
            rsi_score = 100  # zone idéale short continuation
        elif 28 <= rsi < 35 or 55 < rsi <= 65:
            rsi_score = 60
        elif rsi > 65:
            rsi_score = 20   # suracheté = pas idéal pour SHORT momentum
        else:
            rsi_score = 10   # RSI < 28 = survendu extrême
    else:
        rsi_score = 30

    # ── SCORE VOLUME v4.1 — seuils relevés pour futures momentum ─────────────
    if relative_vol >= 1.8:
        vol_score = 100
    elif relative_vol >= 1.4:
        vol_score = 80
    elif relative_vol >= 1.1:
        vol_score = 50
    else:
        vol_score = 10

    # ── SCORE VOLATILITE ─────────────────────────────────────────────────────
    atr_pct = (atr / current) * 100
    if 0.5 <= atr_pct <= 3.0:
        vola_score = 100
    elif atr_pct < 0.5:
        vola_score = 20
    else:
        vola_score = 50

    # ── SCORE POSITION RANGE v4.1 ────────────────────────────────────────────
    # LONG : idéal si prix dans milieu-haut du range (0.4 à 0.85)
    # SHORT : idéal si prix dans milieu-bas du range (0.15 à 0.6)
    if direction == "LONG":
        if 0.4 <= position_range <= 0.85:
            range_score = 100
        elif 0.2 <= position_range < 0.4:
            range_score = 50
        elif position_range > 0.85:
            range_score = 30   # trop haut dans le range = entrée tardive
        else:
            range_score = 10
    elif direction == "SHORT":
        if 0.15 <= position_range <= 0.6:
            range_score = 100
        elif 0.6 < position_range <= 0.8:
            range_score = 50
        elif position_range < 0.15:
            range_score = 30   # trop bas dans le range = entrée tardive
        else:
            range_score = 10
    else:
        range_score = 30

    # ── SCORE GLOBAL v4.1 ────────────────────────────────────────────────────
    global_score = round(
        mom_score    * 0.22 +
        ema_score    * 0.20 +
        rsi_score    * 0.18 +
        vol_score    * 0.18 +
        vola_score   * 0.12 +
        range_score  * 0.10, 1
    )

    # ── PENALITES v4.3 ───────────────────────────────────────────────────────
    # 1. RSI extrême dans le sens du trade
    if direction == "LONG":
        if rsi > 68:
            global_score -= 10
        if rsi > 72:
            global_score -= 10  # double pénalité au-delà de 72
    if direction == "SHORT":
        if rsi < 32:
            global_score -= 10
        if rsi < 28:
            global_score -= 10  # double pénalité en-dessous de 28

    # 2. Distance EMA21 graduée
    if distance_ema21 > 3:
        global_score -= 5
    if distance_ema21 > 6:
        global_score -= 10
    if distance_ema21 > 10:
        global_score -= 10

    # Pump/dump trop tardif
    if direction == "LONG" and price_change > 12:
        global_score -= 15
    if direction == "SHORT" and price_change < -12:
        global_score -= 15

    # 3. Détection 3 bougies consécutives explosives (anti-FOMO)
    last_3_changes = []
    for i in range(-3, 0):
        if closes[i-1] > 0:
            chg = (closes[i] - closes[i-1]) / closes[i-1] * 100
            last_3_changes.append(chg)

    if len(last_3_changes) == 3:
        if all(c > 2.5 for c in last_3_changes) and direction == "LONG":
            global_score -= 20  # 3 bougies vertes explosives = entrée LONG tardive
        if all(c < -2.5 for c in last_3_changes) and direction == "SHORT":
            global_score -= 20  # 3 bougies rouges explosives = short tardif

    # 4. Proximité support/résistance (position dans range 24h)
    # Trop près du haut du range pour LONG = résistance proche
    # Trop près du bas du range pour SHORT = support proche
    if direction == "LONG" and position_range > 0.88:
        global_score -= 15  # proche résistance 24h
    if direction == "SHORT" and position_range < 0.12:
        global_score -= 15  # proche support 24h

    # Direction NEUTRAL = score plafonné à 45 (jamais CANDIDAT)
    if direction == "NEUTRAL":
        global_score = min(global_score, 45)

    # Pénalité/bonus market regime BTC v4.3
    if market_regime == "volatile":
        global_score -= 15  # marché chaotique = pénalité forte mais pas bloquante
    if market_regime == "bearish" and direction == "LONG":
        global_score -= 10
    if market_regime == "bullish" and direction == "SHORT":
        global_score -= 10
    if market_regime == "neutral":
        global_score -= 5   # marché sans direction = légère pénalité
    # Bonus alignement tendance macro
    if market_regime == "bearish" and direction == "SHORT":
        global_score += 8
    if market_regime == "bullish" and direction == "LONG":
        global_score += 8

    global_score = round(max(0, global_score), 1)

    # ── FLAG ─────────────────────────────────────────────────────────────────
    if global_score >= 58:
        flag = "CANDIDAT"
    elif global_score >= 50:
        flag = "WATCHLIST"
    else:
        flag = "REJET"

    return {
        "symbol":          symbol,
        "score":           global_score,
        "flag":            flag,
        "direction":       direction,
        "trend_strength":  trend_strength,
        "market_regime":   market_regime,
        "rsi":             rsi,
        "ema9":            ema9,
        "ema21":           ema21,
        "ema_spread":      ema_spread,
        "atr":             atr,
        "atr_pct":         round(atr_pct, 2),
        "volume_relatif":  relative_vol,
        "distance_high":   distance_high,
        "distance_ema21":  distance_ema21,
        "position_range":  position_range,
        "momentum_24h":    price_change,
        "funding_rate":    funding_rate,
        "current_price":   current,
    }

# ─── ENDPOINT /prescore v4 ────────────────────────────────────────────────────
# Garde le signe de priceChangePercent + filtre directionnel dès le préscore

@app.route("/prescore", methods=["POST"])
def prescore():
    data    = request.json
    tickers = data.get("tickers", [])

    scored = []
    for t in tickers:
        symbol           = t.get("symbol", "")
        price_change_pct = float(t.get("priceChangePercent", 0))  # AVEC signe
        volume           = float(t.get("quoteVolume", 0))
        high             = float(t.get("highPrice", 1))
        low              = float(t.get("lowPrice",  1))
        last             = float(t.get("lastPrice", 1))
        open_price       = float(t.get("openPrice", last))

        # Filtre minimum : variation absolue > 1.5% et volume suffisant
        if abs(price_change_pct) < 1.5 or volume < 1_000_000:
            continue

        # Filtre directionnel : on sépare pompes et dumps
        # On accepte les deux sens mais on garde l'info pour le scoring
        range_pct     = ((high - low) / low) * 100 if low > 0 else 0
        distance_high = ((high - last) / high) * 100 if high > 0 else 0

        # Momentum : récompense le mouvement dans sa direction (pas abs())
        directional_pct = price_change_pct  # +12 = pump, -8 = dump
        mom_score   = min(100, (abs(directional_pct) / 10) * 100)
        vol_score   = min(100, (volume / 50_000_000) * 100)
        range_score = min(100, (range_pct / 8) * 100)

        # Pénalité pump/dump trop fort dès le préscore
        penalty = 0
        if abs(price_change_pct) > 15:
            penalty = 20
        elif abs(price_change_pct) > 12:
            penalty = 10

        prescore_val = round(
            max(0, mom_score * 0.40 + vol_score * 0.35 + range_score * 0.25 - penalty), 1
        )

        scored.append({
            "symbol":           symbol,
            "prescore":         prescore_val,
            "price_change_pct": price_change_pct,
            "volume":           volume,
            "range_pct":        round(range_pct, 2),
            "distance_high":    round(distance_high, 2),
            "lastPrice":        last,
            "highPrice":        high,
            "lowPrice":         low,
            "openPrice":        open_price,
            "ticker":           t,
        })

    scored.sort(key=lambda x: x["prescore"], reverse=True)
    top8 = scored[:8]

    # Format prêt pour /score_batch — évite les problèmes de mapping Make
    symbols_for_batch = [{"symbol": s["symbol"], "ticker": s["ticker"]} for s in top8]
    # Liste simple des symbols pour Make (JSON string compatible)
    symbol_list = ",".join([s["symbol"] for s in top8])

    return jsonify({
        "total_analyzed": len(tickers),
        "after_filter":   len(scored),
        "top8":           top8,
        "symbols":        symbols_for_batch,
        "symbol_list":    symbol_list
    })

# ─── ENDPOINT /score ──────────────────────────────────────────────────────────

@app.route("/score", methods=["POST"])
def score():
    data   = request.json
    symbol = data.get("symbol", "")
    ticker = data.get("ticker", {})

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    btc_klines    = get_klines("BTCUSDT", limit=50)
    market_regime = detect_market_regime(btc_klines)

    result = score_symbol(symbol, ticker, market_regime)

    if not result:
        return jsonify({"error": f"Could not score {symbol}"}), 500

    return jsonify(result)

# ─── ENDPOINT /score_batch ────────────────────────────────────────────────────

@app.route("/score_batch", methods=["POST"])
def score_batch():
    data    = request.json
    # Accepte soit un array "symbols" soit une string "symbol_list" séparée par virgules
    symbols = data.get("symbols", data.get("top8", []))
    symbol_list_str = data.get("symbol_list", "")
    if not symbols and symbol_list_str:
        symbols = [{"symbol": s.strip(), "ticker": {}} for s in symbol_list_str.split(",") if s.strip()]

    btc_klines    = get_klines("BTCUSDT", limit=50)
    market_regime = detect_market_regime(btc_klines)

    results = []
    for item in symbols:
        symbol = item.get("symbol", "")
        ticker = item.get("ticker", {})
        result = score_symbol(symbol, ticker, market_regime)
        if result:
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    candidats = [r for r in results if r["flag"] == "CANDIDAT"]
    watchlist = [r for r in results if r["flag"] == "WATCHLIST"]

    return jsonify({
        "market_regime": market_regime,
        "candidats":     candidats,
        "watchlist":     watchlist,
        "rejets_count":  len([r for r in results if r["flag"] == "REJET"]),
        "total_pairs":   len(results)
    })

# ─── ENDPOINT /signal_text ───────────────────────────────────────────────────
# Retourne les candidats formatés en texte brut pour GPT — pas de problème Make

@app.route("/signal_text", methods=["POST"])
def signal_text():
    data    = request.json
    symbols = data.get("symbols", data.get("top8", []))
    symbol_list_str = data.get("symbol_list", "")
    if not symbols and symbol_list_str:
        symbols = [{"symbol": s.strip(), "ticker": {}} for s in symbol_list_str.split(",") if s.strip()]

    btc_klines    = get_klines("BTCUSDT", limit=50)
    market_regime = detect_market_regime(btc_klines)

    results = []
    for item in symbols:
        symbol = item.get("symbol", "")
        ticker = item.get("ticker", {})
        result = score_symbol(symbol, ticker, market_regime)
        if result:
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    candidats = [r for r in results if r["flag"] == "CANDIDAT"]

    if not candidats:
        return jsonify({"text": "SKIP", "count": 0})

    lines = [f"market_regime: {market_regime}\n"]
    for r in candidats:
        lines.append(
            f"symbol: {r['symbol']} | score: {r['score']} | flag: {r['flag']} | "
            f"direction: {r['direction']} | trend_strength: {r['trend_strength']} | "
            f"rsi: {r['rsi']} | ema_spread: {r['ema_spread']} | "
            f"volume_relatif: {r['volume_relatif']} | atr: {r['atr']} | atr_pct: {r['atr_pct']} | "
            f"momentum_24h: {r['momentum_24h']} | distance_ema21: {r['distance_ema21']} | "
            f"position_range: {r['position_range']} | current_price: {r['current_price']} | "
            f"funding_rate: {r['funding_rate']} | market_regime: {r['market_regime']}"
        )

    return jsonify({"text": "\n".join(lines), "count": len(candidats)})

@app.route("/full_analysis", methods=["POST"])
def full_analysis():
    data    = request.json

    # Récupère les 30 tickers directement depuis Binance — pas de dépendance Make
    symbols_config = [
        "BTCUSDT","ETHUSDT","SOLUSDT","LINKUSDT","AVAXUSDT","INJUSDT","FETUSDT",
        "RNDRUSDT","ARBUSDT","SUIUSDT","WLDUSDT","TIAUSDT","JUPUSDT","SEIUSDT",
        "GMXUSDT","PENDLEUSDT","ETHFIUSDT","STRKUSDT","ALTUSDT","PIXELUSDT",
        "PORTALUSDT","MANTAUSDT","NEARUSDT","APTUSDT","DOGEUSDT","WIFUSDT",
        "PEPEUSDT","BONKUSDT","FLOKIUSDT","BOMEUSDT"
    ]

    # Fetch ticker 24h pour toutes les paires
    symbols_str = "%2C".join(symbols_config)
    batch_url = "https://api.binance.com/api/v3/ticker/24hr?symbols=[" + ",".join([f'"{s}"' for s in symbols_config]) + "]"
    tickers_data = fetch_binance(batch_url)

    if not tickers_data:
        return jsonify({"text": "SKIP", "count": 0, "market_regime": "unknown", "error": "Binance unreachable"})

    # Prescore
    scored = []
    for t in tickers_data:
        symbol           = t.get("symbol", "")
        price_change_pct = float(t.get("priceChangePercent", 0))
        volume           = float(t.get("quoteVolume", 0))
        high             = float(t.get("highPrice", 1))
        low              = float(t.get("lowPrice", 1))

        if abs(price_change_pct) < 1.5 or volume < 1_000_000:
            continue

        range_pct    = ((high - low) / low) * 100 if low > 0 else 0
        mom_score    = min(100, (abs(price_change_pct) / 10) * 100)
        vol_score    = min(100, (volume / 50_000_000) * 100)
        range_score  = min(100, (range_pct / 8) * 100)
        penalty      = 20 if abs(price_change_pct) > 15 else (10 if abs(price_change_pct) > 12 else 0)
        prescore_val = round(max(0, mom_score * 0.40 + vol_score * 0.35 + range_score * 0.25 - penalty), 1)
        scored.append({"symbol": symbol, "prescore": prescore_val, "ticker": t})

    scored.sort(key=lambda x: x["prescore"], reverse=True)
    top8 = scored[:8]

    # Market regime
    btc_klines    = get_klines("BTCUSDT", limit=50)
    market_regime = detect_market_regime(btc_klines)

    # Scoring complet
    results = []
    for item in top8:
        result = score_symbol(item["symbol"], item["ticker"], market_regime)
        if result:
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    candidats = [r for r in results if r["flag"] == "CANDIDAT"]

    if not candidats:
        return jsonify({"text": "SKIP", "count": 0, "market_regime": market_regime})

    lines = [f"market_regime_btc: {market_regime}\n"]
    for r in candidats:
        lines.append(
            f"symbol: {r['symbol']} | score: {r['score']} | direction: {r['direction']} | "
            f"trend_strength: {r['trend_strength']} | rsi: {r['rsi']} | "
            f"ema_spread: {r['ema_spread']} | volume_relatif: {r['volume_relatif']} | "
            f"atr: {r['atr']} | atr_pct: {r['atr_pct']} | momentum_24h: {r['momentum_24h']} | "
            f"distance_ema21: {r['distance_ema21']} | position_range: {r['position_range']} | "
            f"current_price: {r['current_price']} | funding_rate: {r['funding_rate']} | "
            f"market_regime: {r['market_regime']}"
        )

    return jsonify({"text": "\n".join(lines), "count": len(candidats), "market_regime": market_regime})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "crypto-scorer", "version": "4.3"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

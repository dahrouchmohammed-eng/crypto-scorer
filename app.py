from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# ─── CALCULS TECHNIQUES ───────────────────────────────────────────────────────

def calculate_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
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
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)
    return round(np.mean(trs[-period:]), 6)

def calculate_relative_volume(volumes, period=20):
    volumes = np.array(volumes, dtype=float)
    avg_volume = np.mean(volumes[-period-1:-1])
    if avg_volume == 0:
        return 0
    return round(volumes[-1] / avg_volume, 2)

def detect_market_regime(btc_klines):
    closes = [float(k[4]) for k in btc_klines]
    ema9  = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    rsi   = calculate_rsi(closes)
    current = closes[-1]
    if current > ema9 > ema21 and rsi > 50:
        return "bullish"
    elif current < ema9 < ema21 and rsi < 50:
        return "bearish"
    else:
        return "neutral"

# ─── ENDPOINT /prescore ───────────────────────────────────────────────────────
# Filtre léger sur ticker 24h uniquement — pas de klines
# Retourne TOP 8 selon variation + volume + range

@app.route("/prescore", methods=["POST"])
def prescore():
    data   = request.json
    tickers = data.get("tickers", [])   # liste de ticker 24h Binance

    scored = []
    for t in tickers:
        symbol           = t.get("symbol", "")
        price_change_pct = abs(float(t.get("priceChangePercent", 0)))
        volume           = float(t.get("quoteVolume", 0))
        high             = float(t.get("highPrice", 1))
        low              = float(t.get("lowPrice", 1))
        last             = float(t.get("lastPrice", 1))

        # Filtre minimum — écarte les paires mortes
        if price_change_pct < 1.5:
            continue
        if volume < 1_000_000:
            continue

        # Range 24h relatif
        range_pct = ((high - low) / low) * 100 if low > 0 else 0

        # Distance au high 24h
        distance_high = ((high - last) / high) * 100 if high > 0 else 0

        # Score présélection (0-100)
        mom_score    = min(100, (price_change_pct / 10) * 100)
        vol_score    = min(100, (volume / 50_000_000) * 100)
        range_score  = min(100, (range_pct / 8) * 100)

        prescore_val = round(
            mom_score   * 0.40 +
            vol_score   * 0.35 +
            range_score * 0.25,
            1
        )

        scored.append({
            "symbol":           symbol,
            "prescore":         prescore_val,
            "price_change_pct": float(t.get("priceChangePercent", 0)),
            "volume":           volume,
            "range_pct":        round(range_pct, 2),
            "distance_high":    round(distance_high, 2),
            "lastPrice":        last,
            "highPrice":        high,
            "lowPrice":         low,
            "openPrice":        float(t.get("openPrice", last)),
        })

    # Trier par prescore et garder TOP 8
    scored.sort(key=lambda x: x["prescore"], reverse=True)
    top8 = scored[:8]

    return jsonify({
        "total_analyzed": len(tickers),
        "after_filter":   len(scored),
        "top8":           top8
    })


# ─── ENDPOINT /score ──────────────────────────────────────────────────────────
# Scoring complet avec klines H1 + funding

@app.route("/score", methods=["POST"])
def score():
    data       = request.json
    pairs      = data.get("pairs", [])       # [{symbol, klines, ticker, funding}]
    btc_klines = data.get("btc_klines", [])  # klines BTC pour market regime

    market_regime = detect_market_regime(btc_klines) if len(btc_klines) >= 30 else "unknown"

    results = []
    for pair in pairs:
        symbol       = pair.get("symbol")
        klines       = pair.get("klines", [])
        ticker       = pair.get("ticker", {})
        funding_rate = float(pair.get("funding_rate", 0))

        if len(klines) < 30:
            continue

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

        high_24h       = float(ticker.get("highPrice", current))
        distance_high  = round(((high_24h - current) / high_24h) * 100, 2)
        price_change   = float(ticker.get("priceChangePercent", 0))

        direction = "LONG" if ema9 > ema21 and rsi < 70 else "SHORT" if ema9 < ema21 and rsi > 30 else "NEUTRAL"

        # Scores composantes
        mom_score  = min(100, (abs(price_change) / 10) * 100) if abs(price_change) >= 2 else 0

        if ema9 > ema21 * 1.002:
            ema_score = 80
        elif ema9 > ema21:
            ema_score = 60
        elif ema9 < ema21 * 0.998:
            ema_score = 20
        else:
            ema_score = 40

        if 45 <= rsi <= 65:
            rsi_score = 100
        elif 35 <= rsi < 45 or 65 < rsi <= 75:
            rsi_score = 60
        elif rsi < 35:
            rsi_score = 30
        else:
            rsi_score = 10

        if relative_vol >= 2.0:
            vol_score = 100
        elif relative_vol >= 1.5:
            vol_score = 80
        elif relative_vol >= 1.0:
            vol_score = 50
        else:
            vol_score = 20

        atr_pct = (atr / current) * 100
        if 0.5 <= atr_pct <= 3.0:
            vola_score = 100
        elif atr_pct < 0.5:
            vola_score = 20
        else:
            vola_score = 50

        global_score = round(
            mom_score  * 0.25 +
            ema_score  * 0.20 +
            rsi_score  * 0.20 +
            vol_score  * 0.20 +
            vola_score * 0.15,
            1
        )

        if global_score >= 65:
            flag = "CANDIDAT"
        elif global_score >= 50:
            flag = "WATCHLIST"
        else:
            flag = "REJET"

        results.append({
            "symbol":          symbol,
            "score":           global_score,
            "flag":            flag,
            "direction":       direction,
            "rsi":             rsi,
            "ema9":            ema9,
            "ema21":           ema21,
            "atr":             atr,
            "atr_pct":         round(atr_pct, 2),
            "volume_relatif":  relative_vol,
            "distance_high":   distance_high,
            "momentum_24h":    price_change,
            "funding_rate":    funding_rate,
            "current_price":   current,
        })

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


# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "crypto-scorer", "version": "2.0"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

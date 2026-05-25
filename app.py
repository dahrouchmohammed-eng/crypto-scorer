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

def detect_market_regime(btc_data):
    closes = [float(k[4]) for k in btc_data]
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    rsi = calculate_rsi(closes)
    current = closes[-1]
    if current > ema9 > ema21 and rsi > 50:
        return "bullish"
    elif current < ema9 < ema21 and rsi < 50:
        return "bearish"
    else:
        return "neutral"

# ─── SCORING ──────────────────────────────────────────────────────────────────

def score_pair(klines, ticker, funding_rate):
    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    rsi           = calculate_rsi(closes)
    ema9          = calculate_ema(closes, 9)
    ema21         = calculate_ema(closes, 21)
    atr           = calculate_atr(highs, lows, closes)
    relative_vol  = calculate_relative_volume(volumes)
    current_price = closes[-1]

    # Distance au high 24h
    high_24h = float(ticker.get("highPrice", current_price))
    distance_high = round(((high_24h - current_price) / high_24h) * 100, 2)

    # Variation 24h
    price_change_pct = float(ticker.get("priceChangePercent", 0))

    # Direction suggérée
    direction = "LONG" if ema9 > ema21 and rsi < 70 else "SHORT" if ema9 < ema21 and rsi > 30 else "NEUTRAL"

    # ── SCORES COMPOSANTES (0-100) ────────────────────────────────────────────

    # 1. Momentum (25%) — variation 24h
    mom = abs(price_change_pct)
    momentum_score = min(100, (mom / 10) * 100) if mom >= 2 else 0

    # 2. Tendance EMA (20%) — alignement + croisement
    if ema9 > ema21 * 1.002:
        ema_score = 80
    elif ema9 > ema21:
        ema_score = 60
    elif ema9 < ema21 * 0.998:
        ema_score = 20
    else:
        ema_score = 40

    # 3. RSI (20%) — zone optimale 45-65
    if 45 <= rsi <= 65:
        rsi_score = 100
    elif 35 <= rsi < 45 or 65 < rsi <= 75:
        rsi_score = 60
    elif rsi < 35:
        rsi_score = 30  # potentiel rebond
    else:
        rsi_score = 10  # suracheté

    # 4. Volume relatif (20%)
    if relative_vol >= 2.0:
        volume_score = 100
    elif relative_vol >= 1.5:
        volume_score = 80
    elif relative_vol >= 1.0:
        volume_score = 50
    else:
        volume_score = 20

    # 5. Volatilité/ATR (15%) — exploitabilité
    atr_pct = (atr / current_price) * 100
    if 0.5 <= atr_pct <= 3.0:
        volatility_score = 100
    elif atr_pct < 0.5:
        volatility_score = 20  # trop calme
    else:
        volatility_score = 50  # trop volatile

    # ── SCORE GLOBAL ─────────────────────────────────────────────────────────
    global_score = round(
        momentum_score   * 0.25 +
        ema_score        * 0.20 +
        rsi_score        * 0.20 +
        volume_score     * 0.20 +
        volatility_score * 0.15,
        1
    )

    # ── FLAG ─────────────────────────────────────────────────────────────────
    if global_score >= 65:
        flag = "CANDIDAT"
    elif global_score >= 50:
        flag = "WATCHLIST"
    else:
        flag = "REJET"

    return {
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
        "momentum_24h":    price_change_pct,
        "funding_rate":    funding_rate,
        "scores_detail": {
            "momentum":   momentum_score,
            "ema":        ema_score,
            "rsi":        rsi_score,
            "volume":     volume_score,
            "volatilite": volatility_score
        }
    }

# ─── ENDPOINT PRINCIPAL ───────────────────────────────────────────────────────

@app.route("/score", methods=["POST"])
def score():
    data = request.json

    pairs      = data.get("pairs", [])        # [{symbol, klines, ticker, funding}]
    btc_klines = data.get("btc_klines", [])   # klines BTC pour market regime

    market_regime = detect_market_regime(btc_klines) if btc_klines else "unknown"

    results = []
    for pair in pairs:
        symbol       = pair.get("symbol")
        klines       = pair.get("klines", [])
        ticker       = pair.get("ticker", {})
        funding_rate = pair.get("funding_rate", 0)

        if len(klines) < 30:
            continue

        scored = score_pair(klines, ticker, funding_rate)
        scored["symbol"] = symbol
        results.append(scored)

    # Trier par score décroissant
    results.sort(key=lambda x: x["score"], reverse=True)

    # Séparer candidats / watchlist / rejets
    candidats  = [r for r in results if r["flag"] == "CANDIDAT"]
    watchlist  = [r for r in results if r["flag"] == "WATCHLIST"]
    rejets     = [r for r in results if r["flag"] == "REJET"]

    return jsonify({
        "market_regime": market_regime,
        "candidats":     candidats,
        "watchlist":     watchlist,
        "rejets_count":  len(rejets),
        "total_pairs":   len(results)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "crypto-scorer"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

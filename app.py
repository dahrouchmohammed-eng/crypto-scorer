from flask import Flask, request, jsonify
import numpy as np
import urllib.request
import json
import time
import os

app = Flask(__name__)

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
    except:
        pass
    return {}

def save_cooldown(data):
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(data, f)
    except:
        pass

def is_on_cooldown(symbol):
    cooldown = load_cooldown()
    if symbol in cooldown:
        elapsed = time.time() - cooldown[symbol]
        if elapsed < COOLDOWN_SECONDS:
            remaining = round((COOLDOWN_SECONDS - elapsed) / 3600, 1)
            return True, remaining
    return False, 0

def set_cooldown_symbols(symbols):
    cooldown = load_cooldown()
    now = time.time()
    for symbol in symbols:
        cooldown[symbol] = now
    cooldown = {k: v for k, v in cooldown.items() if time.time() - v < COOLDOWN_SECONDS}
    save_cooldown(cooldown)

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
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    return fetch_binance(url)

def get_funding_rate(symbol):
    try:
        url  = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        data = fetch_binance(url)
        if data and len(data) > 0:
            return float(data[0].get("fundingRate", 0))
    except:
        pass
    return 0.0

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

    atr_pct      = (atr / current) * 100 if current > 0 else 0
    variation_2h = abs((closes[-1] - closes[-3]) / closes[-3] * 100) if len(closes) >= 3 else 0
    variation_4h = abs((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else 0

    if atr_pct > 4 or variation_2h > 3 or variation_4h > 5:
        return "volatile"
    if current > ema9 > ema21 and rsi > 50:
        return "bullish"
    elif current < ema9 < ema21 and rsi < 50:
        return "bearish"
    return "neutral"

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

def score_symbol(symbol, ticker_data=None, market_regime="unknown"):
    klines = get_klines(symbol)
    if not klines or len(klines) < 55:
        return None

    funding_rate = get_funding_rate(symbol)

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

    # ── ENTRY TYPE — indépendant de direction ────────────────────────────────
    # Un setup peut être LONG techniquement mais encore EARLY en momentum
    if early_long or early_short:
        entry_type = "EARLY"
        # Override direction si NEUTRAL avec early détecté
        if direction == "NEUTRAL":
            direction = "LONG" if early_long else "SHORT"
    elif direction != "NEUTRAL":
        entry_type = "MOMENTUM"
    else:
        entry_type = "NEUTRAL"

    # ── SCORE MOMENTUM ───────────────────────────────────────────────────────
    if direction == "LONG":
        if entry_type == "EARLY":
            mom_score = 40  # score modéré pour early entry
        else:
            mom_score = min(100, (price_change / 10) * 100) if price_change >= 2 else 0
    elif direction == "SHORT":
        if entry_type == "EARLY":
            mom_score = 40
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

    # 7. Direction NEUTRAL → plafonné à 45
    if direction == "NEUTRAL":
        global_score = min(global_score, 45)

    # 8. Market regime BTC
    if market_regime == "volatile":                         global_score -= 15
    if market_regime == "bearish" and direction == "LONG":  global_score -= 10
    if market_regime == "bullish" and direction == "SHORT": global_score -= 10
    if market_regime == "neutral":                          global_score -= 5
    if market_regime == "bearish" and direction == "SHORT": global_score += 8
    if market_regime == "bullish" and direction == "LONG":  global_score += 8

    # 9. Pénalité légère early entry (moins fiable que momentum)
    if entry_type == "EARLY": global_score -= 5

    global_score = round(max(0, global_score), 1)

    # ── FLAG ─────────────────────────────────────────────────────────────────
    if global_score >= 58:
        flag = "CANDIDAT"
    elif global_score >= 52:
        flag = "WATCHLIST"
    else:
        flag = "REJET"

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
    memecoins = ["DOGEUSDT","WIFUSDT","PEPEUSDT","BONKUSDT","FLOKIUSDT","BOMEUSDT"]
    majors    = ["BTCUSDT","ETHUSDT","SOLUSDT"]
    if symbol in memecoins:    sl_max_pct = 2.0
    elif symbol in majors:     sl_max_pct = 4.0
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
    memecoins = ["DOGEUSDT","WIFUSDT","PEPEUSDT","BONKUSDT","FLOKIUSDT","BOMEUSDT"]
    majors    = ["BTCUSDT","ETHUSDT","SOLUSDT"]
    if symbol in memecoins:    base_leverage = 5
    elif symbol in majors:     base_leverage = 10
    else:                      base_leverage = 7

    leverage_caps = [base_leverage]
    if trend_strength == "weak":    leverage_caps.append(3)
    if market_regime == "neutral":  leverage_caps.append(5)
    if market_regime == "volatile": leverage_caps.append(3)
    if entry_type == "EARLY":       leverage_caps.append(3)
    if flag == "WATCHLIST":         leverage_caps.append(3)
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
    if entry_type == "EARLY":                               confidence -= 5
    if distance_ema21 > 6:                                  confidence -= 5
    if (direction == "LONG"  and not (0.4 <= position_range <= 0.85)) or \
       (direction == "SHORT" and not (0.15 <= position_range <= 0.6)): confidence -= 5

    # Plafonds confiance
    if trend_strength == "weak": confidence = min(confidence, 68)
    if flag == "WATCHLIST":      confidence = min(confidence, 60)
    confidence = round(max(45, min(88, confidence)), 1)

    return {
        "symbol":          symbol,
        "score":           global_score,
        "flag":            flag,
        "direction":       direction,
        "entry_type":      entry_type,
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
        "momentum_24h":    price_change,
        "funding_rate":    funding_rate,
        "current_price":   current,
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
        symbols_config = [
            "BTCUSDT","ETHUSDT","SOLUSDT","LINKUSDT","AVAXUSDT","INJUSDT","FETUSDT",
            "RNDRUSDT","ARBUSDT","SUIUSDT","WLDUSDT","TIAUSDT","JUPUSDT","SEIUSDT",
            "GMXUSDT","PENDLEUSDT","ETHFIUSDT","STRKUSDT","ALTUSDT","PIXELUSDT",
            "PORTALUSDT","MANTAUSDT","NEARUSDT","APTUSDT","DOGEUSDT","WIFUSDT",
            "PEPEUSDT","BONKUSDT","FLOKIUSDT","BOMEUSDT"
        ]

        batch_url    = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbols=[" + ",".join([f'"{s}"' for s in symbols_config]) + "]"
        tickers_data = fetch_binance(batch_url)

        if not tickers_data:
            return jsonify({"text": "SKIP", "count": 0, "market_regime": "unknown", "error": "Binance unreachable"})

        # ── DOUBLE PORTE PRESCORE v4.4 ────────────────────────────────────────
        scored = []
        for t in tickers_data:
            symbol           = t.get("symbol", "")
            price_change_pct = float(t.get("priceChangePercent", 0))
            volume           = float(t.get("quoteVolume", 0))
            high             = float(t.get("highPrice", 1))
            low              = float(t.get("lowPrice",  1))

            if volume < 1_000_000:
                continue

            range_pct = ((high - low) / low) * 100 if low > 0 else 0

            # Porte 1 : Momentum classique
            momentum_gate = abs(price_change_pct) >= 2.0
            # Porte 2 : Early setup — compression + range actif mais mouvement pas encore fort
            early_gate    = abs(price_change_pct) < 2.0 and range_pct >= 1.5

            if not momentum_gate and not early_gate:
                continue

            mom_score    = min(100, (abs(price_change_pct) / 10) * 100)
            vol_score    = min(100, (volume / 50_000_000) * 100)
            range_score  = min(100, (range_pct / 8) * 100)
            penalty      = 20 if abs(price_change_pct) > 15 else (10 if abs(price_change_pct) > 12 else 0)
            # Bonus porte early pour compenser le faible momentum
            early_bonus  = 10 if early_gate else 0
            prescore_val = round(max(0, mom_score * 0.40 + vol_score * 0.35 + range_score * 0.25 - penalty + early_bonus), 1)

            scored.append({"symbol": symbol, "prescore": prescore_val, "ticker": t, "gate": "early" if early_gate else "momentum"})

        scored.sort(key=lambda x: x["prescore"], reverse=True)
        # Double porte : garantir représentation des early setups
        momentum_top = [x for x in scored if x["gate"] == "momentum"][:5]
        early_top    = [x for x in scored if x["gate"] == "early"][:5]
        top_candidates = momentum_top + early_top  # max 10 paires analysées

        # Market regime BTC
        btc_klines    = get_klines("BTCUSDT", limit=50)
        market_regime = detect_market_regime(btc_klines)

        # Scoring complet avec filtre cooldown (lecture seule)
        results          = []
        cooldown_skipped = []
        for item in top_candidates:
            on_cd, remaining = is_on_cooldown(item["symbol"])
            if on_cd:
                cooldown_skipped.append(f"{item['symbol']} ({remaining}h)")
                continue
            result = score_symbol(item["symbol"], item["ticker"], market_regime)
            if result:
                results.append(result)

        results.sort(key=lambda x: x["score"], reverse=True)
        candidats = [
            r for r in results
            if r["flag"] == "CANDIDAT"
            and r["rr_valid"]
            and r["direction"] != "NEUTRAL"
        ]
        candidats = limit_weak_candidates(candidats)

        # Fallback : si aucun CANDIDAT valide en R/R, envoyer la meilleure WATCHLIST à GPT
        # GPT la traitera avec règles strictes (confiance max 60%, levier 3x)
        watchlist_fallback = False
        if not candidats:
            watchlist = [
                r for r in results
                if r["flag"] == "WATCHLIST"
                and r["rr_valid"]
                and r["direction"] != "NEUTRAL"
            ]
            watchlist = limit_weak_candidates(watchlist)
            if watchlist:
                candidats = watchlist[:1]
                watchlist_fallback = True

        if not candidats:
            return jsonify({
                "text":             "SKIP",
                "count":            0,
                "market_regime":    market_regime,
                "cooldown_skipped": cooldown_skipped
            })

        # Formatage texte pour GPT — niveaux précalculés par Python
        fallback_note = "\n⚠️ MODE WATCHLIST : aucun CANDIDAT disponible. Signal de calibration uniquement.\n" if watchlist_fallback else ""
        lines = [f"market_regime_btc: {market_regime}\n{fallback_note}"]
        for r in candidats:
            lines.append(
                f"symbol: {r['symbol']} | flag: {r['flag']} | score: {r['score']} | "
                f"direction: {r['direction']} | entry_type: {r['entry_type']} | "
                f"trend_strength: {r['trend_strength']} | rsi: {r['rsi']} | "
                f"ema_spread: {r['ema_spread']} | ema50_trend: {r['ema50_trend']} | "
                f"volume_relatif: {r['volume_relatif']} | atr_pct: {r['atr_pct']} | "
                f"momentum_24h: {r['momentum_24h']} | distance_ema21: {r['distance_ema21']} | "
                f"position_range: {r['position_range']} | market_regime: {r['market_regime']} | "
                f"entry_low: {r['entry_low']} | entry_high: {r['entry_high']} | entry_avg: {r['entry_avg']} | "
                f"stop_loss: {r['stop_loss']} | tp1: {r['tp1']} | tp2: {r['tp2']} | tp3: {r['tp3']} | tp4: {r['tp4']} | "
                f"risk_reward: {r['risk_reward']} | rr_valid: {r['rr_valid']} | "
                f"max_leverage: {r['max_leverage']} | confidence: {r['confidence']}"
            )

        return jsonify({
            "text":             "\n".join(lines),
            "count":            len(candidats),
            "market_regime":    market_regime,
            "cooldown_skipped": cooldown_skipped
        })

    except Exception as e:
        return jsonify({"error": str(e), "text": "SKIP", "count": 0}), 500

# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "crypto-scorer", "version": "4.4-futures"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

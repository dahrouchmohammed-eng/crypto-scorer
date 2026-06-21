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
from datetime import datetime, timezone
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

# v6.4.0 — Parallélisme réduit pour /evaluate_signals.
# L'évaluation fait plusieurs pages klines 5m par signal → Binance/Bybit rate-limit
# rapidement si MAX_WORKERS=8 est utilisé sur des batches de 60-100 signaux.
# EVAL_MAX_WORKERS=3 évite les klines indisponibles par saturation provider.
EVAL_MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", "2"))

# ─── SOURCES DE DONNÉES ─────────────────────────────────────────────────────────
# Binance Futures (fapi.binance.com) activé par défaut — source principale.
# Bybit Futures en fallback secondaire si Binance est bloqué.
# Désactivable via variable d'environnement : BINANCE_ENABLED=false
BINANCE_ENABLED = os.environ.get("BINANCE_ENABLED", "true").lower() == "true"

# Timeouts courts : si une source est bloquée, on ne veut pas attendre longtemps.
HTTP_TIMEOUT  = int(os.environ.get("HTTP_TIMEOUT", "8"))
HTTP_RETRIES  = int(os.environ.get("HTTP_RETRIES", "2"))

# ─── CONFIG V6.0.5 ─────────────────────────────────────────────────────────────
# V6.0.6e : Fix source taker_buy/sell_ratio
#   FIX — taker_buy_ratio et taker_sell_ratio lus depuis v6_futures_raw (source fiable)
#         (avant : r.get("taker_buy_ratio") = champ inexistant au niveau signal)
#   --- héritées v6.0.6d ---
#   oi_change_pct / funding_signal / derivatives_note
#   DBG1 — oi_change_pct : variation OI brute (depuis v6_futures_raw)
#   DBG2 — taker_sell_ratio : calculé depuis taker_buy_ratio
#   DBG3 — funding_signal + derivatives_note
#   --- héritées v6.0.6c ---
#   Fix OI threshold hardcodé
#   BUG — compute_futures_score_v6 utilisait encore des seuils OI hardcodés (0.05/0.10/0.20)
#         malgré OI_BONUS_TABLE recalibrée → oi=+0 persistant
#   FIX — Bloc OI remplacé par _tiered_pts(OI_BONUS_TABLE/OI_MALUS_TABLE) + price_aligned
#   --- héritées v6.0.6b ---
#   Recalibrage OI (÷10) + Taker (53%/58%/65%)
#   CAL1 — OI : seuils ÷10 (0.5%/1%/2% au lieu de 5%/10%/20%)
#   CAL2 — Taker : seuils abaissés (53%/58%/65% au lieu de 60%/70%)
#   CAL3 — Taker malus enrichi : 3 paliers (47%/42%/35%)
#   Raison : oi=+0 et taker=+0 systématiques → données bien reçues mais seuils hors marché
#   --- héritées v6.0.6 ---
#   Règle confirmation futures MOMENTUM
#   CF1 — futures_support = nb indicateurs positifs (OI, Taker, Funding, L/S)
#   CF2 — MOMENTUM + vol < 0.50 + futures_support < 2 → WATCHLIST
#   CF3 — Exception : taker >= +8 → CANDIDAT maintenu (flux court terme suffisant)
#   CF4 — vol < 0.30 → WATCHLIST direct sans exception
#   CF5 — caps : confidence ≤60 (vol<0.30) ou ≤65 (vol<0.50) + levier ≤3x
#   --- héritées v6.0.5c ---
#   Tech gate non bloquant / pénalité volume MOMENTUM
#   FIX — gate < 58 retournait v6_accepted=False → flag REJET systématique
#         Corrigé : v6_accepted=True, score technique conservé, flag naturel
#         WATCHLIST si score ≥52, REJET si score <52 (comportement voulu)
#         ETH 54.0 → WATCHLIST ✓ | NEAR 53.8 → WATCHLIST ✓ | SOL 47.1 → REJET ✓
#   --- héritées v6.0.5b ---
#   Pénalité volume MOMENTUM / garde-fou post-V6
#   VOL1 — volume < 0.50 : pénalité -8 pts (score)
#   VOL2 — volume < 0.30 : pénalité -15 pts + rétrogradation CANDIDAT → WATCHLIST
#   VOL3 — volume < 0.15 : pénalité -20 pts + WATCHLIST + levier cap 3x
#   VOL4 — EARLY exclu (volume faible normal avant départ)
#   VOL5 — BREAKOUT exclu (déjà protégé par relative_vol >= 1.3)
#   VOL6 — vol_penalty_note exposé dans JSON et texte GPT
#   --- héritées v6.0.4c ---
#   liquidations désactivées / FUTURES_RAW ±38 / 4 appels parallèles
#   FIX — /fapi/v1/allForceOrders déprécié par Binance ("endpoint out of maintenance")
#         Appel remplacé par stub None — zéro appel réseau inutile par cycle
#         available_count repassé à 5 (liquidations exclues)
#         Remplacement prévu via CoinGlass en v6.1
#   --- héritées v6.0.4 ---
#   fetch_liquidations_v6 conservé dans le code mais non appelé
#   LIQ1 — fetch_liquidations_v6 : long_liq/short_liq/imbalance sur fenêtre configurable
#   LIQ2 — scoring liquidations : +12/+6 si alignées, -12/-6 si opposées, -6 cascade extrême
#   FIX — FUTURES_RAW_MIN/MAX revenu à ±38 car liquidations exclues du scoring
#   LIQ4 — v6_futures_raw exposé dans tous les retours (données brutes visibles dans JSON/GPT)
#   LIQ5 — taker_sell_ratio calculé et exposé
#   LIQ6 — _safe_float() helper robuste pour payloads API incomplets
#   LIQ7 — Tous les seuils liquidations externalisés en variables d'env (LIQ_NOTIONAL_*)
#   --- héritées v6.0.3 ---
#   RR hybride / SL technique / TP en R / contrôle réalisme / deux RR
#   RR1 — SL reste TECHNIQUE (ATR plafonné), jamais éloigné pour gonfler le RR
#   RR2 — TP en multiples de R : TP1=1R, TP2=2R, TP3=3R, TP4=5R, TP5=target_rr
#   RR3 — CONTRÔLE DE RÉALISME du TP final : target_rr=8R si atteignable
#         (budget ATR + high/low 7j + late_entry), sinon rabattu à 5R
#   RR4 — Deux RR : risk_reward_tp2 (reporting) / risk_reward_target=TP4 5R (veto stable)
#   RR5 — funding=None si indisponible (plus de faux bonus 'funding neutre')
#   --- héritées v6.0.2 ---
#   check_veto_v6(direction) / veto avant fallback data
#   --- héritées v6.0.1 ---
#   veto tous flags / fallback data / funding directionnel / OI directionnel / gate=58 / candidats=12
# SAFE_MODE=True = conserve une logique prudente tant que Binance Futures reste instable.
SAFE_MODE = True

# ─── CONFIG V6.0 — SEUILS FUTURES (à tuner sans toucher au code) ──────────────

# Poids combinaison score final
V6_WEIGHT_TECH    = 0.70   # poids score technique (moteur existant)
V6_WEIGHT_FUTURES = 0.30   # poids score futures (OI + funding + L/S + taker)

# Couche 1 — VETO bloquant (5 règles)
VETO_FUNDING_EXTREME    = 0.0008   # |funding| > 0.08% → veto
VETO_OI_COLLAPSE_PCT    = -0.10    # OI < -10% sur la fenêtre → veto
VETO_MIN_RR_V6          = 3.0      # RR réaliste < 3 → veto
# BTC danger : utilise le market_danger_level existant (HIGH → veto)
# Liquidité : utilise MIN_QUOTE_VOLUME_USDT existant

# Couche 3 — Barème bonus/malus Open Interest
# Seuils recalibrés v6.0.6b : OI varie de 0.1% à 2%/h en conditions normales.
# Anciens seuils (5%/10%/20%) ne se déclenchaient jamais → oi=+0 systématique.
OI_BONUS_TABLE = [(0.02, 12), (0.01, 8), (0.005, 4)]   # +2% / +1% / +0.5%
OI_MALUS_TABLE = [(-0.02, -12), (-0.01, -6)]            # -2% / -1%

# Couche 3 — Barème bonus/malus Taker buy ratio
# Seuils recalibrés v6.0.6b : marché calme = taker entre 45-55%.
# Anciens seuils (70%/60%) trop stricts → taker=+0 systématique.
TAKER_BONUS_TABLE = [(0.65, 12), (0.58, 8), (0.53, 4)]   # pression acheteuse
TAKER_MALUS_TABLE = [(0.35, -8), (0.42, -4), (0.47, -2)] # pression vendeuse

# Couche 3 — Funding (bonus/malus doux, veto dur géré séparément)
FUNDING_FAVORABLE_PTS   = 4      # funding réellement favorable → +4
FUNDING_HOSTILE_PTS     = -10    # funding extrême dans mauvais sens → -10
FUNDING_EXTREME_SOFT    = 0.0005 # seuil soft (0.05%) pour le malus doux
FUNDING_NEUTRAL_BAND    = 0.0001 # ±0.01% = neutre

# Couche 3 — Long/Short ratio
LS_TOP_ALIGNED_BONUS    = 6    # top traders alignés avec notre direction → +6
LS_CROWD_MALUS          = -8   # foule trop chargée dans notre sens → -8
LS_CROWD_THRESHOLD_LONG = 2.0  # ratio > 2 = trop de longs dans la foule
LS_CROWD_THRESHOLD_SHORT= 0.5  # ratio < 0.5 = trop de shorts dans la foule

# Couche 3 — Liquidations forcées (Binance Futures allForceOrders)
# Dans les liquidation orders :
#   side=SELL ≈ liquidation de LONGS (vente forcée)  → pression vendeuse
#   side=BUY  ≈ liquidation de SHORTS (achat forcé) → pression acheteuse / squeeze
LIQ_LOOKBACK_MINUTES      = int(os.environ.get("LIQ_LOOKBACK_MINUTES", "60"))
LIQ_API_LIMIT             = int(os.environ.get("LIQ_API_LIMIT", "1000"))
LIQ_NOTIONAL_SOFT_USDT    = float(os.environ.get("LIQ_NOTIONAL_SOFT_USDT", "50000"))
LIQ_NOTIONAL_STRONG_USDT  = float(os.environ.get("LIQ_NOTIONAL_STRONG_USDT", "250000"))
LIQ_NOTIONAL_EXTREME_USDT = float(os.environ.get("LIQ_NOTIONAL_EXTREME_USDT", "1000000"))
# NOTE calibration : ces seuils sont calibrés pour BTC/ETH/SOL.
# Pour les altcoins du universe (AVAX, LINK, etc.), les volumes de liquidations sont
# souvent 10x–50x inférieurs. Surveiller les logs liq_count + total_liq_usdt sur 3–5 jours
# post-déploiement et abaisser via variables d'env si le module reste systématiquement inactif.
# Exemple altcoins : LIQ_NOTIONAL_SOFT_USDT=5000 / LIQ_NOTIONAL_STRONG_USDT=25000
LIQ_IMBALANCE_STRONG      = float(os.environ.get("LIQ_IMBALANCE_STRONG", "0.65"))
LIQ_EXTREME_MALUS_PTS     = -6   # volatilité/cascade : on réduit un peu le score même si aligné

# Normalisation score futures (bornes du raw avant normalisation 0–100)
# Revenu à ±38 car le module liquidations est désactivé en v6.0.4c.
FUTURES_RAW_MIN = -38.0
FUTURES_RAW_MAX =  38.0

# ─── CONFIG V6.0.7 — CALIBRATION LÉGÈRE ─────────────────────────────────────
# Objectif : ne pas ouvrir tous les filtres, mais corriger les 2 biais détectés
# par le forward backtest : taker trop bloquant et futures_score trop linéaire.
TAKER_SOFT_PENALTY_PTS       = float(os.environ.get("TAKER_SOFT_PENALTY_PTS", "4"))
TAKER_SOFT_CONF_CAP          = float(os.environ.get("TAKER_SOFT_CONF_CAP", "75"))
TAKER_SOFT_LEVERAGE_CAP      = int(os.environ.get("TAKER_SOFT_LEVERAGE_CAP", "5"))

FUTURES_HEALTHY_MIN          = float(os.environ.get("FUTURES_HEALTHY_MIN", "50"))
FUTURES_HEALTHY_MAX          = float(os.environ.get("FUTURES_HEALTHY_MAX", "65"))
FUTURES_CAUTION_MAX          = float(os.environ.get("FUTURES_CAUTION_MAX", "70"))
FUTURES_OVERHEATED_THRESHOLD = float(os.environ.get("FUTURES_OVERHEATED_THRESHOLD", "70"))
FUTURES_HEALTHY_BONUS_PTS    = float(os.environ.get("FUTURES_HEALTHY_BONUS_PTS", "2"))
FUTURES_OVERHEATED_PENALTY   = float(os.environ.get("FUTURES_OVERHEATED_PENALTY", "6"))
FUTURES_OVERHEATED_CONF_CAP  = float(os.environ.get("FUTURES_OVERHEATED_CONF_CAP", "72"))
FUTURES_OVERHEATED_LEV_CAP   = int(os.environ.get("FUTURES_OVERHEATED_LEV_CAP", "5"))
FUTURES_LATE_POSITION_RANGE  = float(os.environ.get("FUTURES_LATE_POSITION_RANGE", "0.70"))

# ─── CONFIG V6.1 — DECISION ENGINE CENTRALISÉ ───────────────────────────────
DECISION_VERSION = os.environ.get("DECISION_VERSION", "v6.5.3")
V61_LATE_ENTRY_RISK_MIN = float(os.environ.get("V61_LATE_ENTRY_RISK_MIN", "55"))
V61_PROMOTE_MAX_LATE_RISK = float(os.environ.get("V61_PROMOTE_MAX_LATE_RISK", "45"))
V61_PROMOTE_MIN_VOLUME = float(os.environ.get("V61_PROMOTE_MIN_VOLUME", "0.50"))
V61_PROMOTE_LONG_MAX_POSITION = float(os.environ.get("V61_PROMOTE_LONG_MAX_POSITION", "0.75"))
V61_PROMOTE_SHORT_MIN_POSITION = float(os.environ.get("V61_PROMOTE_SHORT_MIN_POSITION", "0.25"))

# ─── CONFIG V6.4.4 — BTC CONTEXT LAYER / BUCKET ENGINE ──────────────────────
# Seuils externalisés pour recalibrage sans toucher à la logique métier.
BTC_BULL_IMPULSE_VAR4H = float(os.environ.get("BTC_BULL_IMPULSE_VAR4H", "1.5"))
BTC_BULL_IMPULSE_VAR2H = float(os.environ.get("BTC_BULL_IMPULSE_VAR2H", "0.5"))
BTC_BULL_IMPULSE_RSI   = float(os.environ.get("BTC_BULL_IMPULSE_RSI", "60"))

BTC_BULL_SOFT_VAR4H = float(os.environ.get("BTC_BULL_SOFT_VAR4H", "0.3"))
BTC_BULL_SOFT_RSI   = float(os.environ.get("BTC_BULL_SOFT_RSI", "52"))

BTC_SWITCH_VAR30M_MIN = float(os.environ.get("BTC_SWITCH_VAR30M_MIN", "-0.1"))
BTC_SWITCH_VAR2H_MIN  = float(os.environ.get("BTC_SWITCH_VAR2H_MIN", "-0.2"))
BTC_SWITCH_VAR4H_MIN  = float(os.environ.get("BTC_SWITCH_VAR4H_MIN", "-0.3"))
BTC_SWITCH_RSI_MIN    = float(os.environ.get("BTC_SWITCH_RSI_MIN", "45"))
BTC_SWITCH_PREV_VAR4H_MAX = float(os.environ.get("BTC_SWITCH_PREV_VAR4H_MAX", "-0.1"))

BTC_BEAR_EXHAUSTION_VAR4H_MIN = float(os.environ.get("BTC_BEAR_EXHAUSTION_VAR4H_MIN", "-0.7"))
BTC_BEAR_CONT_VAR4H_MAX = float(os.environ.get("BTC_BEAR_CONT_VAR4H_MAX", "-0.5"))
BTC_BEAR_CONT_VAR2H_MAX = float(os.environ.get("BTC_BEAR_CONT_VAR2H_MAX", "-0.2"))

LONG_PREMIUM_PR_BULL_IMPULSE = float(os.environ.get("LONG_PREMIUM_PR_BULL_IMPULSE", "0.80"))
LONG_PREMIUM_PR_BULL_SOFT    = float(os.environ.get("LONG_PREMIUM_PR_BULL_SOFT", "0.70"))
LONG_PREMIUM_PR_DEFAULT      = float(os.environ.get("LONG_PREMIUM_PR_DEFAULT", "0.65"))

# ─── CONFIG V6.5.3 — CONTEXTUAL BUCKET RULES / SIGNALS BETA ─────────────────
# Le score brut reste informatif. Ces règles ne changent pas le scoring :
# elles pré-qualifient ou déclassent selon setup_family × btc_phase × conditions.
ENABLE_SHORT_MOMENTUM_CONTINUATION_BETA = os.environ.get(
    "ENABLE_SHORT_MOMENTUM_CONTINUATION_BETA", "true"
).lower() == "true"
SHORT_MOMENTUM_BETA_CONF_CAP = float(os.environ.get("SHORT_MOMENTUM_BETA_CONF_CAP", "62"))
SHORT_MOMENTUM_BETA_LEVERAGE_CAP = int(os.environ.get("SHORT_MOMENTUM_BETA_LEVERAGE_CAP", "3"))

TELEGRAM_ALLOWED_BUCKETS = {"LONG_PREMIUM", "SHORT_MOMENTUM_CONTINUATION_PREMIUM"}

WATCHLIST_BUCKET_PRIORITY = {
    "LONG_EARLY_NEUTRAL_PREMIUM":             100,
    "WATCHLIST_PREMIUM_SCORE_HIGH_EARLY":      95,
    "SHORT_MOMENTUM_CONTINUATION_PREMIUM":     92,
    "WATCHLIST_PREMIUM_LONG_STRONG":           90,
    "SHORT_EARLY_BEAR_CONTEXT_DIAGNOSTIC":     82,
    "SHORT_PREMIUM_CANDIDATE_DISABLED":        81,
    "WATCHLIST_SHORT_MOMENTUM_BEARISH":        80,
    "WATCHLIST_LONG_STRONG_DIAGNOSTIC":        60,
    "WATCHLIST_LONG_LATE_MOMENTUM":            55,
    "REJECT_LONG_LATE_MOMENTUM":               54,
    "REJECT_LONG_STRONG_DIAGNOSTIC":           53,
    "REJECT_PREMIUM_BUCKET_CLEANUP":           52,
    "WATCHLIST_SCORE_HIGH_REVIEW":             50,
    "WATCHLIST_LONG_STRONG_REVIEW":            45,
    "WATCHLIST_SHORT_MOMENTUM_BLOCKED":        20,
    "WATCHLIST_NON_PREMIUM_CANDIDATE":         10,
    "REJECT_SHORT_EARLY_BULL_CONTEXT":          6,
    "REJECT_SHORT_BTC30_POSITIVE":              5,
    "REJECT_SHORT_MOMENTUM_BULLISH":            5,
    "STANDARD":                                 0,
}

# Paramètres API futures data
OI_PERIOD   = "1h"   # granularité OI history
OI_LOOKBACK = 6      # nb de points pour mesurer la variation d'OI

# Seuil minimum score technique pour déclencher l'appel futures (économie réseau)
# À 58 : seuls les vrais candidats et watchlist solides déclenchent les appels API futures
# (monter à 62-65 progressivement si les timeouts augmentent)
FUTURES_TECH_SCORE_GATE = 58

# Universe dynamique : on part de toutes les paires USDT disponibles via ticker 24h,
# puis on garde uniquement les plus liquides / actives.
MAX_DYNAMIC_UNIVERSE = 90
MAX_FULL_ANALYSIS_CANDIDATES = 12   # réduit à 12 pour v6.0.1 (moins d'appels API futures)
MIN_QUOTE_VOLUME_USDT = 15_000_000

# Nombre de signaux envoyés à GPT. Python est maître de la décision ;
# GPT ne fait que recopier le nombre de candidats reçus, sans en ajouter ni retirer.
MAX_SIGNALS_DEFAULT       = 2   # cas nominal
MAX_SIGNALS_NEUTRAL       = 2   # market_regime neutral
MAX_SIGNALS_VOLATILE      = 1   # market_regime volatile -> prudence
MAX_SIGNALS_SPOT_FALLBACK = 1   # source spot -> prudence
MAX_SIGNALS_ABSOLUTE      = 4   # plafond dur

# Fenêtres d'évaluation (forward backtest, utilisées par /evaluate_signals) :
# - FILL : délai pour que le prix touche la zone d'entrée, sinon NO_FILL.
# - RESOLVE : délai max après remplissage pour toucher SL/TP, sinon EXPIRED.
FILL_WINDOW_SECONDS    = int(os.environ.get("FILL_WINDOW_SECONDS", str(4 * 3600)))    # 4h
RESOLVE_WINDOW_SECONDS = int(os.environ.get("RESOLVE_WINDOW_SECONDS", str(72 * 3600)))  # 72h

TIER1_ALWAYS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
EXCLUDED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
EXCLUDED_SYMBOLS = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT"}

# v6.4.2.2 — Dynamic universe guard :
# On conserve l'univers dynamique, mais on exclut les instruments non-crypto /
# equity-like / ETF-like qui peuvent apparaître chez certains providers et polluer
# le scoring crypto. Cette liste n'est PAS une whitelist : toutes les autres
# paires crypto USDT restent autorisées si elles passent les filtres de liquidité.
EXCLUDED_NON_CRYPTO_SYMBOLS = {
    # Observés dans les runs récents
    "SOXLUSDT", "MRVLUSDT", "BTWUSDT",
    # Métaux précieux / instruments non crypto
    "XAUUSDT", "XAGUSDT",

    # Actions tokenisées / equity-like fréquentes selon providers
    "AAPLUSDT", "AMZNUSDT", "AMDUSDT", "COINUSDT", "GOOGLUSDT",
    "METAUSDT", "MSFTUSDT", "MSTRUSDT", "NFLXUSDT", "NVDAUSDT",
    "PLTRUSDT", "TSLAUSDT",

    # ETF / leveraged equity-like
    "SOXSUSDT", "SPYUSDT", "QQQUSDT", "TQQQUSDT", "SQQQUSDT",
}

# Classes d'actifs pour SL / levier (factorisé : était dupliqué dans score_symbol)
MEMECOINS = {"DOGEUSDT", "WIFUSDT", "PEPEUSDT", "BONKUSDT", "FLOKIUSDT", "BOMEUSDT"}
MAJORS    = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


# ─── COOLDOWN ─────────────────────────────────────────────────────────────────
# Cooldown déclenché APRÈS envoi Telegram via /set_cooldown
# PAS dans /full_analysis — pour ne pas bloquer des candidats non sélectionnés

COOLDOWN_FILE    = "/tmp/cooldown.json"
COOLDOWN_SECONDS = 4 * 3600

# ── Anti-doublon intra-session (v6.0.6g) ──────────────────────────────────────
# Track les setup_ids émis en mémoire avec timestamp.
# Bloque un setup_id identique dans les 60 minutes suivant son émission,
# sans attendre que Make appelle /set_cooldown.
# Reset automatique au redémarrage du service (comportement voulu).
_SETUP_ID_LOCK = threading.Lock()
_emitted_setup_ids: dict = {}   # {setup_id: timestamp_emitted}
SETUP_ID_BLOCK_SECONDS = 3600   # 60 minutes

def is_setup_id_blocked(setup_id: str) -> bool:
    with _SETUP_ID_LOCK:
        ts = _emitted_setup_ids.get(setup_id)
        if ts is None:
            return False
        if time.time() - ts > SETUP_ID_BLOCK_SECONDS:
            del _emitted_setup_ids[setup_id]
            return False
        return True

def mark_setup_id_emitted(setup_id: str):
    with _SETUP_ID_LOCK:
        _emitted_setup_ids[setup_id] = time.time()
        # Purger les anciens
        now = time.time()
        expired = [k for k, v in _emitted_setup_ids.items()
                   if now - v > SETUP_ID_BLOCK_SECONDS]
        for k in expired:
            del _emitted_setup_ids[k]


def build_dedup_key(r):
    """
    Clé anti-doublon stable calculée AVANT build_signal_record.
    Objectif : éviter de renvoyer le même symbole dans le même sens pendant
    la fenêtre de blocage, même si le setup_id horaire change.
    """
    symbol = normalize_symbol(r.get("symbol", ""))
    direction = str(r.get("direction", "")).upper().strip()
    if not symbol or not direction:
        return ""
    return f"{symbol}-{direction}"

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

        # Garder uniquement les cooldowns symboles au format numérique.
        # Les clés techniques setup_ids/setup_ids_ts ne doivent jamais passer
        # dans le calcul now - v, sinon le JSON de cooldown peut provoquer un 500.
        cleaned = {
            k: v for k, v in cooldown.items()
            if k not in ("setup_ids", "setup_ids_ts")
            and isinstance(v, (int, float))
            and now - v < COOLDOWN_SECONDS
        }

        for symbol in symbols:
            symbol = normalize_symbol(symbol)
            if symbol:
                cleaned[symbol] = now

        # Préserver les setup_ids / dedup_keys séparément.
        setup_ids_ts = cooldown.get("setup_ids_ts", {})
        if isinstance(setup_ids_ts, dict):
            setup_ids_ts = {
                str(k): v for k, v in setup_ids_ts.items()
                if isinstance(v, (int, float)) and now - v < COOLDOWN_SECONDS
            }
            cleaned["setup_ids"] = list(setup_ids_ts.keys())
            cleaned["setup_ids_ts"] = setup_ids_ts

        save_cooldown(cleaned)

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
            "btc_market_state": entry.get("btc_market_state"),
            "btc_market_state_reason": entry.get("btc_market_state_reason"),
            "futures_zone": entry.get("futures_zone"),
            "futures_overheated": entry.get("futures_overheated"),
            "taker_not_confirmed": entry.get("taker_not_confirmed"),
            "decision_version": entry.get("decision_version"),
            "healthy_futures_zone": entry.get("healthy_futures_zone"),
            "overheated_futures": entry.get("overheated_futures"),
            "late_entry_risk_v6_1": entry.get("late_entry_risk_v6_1"),
            "crowded_risk": entry.get("crowded_risk"),
            "watchlist_promotion_candidate": entry.get("watchlist_promotion_candidate"),
            "signal_downgrade_candidate": entry.get("signal_downgrade_candidate"),
            "final_decision_reason": entry.get("final_decision_reason"),
            "signal_quality_bucket": entry.get("signal_quality_bucket"),
            "regime_rule_applied": entry.get("regime_rule_applied"),
            "pr_threshold_used": entry.get("pr_threshold_used"),
            "telegram_rule_notes": entry.get("telegram_rule_notes"),
            "v652_actions": entry.get("v652_actions"),
            "v652_notes": entry.get("v652_notes"),
            "v652_short_beta_ok": entry.get("v652_short_beta_ok"),
            # ── v6.5.0 — audit complet contexte / setup / participation / RR ──
            "btc_phase": entry.get("btc_phase"),
            "btc_context_bias": entry.get("btc_context_bias"),
            "btc_market_state": entry.get("btc_market_state"),
            "btc_impulse_age": entry.get("btc_impulse_age"),
            "btc_last_pivot_type": entry.get("btc_last_pivot_type"),
            "btc_last_pivot_age": entry.get("btc_last_pivot_age"),
            "btc_last_pivot_distance_pct": entry.get("btc_last_pivot_distance_pct"),
            "btc_last_pivot_method": entry.get("btc_last_pivot_method"),
            "setup_family": entry.get("setup_family"),
            "setup_context_alignment": entry.get("setup_context_alignment"),
            "setup_maturity": entry.get("setup_maturity"),
            "volume_regime": entry.get("volume_regime"),
            "volume_quality": entry.get("volume_quality"),
            "volume_context": entry.get("volume_context"),
            "derivatives_alignment": entry.get("derivatives_alignment"),
            "crowding_state": entry.get("crowding_state"),
            "participation_score": entry.get("participation_score"),
            "participation_warning": entry.get("participation_warning"),
            "rr_tp1": entry.get("rr_tp1"),
            "rr_tp2": entry.get("rr_tp2"),
            "rr_tp3": entry.get("rr_tp3"),
            "rr_tp4": entry.get("rr_tp4"),
            "rr_tp5": entry.get("rr_tp5"),
        }
        with _SIGNAL_LOG_LOCK:
            with open(SIGNAL_LOG_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("log_signal a échoué: %s", e)

# ─── BINANCE API ──────────────────────────────────────────────────────────────

def build_signal_record(r, market_regime, data_source_run, emitted_ts, market_details=None):
    """
    Construit un objet signal structuré et plat, destiné à l'archivage
    (Google Sheet via Make) et au futur /evaluate_signals.
    Tous les champs sont déjà calculés par Python : on ne fait que les exposer
    proprement, un objet par signal réellement envoyé.

    signal_id = symbol-timestamp : identifiant unique pour retrouver la ligne
    dans la Sheet et mettre à jour son statut (OPEN -> WIN/LOSS/NO_FILL).
    ref_price = prix au moment de l'émission : sert de référence à l'évaluation.
    """
    market_details = market_details or {}
    symbol = r.get("symbol", "")
    ref_price = r.get("current_price")
    src_quality = r.get("source_quality", source_quality_label(r.get("data_source", data_source_run)))
    fill_deadline_ts = emitted_ts + FILL_WINDOW_SECONDS
    resolve_deadline_ts = emitted_ts + RESOLVE_WINDOW_SECONDS

    # setup_id : regroupe les signaux du même symbole+direction dans un bloc de 4h.
    # Permet de calculer le winrate par setup indépendant (≠ winrate par signal brut).
    # Format : BTCUSDT-SHORT-202606011 (jour + bloc 4h : 0,4,8,12,16,20)
    dt_emit   = datetime.fromtimestamp(emitted_ts, tz=timezone.utc)
    hour_block = (dt_emit.hour // 4) * 4
    setup_id  = f"{symbol}-{r.get('direction','')}-{dt_emit.strftime('%Y%m%d')}{hour_block:02d}"

    return {
        "signal_id":       f"{symbol}-{emitted_ts}",
        "setup_id":        setup_id,
        "dedup_key":       r.get("dedup_key") or build_dedup_key(r),
        "timestamp":       datetime.fromtimestamp(emitted_ts, tz=timezone.utc).isoformat(),
        "symbol":          symbol,
        "direction":       r.get("direction"),
        "entry_type":      r.get("entry_type"),
        "entry_low":       r.get("entry_low"),
        "entry_high":      r.get("entry_high"),
        "entry_avg":       r.get("entry_avg"),
        "ref_price":       ref_price,
        "stop_loss":       r.get("stop_loss"),
        "tp1":             r.get("tp1"),
        "tp2":             r.get("tp2"),
        "tp3":             r.get("tp3"),
        "tp4":             r.get("tp4"),
        "tp5":             r.get("tp5"),
        "score":           r.get("score"),
        "confidence":      r.get("confidence"),
        "max_leverage":    r.get("max_leverage"),
        "risk_reward":     r.get("risk_reward"),
        "risk_reward_tp2":    r.get("risk_reward_tp2"),
        "risk_reward_target": r.get("risk_reward_target"),
        "target_rr":          r.get("target_rr"),
        "rr_tp1":             r.get("rr_tp1"),
        "rr_tp2":             r.get("rr_tp2"),
        "rr_tp3":             r.get("rr_tp3"),
        "rr_tp4":             r.get("rr_tp4"),
        "rr_tp5":             r.get("rr_tp5"),
        "flag":            r.get("flag"),
        "source_quality":  src_quality,
        "data_source":     r.get("data_source", data_source_run),
        "market_regime":   market_regime,
        # ── Contexte BTC instant T v6.4.2 — data enrichment uniquement ─────
        "btc_rsi":              market_details.get("btc_rsi"),
        "btc_variation_15m":    market_details.get("btc_variation_15m"),
        "btc_variation_30m":    market_details.get("btc_variation_30m"),
        "btc_variation_2h":     market_details.get("btc_variation_2h"),
        "btc_variation_4h":     market_details.get("btc_variation_4h"),
        "btc_variation_12h":    market_details.get("btc_variation_12h"),
        "btc_atr_pct":          market_details.get("btc_atr_pct"),
        "market_danger_level":  market_details.get("market_danger_level"),
        "market_danger_score":  market_details.get("market_danger_score"),
        "btc_note":             market_details.get("btc_note"),
        "btc_market_state":     market_details.get("btc_market_state", "BTC_NEUTRAL_COMPRESS"),
        "btc_market_state_reason": market_details.get("btc_market_state_reason", ""),
        "btc_context_bias": market_details.get("btc_context_bias"),
        "btc_phase": market_details.get("btc_phase"),
        "btc_trend_slope_2h": market_details.get("btc_trend_slope_2h"),
        "btc_trend_slope_4h": market_details.get("btc_trend_slope_4h"),
        "btc_trend_slope_12h": market_details.get("btc_trend_slope_12h"),
        "btc_impulse_age": market_details.get("btc_impulse_age"),
        "btc_last_pivot_type": market_details.get("btc_last_pivot_type"),
        "btc_last_pivot_age": market_details.get("btc_last_pivot_age"),
        "btc_last_pivot_distance_pct": market_details.get("btc_last_pivot_distance_pct"),
        "btc_last_pivot_method": market_details.get("btc_last_pivot_method"),
        "btc_pullback_depth": market_details.get("btc_pullback_depth"),
        "btc_range_position": market_details.get("btc_range_position"),
        "btc_rejection_state": market_details.get("btc_rejection_state"),
        "btc_support_distance_pct": market_details.get("btc_support_distance_pct"),
        "btc_resistance_distance_pct": market_details.get("btc_resistance_distance_pct"),
        "btc_volatility_regime": market_details.get("btc_volatility_regime"),
        "btc_context_score": market_details.get("btc_context_score"),
        "trend_strength":  r.get("trend_strength"),
        "rsi":             r.get("rsi"),
        "position_range":  r.get("position_range"),
        # ── Champs setup / participation v6.5.0 ─────────────────────────────
        "setup_family": r.get("setup_family"),
        "setup_variant": r.get("setup_variant"),
        "setup_maturity": r.get("setup_maturity"),
        "setup_directional_quality": r.get("setup_directional_quality"),
        "setup_context_alignment": r.get("setup_context_alignment"),
        "setup_late_risk_label": r.get("setup_late_risk_label"),
        "volume_regime": r.get("volume_regime"),
        "volume_quality": r.get("volume_quality"),
        "volume_context": r.get("volume_context"),
        "volume_vs_move": r.get("volume_vs_move"),
        "oi_regime": r.get("oi_regime"),
        "taker_regime": r.get("taker_regime"),
        "funding_regime": r.get("funding_regime"),
        "crowding_state": r.get("crowding_state"),
        "derivatives_alignment": r.get("derivatives_alignment"),
        "participation_score": r.get("participation_score"),
        "participation_warning": r.get("participation_warning"),
        # ── Champs qualité signal v6.0.6 ────────────────────────────────────
        "prescore":           r.get("prescore"),
        "volume_relatif":     r.get("volume_relatif"),
        "vol_penalty_note":   r.get("vol_penalty_note", ""),
        "late_entry_risk":    r.get("late_entry_risk"),
        "v6_score_futures":   r.get("v6_score_futures"),
        "tp_realism_note":    r.get("tp_realism_note", ""),
        "v6_futures_detail":  (
            f"oi={r.get('v6_futures_detail',{}).get('oi',0):+d} "
            f"taker={r.get('v6_futures_detail',{}).get('taker',0):+d} "
            f"funding={r.get('v6_futures_detail',{}).get('funding',0):+d} "
            f"ls={r.get('v6_futures_detail',{}).get('long_short',0):+d}"
        ) if r.get("v6_futures_detail") else "",
        "v6_data_errors":     " | ".join(r.get("v6_data_errors", [])) if isinstance(r.get("v6_data_errors"), list) else r.get("v6_data_errors", ""),
        # ── Champs debug v6.0.6e/f ───────────────────────────────────────────
        "oi_change_pct":      r.get("v6_futures_raw", {}).get("oi_change_pct"),
        "taker_buy_ratio":    r.get("v6_futures_raw", {}).get("taker_buy_ratio"),
        "taker_sell_ratio":   r.get("v6_futures_raw", {}).get("taker_sell_ratio") or (
            round(1 - r.get("v6_futures_raw", {}).get("taker_buy_ratio"), 4)
            if r.get("v6_futures_raw", {}).get("taker_buy_ratio") is not None else None
        ),
        "funding_signal":     r.get("funding_signal", ""),
        "derivatives_note":   r.get("derivatives_note", ""),
        # ── Champs décision v6.0.6f ──────────────────────────────────────────
        "executable_signal":  r.get("executable_signal", False),
        "taker_score":        r.get("taker_score", 0),
        "oi_score":           r.get("oi_score", 0),
        "funding_score":      r.get("funding_score", 0),
        "long_short_score":   r.get("long_short_score", 0),
        "futures_support":    r.get("futures_support", 0),
        "risk_guard_reason":  r.get("risk_guard_reason", "aucun"),
        "decision_explain":   r.get("decision_explain", ""),
        # ── Champs calibration v6.0.7 ───────────────────────────────────────
        "futures_zone":       r.get("futures_zone", "unavailable"),
        "healthy_futures_confirmation": r.get("healthy_futures_confirmation", False),
        "futures_overheated": r.get("futures_overheated", False),
        "taker_not_confirmed": r.get("taker_not_confirmed", False),
        "calibration_flags":  " | ".join(r.get("calibration_flags", [])) if isinstance(r.get("calibration_flags"), list) else r.get("calibration_flags", ""),
        "confidence_cap_reason": r.get("confidence_cap_reason", ""),
        "leverage_cap_reason": r.get("leverage_cap_reason", ""),
        # ── Champs decision_engine v6.1 ─────────────────────────────────────
        "decision_version": r.get("decision_version", DECISION_VERSION),
        "healthy_futures_zone": r.get("healthy_futures_zone", False),
        "overheated_futures": r.get("overheated_futures", False),
        "late_entry_risk_v6_1": r.get("late_entry_risk_v6_1", False),
        "crowded_risk": r.get("crowded_risk", False),
        "watchlist_promotion_candidate": r.get("watchlist_promotion_candidate", False),
        "signal_downgrade_candidate": r.get("signal_downgrade_candidate", False),
        "final_decision_reason": r.get("final_decision_reason", r.get("decision_explain", "")),
        # ── Champs sélection Telegram v6.4.3 ───────────────────────────────
        "signal_quality_bucket": r.get("signal_quality_bucket", "STANDARD"),
        "regime_rule_applied": r.get("regime_rule_applied", "STANDARD_NO_CONTEXT_RULE"),
        "pr_threshold_used": r.get("pr_threshold_used"),
        "telegram_rule_notes": r.get("telegram_rule_notes", ""),
        "v652_actions": r.get("v652_actions", ""),
        "v652_notes": r.get("v652_notes", ""),
        "v652_short_beta_ok": r.get("v652_short_beta_ok", False),
        # ── Champs d'évaluation (forward backtest) ──────────────────────────
        "outcome":         "OPEN",
        "fill_deadline":   datetime.fromtimestamp(fill_deadline_ts, tz=timezone.utc).isoformat(),
        "resolve_deadline":datetime.fromtimestamp(resolve_deadline_ts, tz=timezone.utc).isoformat(),
        # Remplis par /evaluate_signals :
        "filled_at":       "",
        "closed_at":       "",
        "exit_price":      "",
        "fill_time_minutes": "",   # durée entre émission et fill — confirme si entrées trop proches
        "evaluation_note": "",
        "bars_checked":    "",
    }


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

# ─── MODULE FUTURES V6.0 — OI / LONG-SHORT RATIO / TAKER BUY/SELL ────────────
# Source : Binance USD-M Futures public (fapi.binance.com), sans clé API.
# Architecture 3 couches :
#   Couche 1 — VETO         : 5 règles bloquantes
#   Couche 2 — TECHNIQUE    : moteur existant (score_symbol), poids 70%
#   Couche 3 — FUTURES      : bonus/malus OI+funding+L/S+taker, poids 30%
# Score final = 0.70 * technique + 0.30 * futures   (après veto)

def _fetch_futures_endpoint(path, params):
    """GET sur fapi.binance.com avec retry (réutilise fetch_binance)."""
    qs = urllib.parse.urlencode(params)
    url = f"https://fapi.binance.com{path}?{qs}"
    return fetch_binance(url)

def _safe_float(value, default=0.0):
    """Conversion float robuste pour les payloads API parfois incomplets."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def fetch_liquidations_v6(symbol):
    """
    Liquidations désactivées depuis v6.0.4b/v6.0.4c.

    Ancien endpoint Binance /fapi/v1/allForceOrders déprécié :
    {"code":400,"msg":"The endpoint has been out of maintenance"}

    Cette fonction est conservée uniquement pour compatibilité interne,
    mais ne fait AUCUN appel réseau. Réactivation prévue via CoinGlass
    ou un collecteur WebSocket Binance forceOrder en v6.1.
    """
    return {
        "long_liq_usdt": None,
        "short_liq_usdt": None,
        "total_liq_usdt": None,
        "liq_imbalance": None,
        "largest_liq_usdt": None,
        "liq_count": None,
        "liq_window_min": LIQ_LOOKBACK_MINUTES,
        "error": "liquidations:disabled"
    }


def fetch_futures_data_v6(symbol):
    """
    Récupère en parallèle les indicateurs futures Binance pour un symbole :
    OI, taker buy/sell, long/short global, top traders long/short.

    Les champs liquidations sont conservés à None pour compatibilité JSON,
    mais aucun appel réseau liquidations n'est effectué en v6.0.4c.
    """
    result = {
        "oi_change_pct": None,
        "taker_buy_ratio": None,
        "taker_sell_ratio": None,
        "ls_global_long_ratio": None,
        "ls_top_long_ratio": None,
        "long_liq_usdt": None,
        "short_liq_usdt": None,
        "total_liq_usdt": None,
        "liq_imbalance": None,
        "largest_liq_usdt": None,
        "liq_count": None,
        "liq_window_min": LIQ_LOOKBACK_MINUTES,
        "errors": []
    }

    def _oi():
        data = _fetch_futures_endpoint(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": OI_PERIOD, "limit": OI_LOOKBACK}
        )
        if data and len(data) >= 2:
            try:
                old = float(data[0]["sumOpenInterest"])
                new = float(data[-1]["sumOpenInterest"])
                return (new - old) / old if old else None
            except Exception as e:
                result["errors"].append(f"oi:{e}")
        else:
            result["errors"].append("oi:no_data")
        return None

    def _taker():
        data = _fetch_futures_endpoint(
            "/futures/data/takerlongshortRatio",
            {"symbol": symbol, "period": OI_PERIOD, "limit": 1}
        )
        if data and len(data) >= 1:
            try:
                r = float(data[-1]["buySellRatio"])
                return r / (1 + r)   # convertir en part d'achat 0..1
            except Exception as e:
                result["errors"].append(f"taker:{e}")
        else:
            result["errors"].append("taker:no_data")
        return None

    def _ls_global():
        data = _fetch_futures_endpoint(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": OI_PERIOD, "limit": 1}
        )
        if data and len(data) >= 1:
            try:
                return float(data[-1]["longShortRatio"])
            except Exception as e:
                result["errors"].append(f"ls_global:{e}")
        return None

    def _ls_top():
        data = _fetch_futures_endpoint(
            "/futures/data/topLongShortPositionRatio",
            {"symbol": symbol, "period": OI_PERIOD, "limit": 1}
        )
        if data and len(data) >= 1:
            try:
                return float(data[-1]["longShortRatio"])
            except Exception as e:
                result["errors"].append(f"ls_top:{e}")
        return None

    # Appels parallèles : 4 endpoints réseau.
    # Les liquidations sont désactivées et restent à None dans result.
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_oi     = ex.submit(_oi)
        f_taker  = ex.submit(_taker)
        f_lsg    = ex.submit(_ls_global)
        f_lst    = ex.submit(_ls_top)

        result["oi_change_pct"]        = f_oi.result()
        # f_taker.result() bloque jusqu'à complétion — taker_sell_ratio calculé juste après,
        # une fois la valeur disponible (pas de risque de race condition).
        result["taker_buy_ratio"]      = f_taker.result()
        result["taker_sell_ratio"]     = round(1 - result["taker_buy_ratio"], 4) if result["taker_buy_ratio"] is not None else None
        result["ls_global_long_ratio"] = f_lsg.result()
        result["ls_top_long_ratio"]    = f_lst.result()

    return result


def check_veto_v6(fd, direction, rr, market_danger_level, quote_volume_24h):
    """
    Couche 1 : 5 règles bloquantes.
    Retourne (passed: bool, reasons: list[str]).
    """
    reasons = []

    # Règle 1 : RR trop faible
    if rr is None or rr < VETO_MIN_RR_V6:
        reasons.append(f"RR {rr} < {VETO_MIN_RR_V6} (RR minimum requis)")

    # Règle 2 : Funding extrême DIRECTIONNEL
    # funding très positif = longs crowded → veto seulement pour LONG
    # funding très négatif = shorts crowded → veto seulement pour SHORT
    # (l'inverse peut être un avantage : funding positif + SHORT = squeeze possible)
    funding_rate = fd.get("funding_rate")
    if funding_rate is not None:
        hostile_funding = (
            (direction == "LONG"  and funding_rate  >  VETO_FUNDING_EXTREME) or
            (direction == "SHORT" and funding_rate  < -VETO_FUNDING_EXTREME)
        )
        if hostile_funding:
            reasons.append(f"funding hostile {funding_rate:+.4%} pour {direction} (seuil ±{VETO_FUNDING_EXTREME:.4%})")

    # Règle 3 : BTC en danger (réutilise market_danger_level existant)
    if market_danger_level == "HIGH":
        reasons.append("BTC market_danger_level HIGH")

    # Règle 4 : Liquidité insuffisante
    if quote_volume_24h is not None and quote_volume_24h < MIN_QUOTE_VOLUME_USDT:
        reasons.append(f"liquidité 24h {quote_volume_24h:,.0f} < {MIN_QUOTE_VOLUME_USDT:,.0f}")

    # Règle 5 : OI en chute libre
    oi_chg = fd.get("oi_change_pct")
    if oi_chg is not None and oi_chg < VETO_OI_COLLAPSE_PCT:
        reasons.append(f"OI en chute {oi_chg:+.1%} (< {VETO_OI_COLLAPSE_PCT:.0%})")

    return len(reasons) == 0, reasons


def _tiered_pts(value, bonus_table, malus_table):
    """Applique le premier palier atteint (tables triées du + fort au + faible)."""
    for threshold, pts in bonus_table:
        if value >= threshold:
            return pts
    for threshold, pts in malus_table:
        if value <= threshold:
            return pts
    return 0


def compute_futures_score_v6(fd, direction, volume_relatif=None):
    """Couche 3 : score futures v6.0.6f — funding neutre=0, OI conditionnel."""
    raw = 0
    breakdown = {}
    is_long = (direction == "LONG")
    momentum_1h = fd.get("momentum_1h", 0) or 0

    # — Open Interest DIRECTIONNEL —
    # OI seul ne suffit pas : il faut que le prix confirme la direction
    # OI↑ + prix dans notre sens = soutenu → bonus (OI_BONUS_TABLE)
    # OI↑ + prix contre notre sens = divergence → malus doux (moitié du bonus)
    # OI CONDITIONNEL v6.0.6f : bonus complet si taker>0 OU vol>=0.50
    oi_chg = fd.get("oi_change_pct")
    if oi_chg is not None:
        price_aligned = (is_long and momentum_1h >= 0) or (not is_long and momentum_1h <= 0)
        taker_raw = fd.get("taker_buy_ratio")
        taker_dir = taker_raw if is_long else (1 - taker_raw) if taker_raw is not None else None
        taker_positive_oi = taker_dir is not None and taker_dir >= 0.53
        vol_ok = (volume_relatif or 0) >= 0.50
        if oi_chg > 0:
            base_pts = _tiered_pts(oi_chg, OI_BONUS_TABLE, [])
            if price_aligned:
                pts = base_pts if (taker_positive_oi or vol_ok) else min(base_pts, 2)
            else:
                pts = -max(2, int(abs(base_pts) / 2)) if base_pts > 0 else 0
        else:
            pts = _tiered_pts(oi_chg, [], OI_MALUS_TABLE)
        raw += pts
        breakdown["oi"] = pts

    # — Taker buy/sell (directionnel : LONG veut buy, SHORT veut sell) —
    tbr = fd.get("taker_buy_ratio")
    if tbr is not None:
        directionnal_tbr = tbr if is_long else (1 - tbr)
        pts = _tiered_pts(directionnal_tbr, TAKER_BONUS_TABLE, TAKER_MALUS_TABLE)
        raw += pts
        breakdown["taker"] = pts

    # FUNDING v6.0.6f : neutre = 0 pt (plus de faux +4 systematique)
    funding_rate = fd.get("funding_rate")
    if funding_rate is not None:
        signed = funding_rate if is_long else -funding_rate
        pts = 0
        if abs(funding_rate) >= FUNDING_EXTREME_SOFT and signed > 0:
            pts = FUNDING_HOSTILE_PTS        # extrême et hostile
        elif signed < -FUNDING_NEUTRAL_BAND:
            pts = FUNDING_FAVORABLE_PTS      # réellement favorable
        # funding neutre = 0 pt
        raw += pts
        breakdown["funding"] = pts

    # — Long/Short ratios —
    ls_pts = 0
    ls_top = fd.get("ls_top_long_ratio")
    if ls_top is not None:
        top_is_long = ls_top > 1.0
        if top_is_long == is_long:
            ls_pts += LS_TOP_ALIGNED_BONUS    # top traders alignés avec nous
    ls_global = fd.get("ls_global_long_ratio")
    if ls_global is not None:
        if is_long and ls_global > LS_CROWD_THRESHOLD_LONG:
            ls_pts += LS_CROWD_MALUS          # foule trop longue
        elif (not is_long) and ls_global < LS_CROWD_THRESHOLD_SHORT:
            ls_pts += LS_CROWD_MALUS          # foule trop courte
    raw += ls_pts
    breakdown["long_short"] = ls_pts

    # — Liquidations forcées — DÉSACTIVÉ v6.0.4b
    # Endpoint Binance /fapi/v1/allForceOrders déprécié ("out of maintenance").
    # Champs conservés en sortie (None) pour compatibilité JSON, mais non scorés.
    # Réactivation prévue via CoinGlass en v6.1.
    # (liq_pts = 0, pas ajouté au raw, pas dans breakdown)

    # Normalisation 0..100
    raw_clamped  = max(FUTURES_RAW_MIN, min(FUTURES_RAW_MAX, raw))
    normalized   = (raw_clamped - FUTURES_RAW_MIN) / (FUTURES_RAW_MAX - FUTURES_RAW_MIN) * 100

    return round(normalized, 1), raw, breakdown


def apply_v6_layer(symbol, direction, technical_score, rr, market_danger_level,
                   quote_volume_24h, result_dict):
    """
    Point d'entrée v6 : orchestre les 3 couches et retourne le score final.

    result_dict : le dict déjà construit par score_symbol (pour récupérer funding_rate).
    Retourne un dict enrichi avec :
      v6_accepted      : bool (False si veto)
      v6_veto_reasons  : list[str]
      v6_score_final   : float (score combiné 70/30) ou None si veto
      v6_score_futures : float ou None
      v6_futures_detail: dict breakdown
      v6_data_errors   : list[str]
    """
    # Gate technique : si score trop faible, pas d'appels API futures (économie réseau),
    # mais v6_accepted=True — le flag naturel est conservé (WATCHLIST si ≥52, REJET si <52).
    # Avant v6.0.5c : v6_accepted=False forçait REJET → trop dur pour les signaux
    # affaiblis par la pénalité volume qui méritent encore une surveillance.
    if technical_score < FUTURES_TECH_SCORE_GATE:
        return {
            "v6_accepted": True,
            "v6_veto_reasons": [],
            "v6_score_final": technical_score,
            "v6_score_futures": None,
            "v6_futures_detail": {},
            "v6_futures_raw": {},
            "v6_data_errors": [f"skipped futures: tech score {technical_score} < gate {FUTURES_TECH_SCORE_GATE}"]
        }

    # Récupération données futures (4 appels réseau : OI, taker, LS global, LS top)
    # Liquidations désactivées : champs conservés à None, sans appel réseau.
    fd = fetch_futures_data_v6(symbol)
    # Injecter funding_rate déjà récupéré dans score_symbol (évite un 5e appel)
    fd["funding_rate"]   = result_dict.get("funding_rate")
    fd["momentum_1h"]    = result_dict.get("momentum_1h", 0)
    volume_relatif_v6    = result_dict.get("volume_relatif")

    # ── COUCHE 1 — VETO (TOUJOURS appliqué, même si futures data incomplète) ──
    # Les vetos RR / BTC danger / liquidité / funding hostile ne dépendent pas
    # des données OI. Le veto OI collapse est lui conditionnel (is not None interne).
    # On lance donc le veto AVANT le contrôle de disponibilité data.
    veto_ok, veto_reasons = check_veto_v6(fd, direction, rr, market_danger_level, quote_volume_24h)
    if not veto_ok:
        return {
            "v6_accepted": False,
            "v6_veto_reasons": veto_reasons,
            "v6_score_final": None,
            "v6_score_futures": None,
            "v6_futures_detail": {},
            "v6_futures_raw": {k: fd.get(k) for k in ["oi_change_pct", "taker_buy_ratio", "taker_sell_ratio", "ls_global_long_ratio", "ls_top_long_ratio", "long_liq_usdt", "short_liq_usdt", "total_liq_usdt", "liq_imbalance", "largest_liq_usdt", "liq_count", "liq_window_min"]},
            "v6_data_errors": fd.get("errors", [])
        }

    # Qualité data : si < 3 familles d'indicateurs sur 5 disponibles
    # (liquidations exclues : endpoint Binance déprécié depuis v6.0.4b)
    available_count = sum([
        fd.get("oi_change_pct")        is not None,
        fd.get("taker_buy_ratio")      is not None,
        fd.get("ls_global_long_ratio") is not None,
        fd.get("ls_top_long_ratio")    is not None,
        fd.get("funding_rate")         is not None,
    ])
    if available_count < 3:
        logger.warning("V6 data insuffisante %s (%d/5 indicateurs) — score technique conservé (veto déjà passé)",
                       symbol, available_count)
        return {
            "v6_accepted": True,
            "v6_veto_reasons": [],
            "v6_score_final": technical_score,   # score inchangé, pas de pénalité
            "v6_score_futures": None,
            "v6_futures_detail": {},
            "v6_futures_raw": {k: fd.get(k) for k in ["oi_change_pct", "taker_buy_ratio", "taker_sell_ratio", "ls_global_long_ratio", "ls_top_long_ratio", "long_liq_usdt", "short_liq_usdt", "total_liq_usdt", "liq_imbalance", "largest_liq_usdt", "liq_count", "liq_window_min"]},
            "v6_data_errors": fd.get("errors", []) + [f"insufficient futures data ({available_count}/5)"]
        }

    # ── COUCHE 3 — Score futures ──────────────────────────────────────────────
    futures_norm, futures_raw, futures_breakdown = compute_futures_score_v6(fd, direction, volume_relatif=volume_relatif_v6)

    # Combinaison 70/30
    score_final = round(V6_WEIGHT_TECH * technical_score + V6_WEIGHT_FUTURES * futures_norm, 1)

    return {
        "v6_accepted": True,
        "v6_veto_reasons": [],
        "v6_score_final": score_final,
        "v6_score_futures": futures_norm,
        "v6_futures_detail": futures_breakdown,
        "v6_futures_raw": {k: fd.get(k) for k in ["oi_change_pct", "taker_buy_ratio", "taker_sell_ratio", "ls_global_long_ratio", "ls_top_long_ratio", "long_liq_usdt", "short_liq_usdt", "total_liq_usdt", "liq_imbalance", "largest_liq_usdt", "liq_count", "liq_window_min"]},
        "v6_data_errors": fd.get("errors", [])
    }



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
    """
    Volume relatif : dernière bougie CLÔTURÉE / moyenne des `period` bougies précédentes.

    v6.3 — Fix bougie incomplète :
    La bougie H1 en cours n'est pas clôturée au moment du scan (run peut tomber
    à n'importe quel moment dans l'heure). Son volume partiel est systématiquement
    plus faible que les bougies closes → relative_vol artificellement très bas
    (0.01x–0.07x observé en prod), ce qui déclenche des rejets en masse.

    Fix : on utilise volumes[-2] (dernière bougie close) comme référence courante,
    et volumes[-period-2:-2] pour la moyenne des `period` bougies closes précédentes.
    Nécessite period + 2 bougies minimum.
    """
    volumes = np.array(volumes, dtype=float)
    if len(volumes) < period + 2:
        return 0
    avg = np.mean(volumes[-period-2:-2])
    if avg == 0:
        return 0
    return round(volumes[-2] / avg, 2)

def compute_btc_micro_context(btc_klines_5m):
    """
    v6.4.2.2 — Contexte BTC micro en 5m pour logging uniquement.

    Important : cette fonction NE pilote PAS le market_regime ni le scoring.
    Elle sert à alimenter WATCHLIST_LOG pour analyser les squeezes BTC
    au moment T du signal.
    """
    empty = {
        "btc_variation_15m": 0,
        "btc_variation_30m": 0,
    }
    try:
        if not btc_klines_5m or len(btc_klines_5m) < 7:
            return empty
        closes = [float(k[4]) for k in btc_klines_5m]
        # Sur intervalle 5m : 15m = 3 intervalles, 30m = 6 intervalles.
        variation_15m = ((closes[-1] - closes[-4]) / closes[-4] * 100) if len(closes) >= 4 and closes[-4] > 0 else 0
        variation_30m = ((closes[-1] - closes[-7]) / closes[-7] * 100) if len(closes) >= 7 and closes[-7] > 0 else 0
        return {
            "btc_variation_15m": round(variation_15m, 2),
            "btc_variation_30m": round(variation_30m, 2),
        }
    except Exception as e:
        logger.warning("compute_btc_micro_context échec: %s", e)
        return empty




def compute_btc_market_state_details(market_details):
    """
    v6.4.4-final-clean — BTC context layer auditable.

    Retourne (state, reason). Les seuils sont externalisés en constantes/env vars
    pour recalibrage sans modifier la logique métier.
    """
    try:
        regime = str(
            market_details.get("market_regime_btc") or
            market_details.get("market_regime") or
            "unknown"
        ).lower()
        rsi     = float(market_details.get("btc_rsi") or 50)
        var_30m = float(market_details.get("btc_variation_30m") or 0)
        var_2h  = float(market_details.get("btc_variation_2h") or 0)
        var_4h  = float(market_details.get("btc_variation_4h") or 0)
    except Exception as exc:
        return "BTC_NEUTRAL_COMPRESS", f"fallback neutral: invalid BTC context ({exc})"

    if regime in ("bearish", "neutral"):
        remontee_courte = var_30m > BTC_SWITCH_VAR30M_MIN and var_2h > BTC_SWITCH_VAR2H_MIN
        remontee_4h = var_4h > BTC_SWITCH_VAR4H_MIN
        rsi_non_bearish = rsi > BTC_SWITCH_RSI_MIN
        bearish_precedent = var_4h < BTC_SWITCH_PREV_VAR4H_MAX

        if remontee_4h and remontee_courte and rsi_non_bearish and (regime == "bearish" or bearish_precedent):
            return (
                "BTC_SWITCH_RISK",
                f"{regime}: var4h={var_4h:+.2f}>{BTC_SWITCH_VAR4H_MIN}, "
                f"var2h={var_2h:+.2f}>{BTC_SWITCH_VAR2H_MIN}, "
                f"var30m={var_30m:+.2f}>{BTC_SWITCH_VAR30M_MIN}, rsi={rsi:.1f}>{BTC_SWITCH_RSI_MIN}"
            )

        if regime == "bearish" and var_4h > BTC_BEAR_EXHAUSTION_VAR4H_MIN and var_2h > var_4h * 0.5:
            return (
                "BTC_BEAR_EXHAUSTION",
                f"bearish exhaustion: var4h={var_4h:+.2f}>{BTC_BEAR_EXHAUSTION_VAR4H_MIN}, "
                f"var2h={var_2h:+.2f} > 0.5*var4h"
            )

    if var_4h > BTC_BULL_IMPULSE_VAR4H and rsi > BTC_BULL_IMPULSE_RSI and var_2h > BTC_BULL_IMPULSE_VAR2H:
        return (
            "BTC_BULL_IMPULSE",
            f"bull impulse: var4h={var_4h:+.2f}>{BTC_BULL_IMPULSE_VAR4H}, "
            f"var2h={var_2h:+.2f}>{BTC_BULL_IMPULSE_VAR2H}, rsi={rsi:.1f}>{BTC_BULL_IMPULSE_RSI}"
        )

    if regime == "bullish" or (var_4h > BTC_BULL_SOFT_VAR4H and rsi > BTC_BULL_SOFT_RSI):
        return (
            "BTC_BULL_SOFT",
            f"bull soft: regime={regime} or var4h={var_4h:+.2f}>{BTC_BULL_SOFT_VAR4H} and rsi={rsi:.1f}>{BTC_BULL_SOFT_RSI}"
        )

    if regime == "bearish" and var_4h < BTC_BEAR_CONT_VAR4H_MAX and var_2h < BTC_BEAR_CONT_VAR2H_MAX and var_30m < 0:
        return (
            "BTC_BEAR_CONTINUATION",
            f"bear continuation: var4h={var_4h:+.2f}<{BTC_BEAR_CONT_VAR4H_MAX}, "
            f"var2h={var_2h:+.2f}<{BTC_BEAR_CONT_VAR2H_MAX}, var30m={var_30m:+.2f}<0"
        )

    return "BTC_NEUTRAL_COMPRESS", f"neutral compress: regime={regime}, var4h={var_4h:+.2f}, var2h={var_2h:+.2f}, var30m={var_30m:+.2f}, rsi={rsi:.1f}"


def compute_btc_market_state(market_details):
    """Compatibilité : retourne uniquement l'état BTC."""
    state, _reason = compute_btc_market_state_details(market_details)
    return state



def _find_btc_structural_pivot(highs, lows, closes, pivot_type, lookback=36, left=3, right=3, fallback_window=6):
    """
    v6.5.0.1 — pivot BTC structurel sans appel API supplémentaire.

    Utilise uniquement le tableau btc_klines_1h déjà chargé par /full_analysis.
    - Cherche d'abord le dernier swing high / swing low significatif sur les N dernières bougies 1h.
    - Validation locale : 3 bougies à gauche et 3 bougies à droite.
    - Fallback : rolling max/min 6 bougies si aucun pivot propre n'est trouvé.

    Retourne un dict exploitable pour btc_impulse_age :
    - pivot_type
    - pivot_age : nombre de bougies 1h depuis le pivot
    - pivot_price
    - pivot_distance_pct : distance du prix actuel au pivot
    - pivot_method : swing_3_3 ou fallback_rolling_6
    """
    out = {
        "pivot_type": pivot_type,
        "pivot_age": 0,
        "pivot_price": None,
        "pivot_distance_pct": None,
        "pivot_method": "unavailable",
    }
    try:
        if not closes or len(closes) < max(left + right + 2, fallback_window + 1):
            return out

        n = len(closes)
        current = float(closes[-1])
        if current <= 0:
            return out

        pivot_type = "swing_low" if str(pivot_type).lower() == "swing_low" else "swing_high"
        scan_start = max(left, n - int(lookback))
        scan_end_exclusive = n - right
        pivot_idx = None
        pivot_price = None
        pivot_method = "swing_3_3"

        # Dernier pivot validé localement. On scanne du plus récent au plus ancien.
        for i in range(scan_end_exclusive - 1, scan_start - 1, -1):
            h_window = highs[i - left:i + right + 1]
            l_window = lows[i - left:i + right + 1]
            if pivot_type == "swing_high":
                candidate = float(highs[i])
                if candidate >= max(h_window):
                    pivot_idx = i
                    pivot_price = candidate
                    break
            else:
                candidate = float(lows[i])
                if candidate <= min(l_window):
                    pivot_idx = i
                    pivot_price = candidate
                    break

        # Fallback rolling 6 : garantit une valeur même en tendance très lisse.
        if pivot_idx is None:
            fw = min(int(fallback_window), n)
            offset = n - fw
            if pivot_type == "swing_high":
                local = highs[-fw:]
                local_idx = max(range(len(local)), key=lambda j: float(local[j]))
                pivot_idx = offset + local_idx
                pivot_price = float(highs[pivot_idx])
            else:
                local = lows[-fw:]
                local_idx = min(range(len(local)), key=lambda j: float(local[j]))
                pivot_idx = offset + local_idx
                pivot_price = float(lows[pivot_idx])
            pivot_method = "fallback_rolling_6"

        age = max(0, n - 1 - int(pivot_idx))
        if pivot_type == "swing_high":
            distance_pct = round((float(pivot_price) - current) / current * 100, 2)
        else:
            distance_pct = round((current - float(pivot_price)) / current * 100, 2)

        out.update({
            "pivot_type": pivot_type,
            "pivot_age": int(age),
            "pivot_price": round(float(pivot_price), 8),
            "pivot_distance_pct": distance_pct,
            "pivot_method": pivot_method,
        })
        return out
    except Exception as exc:
        logger.warning("_find_btc_structural_pivot échec: %s", exc)
        return out


def compute_btc_context_v65(btc_klines_1h, market_details):
    """
    v6.5.0.1 — Market Context Engine (instrumentation only).

    Enrichit le contexte BTC sans modifier directement les décisions Telegram.
    Utilise les klines BTC 1h déjà chargées dans /full_analysis : zéro appel API supplémentaire.

    Refinement v6.5.0.1 :
    - btc_impulse_age ne mesure plus des bougies consécutives dans une direction.
    - Il mesure l'âge du dernier pivot structurel 1h compatible avec le biais BTC.
    - Pivot significatif : swing high / swing low sur 36 bougies 1h, validation 3/3.
    - Fallback : rolling max/min 6 bougies si aucun pivot propre n'est trouvé.
    """
    market_details = market_details or {}
    out = {
        "btc_context_bias": "uncertain",
        "btc_phase": "BTC_UNCLEAR",
        "btc_trend_slope_2h": market_details.get("btc_variation_2h", 0),
        "btc_trend_slope_4h": market_details.get("btc_variation_4h", 0),
        "btc_trend_slope_12h": market_details.get("btc_variation_12h", 0),
        "btc_impulse_age": 0,
        "btc_last_pivot_type": None,
        "btc_last_pivot_age": 0,
        "btc_last_pivot_distance_pct": None,
        "btc_last_pivot_method": "unavailable",
        "btc_pullback_depth": 0,
        "btc_range_position": 0.5,
        "btc_rejection_state": "none",
        "btc_support_distance_pct": None,
        "btc_resistance_distance_pct": None,
        "btc_volatility_regime": "unknown",
        "btc_context_score": 0,
    }
    try:
        if not btc_klines_1h or len(btc_klines_1h) < 30:
            out["btc_phase"] = "BTC_UNCLEAR"
            return out

        closes = [float(k[4]) for k in btc_klines_1h]
        highs = [float(k[2]) for k in btc_klines_1h]
        lows = [float(k[3]) for k in btc_klines_1h]
        current = closes[-1]
        if current <= 0:
            out["btc_phase"] = "BTC_UNCLEAR"
            return out

        var_30m = _safe_float(market_details.get("btc_variation_30m"), 0.0)
        var_2h = _safe_float(market_details.get("btc_variation_2h"), 0.0)
        var_4h = _safe_float(market_details.get("btc_variation_4h"), 0.0)
        var_12h = _safe_float(market_details.get("btc_variation_12h"), 0.0)
        atr_pct = _safe_float(market_details.get("btc_atr_pct"), 0.0)
        base_state = str(market_details.get("btc_market_state", "BTC_NEUTRAL_COMPRESS"))

        lookback = min(len(closes), 72)
        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])
        range_span = recent_high - recent_low
        range_pos = round((current - recent_low) / range_span, 3) if range_span > 0 else 0.5

        support_distance = round((current - recent_low) / current * 100, 2) if current > 0 else None
        resistance_distance = round((recent_high - current) / current * 100, 2) if current > 0 else None

        # Pullback depth structurel sur les klines 1h déjà chargées.
        if var_4h >= 0:
            pullback_depth = round((recent_high - current) / recent_high * 100, 2) if recent_high > 0 else 0
        else:
            pullback_depth = round((current - recent_low) / recent_low * 100, 2) if recent_low > 0 else 0

        rejection_state = "none"
        for k in btc_klines_1h[-3:]:
            o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
            rng = max(h - l, 1e-12)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            if upper_wick / rng >= 0.45 and c < o:
                rejection_state = "upper_rejection"
            if lower_wick / rng >= 0.45 and c > o and rejection_state == "none":
                rejection_state = "lower_rejection"

        if atr_pct >= 2.5:
            vol_regime = "high"
        elif atr_pct >= 1.2:
            vol_regime = "moderate"
        else:
            vol_regime = "low"

        phase = base_state or "BTC_UNCLEAR"
        bias = "neutral"
        score = 0

        if base_state in ("BTC_BULL_IMPULSE", "BTC_BULL_SOFT"):
            bias = "bullish"
            score += 2
            if pullback_depth >= 1.0 and var_30m < 0:
                phase = "BTC_BULL_PULLBACK"
            elif range_pos >= 0.80 and rejection_state == "upper_rejection":
                phase = "BTC_BULL_EXHAUSTION"
            elif base_state == "BTC_BULL_IMPULSE":
                phase = "BTC_BULL_IMPULSE"
            else:
                phase = "BTC_BULL_SOFT"
        elif (
            base_state == "BTC_BEAR_CONTINUATION"
            and var_4h < 0
            and var_2h > 0
            and var_30m > 0
        ):
            # BTC_BEAR_PULLBACK : bearish de fond mais rebond court terme en cours
            # → danger pour SHORT immédiat, attendre rejet.
            bias = "transition"
            score -= 1
            phase = "BTC_BEAR_PULLBACK"
        elif base_state == "BTC_BEAR_CONTINUATION":
            bias = "bearish"
            score -= 2
            phase = "BTC_BEAR_CONTINUATION"
        elif base_state == "BTC_BEAR_EXHAUSTION":
            bias = "transition"
            score -= 1
            phase = "BTC_BEAR_EXHAUSTION"
        elif base_state == "BTC_SWITCH_RISK":
            bias = "transition"
            phase = "BTC_SWITCH_RISK"
        else:
            if var_12h < -1.0 or (var_4h < -0.3 and var_2h <= 0):
                bias = "bearish"
                phase = "BTC_NEUTRAL_AFTER_BEAR"
                score -= 1
            elif var_12h > 1.0 or (var_4h > 0.3 and var_2h >= 0):
                bias = "bullish"
                phase = "BTC_NEUTRAL_AFTER_BULL"
                score += 1
            elif abs(var_4h) < 0.35 and abs(var_2h) < 0.25 and vol_regime == "low":
                bias = "neutral"
                phase = "BTC_NEUTRAL_ACCUMULATION"
            else:
                bias = "neutral"
                phase = "BTC_RANGE_CHOP"

        if not phase:
            phase = "BTC_UNCLEAR"

        if rejection_state == "upper_rejection":
            score -= 1
            if phase in ("BTC_BULL_SOFT", "BTC_NEUTRAL_AFTER_BULL"):
                phase = "BTC_BULL_EXHAUSTION"
        elif rejection_state == "lower_rejection":
            score += 1
            if phase == "BTC_BEAR_CONTINUATION":
                phase = "BTC_BEAR_EXHAUSTION"

        # Pivot à utiliser pour l'âge d'impulsion :
        # - contextes bull / after bull : dernier swing high
        # - contextes bear / after bear : dernier swing low
        # - transition / range : choix selon pente 12h/4h, fallback low si structure bearish.
        if phase in ("BTC_BULL_IMPULSE", "BTC_BULL_PULLBACK", "BTC_BULL_EXHAUSTION", "BTC_BULL_SOFT", "BTC_NEUTRAL_AFTER_BULL"):
            pivot_type = "swing_high"
        elif phase in ("BTC_BEAR_CONTINUATION", "BTC_BEAR_PULLBACK", "BTC_BEAR_EXHAUSTION", "BTC_NEUTRAL_AFTER_BEAR"):
            pivot_type = "swing_low"
        else:
            pivot_type = "swing_high" if (var_12h >= 0 or var_4h >= 0) else "swing_low"

        pivot = _find_btc_structural_pivot(
            highs=highs,
            lows=lows,
            closes=closes,
            pivot_type=pivot_type,
            lookback=36,
            left=3,
            right=3,
            fallback_window=6,
        )
        impulse_age = int(pivot.get("pivot_age") or 0)

        out.update({
            "btc_context_bias": bias,
            "btc_phase": phase,
            "btc_trend_slope_2h": round(var_2h, 2),
            "btc_trend_slope_4h": round(var_4h, 2),
            "btc_trend_slope_12h": round(var_12h, 2),
            "btc_impulse_age": impulse_age,
            "btc_last_pivot_type": pivot.get("pivot_type"),
            "btc_last_pivot_age": impulse_age,
            "btc_last_pivot_distance_pct": pivot.get("pivot_distance_pct"),
            "btc_last_pivot_method": pivot.get("pivot_method"),
            "btc_pullback_depth": pullback_depth,
            "btc_range_position": range_pos,
            "btc_rejection_state": rejection_state,
            "btc_support_distance_pct": support_distance,
            "btc_resistance_distance_pct": resistance_distance,
            "btc_volatility_regime": vol_regime,
            "btc_context_score": score,
        })
        return out
    except Exception as exc:
        logger.warning("compute_btc_context_v65 échec: %s", exc)
        out["btc_phase"] = "BTC_UNCLEAR"
        return out

def compute_rr_levels(direction, entry_avg, stop_loss, tp_values):
    """
    v6.5.0 — RR Theoretical Engine.
    Calcule le RR théorique de chaque TP. Instrumentation uniquement.
    """
    keys = ["rr_tp1", "rr_tp2", "rr_tp3", "rr_tp4", "rr_tp5"]
    out = {k: None for k in keys}
    try:
        direction = str(direction or "").upper()
        entry = float(entry_avg)
        sl = float(stop_loss)
        risk = abs(entry - sl)
        if direction not in ("LONG", "SHORT") or risk <= 0:
            return out
        for idx, tp in enumerate(tp_values, start=1):
            if tp is None or tp == "":
                continue
            tp_f = float(tp)
            rr = (tp_f - entry) / risk if direction == "LONG" else (entry - tp_f) / risk
            out[f"rr_tp{idx}"] = round(rr, 2)
        return out
    except Exception:
        return out


def classify_setup_v65(direction, entry_type, trend_strength, late_entry_risk, late_entry_level,
                       position_range, relative_vol, momentum_1h, momentum_3h,
                       btc_phase, btc_context_bias, distance_ema21, rsi):
    """
    v6.5.0 — Setup Classification Engine.
    Classification plus financière du setup, sans changer la décision.
    """
    direction = str(direction or "NEUTRAL").upper()
    entry_type = str(entry_type or "NEUTRAL").upper()
    trend_strength = str(trend_strength or "weak").lower()
    btc_context_bias = str(btc_context_bias or "uncertain")
    pr = _safe_float(position_range, 0.5)
    vol = _safe_float(relative_vol, 0.0)
    mom1 = _safe_float(momentum_1h, 0.0)
    mom3 = _safe_float(momentum_3h, 0.0)
    late = _safe_float(late_entry_risk, 0.0)
    rsi_v = _safe_float(rsi, 50.0)

    setup_family = "NEUTRAL_NO_TRADE"
    if direction == "LONG":
        if entry_type == "EARLY":
            setup_family = "LONG_EARLY_REVERSAL" if btc_context_bias in ("bearish", "transition") else "LONG_EARLY_CONTINUATION"
        elif entry_type == "BREAKOUT":
            setup_family = "LONG_BREAKOUT"
        elif entry_type == "MOMENTUM":
            setup_family = "LONG_LATE_MOMENTUM" if (late >= 55 or pr >= 0.80 or rsi_v >= 70) else "LONG_MOMENTUM_CONTINUATION"
    elif direction == "SHORT":
        if entry_type == "EARLY":
            setup_family = "SHORT_EARLY_BREAKDOWN" if btc_context_bias in ("bearish", "neutral") else "SHORT_EARLY_REVERSAL"
        elif entry_type == "BREAKOUT":
            setup_family = "SHORT_BREAKDOWN"
        elif entry_type == "MOMENTUM":
            setup_family = "SHORT_LATE_DUMP" if (late >= 55 or pr <= 0.15 or rsi_v <= 30) else "SHORT_MOMENTUM_CONTINUATION"

    if late_entry_level == "HIGH" or late >= 70:
        setup_maturity = "exhausted"
    elif late >= 55:
        setup_maturity = "late"
    elif late >= 35:
        setup_maturity = "mature"
    elif entry_type == "EARLY":
        setup_maturity = "early"
    else:
        setup_maturity = "healthy"

    if direction == "LONG":
        alignment = "aligned" if btc_context_bias == "bullish" else ("countertrend" if btc_context_bias == "bearish" else ("transition" if btc_context_bias == "transition" else "unclear"))
    elif direction == "SHORT":
        alignment = "aligned" if btc_context_bias == "bearish" else ("countertrend" if btc_context_bias == "bullish" else ("transition" if btc_context_bias == "transition" else "unclear"))
    else:
        alignment = "unclear"

    if vol < 0.30:
        setup_variant = "thin_participation"
    elif vol >= 1.50 and setup_maturity in ("late", "exhausted"):
        setup_variant = "late_volume_expansion"
    elif "BREAK" in setup_family and vol >= 1.20:
        setup_variant = "volume_confirmed_break"
    elif entry_type == "EARLY" and vol < 0.80:
        setup_variant = "quiet_early"
    elif abs(mom1) > 2 or abs(mom3) > 5:
        setup_variant = "fast_move"
    else:
        setup_variant = "standard"

    late_label = "HIGH" if late >= 70 else ("MEDIUM" if late >= 40 else "LOW")

    return {
        "setup_family": setup_family,
        "setup_variant": setup_variant,
        "setup_maturity": setup_maturity,
        "setup_directional_quality": trend_strength,
        "setup_context_alignment": alignment,
        "setup_late_risk_label": late_label,
    }


def classify_participation_v65(entry_type, direction, relative_vol, position_range, setup_maturity,
                               taker_pts, oi_pts, funding_pts, long_short_pts,
                               futures_zone, funding_signal, derivatives_bias):
    """
    v6.5.0 — Participation & Derivatives Engine.
    Qualifie la participation au lieu de lire volume fort/faible de façon plate.
    """
    entry_type = str(entry_type or "").upper()
    direction = str(direction or "").upper()
    vol = _safe_float(relative_vol, 0.0)
    taker_pts = int(taker_pts or 0)
    oi_pts = int(oi_pts or 0)
    funding_pts = int(funding_pts or 0)
    long_short_pts = int(long_short_pts or 0)
    futures_zone = str(futures_zone or "unavailable")

    if vol < 0.30:
        volume_regime = "very_low"
    elif vol < 0.50:
        volume_regime = "low"
    elif vol < 0.80:
        volume_regime = "moderate"
    elif vol < 1.20:
        volume_regime = "healthy"
    elif vol < 1.80:
        volume_regime = "high"
    else:
        volume_regime = "excessive"

    if entry_type == "EARLY" and vol < 0.80:
        volume_quality = "constructive"
        volume_context = "quiet early participation"
    elif entry_type == "BREAKOUT" and vol >= 1.20:
        volume_quality = "constructive"
        volume_context = "breakout volume confirmation"
    elif entry_type == "MOMENTUM" and vol < 0.50:
        volume_quality = "weak"
        volume_context = "momentum without enough volume"
    elif setup_maturity in ("late", "exhausted") and vol >= 1.20:
        volume_quality = "late"
        volume_context = "late volume expansion"
    elif vol >= 1.80:
        volume_quality = "danger"
        volume_context = "excessive volume / possible crowding"
    else:
        volume_quality = "neutral"
        volume_context = "volume neutral"

    if taker_pts > 0 and oi_pts >= 0 and long_short_pts >= 0:
        derivatives_alignment = "confirmed"
    elif taker_pts > 0 or oi_pts > 0 or long_short_pts > 0 or funding_pts > 0:
        derivatives_alignment = "partially_confirmed"
    elif taker_pts < 0 or oi_pts < 0 or long_short_pts < 0:
        derivatives_alignment = "opposed"
    elif futures_zone == "unavailable":
        derivatives_alignment = "unavailable"
    else:
        derivatives_alignment = "not_confirmed"

    crowding_score = 0
    if funding_signal in ("longs crowded", "shorts crowded"):
        crowded_with_trade = (direction == "LONG" and funding_signal == "longs crowded") or (direction == "SHORT" and funding_signal == "shorts crowded")
        crowding_score += 2 if crowded_with_trade else 1
    if long_short_pts < 0:
        crowding_score += 1
    if futures_zone == "overheated":
        crowding_score += 2
    if volume_quality in ("late", "danger"):
        crowding_score += 1

    if crowding_score >= 4:
        crowding_state = "extreme_crowding"
    elif crowding_score >= 2:
        crowding_state = "crowded"
    elif crowding_score == 1:
        crowding_state = "mild_crowding"
    else:
        crowding_state = "not_crowded"

    participation_score = 0
    participation_score += {"constructive": 2, "neutral": 0, "weak": -1, "late": -2, "danger": -3}.get(volume_quality, 0)
    participation_score += {"confirmed": 2, "partially_confirmed": 1, "not_confirmed": 0, "unavailable": -1, "opposed": -2}.get(derivatives_alignment, 0)
    if crowding_state in ("crowded", "extreme_crowding"):
        participation_score -= 1

    warnings = []
    if volume_quality in ("weak", "late", "danger"):
        warnings.append(volume_context)
    if derivatives_alignment in ("opposed", "unavailable"):
        warnings.append(f"derivatives {derivatives_alignment}")
    if crowding_state in ("crowded", "extreme_crowding"):
        warnings.append(crowding_state)

    return {
        "volume_regime": volume_regime,
        "volume_quality": volume_quality,
        "volume_context": volume_context,
        "volume_vs_move": f"{entry_type.lower()}:{volume_quality}",
        "oi_regime": "positive" if oi_pts > 0 else ("negative" if oi_pts < 0 else "neutral"),
        "taker_regime": "positive" if taker_pts > 0 else ("negative" if taker_pts < 0 else "neutral"),
        "funding_regime": str(funding_signal or "neutral"),
        "crowding_state": crowding_state,
        "derivatives_alignment": derivatives_alignment,
        "participation_score": participation_score,
        "participation_warning": " | ".join(warnings),
    }


def detect_market_regime(btc_klines, return_details=False):
    """
    Régime BTC enrichi V4.7/V5.
    Retour simple par défaut pour compatibilité.
    Si return_details=True, retourne (regime, details).

    v6.4.2.2 — IMPORTANT : cette fonction reste basée sur BTC 1h, comme en v6.4.1,
    pour ne pas modifier le scoring. Les variations 15m/30m sont ajoutées ensuite
    via compute_btc_micro_context() et ne servent qu'au logging WATCHLIST_LOG.
    """
    empty = {
        "market_danger_score": 50,
        "market_danger_level": "UNKNOWN",
        "btc_rsi": 50,
        "btc_atr_pct": 0,
        "btc_variation_15m": 0,
        "btc_variation_30m": 0,
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
    # Formules historiques v6.4.1 sur bougies 1h.
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
        "btc_variation_15m": 0,
        "btc_variation_30m": 0,
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
    if symbol in EXCLUDED_NON_CRYPTO_SYMBOLS:
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

def cap_signal_count(candidates, market_regime, data_source_run):
    """
    Plafonne le nombre de signaux envoyés à GPT selon le régime et la source.
    Python reste maître de la décision : GPT ne fait que recopier ce qui reste.
    L'ordre (tri par score décroissant) est déjà appliqué en amont.

    v6.5.3 :
    - ouverture beta d'un seul bucket SHORT vers Telegram ;
    - max 1 SHORT_MOMENTUM_CONTINUATION_PREMIUM par run.
    """
    if not candidates:
        return candidates

    cap = MAX_SIGNALS_DEFAULT
    if market_regime == "neutral":
        cap = min(cap, MAX_SIGNALS_NEUTRAL)
    if market_regime == "volatile":
        cap = min(cap, MAX_SIGNALS_VOLATILE)
    if data_source_run == "SPOT_FALLBACK":
        cap = min(cap, MAX_SIGNALS_SPOT_FALLBACK)
    # Un setup en tendance faible n'est jamais accompagné d'un second signal.
    if candidates[0].get("trend_strength") == "weak":
        cap = 1
    cap = min(cap, MAX_SIGNALS_ABSOLUTE)

    selected = []
    short_beta_added = False

    for r in candidates:
        bucket = r.get("signal_quality_bucket")
        if bucket == "SHORT_MOMENTUM_CONTINUATION_PREMIUM":
            if short_beta_added:
                continue
            short_beta_added = True
        selected.append(r)
        if len(selected) >= cap:
            break

    return selected

def build_weakness_notes(r):
    """
    Faiblesses pré-rédigées par Python à partir des champs déjà calculés.
    GPT recopie ces notes dans la section Raison au lieu de décider lui-même
    quelle faiblesse signaler. Phrases neutres, sans mot valorisant ni underscore.
    """
    notes = []
    sq = r.get("source_quality", "")
    if "SPOT FALLBACK" in sq:
        notes.append("données spot, dérivés non vérifiés, fiabilité réduite pour un trade à levier")
    elif "BYBIT FUTURES" in sq:
        notes.append("source futures alternative (Bybit)")

    if r.get("funding_signal") == "unavailable":
        notes.append("funding indisponible")
    if r.get("derivatives_bias") == "caution":
        notes.append("le funding appelle à la prudence")

    ts = r.get("trend_strength")
    if ts == "weak":
        notes.append("tendance faible, à gérer rapidement")
    elif ts == "moderate":
        notes.append("tendance modérée")

    if r.get("entry_type") == "MOMENTUM" and (r.get("late_entry_risk") or 0) >= 40:
        notes.append("mouvement déjà entamé, risque d'entrée tardive")

    mr, d = r.get("market_regime"), r.get("direction")
    if (mr == "bearish" and d == "LONG") or (mr == "bullish" and d == "SHORT"):
        notes.append("setup contre-régime BTC, levier réduit par prudence")

    if r.get("flag") == "WATCHLIST":
        notes.append("signal de calibration, prudence")

    return " ; ".join(notes) if notes else "aucune faiblesse majeure détectée par le moteur"
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


def apply_contextual_v652_rules(
    *,
    flag,
    confidence,
    max_leverage,
    direction,
    entry_type,
    global_score,
    trend_strength,
    market_regime,
    btc_market_state,
    market_details,
    relative_vol,
    position_range,
    market_danger_level,
    late_entry_risk,
    late_entry_level,
    rr_valid,
    data_source,
    hard_reject,
    setup_family,
    setup_maturity,
    futures_zone,
    futures_overheated,
    v6_score_futures,
    crowding_state,
    derivatives_alignment,
):
    """
    v6.5.3 — Préqualification contextuelle, appelée AVANT le bucket engine.

    Rôle :
    - ne remplace pas le score brut ;
    - ne remplace pas apply_contextual_bucket_engine ;
    - prépare des overrides contextuels que le bucket engine applique après ses règles v6.4.4 ;
    - permet d'ouvrir UN bucket SHORT très spécifique en beta Telegram :
      SHORT_MOMENTUM_CONTINUATION_PREMIUM.

    Invariant projet :
    Un setup n'est pas bon/mauvais de façon universelle.
    La décision dépend du contexte BTC, de la maturité, du volume, du PR et des dérivés.
    """
    btc_phase = str(market_details.get("btc_phase") or btc_market_state or "BTC_UNCLEAR")
    btc_var_30m = _safe_float(market_details.get("btc_variation_30m"), 0.0)
    vol = _safe_float(relative_vol, 0.0)
    pr = _safe_float(position_range, 0.5)
    late = _safe_float(late_entry_risk, 0.0)
    setup_family = str(setup_family or "")
    setup_maturity = str(setup_maturity or "")
    crowding_state = str(crowding_state or "unknown")
    derivatives_alignment = str(derivatives_alignment or "unknown")

    actions = []
    notes = []
    forced_bucket = None
    force_flag = None
    force_confidence_cap = None
    force_leverage_cap = None
    risk_guard = None
    decision = None

    # P1 — Ouverture beta Signals : uniquement ce short momentum précis.
    short_momentum_beta_ok = (
        ENABLE_SHORT_MOMENTUM_CONTINUATION_BETA
        and setup_family == "SHORT_MOMENTUM_CONTINUATION"
        and direction == "SHORT"
        and entry_type == "MOMENTUM"
        and btc_phase in ("BTC_NEUTRAL_AFTER_BEAR", "BTC_RANGE_CHOP", "BTC_BEAR_CONTINUATION")
        and btc_var_30m < 0
        and late < 35
        and pr < 0.35
        and vol < 0.80
        and market_danger_level != "HIGH"
        and crowding_state not in ("crowded", "extreme_crowding")
        and setup_maturity not in ("late", "exhausted")
        and data_source != "SPOT_FALLBACK"
        and rr_valid
        and not hard_reject
    )
    if short_momentum_beta_ok:
        actions.append("promote_short_momentum_continuation_beta")
        forced_bucket = "SHORT_MOMENTUM_CONTINUATION_PREMIUM"
        force_flag = "CANDIDAT"
        force_confidence_cap = SHORT_MOMENTUM_BETA_CONF_CAP
        force_leverage_cap = SHORT_MOMENTUM_BETA_LEVERAGE_CAP
        risk_guard = "short momentum continuation premium beta"
        decision = (
            "CANDIDAT v6.5.3 beta : SHORT_MOMENTUM_CONTINUATION_PREMIUM "
            f"validé par {btc_phase}, btc30m={btc_var_30m:+.2f}%, "
            f"late={late:.1f}, PR={pr:.3f}, volume={vol:.2f}x, crowding={crowding_state}."
        )
        notes.append("SHORT momentum continuation premium beta ouvert vers Signals")

    # P2 — LONG_LATE_MOMENTUM : downgrade par défaut, exception bull pullback très propre.
    long_late_exception = (
        setup_family == "LONG_LATE_MOMENTUM"
        and btc_phase == "BTC_BULL_PULLBACK"
        and late < 25
        and pr < 0.65
        and market_danger_level != "HIGH"
        and rr_valid
        and crowding_state not in ("crowded", "extreme_crowding")
    )
    if setup_family == "LONG_LATE_MOMENTUM" and not long_late_exception:
        actions.append("downgrade_long_late_momentum")
        forced_bucket = forced_bucket or "WATCHLIST_LONG_LATE_MOMENTUM"
        force_flag = "WATCHLIST" if force_flag is None else force_flag
        force_confidence_cap = min(force_confidence_cap or 60, 60)
        force_leverage_cap = min(force_leverage_cap or 3, 3)
        risk_guard = risk_guard or "long late momentum"
        decision = decision or (
            f"WATCHLIST v6.5.3 : LONG_LATE_MOMENTUM downgradé "
            f"(btc_phase={btc_phase}, late={late:.1f}, PR={pr:.3f}, volume={vol:.2f}x)."
        )
        notes.append("LONG late momentum downgradé sauf bull pullback très propre")

    # P3 — SHORT_EARLY contextualisé.
    is_short_early = (
        direction == "SHORT"
        and (entry_type == "EARLY" or setup_family.startswith("SHORT_EARLY"))
    )
    if is_short_early:
        if btc_phase in ("BTC_BULL_SOFT", "BTC_BULL_IMPULSE", "BTC_SWITCH_RISK"):
            actions.append("reject_short_early_bull_context")
            forced_bucket = "REJECT_SHORT_EARLY_BULL_CONTEXT"
            force_flag = "REJET"
            force_confidence_cap = min(force_confidence_cap or 55, 55)
            force_leverage_cap = min(force_leverage_cap or 3, 3)
            risk_guard = "short early mauvais contexte BTC"
            decision = (
                f"REJET v6.5.3 : SHORT_EARLY en {btc_phase}, "
                "contexte BTC défavorable au short anticipé."
            )
            notes.append("SHORT_EARLY rejeté en contexte bull/switch")
        elif btc_phase in ("BTC_BEAR_CONTINUATION", "BTC_NEUTRAL_AFTER_BEAR") and btc_var_30m < 0:
            actions.append("short_early_bear_context_diagnostic")
            forced_bucket = forced_bucket or "SHORT_EARLY_BEAR_CONTEXT_DIAGNOSTIC"
            force_flag = "WATCHLIST" if force_flag is None else force_flag
            force_confidence_cap = min(force_confidence_cap or 60, 60)
            force_leverage_cap = min(force_leverage_cap or 3, 3)
            risk_guard = risk_guard or "short early diagnostic uniquement"
            decision = decision or (
                f"WATCHLIST_DIAG v6.5.3 : SHORT_EARLY en {btc_phase} avec btc30m négatif. "
                "Diagnostic uniquement, pas premium en v6.5.3."
            )
            notes.append("SHORT_EARLY bear context conservé en diagnostic seulement")
        else:
            actions.append("short_early_no_premium")
            forced_bucket = forced_bucket or "SHORT_PREMIUM_CANDIDATE_DISABLED"
            force_flag = "WATCHLIST" if force_flag is None else force_flag
            force_confidence_cap = min(force_confidence_cap or 60, 60)
            force_leverage_cap = min(force_leverage_cap or 3, 3)
            risk_guard = risk_guard or "short early non premium"
            decision = decision or (
                f"WATCHLIST v6.5.3 : SHORT_EARLY non premium "
                f"(btc_phase={btc_phase}, btc30m={btc_var_30m:+.2f}%)."
            )
            notes.append("SHORT_PREMIUM_CANDIDATE générique désactivé")

    # P4 — LONG strong premium durci.
    long_strong_valid = (
        direction == "LONG"
        and trend_strength == "strong"
        and setup_maturity not in ("late", "exhausted")
        and late < 35
        and pr < 0.80
        and setup_family != "LONG_LATE_MOMENTUM"
        and global_score >= 54
    )
    if direction == "LONG" and trend_strength == "strong" and not long_strong_valid:
        actions.append("downgrade_long_strong_diagnostic")
        forced_bucket = forced_bucket or "WATCHLIST_LONG_STRONG_DIAGNOSTIC"
        force_flag = "WATCHLIST" if force_flag is None else force_flag
        force_confidence_cap = min(force_confidence_cap or 60, 60)
        force_leverage_cap = min(force_leverage_cap or 3, 3)
        risk_guard = risk_guard or "long strong conditions premium insuffisantes"
        decision = decision or (
            f"WATCHLIST v6.5.3 : LONG strong non premium "
            f"(setup={setup_family}, maturity={setup_maturity}, score={global_score:.1f}, "
            f"late={late:.1f}, PR={pr:.3f})."
        )
        notes.append("WATCHLIST_PREMIUM_LONG_STRONG durci")

    # P5 — Volume élevé + setup tardif = downgrade ; volume <0.30 reste autorisé/golden.
    if vol > 1.20 and setup_maturity in ("late", "exhausted"):
        actions.append("downgrade_late_high_volume")
        forced_bucket = forced_bucket or "WATCHLIST_LATE_HIGH_VOLUME"
        force_flag = "WATCHLIST" if force_flag is None else force_flag
        force_confidence_cap = min(force_confidence_cap or 60, 60)
        force_leverage_cap = min(force_leverage_cap or 3, 3)
        risk_guard = risk_guard or "volume élevé sur setup tardif"
        decision = decision or (
            f"WATCHLIST v6.5.3 : volume élevé ({vol:.2f}x) sur setup {setup_maturity}, "
            "risque de participation tardive/crowded."
        )
        notes.append("volume >1.20 + late/exhausted downgradé")

    # Correction Signals : LONG_PREMIUM en BTC_BULL_SOFT trop permissif.
    bullsoft_longpremium_blockers = []
    fut_score = v6_score_futures
    if (
        direction == "LONG"
        and entry_type == "MOMENTUM"
        and btc_market_state == "BTC_BULL_SOFT"
        and setup_family in ("LONG_MOMENTUM_CONTINUATION", "LONG_LATE_MOMENTUM")
    ):
        if futures_overheated or futures_zone == "overheated" or (fut_score is not None and fut_score > 70):
            bullsoft_longpremium_blockers.append(
                f"futures overheated/fut_score={fut_score}"
            )
        if crowding_state in ("crowded", "extreme_crowding"):
            bullsoft_longpremium_blockers.append(f"crowding={crowding_state}")
        if fut_score is None:
            if global_score < 54.5:
                bullsoft_longpremium_blockers.append(f"score faible sans futures ({global_score:.1f}<54.5)")
            if vol >= 0.70:
                bullsoft_longpremium_blockers.append(f"volume sans futures {vol:.2f}x>=0.70")
            if pr >= 0.67:
                bullsoft_longpremium_blockers.append(f"PR sans futures {pr:.3f}>=0.67")
        if setup_family == "LONG_LATE_MOMENTUM":
            bullsoft_longpremium_blockers.append("LONG_LATE_MOMENTUM")
        if derivatives_alignment in ("opposed",):
            bullsoft_longpremium_blockers.append("derivatives opposed")

    if bullsoft_longpremium_blockers:
        actions.append("downgrade_bullsoft_longpremium")
        # Ce guard ne doit s'appliquer que si le bucket engine tente LONG_PREMIUM.
        notes.append("LONG_PREMIUM BTC_BULL_SOFT guard: " + " | ".join(bullsoft_longpremium_blockers))

    return {
        "version": "v6.5.3",
        "actions": actions,
        "forced_bucket": forced_bucket,
        "force_flag": force_flag,
        "force_confidence_cap": force_confidence_cap,
        "force_leverage_cap": force_leverage_cap,
        "risk_guard_reason": risk_guard,
        "decision_explain": decision,
        "notes": " | ".join(notes),
        "bullsoft_longpremium_blockers": bullsoft_longpremium_blockers,
        "short_momentum_beta_ok": short_momentum_beta_ok,
    }

def apply_contextual_bucket_engine(
    *,
    flag,
    confidence,
    max_leverage,
    direction,
    entry_type,
    global_score,
    trend_strength,
    market_regime,
    btc_market_state,
    market_details,
    relative_vol,
    position_range,
    market_danger_level,
    late_entry_level,
    rr_valid,
    data_source,
    hard_reject,
    v6,
    risk_guard_reason,
    decision_explain,
    taker_pts=0,
    futures_support=0,
    v652_context=None,
):
    # ── V6.4.4 — TELEGRAM BUCKET ENGINE ADAPTATIF ───────────────────────────
    #
    # Philosophie : chaque règle est conditionnée au btc_market_state.
    # Le moteur doit performer dans tous les cycles de marché.
    # Données de calibration : 1 652 trades résolus (v6.3.4 → v6.4.3.1).
    #
    # Changements vs v6.4.3.1 :
    #   [R1] SHORT btc30m>=0 → REJET       : inchangée (valide tous régimes)
    #   [R2] SHORT MOMENTUM → bloqué       : ADAPTATIF par btc_market_state
    #   [R3] LONG trend=strong → WL        : ADAPTATIF + bucket PREMIUM en bullish
    #   [R4] score>=70 → WL                : ADAPTATIF EARLY vs MOMENTUM
    #   [R5a] LONG_EARLY_NEUTRAL_PREMIUM   : nouveau bucket 98% WR (56 trades)
    #   [R5b] LONG_PREMIUM → Telegram      : PR conditionnelle au btc_market_state
    #   [R5c] SHORT_PREMIUM_CANDIDATE      : élargi avec BTC_BEAR_CONTINUATION
    #   [R6] Porte Telegram stricte        : inchangée

    v652_context = v652_context or {}
    signal_quality_bucket = "STANDARD"
    telegram_rule_notes = ""
    regime_rule_applied = "STANDARD_NO_CONTEXT_RULE"
    btc_var_30m = _safe_float(market_details.get("btc_variation_30m"), 0.0)
    btc_var_2h  = _safe_float(market_details.get("btc_variation_2h"), 0.0)
    btc_var_4h  = _safe_float(market_details.get("btc_variation_4h"), 0.0)
    btc_phase   = str(market_details.get("btc_phase") or btc_market_state or "BTC_UNCLEAR")

    # Flags état BTC (calculé en amont dans compute_btc_market_state)
    is_bull_impulse    = (btc_market_state == "BTC_BULL_IMPULSE")
    is_bull_soft       = (btc_market_state in ("BTC_BULL_SOFT", "BTC_BULL_IMPULSE"))
    is_bear_cont       = (btc_market_state == "BTC_BEAR_CONTINUATION")
    is_switch_risk     = (btc_market_state == "BTC_SWITCH_RISK")
    is_bear_exhaustion = (btc_market_state == "BTC_BEAR_EXHAUSTION")
    is_neutral_comp    = (btc_market_state == "BTC_NEUTRAL_COMPRESS")
    is_bearish_short_context = (
        is_bear_cont
        or btc_phase in ("BTC_BEAR_CONTINUATION", "BTC_NEUTRAL_AFTER_BEAR")
        or (market_regime == "bearish" and btc_var_30m < 0)
    )

    # PR adaptatif selon état BTC :
    # BTC_BULL_IMPULSE : PR jusqu'à 0.80 (continuation justifiée)
    # BTC_BULL_SOFT    : PR < 0.70 (zone saine confirmée par v40)
    # neutral/autres   : PR < 0.65 (plus strict)
    if is_bull_impulse:
        pr_threshold_lp = LONG_PREMIUM_PR_BULL_IMPULSE
    elif is_bull_soft:
        pr_threshold_lp = LONG_PREMIUM_PR_BULL_SOFT
    else:
        pr_threshold_lp = LONG_PREMIUM_PR_DEFAULT
    pr_threshold_used = pr_threshold_lp

    # ── Bucket LONG_PREMIUM (Telegram) ───────────────────────────────────────
    # Zone optimale v40 : score 52-58, vol 0.30-1.20, PR adaptatif
    # WR : PR<0.70 = 64% · PR<0.65 = 73% · PR<0.60 = 76%
    is_long_premium = (
        flag != "REJET"                    # ne jamais annuler un vrai rejet dur
        and not hard_reject
        and v6.get("v6_accepted", True)
        and flag != "REJET"
        and not hard_reject
        and v6.get("v6_accepted", True)
        and direction == "LONG"
        and 52 <= global_score < 58
        and relative_vol is not None and 0.30 <= relative_vol < 1.20
        and position_range is not None and position_range < pr_threshold_lp
        and market_regime in ("bullish", "neutral")
        and market_danger_level != "HIGH"
        and trend_strength != "strong"
        and late_entry_level != "HIGH"
        and rr_valid
        and data_source != "SPOT_FALLBACK"
        and not is_switch_risk
    )

    # ── Bucket LONG_EARLY_NEUTRAL_PREMIUM (WATCHLIST_PREMIUM) ────────────────
    # Découverte v39/v40 : LONG EARLY neutral = 98% WR (56 trades, 9 symboles)
    # → WATCHLIST_PREMIUM pour accumulation, pas encore Telegram
    is_long_early_neutral = (
        direction == "LONG"
        and entry_type == "EARLY"
        and (is_neutral_comp or market_regime == "neutral")
        and trend_strength in ("weak", "moderate")
        and market_danger_level != "HIGH"
        and rr_valid
        and data_source != "SPOT_FALLBACK"
    )

    # ── Bucket SHORT_PREMIUM_CANDIDATE (WATCHLIST_PREMIUM) ───────────────────
    # v40 : SHORT EARLY neutral btc30m<0 = 70% WR (20t) → WATCHLIST_PREMIUM
    # Élargi : SHORT EARLY en BTC_BEAR_CONTINUATION (données historiques v6.3.4/6.3.6)
    is_short_premium_candidate = (
        direction == "SHORT"
        and entry_type == "EARLY"
        and btc_var_30m < 0
        and late_entry_level != "HIGH"
        and market_danger_level != "HIGH"
        and rr_valid
        and data_source != "SPOT_FALLBACK"
        and (market_regime == "neutral" or is_bear_cont)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 1 — SHORT btc30m>=0 → REJET (inchangée, valide tous régimes)
    # Données : neutral 14% (14t) · bearish 0% (22t) → justifiée partout
    # ═══════════════════════════════════════════════════════════════════════
    if direction == "SHORT" and btc_var_30m >= 0:
        flag = "REJET"
        confidence = min(confidence, 55)
        max_leverage = min(max_leverage, 3)
        risk_guard_reason = "short BTC 30m non negatif"
        decision_explain = (
            f"REJET v6.4.4 : SHORT bloqué btc_variation_30m={btc_var_30m:+.2f}% >= 0 "
            f"(valide tous régimes : 0-14% WR observé)."
        )
        signal_quality_bucket = "REJECT_SHORT_BTC30_POSITIVE"
        regime_rule_applied = "R1_SHORT_BTC30_POSITIVE_REJECT"
        telegram_rule_notes = "SHORT bloqué: BTC 30m non négatif"

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 2 — SHORT MOMENTUM — ADAPTATIF par btc_market_state
    # BTC_BULL_SOFT/IMPULSE  : 0% WR → REJET contextuel
    # BTC_NEUTRAL_COMPRESS   : 41% WR (125t) → WATCHLIST bloqué
    # BTC_SWITCH_RISK        : danger → REJET contextuel
    # BTC_BEAR_CONTINUATION  : 49-62% WR → WATCHLIST diagnostic (pas REJET)
    # ═══════════════════════════════════════════════════════════════════════
    elif direction == "SHORT" and entry_type == "MOMENTUM":
        if flag == "CANDIDAT":
            flag = "WATCHLIST"
        confidence = min(confidence, 60)
        max_leverage = min(max_leverage, 3)
        risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "short momentum non executable"

        if is_bull_soft or is_switch_risk:
            # BTC haussier ou en transition : SHORT MOMENTUM = suicide
            flag = "REJET"
            confidence = min(confidence, 50)
            decision_explain = (
                f"REJET v6.4.4 : SHORT MOMENTUM en {btc_market_state} "
                f"(0% WR historique en contexte BTC haussier)."
            )
            signal_quality_bucket = "REJECT_SHORT_MOMENTUM_BULLISH"
            regime_rule_applied = "R2_SHORT_MOMENTUM_BULLISH_REJECT"
            telegram_rule_notes = f"SHORT MOMENTUM rejeté: BTC état {btc_market_state}"
        elif is_bearish_short_context:
            # Contexte bearish / after-bear : garder en diagnostic, sauf si le layer v6.5.3
            # l'a déjà préqualifié en SHORT_MOMENTUM_CONTINUATION_PREMIUM beta.
            decision_explain = (
                f"WATCHLIST_DIAG v6.5.3 : SHORT MOMENTUM en {btc_phase} "
                f"(market_regime={market_regime}, btc30m={btc_var_30m:+.2f}%). "
                "Diagnostic bearish/after-bear ; pas Telegram sans préqualification beta stricte."
            )
            signal_quality_bucket = "WATCHLIST_SHORT_MOMENTUM_BEARISH"
            regime_rule_applied = "R2_SHORT_MOMENTUM_BEARISH_DIAG"
            telegram_rule_notes = f"SHORT MOMENTUM {btc_phase}: diagnostic uniquement hors beta stricte"
        else:
            # Range/neutral non bearish : bloquer Telegram sans wording neutral trompeur.
            decision_explain = (
                f"WATCHLIST v6.5.3 : SHORT MOMENTUM non exécutable hors contexte beta "
                f"(market_regime={market_regime}, btc_phase={btc_phase}, btc30m={btc_var_30m:+.2f}%)."
            )
            signal_quality_bucket = "WATCHLIST_SHORT_MOMENTUM_BLOCKED"
            regime_rule_applied = "R2_SHORT_MOMENTUM_BLOCKED"
            telegram_rule_notes = f"SHORT MOMENTUM bloqué Telegram ({market_regime}/{btc_phase})"

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 4 — score>=70 — ADAPTATIF EARLY vs MOMENTUM
    # LONG bullish EARLY     : 74% WR (74t) → WATCHLIST_PREMIUM_SCORE_HIGH_EARLY
    # LONG bullish MOMENTUM  : 44% WR (68t) → WATCHLIST_REVIEW (inchangé)
    # LONG neutral EARLY     : 100% WR (7t, trop petit) → WATCHLIST_PREMIUM
    # SHORT bearish          : 43% WR (7t) → WATCHLIST_REVIEW
    # ═══════════════════════════════════════════════════════════════════════
    elif global_score >= 70:
        if flag == "CANDIDAT":
            flag = "WATCHLIST"
        confidence = min(confidence, 62)
        max_leverage = min(max_leverage, 3)
        risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "score eleve review"

        if direction == "LONG" and entry_type == "EARLY" and is_bull_soft:
            # 74% WR (74t) en bullish EARLY — meilleur bucket non exploité de v40
            decision_explain = (
                f"WATCHLIST_PREMIUM v6.4.4 : score élevé ({global_score:.1f}) "
                f"mais LONG EARLY en {btc_market_state} (74% WR, 74 trades). "
                "WATCHLIST_PREMIUM — confirmation avant Telegram."
            )
            signal_quality_bucket = "WATCHLIST_PREMIUM_SCORE_HIGH_EARLY"
            regime_rule_applied = "R4_SCORE_HIGH_LONG_EARLY_PREMIUM"
            telegram_rule_notes = f"score >=70 LONG EARLY {btc_market_state}: WATCHLIST_PREMIUM"
        elif direction == "LONG" and entry_type == "EARLY" and is_neutral_comp:
            # neutral EARLY : 100% WR mais 7 trades — trop petit, rester WATCHLIST_PREMIUM
            decision_explain = (
                f"WATCHLIST_PREMIUM v6.4.4 : score élevé ({global_score:.1f}) "
                f"LONG EARLY neutral (100% WR, 7 trades — en accumulation)."
            )
            signal_quality_bucket = "WATCHLIST_PREMIUM_SCORE_HIGH_EARLY"
            regime_rule_applied = "R4_SCORE_HIGH_LONG_EARLY_PREMIUM"
            telegram_rule_notes = "score >=70 LONG EARLY neutral: WATCHLIST_PREMIUM"
        else:
            # MOMENTUM ou SHORT : WATCHLIST_REVIEW classique
            decision_explain = (
                f"WATCHLIST v6.4.4 : score élevé ({global_score:.1f}) "
                f"({entry_type}, {market_regime}) — late-entry review."
            )
            signal_quality_bucket = "WATCHLIST_SCORE_HIGH_REVIEW"
            regime_rule_applied = "R4_SCORE_HIGH_REVIEW"
            telegram_rule_notes = "score >=70 bloqué Telegram"

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 3 — LONG trend=strong — ADAPTATIF par btc_market_state
    # bullish (tous états)  : 73% WR (56t) en v6.4.3.1 → WATCHLIST_PREMIUM
    # neutral               : 44% WR (46t) → WATCHLIST_REVIEW (inchangé)
    # bearish               : 58% WR (12t) → WATCHLIST_PREMIUM (observer)
    # ═══════════════════════════════════════════════════════════════════════
    elif direction == "LONG" and trend_strength == "strong":
        if flag == "CANDIDAT":
            flag = "WATCHLIST"
        confidence = min(confidence, 62)
        max_leverage = min(max_leverage, 3)

        if is_bull_soft or market_regime == "bearish":
            # bullish : 73% WR (56t) — promouvoir en WATCHLIST_PREMIUM
            # bearish : 58% WR (12t) — rebond possible sur trend forte
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "long trend strong watchlist premium"
            decision_explain = (
                f"WATCHLIST_PREMIUM v6.4.4 : LONG trend strong en {btc_market_state} "
                f"({'73' if is_bull_soft else '58'}% WR historique). "
                "Pas encore Telegram — accumulation en cours."
            )
            signal_quality_bucket = "WATCHLIST_PREMIUM_LONG_STRONG"
            regime_rule_applied = "R3_LONG_STRONG_PREMIUM"
            telegram_rule_notes = "LONG strong: WATCHLIST_PREMIUM en observation"
        else:
            # neutral : 44% WR — review justifié
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "long trend strong late review"
            decision_explain = (
                f"WATCHLIST v6.4.4 : LONG trend strong en {market_regime} "
                f"(44% WR) — risque entrée tardive."
            )
            signal_quality_bucket = "WATCHLIST_LONG_STRONG_REVIEW"
            regime_rule_applied = "R3_LONG_STRONG_REVIEW"
            telegram_rule_notes = "LONG trend strong non exécutable sans confirmation"

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 5a — LONG_EARLY_NEUTRAL_PREMIUM (nouveau v6.4.4)
    # 98% WR (56 trades, 9 symboles, 3 versions) → WATCHLIST_PREMIUM
    # Pas encore Telegram : attendre 50+ trades supplémentaires
    # ═══════════════════════════════════════════════════════════════════════
    elif is_long_early_neutral:
        if flag == "CANDIDAT":
            flag = "WATCHLIST"
        confidence = min(confidence, 65)
        max_leverage = min(max_leverage, 3)
        decision_explain = (
            f"WATCHLIST_PREMIUM v6.4.4 : LONG EARLY neutral "
            f"(98% WR, 56 trades, 9 symboles) — confirmation avant Telegram."
        )
        signal_quality_bucket = "LONG_EARLY_NEUTRAL_PREMIUM"
        regime_rule_applied = "R5A_LONG_EARLY_NEUTRAL_PREMIUM"
        telegram_rule_notes = "LONG EARLY neutral: WATCHLIST_PREMIUM 98% WR — confirmation en cours"

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 5b — LONG_PREMIUM Telegram (adaptée v6.4.4)
    # PR conditionnelle au btc_market_state :
    # BULL_IMPULSE : PR < 0.80 · BULL_SOFT : PR < 0.70 · autres : PR < 0.65
    # ═══════════════════════════════════════════════════════════════════════
    elif is_long_premium:
        flag = "CANDIDAT"
        max_leverage = min(max_leverage, 3)
        confidence = max(58, min(confidence, 66))
        decision_explain = (
            f"CANDIDAT v6.4.4 LONG_PREMIUM : score {global_score:.1f}, "
            f"volume {relative_vol:.2f}x, position_range {position_range:.3f}, "
            f"regime {market_regime} / {btc_market_state} / PR_seuil={pr_threshold_lp:.2f}."
        )
        signal_quality_bucket = "LONG_PREMIUM"
        if is_bull_impulse:
            regime_rule_applied = "R5B_LONG_PREMIUM_BULL_IMPULSE"
        elif is_bull_soft:
            regime_rule_applied = "R5B_LONG_PREMIUM_BULL_SOFT"
        else:
            regime_rule_applied = "R5B_LONG_PREMIUM_NEUTRAL"
        telegram_rule_notes = (
            f"LONG premium: score 52-58, vol 0.30-1.20, PR<{pr_threshold_lp:.2f} "
            f"({btc_market_state})"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 5c — SHORT_PREMIUM_CANDIDATE (WATCHLIST_PREMIUM élargi v6.4.4)
    # SHORT EARLY neutral btc30m<0 : 70% WR (20t)
    # SHORT EARLY BTC_BEAR_CONTINUATION : données historiques v6.3.4/6.3.6
    # → WATCHLIST_PREMIUM, pas encore Telegram
    # ═══════════════════════════════════════════════════════════════════════
    elif is_short_premium_candidate:
        if flag == "CANDIDAT":
            flag = "WATCHLIST"
        confidence = min(confidence, 62)
        max_leverage = min(max_leverage, 3)
        context = "bearish confirmé" if is_bear_cont else "neutral btc30m<0"
        decision_explain = (
            f"WATCHLIST_PREMIUM v6.4.4 : SHORT EARLY {context} + BTC 30m négatif "
            f"(70% WR, 20 trades) — confirmation avant Telegram."
        )
        signal_quality_bucket = "SHORT_PREMIUM_CANDIDATE"
        regime_rule_applied = "R5C_SHORT_EARLY_PREMIUM_CANDIDATE"
        telegram_rule_notes = f"SHORT EARLY {context}: WATCHLIST_PREMIUM 70% WR"

    # ═══════════════════════════════════════════════════════════════════════
    # V6.5.3 — Overrides contextuels après règles v6.4.4, avant gate Telegram
    # Objectif : corriger les faux LONG_PREMIUM bullsoft, durcir WATCHLIST,
    # et ouvrir uniquement SHORT_MOMENTUM_CONTINUATION_PREMIUM en beta Signals.
    # ═══════════════════════════════════════════════════════════════════════
    v652_actions = set(v652_context.get("actions", []))

    # Correction bullsoft Signals : si le bucket engine vient de créer LONG_PREMIUM
    # mais que le contexte v6.5.3 détecte un LONG MOMENTUM fragile, downgrade.
    if signal_quality_bucket == "LONG_PREMIUM" and "downgrade_bullsoft_longpremium" in v652_actions:
        flag = "WATCHLIST"
        confidence = min(confidence, 60)
        max_leverage = min(max_leverage, 3)
        signal_quality_bucket = "WATCHLIST_LONG_PREMIUM_BULLSOFT_GUARD"
        regime_rule_applied = "V652_LONG_PREMIUM_BULLSOFT_GUARD"
        risk_guard_reason = "long premium bullsoft guard"
        blockers = v652_context.get("bullsoft_longpremium_blockers", [])
        telegram_rule_notes = "LONG_PREMIUM BTC_BULL_SOFT downgradé: " + " | ".join(blockers)
        decision_explain = (
            "WATCHLIST v6.5.3 : LONG_PREMIUM en BTC_BULL_SOFT downgradé "
            f"({'; '.join(blockers)})."
        )

    # Overrides généraux v6.5.3 : s'appliquent si pas de hard REJET déjà posé,
    # sauf short early bull context qui est explicitement un REJET contextuel.
    elif v652_context.get("force_flag"):
        requested_flag = v652_context.get("force_flag")
        requested_bucket = v652_context.get("forced_bucket")

        # Ne jamais annuler un REJET fort existant sauf si on reste REJET.
        if flag != "REJET" or requested_flag == "REJET":
            flag = requested_flag
            if requested_bucket:
                signal_quality_bucket = requested_bucket
            if v652_context.get("force_confidence_cap") is not None:
                confidence = min(confidence, v652_context["force_confidence_cap"])
            if v652_context.get("force_leverage_cap") is not None:
                max_leverage = min(max_leverage, v652_context["force_leverage_cap"])
            risk_guard_reason = v652_context.get("risk_guard_reason") or risk_guard_reason
            decision_explain = v652_context.get("decision_explain") or decision_explain
            regime_rule_applied = "V652_" + (requested_bucket or requested_flag or "CONTEXTUAL_RULE")
            telegram_rule_notes = v652_context.get("notes") or telegram_rule_notes

            # Beta Signals : conviction volontairement plafonnée.
            if requested_bucket == "SHORT_MOMENTUM_CONTINUATION_PREMIUM":
                confidence = max(58, min(confidence, SHORT_MOMENTUM_BETA_CONF_CAP))
                max_leverage = min(max_leverage, SHORT_MOMENTUM_BETA_LEVERAGE_CAP)

    # ═══════════════════════════════════════════════════════════════════════
    # RÈGLE 6 — Porte Telegram stricte (v6.5.3)
    # Buckets autorisés : LONG_PREMIUM + SHORT_MOMENTUM_CONTINUATION_PREMIUM beta.
    # ═══════════════════════════════════════════════════════════════════════
    if flag == "CANDIDAT" and signal_quality_bucket not in TELEGRAM_ALLOWED_BUCKETS:
        flag = "WATCHLIST"
        confidence = min(confidence, 60)
        max_leverage = min(max_leverage, 3)
        if signal_quality_bucket == "STANDARD":
            signal_quality_bucket = "WATCHLIST_NON_PREMIUM_CANDIDATE"
            regime_rule_applied = "R6_NON_PREMIUM_CANDIDATE_BLOCKED"
        telegram_rule_notes = telegram_rule_notes or "CANDIDAT standard bloqué Telegram: bucket non premium"
        risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "telegram bucket non premium"
        decision_explain = (
            f"WATCHLIST v6.4.4 : CANDIDAT standard bloqué Telegram "
            f"car bucket={signal_quality_bucket}, seuls LONG_PREMIUM/SHORT_MOMENTUM_CONTINUATION_PREMIUM autorisés."
        )

    # ── Caps finaux par flag (appliqués après tous les risk guards) ───────────
    if flag == "WATCHLIST":
        confidence   = min(confidence, 60)
        max_leverage = min(max_leverage, 3)
    elif flag == "SHORT_WATCH":
        confidence   = min(confidence, 55)
        max_leverage = min(max_leverage, 1)
    elif flag == "REJET":
        confidence   = min(confidence, 55)
        max_leverage = min(max_leverage, 3)

    # ── v6.5.3 — Nettoyage final buckets premium incohérents ────────────────
    # Cas observé : un hard reject post-v6.5.3 gardait parfois un bucket
    # WATCHLIST_PREMIUM_* hérité des règles v6.4.4. Le comportement était bon
    # mais le tracker devenait trompeur. Un REJET ne doit jamais rester premium.
    if flag == "REJET" and str(signal_quality_bucket).startswith("WATCHLIST_PREMIUM_"):
        if "downgrade_long_late_momentum" in v652_actions:
            signal_quality_bucket = "REJECT_LONG_LATE_MOMENTUM"
            regime_rule_applied = "V653_REJECT_LONG_LATE_MOMENTUM_BUCKET_CLEANUP"
            telegram_rule_notes = "Bucket premium nettoyé: REJET LONG_LATE_MOMENTUM"
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "long late momentum rejeté"
            if decision_explain.startswith("WATCHLIST") or "WATCHLIST_PREMIUM" in decision_explain:
                decision_explain = "REJET v6.5.3 : LONG_LATE_MOMENTUM incompatible avec un bucket premium."
        elif "downgrade_long_strong_diagnostic" in v652_actions:
            signal_quality_bucket = "REJECT_LONG_STRONG_DIAGNOSTIC"
            regime_rule_applied = "V653_REJECT_LONG_STRONG_BUCKET_CLEANUP"
            telegram_rule_notes = "Bucket premium nettoyé: REJET LONG_STRONG diagnostic"
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "long strong diagnostic rejeté"
            if decision_explain.startswith("WATCHLIST") or "WATCHLIST_PREMIUM" in decision_explain:
                decision_explain = "REJET v6.5.3 : LONG strong ne remplit pas les conditions premium."
        else:
            signal_quality_bucket = "REJECT_PREMIUM_BUCKET_CLEANUP"
            regime_rule_applied = "V653_REJECT_PREMIUM_BUCKET_CLEANUP"
            telegram_rule_notes = "Bucket premium nettoyé: REJET final"

    elif flag == "WATCHLIST" and str(signal_quality_bucket).startswith("WATCHLIST_PREMIUM_") and "downgrade_long_strong_diagnostic" in v652_actions:
        signal_quality_bucket = "WATCHLIST_LONG_STRONG_DIAGNOSTIC"
        regime_rule_applied = "V653_WATCHLIST_LONG_STRONG_DIAGNOSTIC_BUCKET_CLEANUP"
        telegram_rule_notes = "LONG strong downgradé: diagnostic, non premium"
        if "WATCHLIST_PREMIUM" in decision_explain:
            decision_explain = "WATCHLIST v6.5.3 : LONG strong downgradé en diagnostic, conditions premium insuffisantes."

    executable_signal = (flag == "CANDIDAT")

    # ── Harmonisation finale decision_explain / flag ──────────────────────────
    # Évite les incohérences "WATCHLIST : ..." dans un signal REJET et vice-versa
    if flag == "REJET" and decision_explain.startswith("WATCHLIST"):
        decision_explain = f"REJET : {risk_guard_reason}."
    elif flag == "WATCHLIST" and decision_explain.startswith("REJET"):
        decision_explain = f"WATCHLIST : {risk_guard_reason}."

    if not decision_explain:
        if flag == "CANDIDAT":
            decision_explain = f"CANDIDAT : taker {taker_pts:+d}, futures_support {futures_support}/4, volume {relative_vol:.2f}x."
        elif flag == "WATCHLIST":
            decision_explain = "WATCHLIST : signal technique sans confirmation futures suffisante."
        else:
            decision_explain = "REJET : score ou regles de garde."


    return {
        "flag": flag,
        "confidence": confidence,
        "max_leverage": max_leverage,
        "signal_quality_bucket": signal_quality_bucket,
        "telegram_rule_notes": telegram_rule_notes,
        "regime_rule_applied": regime_rule_applied,
        "pr_threshold_used": pr_threshold_used,
        "decision_explain": decision_explain,
        "risk_guard_reason": risk_guard_reason,
        "executable_signal": executable_signal,
    }

def score_symbol(symbol, ticker_data=None, market_regime="unknown", market_details=None):
    market_details = market_details or {}
    btc_market_state = market_details.get("btc_market_state", "BTC_NEUTRAL_COMPRESS")
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
    high_7d = max(highs[-lookback_7d:]) if lookback_7d > 0 else high_24h
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

    # ── V6.0.5 — PÉNALITÉ VOLUME RELATIF FAIBLE (MOMENTUM uniquement) ────────
    # EARLY exclu : volume faible normal avant départ du mouvement.
    # BREAKOUT déjà protégé : condition relative_vol >= 1.3 dans la détection.
    # MOMENTUM : un volume faible signale un mouvement de prix non soutenu.
    vol_penalty = 0
    vol_penalty_note = ""
    if entry_type == "MOMENTUM":
        if relative_vol < 0.15:
            vol_penalty = -20
            vol_penalty_note = f"volume très faible ({relative_vol:.2f}x moyenne)"
        elif relative_vol < 0.30:
            vol_penalty = -15
            vol_penalty_note = f"volume faible ({relative_vol:.2f}x moyenne)"
        elif relative_vol < 0.50:
            vol_penalty = -8
            vol_penalty_note = f"volume modérément faible ({relative_vol:.2f}x moyenne)"
    if vol_penalty < 0:
        global_score = max(0, global_score + vol_penalty)
        logger.info("V6.0.5 pénalité volume %s: %s (%+d pts) → score %.1f",
                    symbol, vol_penalty_note, vol_penalty, global_score)

    # 11. SPOT FALLBACK : accepté temporairement, mais moins fiable pour futures.
    if data_source == "SPOT_FALLBACK":
        global_score -= 8

    # 12. SHORT interdit : pas de signal SHORT proche du plus bas 7j avec shorts crowded.
    if short_forbidden:
        global_score = min(global_score, 45)

    global_score = round(max(0, global_score), 1)

    # ── Gardes décisionnelles initialisées tôt ───────────────────────────────
    # v6.4.2.2 — évite un UnboundLocalError quand P4-U trend=weak transforme
    # un CANDIDAT en WATCHLIST avant le bloc post-V6.
    hard_reject = False
    forced_watchlist = False
    risk_guard_reason = "aucun"
    decision_explain = ""

    # ── FLAG ─────────────────────────────────────────────────────────────────
    if global_score >= 58:
        flag = "CANDIDAT"
    elif global_score >= 52:
        flag = "WATCHLIST"
    else:
        flag = "REJET"

    if short_watch:
        flag = "SHORT_WATCH"

    if short_forbidden:
        flag = "REJET"

    if flag == "CANDIDAT" and late_entry_risk >= 55:
        flag = "WATCHLIST"

    if flag == "CANDIDAT" and direction == "LONG" and market_regime == "bearish":
        flag = "WATCHLIST"

    # ── v6.3.3 — P4-U : trend=weak interdit en CANDIDAT ────────────────────
    # Données tracker : 3 des 5 LOSS v6.3.2 avaient trend=weak (SOL×2, XRP).
    # Un signal CANDIDAT avec tendance faible n'a pas la structure directionnelle
    # minimale pour un trade exécutable — même avec un bon score.
    if flag == "CANDIDAT" and trend_strength == "weak":
        flag             = "WATCHLIST"
        forced_watchlist = True
        risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "trend faible"
        logger.info("V6.3.3 P4-U trend_weak CANDIDAT→WATCHLIST %s", symbol)

    # ── V6.0.5 — RÉTROGRADATION + CAP LEVIER si volume MOMENTUM très faible ──
    if entry_type == "MOMENTUM":
        if relative_vol < 0.15:
            # Volume extrêmement faible : rétrogradation forcée + levier plafonné à 3x
            if flag == "CANDIDAT":
                flag = "WATCHLIST"
                logger.info("V6.0.5 rétrogradation WATCHLIST %s: volume critique (%.2fx)", symbol, relative_vol)
        elif relative_vol < 0.30:
            # Volume très faible : rétrogradation forcée
            if flag == "CANDIDAT":
                flag = "WATCHLIST"
                logger.info("V6.0.5 rétrogradation WATCHLIST %s: volume très faible (%.2fx)", symbol, relative_vol)

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

    # SL adaptatif v6.0.6f : plus d'espace pour les volatils
    VOLATILE_SYMBOLS = {"HYPEUSDT","ZECUSDT","NEARUSDT","1000PEPEUSDT","WIFUSDT",
                        "PEPEUSDT","MUUSDT","SNDKUSDT","LABUSDT"}
    sl_atr_mult = 1.2 if symbol in MAJORS else 1.3
    if atr_pct >= 2.5 or symbol in VOLATILE_SYMBOLS:
        sl_atr_mult = max(sl_atr_mult, 1.5)

    if symbol in MEMECOINS:    sl_max_pct = 2.0
    elif symbol in MAJORS:     sl_max_pct = 4.0
    else:                      sl_max_pct = 3.0

    if direction == "LONG":
        sl_raw = round(entry_avg - sl_atr_mult * atr, 8)
        sl_distance = abs(current - sl_raw) / current * 100 if current > 0 else 0
        if sl_distance > sl_max_pct:
            sl_raw = round(current * (1 - sl_max_pct / 100), 8)
        stop_loss = sl_raw
    elif direction == "SHORT":
        sl_raw = round(entry_avg + sl_atr_mult * atr, 8)
        sl_distance = abs(sl_raw - current) / current * 100 if current > 0 else 0
        if sl_distance > sl_max_pct:
            sl_raw = round(current * (1 + sl_max_pct / 100), 8)
        stop_loss = sl_raw
    else:
        stop_loss = round(current, 8)

    # Risk (distance entry → SL technique) — calculé AVANT les TP.
    # Le SL reste technique (ATR plafonné par classe d'actif, déjà calculé plus haut).
    # On ne l'éloigne JAMAIS artificiellement pour améliorer le RR.
    risk = abs(entry_avg - stop_loss)

    # ── CONTRÔLE DE RÉALISME DU TP FINAL (target_rr 5R vs 8R) ─────────────────
    # On ne place pas un TP à 8R s'il est hors d'atteinte par rapport à la
    # volatilité récente et aux extrêmes de range. Dans ce cas, on rabat à 5R.
    # Logique : la distance prix→TP8 doit rester atteignable au regard de l'ATR
    # et ne pas dépasser largement le high/low 7j dans le sens du trade.
    target_rr = 8   # objectif idéal par défaut
    realism_reasons = []

    if risk > 0 and direction in ("LONG", "SHORT"):
        # Distance théorique jusqu'au TP 8R, en % du prix
        tp8_price = entry_avg + 8 * risk if direction == "LONG" else entry_avg - 8 * risk
        dist_tp8_pct = abs(tp8_price - current) / current * 100 if current > 0 else 999

        # 1. Réalisme volatilité : le mouvement 8R doit être faisable par rapport à l'ATR.
        #    Si la distance à TP8 dépasse ~12x l'ATR%, c'est peu probable à horizon du trade.
        atr_budget = atr_pct * 12 if atr_pct > 0 else 0
        if atr_budget > 0 and dist_tp8_pct > atr_budget:
            realism_reasons.append(f"TP8 {dist_tp8_pct:.1f}% > budget ATR {atr_budget:.1f}%")

        # 2. Réalisme structure : TP8 ne doit pas exiger de pulvériser le high/low 7j
        #    de façon excessive (au-delà de +3% du high 7j pour un LONG, ou -3% du low 7j pour un SHORT).
        if direction == "LONG" and high_7d > 0:
            overshoot = (tp8_price - high_7d) / high_7d * 100
            if overshoot > 3.0:
                realism_reasons.append(f"TP8 dépasse high_7d de {overshoot:.1f}%")
        elif direction == "SHORT" and low_7d > 0:
            overshoot = (low_7d - tp8_price) / low_7d * 100
            if overshoot > 3.0:
                realism_reasons.append(f"TP8 sous low_7d de {overshoot:.1f}%")

        # 3. Réalisme momentum : une entrée déjà tardive a peu de réserve pour aller à 8R.
        if late_entry_risk >= 55:
            realism_reasons.append("entrée tardive, réserve de mouvement limitée")

        # Si au moins un critère de réalisme échoue → on rabat l'objectif à 5R.
        if realism_reasons:
            target_rr = 5

    # ── TAKE PROFITS en multiples de R (hybride : R + réalisme) ───────────────
    # TP1=1R sécurisation, TP2=2R trade validé, TP3=3R bon move,
    # TP4=5R objectif cible, TP5=target_rr (8R si réaliste, sinon 5R).
    if direction == "LONG":
        tp1 = round(entry_avg + 1 * risk, 8)
        tp2 = round(entry_avg + 2 * risk, 8)
        tp3 = round(entry_avg + 3 * risk, 8)
        tp4 = round(entry_avg + 5 * risk, 8)
        tp5 = round(entry_avg + target_rr * risk, 8)
    elif direction == "SHORT":
        tp1 = round(entry_avg - 1 * risk, 8)
        tp2 = round(entry_avg - 2 * risk, 8)
        tp3 = round(entry_avg - 3 * risk, 8)
        tp4 = round(entry_avg - 5 * risk, 8)
        tp5 = round(entry_avg - target_rr * risk, 8)
    else:
        tp1 = tp2 = tp3 = tp4 = tp5 = round(current, 8)
        target_rr = 0

    # ── DEUX RR DISTINCTS ─────────────────────────────────────────────────────
    # risk_reward_tp2    : RR opérationnel court terme (TP2) → reporting + rr_valid
    # risk_reward_target : RR cible final (TP4 = 5R, plancher solide) → utilisé par le VETO V6
    #   On base le veto sur TP4 (5R) et non TP5 : 5R est l'objectif cible toujours
    #   présent, alors que TP5 peut être rabattu. Le veto reste cohérent et stable.
    if risk > 0:
        risk_reward_tp2    = round(abs(tp2 - entry_avg) / risk, 2)
        risk_reward_target = round(abs(tp4 - entry_avg) / risk, 2)   # = 5R
    else:
        risk_reward_tp2 = 0
        risk_reward_target = 0

    rr_levels = compute_rr_levels(direction, entry_avg, stop_loss, [tp1, tp2, tp3, tp4, tp5])

    # Compat : risk_reward conservé = RR opérationnel TP2 (utilisé pour rr_valid)
    risk_reward = risk_reward_tp2
    rr_valid    = risk_reward_tp2 >= 2.0

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
    # V6.0.5 — Volume MOMENTUM critique : cap levier 3x
    if entry_type == "MOMENTUM" and relative_vol < 0.15:
        leverage_caps.append(3)
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

    # V5.9 — Quarantaine WIFUSDT : 12 LOSS / 0 WIN sur 12 trades mesurés.
    # En quarantaine temporaire (7 jours) : rejeté sauf setup exceptionnel
    # (score ≥ 75 + source Futures + trend strong). À réévaluer après 30 trades.
    QUARANTINE_SYMBOLS = {"WIFUSDT"}
    if symbol in QUARANTINE_SYMBOLS:
        exceptional = (
            global_score >= 75
            and data_source in ("FUTURES", "BYBIT_FUTURES")
            and trend_strength == "strong"
        )
        if not exceptional:
            flag = "REJET"

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

    # ── V6.0 : COUCHES VETO + FUTURES (après tous les calculs, avant return) ──
    # On passe le ticker 24h volume pour la règle de liquidité du veto.
    quote_volume_24h_v6 = float(ticker_data.get("quoteVolume", 0)) if ticker_data else None
    # On prépare un dict partiel pour check_veto (funding_rate déjà connu)
    # Correction : si funding indisponible, passer None (pas 0.0) pour éviter
    # un faux bonus "funding neutre" dans le score futures.
    _v6_input = {
        "funding_rate": funding_rate if funding_available else None,
        "momentum_1h":  momentum_1h,
        "volume_relatif": relative_vol
    }
    # On injecte oi_change_pct une fois fetch fait dans apply_v6_layer
    # Le veto V6 utilise le RR CIBLE (TP4 = 5R), pas le RR opérationnel TP2 (~2R).
    v6 = apply_v6_layer(
        symbol=symbol,
        direction=direction,
        technical_score=global_score,
        rr=risk_reward_target,
        market_danger_level=market_danger_level,
        quote_volume_24h=quote_volume_24h_v6,
        result_dict=_v6_input
    )

    # Si veto déclenché → REJET systématique, tous flags confondus
    # (CANDIDAT, WATCHLIST, SHORT_WATCH : aucun ne doit passer un veto v6)
    if not v6["v6_accepted"]:
        flag = "REJET"
        logger.info("V6 veto %s: %s", symbol, v6["v6_veto_reasons"])

    # Si accepté → remplacer le score par le score final combiné 70/30
    if v6["v6_accepted"] and v6["v6_score_final"] is not None:
        global_score = v6["v6_score_final"]
        # Réévaluer le flag avec le nouveau score combiné
        if flag not in ("SHORT_WATCH", "REJET"):
            if global_score >= 58:
                flag = "CANDIDAT"
            elif global_score >= 52:
                flag = "WATCHLIST"
            else:
                flag = "REJET"
        # Ré-appliquer les règles de rétrogradation sur le score combiné
        if flag == "CANDIDAT" and late_entry_risk >= 55:
            flag = "WATCHLIST"
        if flag == "CANDIDAT" and direction == "LONG" and market_regime == "bearish":
            flag = "WATCHLIST"

    # ── v6.2 — Point 1 : cap confidence MOMENTUM ≤ 70 ──────────────────────
    # Donnée tracker v6.0.6 : conf >75% sur MOMENTUM = 25% WR (8 trades).
    # Exception : futures zone healthy ET late_entry_risk < 30 — confirmation
    # dérivés saine et timing non tardif justifient de laisser monter jusqu'à 75.
    # IMPORTANT : ce bloc doit rester APRÈS apply_v6_layer(), car il dépend de v6_score_futures.
    if entry_type == "MOMENTUM" and flag == "CANDIDAT":
        _fut_score = v6.get("v6_score_futures")
        _fut_healthy = _fut_score is not None and (50 <= _fut_score <= 65)
        _not_late = late_entry_risk < 30
        if _fut_healthy and _not_late:
            confidence = min(confidence, 75)   # exception : futures saine + timing ok
        else:
            confidence = min(confidence, 70)   # règle générale MOMENTUM

    # ── V6.0.6f — REGLES DE GARDE POST-V6 ──────────────────────────────────────
    # hard_reject=True → REJET immuable
    # forced_watchlist=True → WATCHLIST si CANDIDAT, immuable
    # v6.4.2.2 : variables déjà initialisées avant P4-U ; ne pas les réinitialiser ici.

    futures_detail  = v6.get("v6_futures_detail", {}) or {}
    taker_pts       = futures_detail.get("taker", 0)
    oi_pts          = futures_detail.get("oi", 0)
    funding_pts     = futures_detail.get("funding", 0)
    long_short_pts  = futures_detail.get("long_short", 0)
    # funding ne compte dans futures_support QUE si reellement favorable (>0)
    futures_support = sum([oi_pts > 0, taker_pts > 0, long_short_pts > 0, funding_pts > 0])

    # ── V6.0.7 — Lecture non linéaire du futures score + taker non bloquant ──
    # Le backtest v6.0.6g montre :
    # - taker_score <= 0 ne doit plus être un veto automatique sur MOMENTUM ;
    # - v6_score_futures 50–65 = zone saine ;
    # - v6_score_futures >= 70 = risque de surchauffe / crowded / entrée tardive.
    v6_fut_score = v6.get("v6_score_futures")
    futures_zone = "unavailable"
    healthy_futures_confirmation = False
    futures_overheated = False
    taker_not_confirmed = False
    calibration_flags = []
    confidence_cap_reason = ""
    leverage_cap_reason = ""

    if v6_fut_score is not None:
        if v6_fut_score < FUTURES_HEALTHY_MIN:
            futures_zone = "weak"
        elif FUTURES_HEALTHY_MIN <= v6_fut_score <= FUTURES_HEALTHY_MAX:
            futures_zone = "healthy"
            healthy_futures_confirmation = True
            calibration_flags.append("healthy_futures_confirmation")
            if flag == "CANDIDAT" and not hard_reject:
                global_score = round(min(100, global_score + FUTURES_HEALTHY_BONUS_PTS), 1)
        elif v6_fut_score < FUTURES_CAUTION_MAX:
            futures_zone = "caution"
            calibration_flags.append("futures_caution_zone")
        else:
            futures_zone = "overheated"
            futures_overheated = True
            calibration_flags.append("futures_overheated")
            if flag == "CANDIDAT" and not hard_reject:
                global_score = round(max(0, global_score - FUTURES_OVERHEATED_PENALTY), 1)
                confidence = min(confidence, FUTURES_OVERHEATED_CONF_CAP)
                max_leverage = min(max_leverage, FUTURES_OVERHEATED_LEV_CAP)
                confidence_cap_reason = "futures_overheated"
                leverage_cap_reason = "futures_overheated"

            if entry_type == "MOMENTUM" and position_range >= FUTURES_LATE_POSITION_RANGE:
                forced_watchlist = True
                risk_guard_reason = "futures overheated + position range élevée"
                decision_explain = (
                    f"WATCHLIST : futures score surchauffé ({v6_fut_score:.1f}) "
                    f"+ position_range élevée ({position_range:.3f}), risque d'entrée tardive."
                )

    # Regle 7 v6.0.7 : taker non confirmé = pénalité douce, plus de WATCHLIST automatique.
    if entry_type == "MOMENTUM" and flag == "CANDIDAT" and taker_pts <= 0:
        taker_not_confirmed = True
        calibration_flags.append("taker_not_confirmed")
        global_score = round(max(0, global_score - TAKER_SOFT_PENALTY_PTS), 1)
        confidence = min(confidence, TAKER_SOFT_CONF_CAP)
        max_leverage = min(max_leverage, TAKER_SOFT_LEVERAGE_CAP)
        confidence_cap_reason = confidence_cap_reason or "taker_not_confirmed"
        leverage_cap_reason = leverage_cap_reason or "taker_not_confirmed"

        # On ne force WATCHLIST que si le setup cumule d'autres faiblesses concrètes.
        if (
            relative_vol < 0.50
            or (v6_fut_score is not None and v6_fut_score < FUTURES_HEALTHY_MIN)
            or market_danger_level == "HIGH"
            or trend_strength == "weak"
        ):
            forced_watchlist  = True
            risk_guard_reason = "taker non confirme + faiblesse contextuelle"
            decision_explain  = (
                f"WATCHLIST : taker non confirmé ({taker_pts:+d}) "
                f"avec volume {relative_vol:.2f}x / futures_zone={futures_zone}."
            )
        else:
            decision_explain = (
                f"CANDIDAT : taker non confirmé ({taker_pts:+d}) traité en pénalité douce, "
                f"futures_zone={futures_zone}, volume {relative_vol:.2f}x."
            )

    # Regle 8 : volume faible — v6.3 : volume critique n'est plus un hard_reject systématique.
    # Logique :
    #   vol < 0.15 → jamais CANDIDAT (non exécutable)
    #   vol < 0.15 + contexte sain (futures healthy, BTC stable, RR valide) → WATCHLIST
    #   vol < 0.15 + contexte faible (futures indispo/weak, danger HIGH, trend weak) → REJET
    #   vol < 0.30 → forced WATCHLIST (inchangé)
    #   vol < 0.50 + pas de confirmation → forced WATCHLIST (inchangé)
    if entry_type == "MOMENTUM" and not hard_reject:
        if relative_vol < 0.15:
            # Évaluation du contexte pour décider WATCHLIST vs REJET
            _vol_context_sain = (
                futures_zone in ("healthy", "caution")
                and market_danger_level != "HIGH"
                and trend_strength != "weak"
                and rr_valid
                and late_entry_risk < 40
            )
            if _vol_context_sain:
                forced_watchlist  = True
                risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "volume critique"
                decision_explain  = decision_explain or (
                    f"WATCHLIST : volume critique ({relative_vol:.2f}x) mais contexte sain "
                    f"(futures={futures_zone}, danger={market_danger_level}). Non exécutable."
                )
                logger.info("V6.3 volume critique WATCHLIST %s: %.2fx, zone=%s", symbol, relative_vol, futures_zone)
            else:
                hard_reject       = True
                risk_guard_reason = "volume critique"
                decision_explain  = (
                    f"REJET : volume critique ({relative_vol:.2f}x) "
                    f"+ contexte faible (futures={futures_zone}, danger={market_danger_level})."
                )
                logger.info("V6.3 volume critique REJET %s: %.2fx, zone=%s", symbol, relative_vol, futures_zone)
        elif relative_vol < 0.30:
            forced_watchlist  = True
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "volume tres faible"
            decision_explain  = decision_explain or f"WATCHLIST : volume tres faible ({relative_vol:.2f}x)."
        elif relative_vol < 0.50 and taker_pts < 8 and futures_support < 2:
            forced_watchlist  = True
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "volume faible sans confirmation"
            decision_explain  = decision_explain or f"WATCHLIST : volume faible ({relative_vol:.2f}x) + futures_support {futures_support}/4."

    # Regle 9 : short trop bas dans le range
    if direction == "SHORT" and not hard_reject:
        exception_range = (market_regime == "bearish" and relative_vol >= 1.0
                           and taker_pts >= 8 and late_entry_risk < 40)
        if position_range < 0.18 and not exception_range:
            hard_reject       = True
            risk_guard_reason = "short trop proche du bas de range"
            decision_explain  = f"REJET : short trop bas (position_range={position_range:.3f})."
        elif 0.18 <= position_range < 0.25 and not exception_range:
            forced_watchlist  = True
            max_leverage      = min(max_leverage, 3)
            confidence        = min(confidence, 60)
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "short proche du bas de range"
            decision_explain  = decision_explain or f"WATCHLIST : short proche du bas de range ({position_range:.3f})."

    # Regle 10 : anti-short BTC bullish
    # v6.5.3 : on conserve le guard, mais on évite de tuer préventivement le
    # futur bucket SHORT_MOMENTUM_CONTINUATION_PREMIUM quand le btc_phase
    # indique déjà range / after-bear / bear-continuation avec btc30m négatif.
    if direction == "SHORT" and market_regime == "bullish" and not hard_reject:
        _btc_phase_v652 = str(market_details.get("btc_phase") or "")
        _btc_var_30m_v652 = _safe_float(market_details.get("btc_variation_30m"), 0.0)
        exception_bullish = (
            (
                global_score >= 75 and relative_vol >= 1.2 and taker_pts >= 8
                and position_range >= 0.35 and late_entry_risk < 40
            )
            or (
                entry_type == "MOMENTUM"
                and _btc_phase_v652 in ("BTC_NEUTRAL_AFTER_BEAR", "BTC_RANGE_CHOP", "BTC_BEAR_CONTINUATION")
                and _btc_var_30m_v652 < 0
                and late_entry_risk < 35
                and position_range < 0.35
                and relative_vol < 0.80
                and market_danger_level != "HIGH"
            )
        )
        if not exception_bullish:
            hard_reject       = True
            risk_guard_reason = "short contre BTC bullish"
            decision_explain  = "REJET : short contre regime BTC bullish."

    # Après les ajustements v6.0.7, on réévalue le flag si le score calibré
    # repasse sous les seuils. Cela évite un CANDIDAT dont le score a été abaissé
    # par taker_not_confirmed ou futures_overheated.
    #
    # v6.3.1 — Priorité forced_watchlist :
    # Si une règle de garde a déjà décidé WATCHLIST (volume faible / volume critique
    # en contexte sain / futures score insuffisant), le score <52 ne doit pas
    # réécrire la décision en REJET. Sinon on obtient flag=REJET mais
    # final_decision_reason=WATCHLIST dans les logs, et on perd les cas utiles
    # pour calibration WATCHLIST_LOG. Les vrais hard_reject explicites restent
    # prioritaires plus haut (short contre BTC bullish, volume critique contexte faible, etc.).
    if flag == "CANDIDAT" and not hard_reject and global_score < 58:
        if forced_watchlist:
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "watchlist forcée par garde"
            decision_explain = decision_explain or f"WATCHLIST : garde active malgré score calibré bas ({global_score:.1f}<58)."
        elif global_score >= 52:
            forced_watchlist = True
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "score calibré sous seuil candidat"
            decision_explain = decision_explain or f"WATCHLIST : score calibré sous seuil candidat ({global_score:.1f}<58)."
        else:
            hard_reject = True
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "score calibré insuffisant"
            decision_explain = decision_explain or f"REJET : score calibré insuffisant ({global_score:.1f}<52)."

    # Application finale des gardes
    if hard_reject:
        flag = "REJET"
        confidence   = min(confidence, 55)
        max_leverage = min(max_leverage, 3)
        logger.info("V6.0.6f hard_reject %s: %s", symbol, risk_guard_reason)
    elif forced_watchlist and flag != "WATCHLIST":
        # v6.3.3 — FIX P1 : forced_watchlist s'applique quel que soit le flag.
        # v6.3.1 avait 'flag == "CANDIDAT"' ce qui ignorait les cas REJET naturel.
        flag = "WATCHLIST"
        confidence   = min(confidence, 65)
        max_leverage = min(max_leverage, 3)
        logger.info("V6.3.3 forced_watchlist %s: %s (was %s)", symbol, risk_guard_reason, flag)

    # ── V6.0.6g — Règle 1 : futures score minimum pour MOMENTUM CANDIDAT ─────
    # Un MOMENTUM CANDIDAT avec v6_score_futures < 50 n'a pas assez de
    # confirmation dérivés pour être exécutable.
    # v6.2 : exception supprimée — taker_pts>=8 n'est jamais atteint dans le tracker
    # (taker=4 est la valeur quasi-systématique), ce qui rendait l'exception inopérante
    # tout en laissant passer des signaux à 29% WR (7 trades avec fut<50 en v6.0.6).
    if entry_type == "MOMENTUM" and flag == "CANDIDAT":
        v6_fut_score = v6.get("v6_score_futures")
        if v6_fut_score is not None and v6_fut_score < 50:
            forced_watchlist  = True
            flag              = "WATCHLIST"
            confidence        = min(confidence, 60)
            max_leverage      = min(max_leverage, 3)
            risk_guard_reason = "futures score insuffisant"
            decision_explain  = f"WATCHLIST : futures score insuffisant ({v6_fut_score:.1f}<50), confirmation dérivés trop faible."
            logger.info("V6.2 futures_score_min (no exception) %s: score=%.1f", symbol, v6_fut_score)

    # ── V6.0.6g — Règle 3 : OI fortement négatif non compensé ────────────────
    # OI <= -12 signale un effondrement des positions — dangereux sans taker fort.
    if entry_type == "MOMENTUM" and flag == "CANDIDAT" and oi_pts <= -12:
        exception_oi = (
            taker_pts >= 8 and
            relative_vol >= 1.2
        )
        if not exception_oi:
            forced_watchlist  = True
            flag              = "WATCHLIST"
            confidence        = min(confidence, 60)
            max_leverage      = min(max_leverage, 3)
            risk_guard_reason = risk_guard_reason if risk_guard_reason != "aucun" else "OI fortement négatif non compensé"
            decision_explain  = decision_explain or "WATCHLIST : OI fortement négatif non compensé par un taker fort."
            logger.info("V6.0.6g oi_negative %s: oi_pts=%d taker=%d", symbol, oi_pts, taker_pts)

    # ── V6.1 — Decision engine centralisé : promotion / downgrade auditable ──
    v61_decision = decision_engine_v6_1(
        symbol=symbol,
        flag=flag,
        direction=direction,
        entry_type=entry_type,
        global_score=global_score,
        confidence=confidence,
        max_leverage=max_leverage,
        futures_zone=futures_zone,
        futures_overheated=futures_overheated,
        taker_not_confirmed=taker_not_confirmed,
        taker_pts=taker_pts,
        funding_signal=funding_signal,
        long_short_pts=long_short_pts,
        futures_support=futures_support,
        relative_vol=relative_vol,
        late_entry_risk=late_entry_risk,
        late_entry_level=late_entry_level,
        position_range=position_range,
        market_danger_level=market_danger_level,
        market_regime=market_regime,
        rr_valid=rr_valid,
        data_source=data_source,
        decision_explain=decision_explain,
        risk_guard_reason=risk_guard_reason,   # v6.2 : pour bloquer la promotion si garde forte
    )
    flag = v61_decision["flag"]
    confidence = v61_decision["confidence"]
    max_leverage = v61_decision["max_leverage"]
    decision_explain = v61_decision["decision_explain"]

    # ── v6.5.3 — Instrumentation setup/participation AVANT bucket engine ───
    # v6.5.0 instrumentait ces champs après le bucket engine. En v6.5.3 ils
    # deviennent des inputs de décision contextuelle, sans modifier le score brut.
    setup_v65 = classify_setup_v65(
        direction=direction,
        entry_type=entry_type,
        trend_strength=trend_strength,
        late_entry_risk=late_entry_risk,
        late_entry_level=late_entry_level,
        position_range=position_range,
        relative_vol=relative_vol,
        momentum_1h=momentum_1h,
        momentum_3h=momentum_3h,
        btc_phase=market_details.get("btc_phase"),
        btc_context_bias=market_details.get("btc_context_bias"),
        distance_ema21=distance_ema21,
        rsi=rsi,
    )
    participation_v65 = classify_participation_v65(
        entry_type=entry_type,
        direction=direction,
        relative_vol=relative_vol,
        position_range=position_range,
        setup_maturity=setup_v65.get("setup_maturity"),
        taker_pts=taker_pts,
        oi_pts=oi_pts,
        funding_pts=funding_pts,
        long_short_pts=long_short_pts,
        futures_zone=futures_zone,
        funding_signal=funding_signal,
        derivatives_bias=derivatives_bias,
    )

    # ── V6.5.3 — Contextual pre-qualification layer ────────────────────────
    # Appelée avant apply_contextual_bucket_engine, appliquée dans le bucket
    # engine après les règles v6.4.4 et avant la gate Telegram.
    v652_context = apply_contextual_v652_rules(
        flag=flag,
        confidence=confidence,
        max_leverage=max_leverage,
        direction=direction,
        entry_type=entry_type,
        global_score=global_score,
        trend_strength=trend_strength,
        market_regime=market_regime,
        btc_market_state=btc_market_state,
        market_details=market_details,
        relative_vol=relative_vol,
        position_range=position_range,
        market_danger_level=market_danger_level,
        late_entry_risk=late_entry_risk,
        late_entry_level=late_entry_level,
        rr_valid=rr_valid,
        data_source=data_source,
        hard_reject=hard_reject,
        setup_family=setup_v65.get("setup_family"),
        setup_maturity=setup_v65.get("setup_maturity"),
        futures_zone=futures_zone,
        futures_overheated=futures_overheated,
        v6_score_futures=v6.get("v6_score_futures"),
        crowding_state=participation_v65.get("crowding_state"),
        derivatives_alignment=participation_v65.get("derivatives_alignment"),
    )

    # ── V6.4.4-final-clean + V6.5.3 context — TELEGRAM BUCKET ENGINE ────────
    bucket_decision = apply_contextual_bucket_engine(
        flag=flag,
        confidence=confidence,
        max_leverage=max_leverage,
        direction=direction,
        entry_type=entry_type,
        global_score=global_score,
        trend_strength=trend_strength,
        market_regime=market_regime,
        btc_market_state=btc_market_state,
        market_details=market_details,
        relative_vol=relative_vol,
        position_range=position_range,
        market_danger_level=market_danger_level,
        late_entry_level=late_entry_level,
        rr_valid=rr_valid,
        data_source=data_source,
        hard_reject=hard_reject,
        v6=v6,
        risk_guard_reason=risk_guard_reason,
        decision_explain=decision_explain,
        taker_pts=taker_pts,
        futures_support=futures_support,
        v652_context=v652_context,
    )
    flag = bucket_decision["flag"]
    confidence = bucket_decision["confidence"]
    max_leverage = bucket_decision["max_leverage"]
    signal_quality_bucket = bucket_decision["signal_quality_bucket"]
    telegram_rule_notes = bucket_decision["telegram_rule_notes"]
    regime_rule_applied = bucket_decision["regime_rule_applied"]
    pr_threshold_used = bucket_decision["pr_threshold_used"]
    decision_explain = bucket_decision["decision_explain"]
    risk_guard_reason = bucket_decision["risk_guard_reason"]
    executable_signal = bucket_decision["executable_signal"]

    # Duree estimee calculee par Python
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
        # ── v6.5.0 — contexte BTC enrichi ──────────────────────────────────
        "btc_context_bias": market_details.get("btc_context_bias"),
        "btc_phase": market_details.get("btc_phase"),
        "btc_trend_slope_2h": market_details.get("btc_trend_slope_2h"),
        "btc_trend_slope_4h": market_details.get("btc_trend_slope_4h"),
        "btc_trend_slope_12h": market_details.get("btc_trend_slope_12h"),
        "btc_impulse_age": market_details.get("btc_impulse_age"),
        "btc_last_pivot_type": market_details.get("btc_last_pivot_type"),
        "btc_last_pivot_age": market_details.get("btc_last_pivot_age"),
        "btc_last_pivot_distance_pct": market_details.get("btc_last_pivot_distance_pct"),
        "btc_last_pivot_method": market_details.get("btc_last_pivot_method"),
        "btc_pullback_depth": market_details.get("btc_pullback_depth"),
        "btc_range_position": market_details.get("btc_range_position"),
        "btc_rejection_state": market_details.get("btc_rejection_state"),
        "btc_support_distance_pct": market_details.get("btc_support_distance_pct"),
        "btc_resistance_distance_pct": market_details.get("btc_resistance_distance_pct"),
        "btc_volatility_regime": market_details.get("btc_volatility_regime"),
        "btc_context_score": market_details.get("btc_context_score"),
        # ── v6.5.0 — setup classification / participation ─────────────────
        **setup_v65,
        **participation_v65,
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
        "vol_penalty_note": vol_penalty_note,
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
        "tp5":             tp5,
        "target_rr":          target_rr,
        "risk_reward":     risk_reward,
        "risk_reward_tp2":    risk_reward_tp2,
        "risk_reward_target": risk_reward_target,
        "rr_tp1":             rr_levels.get("rr_tp1"),
        "rr_tp2":             rr_levels.get("rr_tp2"),
        "rr_tp3":             rr_levels.get("rr_tp3"),
        "rr_tp4":             rr_levels.get("rr_tp4"),
        "rr_tp5":             rr_levels.get("rr_tp5"),
        "tp_realism_note":    " ; ".join(realism_reasons) if realism_reasons else "TP8 réaliste",
        "rr_valid":        rr_valid,
        "max_leverage":    max_leverage,
        "confidence":      confidence,
        "duration_label":   duration_label,
        # ── V6.0 : champs futures ────────────────────────────────────────────
        "v6_accepted":       v6["v6_accepted"],
        "v6_veto_reasons":   v6["v6_veto_reasons"],
        "v6_score_final":    v6["v6_score_final"],
        "v6_score_futures":  v6["v6_score_futures"],
        "v6_futures_detail": v6["v6_futures_detail"],
        "v6_futures_raw":    v6.get("v6_futures_raw", {}),
        "v6_data_errors":    v6["v6_data_errors"],
        "executable_signal":  executable_signal,
        "taker_score":        taker_pts,
        "oi_score":           oi_pts,
        "funding_score":      funding_pts,
        "long_short_score":   long_short_pts,
        "futures_support":    futures_support,
        "risk_guard_reason":  risk_guard_reason,
        "decision_explain":   decision_explain,
        # ── Champs calibration v6.0.7 ───────────────────────────────────────
        "futures_zone":       futures_zone,
        "healthy_futures_confirmation": healthy_futures_confirmation,
        "futures_overheated": futures_overheated,
        "taker_not_confirmed": taker_not_confirmed,
        "calibration_flags":  calibration_flags,
        # ── Champs decision_engine v6.1 ─────────────────────────────────────
        "decision_version": v61_decision.get("decision_version", DECISION_VERSION),
        "healthy_futures_zone": v61_decision.get("healthy_futures_zone", False),
        "overheated_futures": v61_decision.get("overheated_futures", False),
        "late_entry_risk_v6_1": v61_decision.get("late_entry_risk_v6_1", False),
        "crowded_risk": v61_decision.get("crowded_risk", False),
        "watchlist_promotion_candidate": v61_decision.get("watchlist_promotion_candidate", False),
        "signal_downgrade_candidate": v61_decision.get("signal_downgrade_candidate", False),
        "final_decision_reason": decision_explain,
        "signal_quality_bucket": signal_quality_bucket,
        "regime_rule_applied": regime_rule_applied,
        "pr_threshold_used": pr_threshold_used,
        "btc_market_state":     btc_market_state,
        "btc_market_state_reason": market_details.get("btc_market_state_reason", ""),
        "telegram_rule_notes": telegram_rule_notes,
        "v652_actions": " | ".join(v652_context.get("actions", [])) if isinstance(v652_context, dict) else "",
        "v652_notes": v652_context.get("notes", "") if isinstance(v652_context, dict) else "",
        "v652_short_beta_ok": v652_context.get("short_momentum_beta_ok", False) if isinstance(v652_context, dict) else False,
        "confidence_cap_reason": confidence_cap_reason,
        "leverage_cap_reason": leverage_cap_reason,
    }


def decision_engine_v6_1(symbol, flag, direction, entry_type, global_score, confidence, max_leverage,
                         futures_zone, futures_overheated, taker_not_confirmed, taker_pts,
                         funding_signal, long_short_pts, futures_support, relative_vol,
                         late_entry_risk, late_entry_level, position_range, market_danger_level,
                         market_regime, rr_valid, data_source, decision_explain,
                         risk_guard_reason="aucun"):
    """
    Decision engine v6.1 : sépare la décision finale du score brut.

    Objectif :
    - promouvoir les WATCHLIST saines qui ressemblent aux meilleurs cas du tracker ;
    - dégrader les CANDIDAT trop tardifs / surchauffés / crowded ;
    - exposer des flags auditables dans Signals et WATCHLIST_LOG.

    Ne modifie pas les niveaux d'entrée, SL, TP, RR. Les caps confiance/levier
    restent gérés par les règles existantes après cette fonction.

    v6.2 — risk_guard_reason : si une règle de garde a explicitement forcé WATCHLIST
    (volume critique, short trop bas, anti-short BTC bullish, futures score insuffisant…),
    la promotion est bloquée — on ne peut pas annuler une décision explicite du moteur.
    """
    # ── v6.2 — Raisons bloquantes pour la promotion ──────────────────────────
    # Ces raisons indiquent que le moteur a délibérément dégradé le signal.
    # Une zone futures saine ne suffit pas à contrebalancer ces raisons.
    _HARD_GUARD_REASONS = {
        "volume critique",              # vol < 0.15 : jamais CANDIDAT, même si WATCHLIST autorisé
        "volume tres faible",           # vol < 0.30 : trop faible pour promouvoir
        "short trop proche du bas de range",
        "short contre BTC bullish",
        "futures score insuffisant",
        "OI fortement négatif non compensé",
        "futures overheated + position range élevée",
        # Note v6.3 : "volume critique" reste bloquant pour la PROMOTION même si
        # la règle 8 autorise désormais WATCHLIST (contexte sain). La condition
        # relative_vol >= V61_PROMOTE_MIN_VOLUME (0.50) bloque déjà en pratique,
        # mais on garde cette ligne comme filet de sécurité logique explicite.
    }
    _promotion_hard_blocked = (
        isinstance(risk_guard_reason, str) and
        any(r in risk_guard_reason for r in _HARD_GUARD_REASONS)
    )

    original_flag = flag
    reasons = []

    healthy_futures_zone = (futures_zone == "healthy")
    overheated_futures = bool(futures_overheated or futures_zone == "overheated")

    late_entry_risk_v6_1 = bool(
        (late_entry_risk is not None and late_entry_risk >= V61_LATE_ENTRY_RISK_MIN)
        or late_entry_level == "HIGH"
        or (
            entry_type == "MOMENTUM"
            and overheated_futures
            and position_range is not None
            and position_range >= FUTURES_LATE_POSITION_RANGE
        )
    )

    crowded_funding = (
        (direction == "LONG" and funding_signal == "longs crowded") or
        (direction == "SHORT" and funding_signal == "shorts crowded")
    )
    crowded_risk = bool(overheated_futures or crowded_funding or long_short_pts < 0)

    watchlist_promotion_candidate = bool(
        flag == "WATCHLIST"
        and direction in ("LONG", "SHORT")
        and rr_valid
        and healthy_futures_zone
        and not late_entry_risk_v6_1
        and not crowded_risk
        and not _promotion_hard_blocked        # v6.2 : ne pas annuler une garde explicite
        and market_danger_level != "HIGH"
        and data_source != "SPOT_FALLBACK"
        and relative_vol is not None
        and relative_vol >= V61_PROMOTE_MIN_VOLUME
        and (
            (direction == "LONG" and position_range <= V61_PROMOTE_LONG_MAX_POSITION) or
            (direction == "SHORT" and position_range >= V61_PROMOTE_SHORT_MIN_POSITION)
        )
    )

    signal_downgrade_candidate = bool(
        flag == "CANDIDAT"
        and (
            overheated_futures
            or late_entry_risk_v6_1
            or crowded_risk
            or (
                entry_type == "MOMENTUM"
                and taker_not_confirmed
                and (relative_vol < 0.50 or futures_support < 2)
            )
        )
    )

    # ── v6.3.3 — P4-V : promotion WATCHLIST→CANDIDAT suspendue ─────────────
    # Données tracker v6.3.2 : WR watchlist promues = 34% (22W/43L).
    # La promotion basée sur vol≥0.50 + futures healthy capturait des setups
    # déjà en fin de mouvement (volume fort corrèle avec LOSS dans les données).
    # watchlist_promotion_candidate reste calculé et loggué pour calibration future,
    # mais ne produit plus de promotion vers CANDIDAT.
    # Réactivation quand WR watchlist promues ≥ 60% sur ≥ 20 trades résolus.
    if watchlist_promotion_candidate:
        logger.info("V6.3.3 P4-V promotion SUSPENDUE %s (loggué uniquement)", symbol)
        # flag reste WATCHLIST

    elif signal_downgrade_candidate:
        flag = "WATCHLIST"
        confidence = min(confidence, 65)
        max_leverage = min(max_leverage, 3)
        if overheated_futures:
            reasons.append("DOWNGRADE : futures surchauffés")
        if late_entry_risk_v6_1:
            reasons.append("DOWNGRADE : risque entrée tardive")
        if crowded_risk:
            reasons.append("DOWNGRADE : risque crowded")
        if taker_not_confirmed and relative_vol < 0.50:
            reasons.append("DOWNGRADE : taker non confirmé avec volume faible")

    if not reasons:
        if decision_explain:
            final_decision_reason = decision_explain
        elif flag == "CANDIDAT":
            final_decision_reason = "CANDIDAT v6.1 : conditions décisionnelles acceptées."
        elif flag == "WATCHLIST":
            final_decision_reason = "WATCHLIST v6.1 : suivi sans exécution automatique."
        else:
            final_decision_reason = "REJET v6.1 : règles de garde ou score insuffisant."
    else:
        final_decision_reason = " | ".join(reasons)

    if flag != original_flag:
        decision_explain = final_decision_reason

    return {
        "flag": flag,
        "confidence": confidence,
        "max_leverage": max_leverage,
        "decision_explain": decision_explain,
        "decision_version": DECISION_VERSION,
        "healthy_futures_zone": healthy_futures_zone,
        "overheated_futures": overheated_futures,
        "late_entry_risk_v6_1": late_entry_risk_v6_1,
        "crowded_risk": crowded_risk,
        "watchlist_promotion_candidate": watchlist_promotion_candidate,
        "signal_downgrade_candidate": signal_downgrade_candidate,
        "final_decision_reason": final_decision_reason,
    }


def validate_decision_config():
    """Sanity check non bloquant de la configuration décisionnelle v6.4.4."""
    warnings = []
    if DECISION_VERSION != "v6.5.3":
        warnings.append(f"DECISION_VERSION inattendu: {DECISION_VERSION}")
    if not (LONG_PREMIUM_PR_DEFAULT <= LONG_PREMIUM_PR_BULL_SOFT <= LONG_PREMIUM_PR_BULL_IMPULSE):
        warnings.append("Seuils PR incohérents: DEFAULT <= BULL_SOFT <= BULL_IMPULSE attendu")
    if not (BTC_BEAR_CONT_VAR4H_MAX < 0 and BTC_BEAR_CONT_VAR2H_MAX < 0):
        warnings.append("Seuils BTC_BEAR_CONT doivent être négatifs")
    if "LONG_PREMIUM" not in TELEGRAM_ALLOWED_BUCKETS:
        warnings.append("LONG_PREMIUM doit rester autorisé Telegram")
    for bucket in (
        "LONG_EARLY_NEUTRAL_PREMIUM",
        "WATCHLIST_PREMIUM_SCORE_HIGH_EARLY",
        "WATCHLIST_PREMIUM_LONG_STRONG",
        "SHORT_MOMENTUM_CONTINUATION_PREMIUM",
        "WATCHLIST_SHORT_MOMENTUM_BEARISH",
        "WATCHLIST_LONG_STRONG_DIAGNOSTIC",
        "REJECT_LONG_LATE_MOMENTUM",
    ):
        if bucket not in WATCHLIST_BUCKET_PRIORITY:
            warnings.append(f"{bucket} absent de WATCHLIST_BUCKET_PRIORITY")
    extra_telegram = TELEGRAM_ALLOWED_BUCKETS - {"LONG_PREMIUM", "SHORT_MOMENTUM_CONTINUATION_PREMIUM"}
    if extra_telegram:
        warnings.append(f"Buckets Telegram non prévus: {sorted(extra_telegram)}")

    if warnings:
        for w in warnings:
            logger.warning("validate_decision_config: %s", w)
    else:
        logger.info("validate_decision_config OK — %s", DECISION_VERSION)
    return warnings

validate_decision_config()

# ─── ENDPOINT /set_cooldown ───────────────────────────────────────────────────

@app.route("/set_cooldown", methods=["POST"])
def set_cooldown_endpoint():
    try:
        data      = request.get_json(silent=True) or {}
        symbols   = data.get("symbols", [])
        setup_ids = data.get("setup_ids", [])   # peut contenir des setup_id ou des dedup_key

        if not symbols or not isinstance(symbols, list):
            return jsonify({"error": "symbols must be a non-empty list"}), 400
        if setup_ids is None:
            setup_ids = []
        if not isinstance(setup_ids, list):
            return jsonify({"error": "setup_ids must be a list"}), 400

        # Cooldown symbole classique.
        set_cooldown_symbols(symbols)

        # Anti-doublon persistant : stocke les dedup_key/setup_id dans le même JSON,
        # mais dans un bloc séparé des timestamps symboles.
        if setup_ids:
            with _COOLDOWN_LOCK:
                cooldown = load_cooldown()
                now = time.time()
                existing = cooldown.get("setup_ids_ts", {})
                if not isinstance(existing, dict):
                    existing = {}

                for sid in setup_ids:
                    sid = str(sid or "").strip().upper()
                    if sid:
                        existing[sid] = now

                # Purger les clés anti-doublon de plus de 4h.
                existing = {
                    k: v for k, v in existing.items()
                    if isinstance(v, (int, float)) and now - v < COOLDOWN_SECONDS
                }
                cooldown["setup_ids"] = list(existing.keys())
                cooldown["setup_ids_ts"] = existing
                save_cooldown(cooldown)

        cooldown = load_cooldown()
        now = time.time()
        symbol_status = {
            k: round((COOLDOWN_SECONDS - (now - v)) / 3600, 1)
            for k, v in cooldown.items()
            if k not in ("setup_ids", "setup_ids_ts") and isinstance(v, (int, float))
        }

        return jsonify({
            "status":    "ok",
            "symbols":   [normalize_symbol(s) for s in symbols],
            "setup_ids": [str(s).strip().upper() for s in setup_ids if str(s).strip()],
            "cooldown":  symbol_status,
            "setup_count": len(cooldown.get("setup_ids", [])) if isinstance(cooldown.get("setup_ids", []), list) else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── ENDPOINT /cooldown_status ────────────────────────────────────────────────

@app.route("/cooldown_status", methods=["GET"])
def cooldown_status():
    try:
        cooldown = load_cooldown()
        status = {}
        now = time.time()

        for symbol, ts in cooldown.items():
            if symbol in ("setup_ids", "setup_ids_ts"):
                continue
            if not isinstance(ts, (int, float)):
                continue

            remaining = max(0, COOLDOWN_SECONDS - (now - ts))
            status[symbol] = {
                "remaining_hours": round(remaining / 3600, 1),
                "expires_in_min":  round(remaining / 60)
            }

        setup_ids = cooldown.get("setup_ids", [])
        if not isinstance(setup_ids, list):
            setup_ids = []

        return jsonify({
            "active_cooldowns": status,
            "active_setup_ids": setup_ids,
            "count": len(status),
            "setup_count": len(setup_ids)
        })
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
                "signals": [],
                "market_regime": "unknown",
                "data_source": "UNAVAILABLE",
                "error": "All providers unreachable: Binance Futures, Bybit Futures, Binance Vision Spot, Binance Spot"
            })

        # ── Market regime BTC enrichi v4.7 ──────────────────────────────────
        # v6.4.2.2 : le scoring reste basé sur BTC 1h comme en v6.4.1.
        # Les champs 15m/30m sont calculés séparément en 5m et ajoutés au JSON
        # uniquement pour la corrélation WATCHLIST_LOG.
        btc_klines, btc_data_source = get_klines("BTCUSDT", limit=80)
        market_regime, market_details = detect_market_regime(btc_klines, return_details=True)

        btc_klines_5m, btc_5m_data_source = get_klines("BTCUSDT", interval="5m", limit=200)
        market_details.update(compute_btc_micro_context(btc_klines_5m))
        market_details["market_regime_btc"] = market_regime
        btc_market_state, btc_market_state_reason = compute_btc_market_state_details(market_details)
        market_details["btc_market_state"] = btc_market_state
        market_details["btc_market_state_reason"] = btc_market_state_reason
        market_details.update(compute_btc_context_v65(btc_klines, market_details))

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

        data_sources_used = {data_source_batch, btc_data_source, btc_5m_data_source}
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

        # ── V6.0.6g — Anti-doublon dedup_key ─────────────────────────────────
        # Double protection : verrou mémoire (intra-session, 60 min)
        # + cooldown fichier (inter-session, via /set_cooldown Make).
        # La clé est calculée ici, car setup_id n'existe qu'après build_signal_record.
        #
        # v6.2 — FIX bug ADAUSDT : 3 signaux identiques émis en 33 secondes.
        # Cause : mark_setup_id_emitted() n'était appelé que dans /set_cooldown
        # (APRÈS l'envoi Telegram), donc le verrou mémoire était vide pendant
        # tout le run courant. Un second run déclenchant dans la même minute
        # (Make retry ou double-trigger) pouvait émettre le même signal.
        # Fix : on appelle mark_setup_id_emitted(dedup_key) ici, AVANT d'ajouter
        # le signal à candidats_dedup, pour que tout run parallèle ou immédiatement
        # suivant soit bloqué dès le filtre is_setup_id_blocked().
        cooldown_data = load_cooldown()
        emitted_setup_ids = set(cooldown_data.get("setup_ids", []))
        candidats_dedup = []
        skipped_dedup = []

        for r in candidats:
            dedup_key = build_dedup_key(r)
            if not dedup_key:
                candidats_dedup.append(r)
                continue

            if dedup_key in emitted_setup_ids or is_setup_id_blocked(dedup_key):
                skipped_dedup.append(dedup_key)
                continue

            # v6.2 FIX : marquer immédiatement en mémoire pour bloquer tout
            # run concurrent ou immédiatement suivant — sans attendre /set_cooldown.
            mark_setup_id_emitted(dedup_key)

            r["dedup_key"] = dedup_key
            candidats_dedup.append(r)

        if skipped_dedup:
            logger.info(
                "V6.0.6g anti-doublon: %d signaux filtrés (dedup_key déjà émise): %s",
                len(skipped_dedup),
                skipped_dedup
            )
        candidats = candidats_dedup

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

        # Python plafonne le nombre de signaux (régime + source). GPT recopie.
        # Les fallbacks watchlist/short_watch sont déjà à 1, le cap ne les change pas.
        candidats = cap_signal_count(candidats, market_regime, data_source_run)

        # ── top_watchlist / top_rejected — visibilité même sur SKIP ──────────
        # P2.5 : les lignes WATCHLIST / REJECT doivent avoir la même structure
        # que les vrais signaux, sinon Make ne peut pas alimenter proprement
        # WATCHLIST_LOG / REJECT_LOG.
        emitted_ts = int(time.time())

        # v6.4.4 — Tri top_watchlist par priorité bucket d'abord, puis score.
        # Les WATCHLIST_PREMIUM doivent apparaître en tête même à score modéré,
        # sinon les nouveaux buckets (LONG_EARLY_NEUTRAL_PREMIUM 98% WR, etc.)
        # disparaissent derrière des STANDARD à score plus élevé.
        # v6.4.4-final-clean — priorité bucket globale validée par validate_decision_config().
        _all_watchlist = sorted(
            [r for r in results if r.get("flag") == "WATCHLIST"],
            key=lambda x: (
                WATCHLIST_BUCKET_PRIORITY.get(x.get("signal_quality_bucket", "STANDARD"), 0),
                x.get("score", 0)
            ),
            reverse=True
        )
        _all_rejected = sorted(
            [r for r in results if r.get("flag") == "REJET"],
            key=lambda x: x.get("score", 0), reverse=True
        )

        def _debug_row(r, signal_type):
            """
            Structure complète pour WATCHLIST_LOG / REJECT_LOG.
            On réutilise build_signal_record() pour générer :
            signal_id, setup_id, timestamp, entry/SL/TP, RR, deadlines, source, etc.
            """
            rec = build_signal_record(r, market_regime, data_source_run, emitted_ts, market_details)
            rec["signal_type"] = signal_type
            rec["outcome"] = "DIAGNOSTIC"
            rec["evaluation_note"] = "Non envoyé Telegram — diagnostic uniquement"
            return rec

        top_watchlist = [
            _debug_row(r, "WATCHLIST")
            for r in _all_watchlist[:5]
        ]
        top_rejected = [
            _debug_row(r, "REJECT")
            for r in _all_rejected[:8]
        ]

        if not candidats:
            return jsonify({
                "text":             "SKIP",
                "count":            0,
                "telegram_count":   0,
                "signals":          [],
                "telegram_signals": [],
                "top_watchlist":    top_watchlist,
                "top_rejected":     top_rejected,
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
            f"market_regime_btc: {market_regime} | btc_market_state: {market_details.get('btc_market_state', 'BTC_NEUTRAL_COMPRESS')} | "
            f"btc_phase: {market_details.get('btc_phase')} | "
            f"btc_context_bias: {market_details.get('btc_context_bias')} | "
            f"btc_context_score: {market_details.get('btc_context_score')} | "
            f"btc_impulse_age: {market_details.get('btc_impulse_age')} | "
            f"btc_last_pivot_type: {market_details.get('btc_last_pivot_type')} | "
            f"btc_last_pivot_age: {market_details.get('btc_last_pivot_age')} | "
            f"btc_last_pivot_distance_pct: {market_details.get('btc_last_pivot_distance_pct')} | "
            f"btc_last_pivot_method: {market_details.get('btc_last_pivot_method')} | "
            f"btc_volatility_regime: {market_details.get('btc_volatility_regime')}\n"
            f"market_danger_level: {market_details.get('market_danger_level')} | "
            f"market_danger_score: {market_details.get('market_danger_score')} | "
            f"btc_rsi: {market_details.get('btc_rsi')} | "
            f"btc_var_15m: {market_details.get('btc_variation_15m')} | "
            f"btc_var_30m: {market_details.get('btc_variation_30m')} | "
            f"btc_var_2h: {market_details.get('btc_variation_2h')} | "
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
                f"vol_penalty: {r.get('vol_penalty_note', '')} | "
                f"market_regime: {r['market_regime']} | market_danger_level: {r.get('market_danger_level')} | "
                f"data_source: {r.get('data_source', data_source_run)} | "
                f"source_quality: {r.get('source_quality', source_quality_label(r.get('data_source', data_source_run)))} | "
                f"weakness_notes: {build_weakness_notes(r)} | "
                f"funding_rate: {r['funding_rate']} | funding_signal: {r['funding_signal']} | "
                f"derivatives_bias: {r['derivatives_bias']} | derivatives_note: {r['derivatives_note']} | "
                f"short_watch: {r.get('short_watch')} | short_watch_reasons: {','.join(r.get('short_watch_reasons', []))} | "
                f"entry_low: {r['entry_low']} | entry_high: {r['entry_high']} | entry_avg: {r['entry_avg']} | "
                f"stop_loss: {r['stop_loss']} | tp1: {r['tp1']} | tp2: {r['tp2']} | tp3: {r['tp3']} | tp4: {r['tp4']} | tp5: {r.get('tp5')} | "
                f"target_rr: {r.get('target_rr')} | tp_realism: {r.get('tp_realism_note')} | "
                f"risk_reward_tp2: {r.get('risk_reward_tp2')} | risk_reward_target: {r.get('risk_reward_target')} | rr_valid: {r['rr_valid']} | "
                f"rr_tp1: {r.get('rr_tp1')} | rr_tp2: {r.get('rr_tp2')} | rr_tp3: {r.get('rr_tp3')} | rr_tp4: {r.get('rr_tp4')} | rr_tp5: {r.get('rr_tp5')} | "
                f"setup_family: {r.get('setup_family')} | setup_maturity: {r.get('setup_maturity')} | setup_context_alignment: {r.get('setup_context_alignment')} | "
                f"btc_phase: {r.get('btc_phase')} | btc_context_bias: {r.get('btc_context_bias')} | btc_context_score: {r.get('btc_context_score')} | "
                f"btc_impulse_age: {r.get('btc_impulse_age')} | btc_last_pivot_type: {r.get('btc_last_pivot_type')} | btc_last_pivot_age: {r.get('btc_last_pivot_age')} | btc_last_pivot_distance_pct: {r.get('btc_last_pivot_distance_pct')} | "
                f"volume_regime: {r.get('volume_regime')} | volume_quality: {r.get('volume_quality')} | derivatives_alignment: {r.get('derivatives_alignment')} | "
                f"crowding_state: {r.get('crowding_state')} | participation_score: {r.get('participation_score')} | participation_warning: {r.get('participation_warning')} | "
                f"max_leverage: {r['max_leverage']} | confidence: {r['confidence']} | "
                f"duration_label: {r['duration_label']} | "
                f"v6_score_futures: {r.get('v6_score_futures')} | "
                f"futures_zone: {r.get('futures_zone')} | "
                f"futures_overheated: {r.get('futures_overheated')} | "
                f"healthy_futures_confirmation: {r.get('healthy_futures_confirmation')} | "
                f"taker_not_confirmed: {r.get('taker_not_confirmed')} | "
                f"calibration_flags: {','.join(r.get('calibration_flags', [])) if isinstance(r.get('calibration_flags'), list) else r.get('calibration_flags', '')} | "
                f"decision_version: {r.get('decision_version')} | "
                f"healthy_futures_zone: {r.get('healthy_futures_zone')} | "
                f"overheated_futures: {r.get('overheated_futures')} | "
                f"late_entry_risk_v6_1: {r.get('late_entry_risk_v6_1')} | "
                f"crowded_risk: {r.get('crowded_risk')} | "
                f"watchlist_promotion_candidate: {r.get('watchlist_promotion_candidate')} | "
                f"signal_downgrade_candidate: {r.get('signal_downgrade_candidate')} | "
                f"final_decision_reason: {r.get('final_decision_reason')} | "
                f"signal_quality_bucket: {r.get('signal_quality_bucket')} | "
                f"btc_market_state: {r.get('btc_market_state', 'BTC_NEUTRAL_COMPRESS')} | "
                f"btc_market_state_reason: {r.get('btc_market_state_reason', '')} | "
                f"regime_rule_applied: {r.get('regime_rule_applied')} | "
                f"pr_threshold_used: {r.get('pr_threshold_used')} | "
                f"telegram_rule_notes: {r.get('telegram_rule_notes')} | "
                f"taker_buy_ratio: {r.get('v6_futures_raw',{}).get('taker_buy_ratio')} | "
                f"taker_sell_ratio: {r.get('v6_futures_raw',{}).get('taker_sell_ratio')} | "
                f"long_liq_usdt: {r.get('v6_futures_raw',{}).get('long_liq_usdt')} | "
                f"short_liq_usdt: {r.get('v6_futures_raw',{}).get('short_liq_usdt')} | "
                f"total_liq_usdt: {r.get('v6_futures_raw',{}).get('total_liq_usdt')} | "
                f"liq_imbalance: {r.get('v6_futures_raw',{}).get('liq_imbalance')} | "
                f"v6_futures_detail: oi={r.get('v6_futures_detail',{}).get('oi',0):+d} "
                f"taker={r.get('v6_futures_detail',{}).get('taker',0):+d} "
                f"funding={r.get('v6_futures_detail',{}).get('funding',0):+d} "
                f"ls={r.get('v6_futures_detail',{}).get('long_short',0):+d}"
            )

        # emitted_ts déjà calculé plus haut pour garder une même horloge run.
        for r in candidats:
            log_signal(r)

        signals = [
            build_signal_record(r, market_regime, data_source_run, emitted_ts, market_details)
            for r in candidats
        ]

        # telegram_signals : uniquement les signaux exécutables (executable_signal=True)
        # Make doit utiliser telegram_count pour décider d'envoyer sur Telegram
        # signals conservé pour Google Sheet / tracking complet
        telegram_signals = [s for s in signals if s.get("executable_signal")]

        # Marquer les dedup_key des signaux exécutables comme émises (anti-doublon mémoire)
        for s in telegram_signals:
            dedup_key = str(s.get("dedup_key") or "").strip().upper()
            if dedup_key:
                mark_setup_id_emitted(dedup_key)
                logger.info("V6.0.6g dedup_key marquée émise: %s", dedup_key)

        return jsonify({
            "text":             "\n".join(lines),
            "count":            len(candidats),
            "telegram_count":   len(telegram_signals),
            "signals":          signals,
            "telegram_signals": telegram_signals,
            "top_watchlist":    top_watchlist,
            "top_rejected":     top_rejected,
            "market_regime":    market_regime,
            "market_danger":    market_details,
            "data_source":      data_source_run,
            "source_quality":   source_quality_run,
            "universe_size":    len(tickers_data),
            "analyzed_count":   len(top_candidates),
            "cooldown_skipped": cooldown_skipped
        })

    except Exception as e:
        return jsonify({"error": str(e), "text": "SKIP", "count": 0}), 500


# ─── EVALUATE SIGNALS ─────────────────────────────────────────────────────────
# Appelé par Google Apps Script avec les lignes DIAGNOSTIC + OPEN de WATCHLIST_LOG.
# Rejoue les klines 5min pour détecter :
#   1. Fill dans la zone d'entrée avant fill_deadline.
#   2. Après fill, TP1 ou SL touché en premier avant resolve_deadline.
#
# v6.3.7 — FIX évaluation 72h + debug NO_FILL :
# - pagination klines 5m jusqu'à min(now, resolve_deadline), au lieu d'une seule
#   page limitée à ~16h/24h ;
# - prise en compte de filled_at déjà présent pour les lignes OPEN ;
# - si filled_at existe, on ne recherche plus le fill, on vérifie directement SL/TP ;
# - fill_time_minutes négatif est neutralisé (champ vide + note timezone), pour
#   éviter de polluer les statistiques avec des durées impossibles ;
# - notes NO_FILL enrichies avec debug fenêtre de fill (max_high/min_low/first/last).

EVAL_KLINE_INTERVAL = "5m"
EVAL_KLINE_SECONDS  = 5 * 60
EVAL_KLINE_PAGE_LIMIT = int(os.environ.get("EVAL_KLINE_PAGE_LIMIT", "1000"))
EVAL_MAX_PAGES = int(os.environ.get("EVAL_MAX_PAGES", "12"))  # 12*1000*5m > 72h Binance ; Bybit capé à 200/page


def _parse_eval_ts(value, fallback=None):
    """Parse ISO, timestamp Unix ou serial Excel/Google Sheets en timestamp Unix UTC."""
    if value is None or value == "":
        return fallback

    if isinstance(value, (int, float)):
        # timestamp Unix probable
        if value > 1_000_000_000:
            return int(value)
        # serial Excel / Google Sheets
        if 40000 < value < 70000:
            import datetime as _dt
            excel_epoch = _dt.datetime(1899, 12, 30, tzinfo=timezone.utc)
            return int((excel_epoch + _dt.timedelta(days=float(value))).timestamp())
        return fallback

    s = str(value).strip()
    if not s:
        return fallback

    # Serial Excel sous forme texte, avec virgule ou point.
    s_num = s.replace(",", ".")
    try:
        serial = float(s_num)
        if 40000 < serial < 70000:
            import datetime as _dt
            excel_epoch = _dt.datetime(1899, 12, 30, tzinfo=timezone.utc)
            return int((excel_epoch + _dt.timedelta(days=serial)).timestamp())
    except Exception:
        pass

    # ISO : tolérant Z, espace au lieu de T, naïf traité UTC.
    try:
        iso = s.replace(" ", "T").replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return fallback


def _safe_fill_time_minutes(filled_at_ts, emit_ts):
    """
    Calcule fill_time_minutes sans polluer la Sheet avec des valeurs négatives.

    Les valeurs négatives viennent d'un décalage timezone / Google Sheet :
    l'outcome WIN/LOSS peut être correct, mais le timestamp d'émission peut être
    interprété après filled_at. Dans ce cas, on renvoie une valeur vide et une
    note de prudence au lieu d'écrire -417, -371, etc.
    """
    try:
        if filled_at_ts is None or emit_ts is None:
            return "", ""
        delta_min = round((int(filled_at_ts) - int(emit_ts)) / 60, 1)
        if delta_min < -1:
            return "", " | fill_time non fiable — timezone"
        if delta_min < 0:
            return 0, ""
        return delta_min, ""
    except Exception:
        return "", " | fill_time non fiable — erreur calcul"


def _emit_ts_from_signal_id(signal_id):
    """
    v6.4.1 — Extrait uniquement un vrai timestamp Unix depuis signal_id.

    Accepte :
    - Unix seconds plausible : 1700000000 → 2100000000
    - Unix milliseconds plausible : 1700000000000 → 2100000000000

    Rejette :
    - YYYYMMDDHHMMSS, ex: 20260610103415
      car ce n'est pas un timestamp Unix.
    """
    try:
        tail = str(signal_id or "").rsplit("-", 1)[-1].strip()

        if not tail:
            return None

        # Rejeter explicitement les formats YYYYMMDDHHMMSS
        if len(tail) == 14 and tail.startswith("20"):
            return None

        ts = int(float(tail))

        # Unix seconds plausible : environ 2023 → 2036
        if 1_700_000_000 <= ts <= 2_100_000_000:
            return ts

        # Unix milliseconds plausible
        if 1_700_000_000_000 <= ts <= 2_100_000_000_000:
            return ts // 1000

    except Exception:
        pass

    return None


def _fmt_eval_ts(ts):
    try:
        if ts is None:
            return ""
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return str(ts)


def _nofill_debug_note(klines, emit_ts, fill_ts, entry_low, entry_high, direction):
    """
    Diagnostic compact ajouté aux NO_FILL.
    Objectif : vérifier si Python a regardé la bonne fenêtre et si les highs/lows
    reçus par API auraient dû déclencher le fill.
    """
    try:
        fill_bars = [
            b for b in (klines or [])
            if int(b.get("ts", 0)) <= int(fill_ts)
        ]
        if not fill_bars:
            return (
                f" | debug: emit={_fmt_eval_ts(emit_ts)} fill_dl={_fmt_eval_ts(fill_ts)} "
                f"bars_fill=0 entry=[{entry_low}-{entry_high}] dir={direction}"
            )

        highs = [float(b["high"]) for b in fill_bars]
        lows = [float(b["low"]) for b in fill_bars]
        first_ts = int(fill_bars[0]["ts"])
        last_ts = int(fill_bars[-1]["ts"])
        max_high = max(highs)
        min_low = min(lows)

        would_fill = any(
            float(b["low"]) <= float(entry_high) and float(b["high"]) >= float(entry_low)
            for b in fill_bars
        )

        return (
            f" | debug: emit={_fmt_eval_ts(emit_ts)} fill_dl={_fmt_eval_ts(fill_ts)} "
            f"first={_fmt_eval_ts(first_ts)} last={_fmt_eval_ts(last_ts)} "
            f"bars_fill={len(fill_bars)} max_high={max_high:.8g} min_low={min_low:.8g} "
            f"entry=[{entry_low}-{entry_high}] dir={direction} would_fill={would_fill}"
        )
    except Exception as e:
        return f" | debug_no_fill_error={e}"




def _fetch_klines_page(symbol, start_ts, limit):
    """
    Récupère une page de klines 5m depuis start_ts.
    Ordre fournisseurs : Binance Futures -> Bybit Linear -> Binance Vision Spot.
    Retour : liste de dicts triés croissant.
    """
    page_limit = max(1, min(int(limit), 1500))

    # Binance Futures
    if BINANCE_ENABLED:
        url = (f"https://fapi.binance.com/fapi/v1/klines"
               f"?symbol={symbol}&interval={EVAL_KLINE_INTERVAL}"
               f"&startTime={int(start_ts)*1000}&limit={page_limit}")
        raw = fetch_binance(url)
        if raw:
            return sorted([
                {"ts": int(k[0]) // 1000, "open": float(k[1]),
                 "high": float(k[2]), "low": float(k[3]), "close": float(k[4])}
                for k in raw
            ], key=lambda x: x["ts"])

    # Bybit Linear : cap 200/page côté API.
    try:
        bybit_limit = min(page_limit, 200)
        url = (f"https://api.bybit.com/v5/market/kline"
               f"?category=linear&symbol={symbol}&interval=5"
               f"&start={int(start_ts)*1000}&limit={bybit_limit}")
        raw = fetch_binance(url)
        if raw and raw.get("result", {}).get("list"):
            return sorted([
                {"ts": int(k[0]) // 1000, "open": float(k[1]),
                 "high": float(k[2]), "low": float(k[3]), "close": float(k[4])}
                for k in raw["result"]["list"]
            ], key=lambda x: x["ts"])
    except Exception as e:
        logger.warning("eval klines bybit page %s: %s", symbol, e)

    # Binance Vision Spot fallback.
    url = (f"https://data-api.binance.vision/api/v3/klines"
           f"?symbol={symbol}&interval={EVAL_KLINE_INTERVAL}"
           f"&startTime={int(start_ts)*1000}&limit={page_limit}")
    raw = fetch_binance(url)
    if raw:
        return sorted([
            {"ts": int(k[0]) // 1000, "open": float(k[1]),
             "high": float(k[2]), "low": float(k[3]), "close": float(k[4])}
            for k in raw
        ], key=lambda x: x["ts"])

    return []


def _get_klines_window(symbol, start_ts, end_ts):
    """
    Paginer les klines 5m de start_ts jusqu'à end_ts inclus.
    Déduplique par timestamp et protège contre les boucles de provider.
    """
    start_ts = int(start_ts)
    end_ts = int(end_ts)
    if end_ts < start_ts:
        end_ts = start_ts

    out_by_ts = {}
    cursor = start_ts
    pages = 0

    while cursor <= end_ts + EVAL_KLINE_SECONDS and pages < EVAL_MAX_PAGES:
        pages += 1
        page = _fetch_klines_page(symbol, cursor, EVAL_KLINE_PAGE_LIMIT)
        if not page:
            break

        last_ts = None
        for bar in page:
            ts = int(bar["ts"])
            last_ts = ts if last_ts is None else max(last_ts, ts)
            if start_ts <= ts <= end_ts:
                out_by_ts[ts] = bar

        if last_ts is None or last_ts <= cursor:
            break
        if last_ts >= end_ts:
            break

        cursor = last_ts + EVAL_KLINE_SECONDS

    return [out_by_ts[k] for k in sorted(out_by_ts.keys())]


def _evaluate_one(sig):
    """
    Évalue un signal DIAGNOSTIC ou OPEN unique.
    Retourne un dict avec signal_id + champs de mise à jour.
    """
    signal_id = str(sig.get("signal_id", "") or "").strip()
    symbol = normalize_symbol(sig.get("symbol", ""))
    direction = str(sig.get("direction", "LONG") or "LONG").strip().upper()

    def _f(key, default=0.0):
        v = sig.get(key, default)
        try:
            return float(str(v).replace(",", "."))
        except (TypeError, ValueError):
            return default

    def _s(key, default=""):
        v = sig.get(key, default)
        return str(v).strip() if v is not None else default

    entry_low = _f("entry_low")
    entry_high = _f("entry_high")
    stop_loss = _f("stop_loss")
    tp1 = _f("tp1")

    now_ts = int(time.time())
    timestamp_str = _s("timestamp")
    fill_dl = _s("fill_deadline")
    resolve_dl = _s("resolve_deadline")
    filled_at_existing = _s("filled_at")

    # ── v6.4.0 — emit_ts : signal_id > timestamp_sheet > fill_deadline ──────
    #
    # Ordre de priorité :
    #   1. signal_id epoch Unix  — généré par Python, jamais retouché par Sheets/Apps Script
    #   2. timestamp_sheet       — fallback si signal_id sans epoch
    #   3. fill_deadline - 4h    — fallback de secours si timestamp incohérent (v6.3.9)
    #
    # Le timestamp Google Sheets subit un décalage +7h observé (double-conversion
    # timezone dans la chaîne Make → Sheets → Apps Script → JSON). Le Unix timestamp
    # dans signal_id contourne complètement cette chaîne.

    emit_from_id    = _emit_ts_from_signal_id(signal_id)
    emit_from_sheet = _parse_eval_ts(timestamp_str, now_ts - 3600)

    logger.info("eval %s: timestamp_raw=%r emit_from_id=%s emit_from_sheet=%s",
                signal_id, timestamp_str,
                _fmt_eval_ts(emit_from_id), _fmt_eval_ts(emit_from_sheet))

    if emit_from_id:
        emit_ts     = emit_from_id
        emit_source = "signal_id"
    else:
        emit_ts     = emit_from_sheet
        emit_source = "timestamp_sheet"

    fill_dl_received = _parse_eval_ts(fill_dl, None)
    if fill_dl_received:
        ecart = abs(fill_dl_received - (emit_ts + FILL_WINDOW_SECONDS))
        if not emit_from_id and ecart > 15 * 60:
            # Correction v6.3.9 : timestamp_sheet incohérent, reconstruire depuis fill_deadline
            emit_ts_corrige = fill_dl_received - FILL_WINDOW_SECONDS
            logger.warning(
                "eval %s: emit_ts incohérent (source=%s ecart=%ds) — "
                "correction: %s → %s (fill_deadline rebuild)",
                signal_id, emit_source, ecart,
                _fmt_eval_ts(emit_ts), _fmt_eval_ts(emit_ts_corrige)
            )
            emit_ts     = emit_ts_corrige
            emit_source = "fill_deadline_rebuild"

    fill_ts    = emit_ts + FILL_WINDOW_SECONDS
    resolve_ts = emit_ts + RESOLVE_WINDOW_SECONDS
    existing_filled_ts = _parse_eval_ts(filled_at_existing, None)

    if not symbol or direction not in ("LONG", "SHORT"):
        return {"signal_id": signal_id, "outcome": "OPEN",
                "evaluation_note": "signal invalide pour évaluation",
                "bars_checked": 0, "filled_at": filled_at_existing,
                "closed_at": "", "exit_price": ""}

    # Fenêtre à charger : jusqu'à maintenant ou resolve_deadline.
    eval_end_ts = min(now_ts, resolve_ts)

    # Pour une ligne déjà OPEN avec filled_at, on démarre au fill existant.
    # Pour une ligne DIAGNOSTIC/non remplie, on démarre à timestamp.
    query_start_ts = existing_filled_ts if existing_filled_ts else emit_ts
    klines = _get_klines_window(symbol, query_start_ts, eval_end_ts)

    if not klines:
        return {"signal_id": signal_id, "outcome": "OPEN",
                "evaluation_note": (
                    f"klines indisponibles, nouvelle tentative plus tard"
                    f" | debug: emit={_fmt_eval_ts(emit_ts)}"
                    f" fill_dl={_fmt_eval_ts(fill_ts)}"
                    f" now={_fmt_eval_ts(now_ts)}"
                    f" query_start={_fmt_eval_ts(query_start_ts)}"
                    f" eval_end={_fmt_eval_ts(eval_end_ts)}"
                ),
                "bars_checked": 0,
                "filled_at": filled_at_existing if existing_filled_ts else "",
                "closed_at": "", "exit_price": ""}

    bars_checked = len(klines)
    fill_price = round((entry_low + entry_high) / 2, 8)

    # ── Phase 1 : fill ──────────────────────────────────────────────────────
    if existing_filled_ts:
        filled_at_ts = existing_filled_ts
        filled_at_iso = datetime.fromtimestamp(filled_at_ts, tz=timezone.utc).isoformat()
        fill_time_min, fill_time_note = _safe_fill_time_minutes(filled_at_ts, emit_ts)
        post_fill = [bar for bar in klines if int(bar["ts"]) > filled_at_ts]
    else:
        filled_at_ts = None
        fill_bar_idx = None

        for i, bar in enumerate(klines):
            if bar["ts"] > fill_ts:
                break
            if bar["low"] <= entry_high and bar["high"] >= entry_low:
                filled_at_ts = int(bar["ts"])
                fill_bar_idx = i
                break

        if filled_at_ts is None and now_ts >= fill_ts:
            return {
                "signal_id": signal_id,
                "outcome": "NO_FILL",
                "filled_at": "",
                "closed_at": "",
                "exit_price": "",
                "bars_checked": bars_checked,
                "evaluation_note": (
                    f"zone [{entry_low}–{entry_high}] non touchée en 4h"
                    + _nofill_debug_note(klines, emit_ts, fill_ts, entry_low, entry_high, direction)
                ),
                "fill_time_minutes": ""
            }

        if filled_at_ts is None:
            return {
                "signal_id": signal_id,
                "outcome": "OPEN",
                "filled_at": "",
                "closed_at": "",
                "exit_price": "",
                "bars_checked": bars_checked,
                "evaluation_note": (
                    f"en attente de fill"
                    + _nofill_debug_note(klines, emit_ts, fill_ts, entry_low, entry_high, direction)
                    + f" | now={_fmt_eval_ts(now_ts)} fill_dl={_fmt_eval_ts(fill_ts)}"
                    + f" now<fill_ts={'oui' if now_ts < fill_ts else 'non'}"
                    + f" ts_raw={repr(timestamp_str)}"
                )
            }

        filled_at_iso = datetime.fromtimestamp(filled_at_ts, tz=timezone.utc).isoformat()
        fill_time_min, fill_time_note = _safe_fill_time_minutes(filled_at_ts, emit_ts)
        post_fill = klines[fill_bar_idx + 1:]

    # ── Phase 2 : SL / TP1 après fill ───────────────────────────────────────
    prev_close = fill_price
    for bar in post_fill:
        ts = int(bar["ts"])
        if ts > resolve_ts:
            break

        bar_hits_sl = (direction == "LONG" and bar["low"] <= stop_loss) or \
                      (direction == "SHORT" and bar["high"] >= stop_loss)
        bar_hits_tp = (direction == "LONG" and bar["high"] >= tp1) or \
                      (direction == "SHORT" and bar["low"] <= tp1)

        if bar_hits_sl and bar_hits_tp:
            dist_to_sl = abs(prev_close - stop_loss)
            dist_to_tp = abs(prev_close - tp1)
            outcome = "WIN" if dist_to_tp < dist_to_sl else "LOSS"
            exit_p = tp1 if outcome == "WIN" else stop_loss
            return {
                "signal_id": signal_id,
                "outcome": outcome,
                "filled_at": filled_at_iso,
                "closed_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "exit_price": exit_p,
                "bars_checked": bars_checked,
                "evaluation_note": f"SL et TP1 dans même bougie — intra-bar bias résolu par proximité. fill={fill_price}{fill_time_note}",
                "fill_time_minutes": fill_time_min
            }

        if bar_hits_tp:
            return {
                "signal_id": signal_id,
                "outcome": "WIN",
                "filled_at": filled_at_iso,
                "closed_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "exit_price": tp1,
                "bars_checked": bars_checked,
                "evaluation_note": f"TP1 touché. fill={fill_price}{fill_time_note}",
                "fill_time_minutes": fill_time_min
            }

        if bar_hits_sl:
            return {
                "signal_id": signal_id,
                "outcome": "LOSS",
                "filled_at": filled_at_iso,
                "closed_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "exit_price": stop_loss,
                "bars_checked": bars_checked,
                "evaluation_note": f"SL touché. fill={fill_price}{fill_time_note}",
                "fill_time_minutes": fill_time_min
            }

        prev_close = bar["close"]

    if now_ts >= resolve_ts:
        return {
            "signal_id": signal_id,
            "outcome": "EXPIRED",
            "filled_at": filled_at_iso,
            "closed_at": "",
            "exit_price": "",
            "bars_checked": bars_checked,
            "evaluation_note": f"72h écoulées sans SL ni TP1. fill={fill_price}{fill_time_note}",
            "fill_time_minutes": fill_time_min
        }

    return {
        "signal_id": signal_id,
        "outcome": "OPEN",
        "filled_at": filled_at_iso,
        "closed_at": "",
        "exit_price": "",
        "bars_checked": bars_checked,
        "evaluation_note": f"rempli à {fill_price}, attente SL/TP{fill_time_note}",
        "fill_time_minutes": fill_time_min
    }


@app.route("/evaluate_signals", methods=["POST"])
def evaluate_signals():
    """
    Reçoit une liste de signaux DIAGNOSTIC/OPEN depuis Google Apps Script.
    Input : {"signals": [{signal_id, symbol, direction, entry_low, entry_high,
                          stop_loss, tp1, timestamp, fill_deadline,
                          resolve_deadline, filled_at}, ...]}
    """
    try:
        body = request.get_json(silent=True) or {}
        signals_raw = body.get("signals", [])

        signals = []
        for item in signals_raw:
            if isinstance(item, dict):
                signals.append(item)
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        signals.append(parsed)
                    elif isinstance(parsed, list):
                        signals.extend([x for x in parsed if isinstance(x, dict)])
                except Exception:
                    logger.warning("evaluate_signals: item string non parseable: %s", item[:100])
            else:
                logger.warning("evaluate_signals: item type inattendu: %s", type(item))

        if not signals:
            return jsonify({"error": "signals list vide ou manquante",
                            "results": [], "evaluated": 0, "still_open": 0, "resolved": 0}), 200

        if len(signals) > 200:
            return jsonify({"error": "trop de signaux (max 200 par appel)",
                            "results": []}), 400

        results = []
        with ThreadPoolExecutor(max_workers=EVAL_MAX_WORKERS) as executor:
            futures = {executor.submit(_evaluate_one, sig): sig for sig in signals}
            for fut in as_completed(futures):
                sig = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.warning("_evaluate_one échec %s: %s", sig.get("signal_id"), e)
                    results.append({
                        "signal_id": sig.get("signal_id", "?"),
                        "outcome": "OPEN",
                        "evaluation_note": f"erreur évaluation: {e}",
                        "bars_checked": 0,
                        "filled_at": sig.get("filled_at", ""),
                        "closed_at": "",
                        "exit_price": ""
                    })

        still_open = sum(1 for r in results if r.get("outcome") == "OPEN")
        resolved = len(results) - still_open
        outcome_counts = {}
        for r in results:
            out = r.get("outcome", "?")
            outcome_counts[out] = outcome_counts.get(out, 0) + 1

        logger.info("evaluate_signals: %d signaux, %d résolus, %d encore OPEN, outcomes=%s",
                    len(signals), resolved, still_open, outcome_counts)

        return jsonify({
            "results": results,
            "evaluated": len(results),
            "still_open": still_open,
            "resolved": resolved,
            "outcome_counts": outcome_counts
        })

    except Exception as e:
        logger.error("evaluate_signals erreur: %s", e)
        return jsonify({"error": str(e), "results": []}), 500


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
        "version": "6.5.3",
        "providers": results
    })


# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "crypto-scorer", "version": "6.5.3"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

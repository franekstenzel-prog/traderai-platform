import io
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
from PIL import Image
import pytesseract


# ----------------------------
# Utilities
# ----------------------------

def _safe_float(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip().replace(",", ".")
    # keep digits dot and minus
    t = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
    if t in ("", "-", ".", "-."):
        return None
    try:
        return float(t)
    except Exception:
        return None


def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def _pct(a: float, b: float) -> float:
    # abs distance as percent of a
    if a == 0:
        return 0.0
    return abs(b - a) / abs(a)


@dataclass
class Candle:
    o: float
    h: float
    l: float
    c: float


# ----------------------------
# 1) Crop chart area (heuristic)
# ----------------------------

def _crop_chart_area(img_bgr: np.ndarray) -> np.ndarray:
    """
    Heuristic: cut top/bottom UI bars and keep the biggest central part.
    Works "good enough" for most screenshots.
    """
    h, w = img_bgr.shape[:2]
    # remove top 8% and bottom 10% (tabs/taskbar)
    y1 = int(0.08 * h)
    y2 = int(0.90 * h)
    x1 = int(0.18 * w)  # remove left sidebar area if present
    x2 = int(0.98 * w)
    cropped = img_bgr[y1:y2, x1:x2].copy()
    return cropped


# ----------------------------
# 2) OCR price axis (right side) -> mapping y -> price
# ----------------------------

def _ocr_axis_mapping(chart_bgr: np.ndarray) -> Tuple[Optional[Tuple[float, float]], List[Tuple[int, float]]]:
    """
    Returns:
      - mapping (a,b) where price ~= a*y + b  (y in pixels, origin at top)
      - raw points [(y, price)] used
    """
    h, w = chart_bgr.shape[:2]
    axis = chart_bgr[:, int(0.86 * w):w].copy()  # right 14% as axis
    gray = cv2.cvtColor(axis, cv2.COLOR_BGR2GRAY)
    # improve contrast for white text on dark background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR with boxes
    data = pytesseract.image_to_data(thr, output_type=pytesseract.Output.DICT, config="--psm 6")
    points: List[Tuple[int, float]] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1
        if conf < 40:
            continue
        val = _safe_float(txt)
        if val is None:
            continue
        # y center of bbox in axis coords
        y = int(data["top"][i] + data["height"][i] / 2)
        points.append((y, val))

    # Need at least 2 distinct points
    # Remove near-duplicates by y
    points = sorted(points, key=lambda t: t[0])
    filtered: List[Tuple[int, float]] = []
    for y, p in points:
        if not filtered or abs(y - filtered[-1][0]) > 10:
            filtered.append((y, p))
    points = filtered

    if len(points) < 2:
        return None, points

    # Fit line price = a*y + b (least squares)
    ys = np.array([p[0] for p in points], dtype=np.float64)
    ps = np.array([p[1] for p in points], dtype=np.float64)
    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, ps, rcond=None)[0]
    return (float(a), float(b)), points


def _y_to_price(mapping: Tuple[float, float], y: float) -> float:
    a, b = mapping
    return a * y + b


# ----------------------------
# 3) Extract candle "close series" roughly from pixels
# ----------------------------

def _extract_close_series(chart_bgr: np.ndarray, mapping: Optional[Tuple[float, float]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Heuristic candle extraction:
    - detect colored candle bodies (green/red) on dark bg
    - for each x-column, find top/bottom of colored pixels
    - group into candles by x spacing
    Returns lists: open/high/low/close in PRICE (if mapping exists) else in "pixel-price" units.
    """
    h, w = chart_bgr.shape[:2]

    # Remove axis area (right side)
    work = chart_bgr[:, : int(0.86 * w)].copy()

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    # Green and red masks (broad)
    # green
    mg1 = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    # red (two ranges)
    mr1 = cv2.inRange(hsv, (0, 40, 40), (12, 255, 255))
    mr2 = cv2.inRange(hsv, (165, 40, 40), (179, 255, 255))
    mr = cv2.bitwise_or(mr1, mr2)

    mask = cv2.bitwise_or(mg1, mr)

    # clean noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # For each x, find y-range of mask pixels
    ys_top = np.full(mask.shape[1], -1, dtype=np.int32)
    ys_bot = np.full(mask.shape[1], -1, dtype=np.int32)

    for x in range(mask.shape[1]):
        col = mask[:, x]
        idx = np.where(col > 0)[0]
        if idx.size == 0:
            continue
        ys_top[x] = int(idx.min())
        ys_bot[x] = int(idx.max())

    # Candle columns are contiguous x ranges with valid ys_top
    xs = np.where(ys_top >= 0)[0]
    if xs.size < 40:
        return [], [], [], []

    # segment into runs
    runs = []
    start = xs[0]
    prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = x
            prev = x
    runs.append((start, prev))

    # Merge tiny gaps: keep only meaningful runs
    # Each run ~ a candle body or wick cluster; we sample by taking center x.
    centers = []
    for a, b in runs:
        if (b - a + 1) < 2:
            continue
        centers.append((a + b) // 2)

    # If too many centers, downsample to ~150
    if len(centers) > 220:
        step = max(1, len(centers) // 180)
        centers = centers[::step]

    closes_px = []
    highs_px = []
    lows_px = []
    opens_px = []

    # Approximate OHLC per "candle" using local neighborhood:
    # open/close: use body color direction if possible; else midpoint.
    for cx in centers:
        x1 = max(0, cx - 1)
        x2 = min(mask.shape[1] - 1, cx + 1)
        col_slice = mask[:, x1:x2 + 1]
        idx = np.where(col_slice > 0)[0]
        if idx.size == 0:
            continue
        top = int(idx.min())
        bot = int(idx.max())
        high = top
        low = bot

        # determine bullish/bearish using hsv green/red counts in the body region
        body = work[top:bot + 1, x1:x2 + 1]
        body_hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        g = cv2.inRange(body_hsv, (35, 40, 40), (95, 255, 255)).sum()
        r = (cv2.inRange(body_hsv, (0, 40, 40), (12, 255, 255)).sum() +
             cv2.inRange(body_hsv, (165, 40, 40), (179, 255, 255)).sum())

        # In pixel coords, smaller y = higher price
        if g >= r:
            # bullish: open near bottom, close near top
            o = low
            c = high
        else:
            # bearish
            o = high
            c = low

        opens_px.append(o)
        highs_px.append(high)
        lows_px.append(low)
        closes_px.append(c)

    if len(closes_px) < 30:
        return [], [], [], []

    # Convert px y -> price
    if mapping is None:
        # fallback: use inverted pixel scale (higher y -> lower "price")
        # make pseudo price so math works
        opens = [float(h - y) for y in opens_px]
        highs = [float(h - y) for y in highs_px]
        lows = [float(h - y) for y in lows_px]
        closes = [float(h - y) for y in closes_px]
        return opens, highs, lows, closes

    opens = [_y_to_price(mapping, y) for y in opens_px]
    highs = [_y_to_price(mapping, y) for y in highs_px]
    lows = [_y_to_price(mapping, y) for y in lows_px]
    closes = [_y_to_price(mapping, y) for y in closes_px]
    return opens, highs, lows, closes


# ----------------------------
# 4) Swings + S/R clustering
# ----------------------------

def _find_swings(highs: List[float], lows: List[float], left: int = 2, right: int = 2):
    sh = []  # (i, price)
    sl = []
    n = len(highs)
    for i in range(left, n - right):
        hh = highs[i]
        ll = lows[i]
        if all(hh > highs[i - k] for k in range(1, left + 1)) and all(hh > highs[i + k] for k in range(1, right + 1)):
            sh.append((i, hh))
        if all(ll < lows[i - k] for k in range(1, left + 1)) and all(ll < lows[i + k] for k in range(1, right + 1)):
            sl.append((i, ll))
    return sh, sl


def _cluster_levels(prices: List[float], tol: float) -> List[float]:
    """
    Cluster similar prices into S/R levels.
    tol is absolute tolerance.
    """
    if not prices:
        return []
    xs = sorted(prices)
    clusters = []
    cur = [xs[0]]
    for p in xs[1:]:
        if abs(p - cur[-1]) <= tol:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)

    levels = []
    for c in clusters:
        # average cluster
        levels.append(sum(c) / len(c))
    return levels


# ----------------------------
# 5) Entry/SL/TP rules (classic)
# ----------------------------

def _pick_entry_sr(signal: str, price_now: float, supports: List[float], resistances: List[float]) -> float:
    if signal == "LONG":
        below = [s for s in supports if s < price_now]
        if below:
            return max(below)  # nearest support below (micro ok)
        # fallback: if no support below, use price_now
        return price_now
    else:
        above = [r for r in resistances if r > price_now]
        if above:
            return min(above)  # nearest resistance above (micro ok)
        return price_now


def _pick_sl(signal: str, entry: float, supports: List[float], resistances: List[float], buffer_abs: float) -> float:
    if signal == "LONG":
        below = [s for s in supports if s <= entry]
        base = max(below) if below else entry
        return base - buffer_abs
    else:
        above = [r for r in resistances if r >= entry]
        base = min(above) if above else entry
        return base + buffer_abs


def _pick_tp(signal: str, entry: float, sl: float, supports: List[float], resistances: List[float], rr_min: float, min_tp_pct: float) -> List[float]:
    # RR fallback target
    if signal == "LONG":
        R = entry - sl
        tp_rr = entry + rr_min * R
        cands = sorted([r for r in resistances if r > entry])
        min_price = entry * (1 + min_tp_pct)

        tp1 = None
        for r in cands:
            if r >= min_price and r >= tp_rr:
                tp1 = r
                break
        if tp1 is None:
            far = [r for r in cands if r >= min_price]
            tp1 = max(tp_rr, far[0]) if far else tp_rr

        tp2 = None
        for r in cands:
            if r > tp1:
                tp2 = r
                break
        if tp2 is None:
            tp2 = tp1 + 0.75 * (tp1 - entry)

        return [tp1, tp2]

    else:
        R = sl - entry
        tp_rr = entry - rr_min * R
        cands = sorted([s for s in supports if s < entry], reverse=True)
        max_price = entry * (1 - min_tp_pct)

        tp1 = None
        for s in cands:
            if s <= max_price and s <= tp_rr:
                tp1 = s
                break
        if tp1 is None:
            far = [s for s in cands if s <= max_price]
            tp1 = min(tp_rr, far[0]) if far else tp_rr

        tp2 = None
        for s in cands:
            if s < tp1:
                tp2 = s
                break
        if tp2 is None:
            tp2 = tp1 - 0.75 * (entry - tp1)

        return [tp1, tp2]


def _position_size(capital: float, risk_fraction: float, entry: float, sl: float, max_leverage_notional: float = 8.0) -> float:
    dist = abs(entry - sl)
    if dist <= 0:
        return 0.0
    risk_amt = capital * risk_fraction
    notional = risk_amt / dist
    return float(min(notional, capital * max_leverage_notional))


# ----------------------------
# Public API
# ----------------------------

def analyze_screen(
    image_bytes: bytes,
    pair: str,
    timeframe: str,
    capital: float,
    risk_fraction: float,
    mode: str,
) -> Dict:
    """
    Always returns a trade plan (no NO_TRADE), even if extraction is imperfect.
    """
    mode_norm = (mode or "swing").strip().lower()
    if mode_norm not in ("scalp", "swing"):
        mode_norm = "swing"

    rr_min = 2.0 if mode_norm == "swing" else 1.2

    # instrument profile: GOLD (XTB) sane min tp distance
    # You can extend profiles later.
    if (pair or "").upper() in ("GOLD", "XAUUSD", "XAUUSD.PRO"):
        min_tp_pct = 0.0025 if mode_norm == "swing" else 0.0018  # 0.25% / 0.18%
        buffer_pct = 0.0012  # 0.12%
    else:
        min_tp_pct = 0.0030 if mode_norm == "swing" else 0.0020
        buffer_pct = 0.0015

    # Decode image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    chart = _crop_chart_area(img_bgr)
    mapping, axis_points = _ocr_axis_mapping(chart)

    opens, highs, lows, closes = _extract_close_series(chart, mapping)

    issues = []
    rationale = []

    if len(closes) < 30:
        # fallback: cannot read candles reliably -> still return plan
        issues.append("Candle extraction weak; using fallback plan.")
        # try to estimate current price from axis points
        price_now = axis_points[len(axis_points)//2][1] if axis_points else 1.0
        # simple deterministic direction: if last axis is higher than first -> SHORT (scale inverted) doesn't apply; choose neutral -> LONG
        signal = "LONG"
        entry = price_now
        buffer_abs = abs(entry) * buffer_pct
        sl = entry - 2.5 * buffer_abs
        tp1 = entry + rr_min * (entry - sl)
        tp2 = tp1 + 0.75 * (tp1 - entry)
        pos = _position_size(capital, risk_fraction, entry, sl)
        return {
            "pair": pair,
            "timeframe": timeframe,
            "mode": mode_norm.upper(),
            "is_chart": True,
            "signal": signal,
            "confidence": 25,
            "setup": "Fallback: chart readable but candle extraction insufficient. Using conservative SR-less plan.",
            "entry": str(round(entry, 2)),
            "stop_loss": str(round(sl, 2)),
            "take_profit": [str(round(tp1, 2)), str(round(tp2, 2))],
            "support_levels": [],
            "resistance_levels": [],
            "position_size": str(round(pos, 2)),
            "risk": str(risk_fraction),
            "invalidation": "If price violates fallback SL.",
            "issues": issues,
            "rationale": rationale,
            "news": {"mode": "not_provided", "impact": "UNKNOWN", "summary": "", "key_points": []},
            "explanation": "Mechanical screen engine v1 (fallback path).",
        }

    # Determine swings and S/R from swings
    sh, sls = _find_swings(highs, lows, left=2, right=2)
    swing_highs = [p for _, p in sh]
    swing_lows = [p for _, p in sls]

    price_now = closes[-1]

    # tolerance based on price scale
    tol = max(abs(price_now) * 0.0012, 0.5)  # ~0.12% or 0.5 absolute

    supports = _cluster_levels(swing_lows[-40:], tol=tol)
    resistances = _cluster_levels(swing_highs[-40:], tol=tol)

    supports = sorted(set(supports))
    resistances = sorted(set(resistances))

    if not supports or not resistances:
        issues.append("Weak S/R detection; levels may be incomplete.")

    # Direction (structure first): simple trend proxy from closes regression slope
    xs = np.arange(len(closes), dtype=np.float64)
    ys = np.array(closes, dtype=np.float64)
    m = float(np.polyfit(xs, ys, 1)[0])
    signal = "LONG" if m >= 0 else "SHORT"

    # ENTRY snapped to nearest S/R (micro OK)
    entry = _pick_entry_sr(signal, price_now, supports, resistances)

    # SL behind that level (classic) with buffer
    buffer_abs = abs(entry) * buffer_pct
    sl_price = _pick_sl(signal, entry, supports, resistances, buffer_abs)

    # If SL invalid (wrong side), force it correct
    if signal == "LONG" and sl_price >= entry:
        sl_price = entry - 2.0 * buffer_abs
    if signal == "SHORT" and sl_price <= entry:
        sl_price = entry + 2.0 * buffer_abs

    # TP on meaningful opposing level (skip micro noise) + RR fallback
    tps = _pick_tp(signal, entry, sl_price, supports, resistances, rr_min=rr_min, min_tp_pct=min_tp_pct)

    # Position sizing
    pos = _position_size(capital, risk_fraction, entry, sl_price)

    # Confidence: based on how many swings + presence of levels
    conf = 35
    if len(sh) >= 6 and len(sls) >= 6:
        conf += 15
    if supports and resistances:
        conf += 10
    conf = int(_clamp(conf, 20, 80))

    setup = "Structure + S/R engine: entry at nearest S/R, SL beyond level, TP at meaningful opposing level."
    rationale = [
        "Primary: swing highs/lows → S/R clustering.",
        "Entry snapped to nearest S/R (micro levels allowed).",
        "Stop placed beyond the level with buffer.",
        "Take profits selected on meaningful opposing levels with RR fallback.",
    ]

    return {
        "pair": pair,
        "timeframe": timeframe,
        "mode": mode_norm.upper(),
        "is_chart": True,
        "signal": signal,
        "confidence": conf,
        "setup": setup,
        "entry": str(round(entry, 2)),
        "stop_loss": str(round(sl_price, 2)),
        "take_profit": [str(round(x, 2)) for x in tps],
        "support_levels": [str(round(x, 2)) for x in supports[-6:]],
        "resistance_levels": [str(round(x, 2)) for x in resistances[-6:]],
        "position_size": str(round(pos, 2)),
        "risk": str(risk_fraction),
        "invalidation": "Structure invalidated beyond SL.",
        "issues": issues,
        "rationale": rationale,
        "news": {"mode": "not_provided", "impact": "UNKNOWN", "summary": "", "key_points": []},
        "explanation": "Mechanical screen engine v1 (no OpenAI decisions).",
    }

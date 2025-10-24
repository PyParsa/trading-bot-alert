# -*- coding: utf-8 -*-
import os, time, json, csv
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from dotenv import load_dotenv

# ===================== تنظیمات عمومی =====================
SYMBOLS     = ["ETC/USDT","AVAX/USDT","LINK/USDT","SAND/USDT","APE/USDT","UNI/USDT","DOGE/USDT","ALGO/USDT"]
TIMEFRAMES  = ["5m","15m","1h","4h"]
CHECK_INTERVAL = 15 * 60            # هر ۱۵ دقیقه اسکن
CANDLE_LIMIT   = 300

# ایچیموکو/فیبو
USE_CHIKOU = True
FIB_LEVELS = [0.5, 0.618, 0.786, 0.886]
FIB_TOL    = 0.01                   # ±۱٪

# ذخیره‌سازی
LOG_FILE   = "signals_log.csv"
SENT_FILE  = "sent_signals.json"    # جلوگیری از ارسال تکراری

# منطقه زمانی تهران (ثابتِ +03:30)
TEHRAN_OFFSET = timedelta(hours=3, minutes=30)
def to_tehran(ts_utc) -> str:
    return (pd.Timestamp(ts_utc, tz="UTC") + TEHRAN_OFFSET).strftime("%Y-%m-%d %H:%M:%S")

# ===================== تلگرام =====================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def tg_send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN/CHAT_ID تنظیم نشده.")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print("⚠️ خطا در ارسال تلگرام:", e)

# ===================== KuCoin =====================
def fetch_ohlcv(symbol, timeframe):
    try:
        ex = ccxt.kucoin({"enableRateLimit": True, "timeout": 15000})
        ex.load_markets()
        if symbol not in ex.symbols:
            print(f"⚠️ {symbol} در KuCoin موجود نیست.")
            return pd.DataFrame()
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
        df = pd.DataFrame(raw, columns=["time","open","high","low","close","volume"])
        # زمان به UTC
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df
    except Exception as e:
        print(f"⚠️ KuCoin error {symbol}@{timeframe}: {e}")
        return pd.DataFrame()

# ===================== اندیکاتورها =====================
from ta.trend import IchimokuIndicator

def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    indi = IchimokuIndicator(df["high"], df["low"])
    df["span_a"] = indi.ichimoku_a()
    df["span_b"] = indi.ichimoku_b()
    df["cloud"]  = np.where(df["span_a"] > df["span_b"], "green", "red")
    if USE_CHIKOU:
        # chikou = Close شیفت به عقب 26 کندل → برای بررسی نسبت به قیمت فعلی، همین ستون کافیست
        df["chikou"] = df["close"].shift(-26)
    return df

def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    # تعریف ساده چکش
    body  = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open","close"]].max(axis=1)
    lower = df[["open","close"]].min(axis=1) - df["low"]
    df["hammer"] = (lower > 1.5 * body) & (upper < 0.4 * body)

    # اینگالف
    prev_o, prev_c = df["open"].shift(1), df["close"].shift(1)
    df["bull_engulf"] = (df["close"] > df["open"]) & (prev_c < prev_o) & (df["close"] >= prev_o) & (df["open"] <= prev_c)
    df["bear_engulf"] = (df["close"] < df["open"]) & (prev_c > prev_o) & (df["close"] <= prev_o) & (df["open"] >= prev_c)
    return df

# ===================== چک‌های استراتژی =====================
def pa_flags(df: pd.DataFrame, i: int, side: str):
    """پرایس‌اکشن (الزامی) — چکش در کندل i یا اینگالف در کندل i+1"""
    if i + 1 >= len(df):
        return False, False, False  # pa, hammer, engulf
    if side == "LONG":
        hammer  = bool(df.loc[i, "hammer"])
        engulf  = bool(df.loc[i + 1, "bull_engulf"])
        pa_ok   = hammer or engulf
        return pa_ok, hammer, engulf
    else:
        hammer  = bool(df.loc[i, "hammer"])
        engulf  = bool(df.loc[i + 1, "bear_engulf"])
        pa_ok   = hammer or engulf
        return pa_ok, hammer, engulf

def fib_zone(df: pd.DataFrame, i: int, lookback: int = 120):
    """نزدیکی قیمت به سطوح فیبوناچی بازگشتی"""
    start = max(0, i - lookback)
    seg = df.iloc[start:i+1]
    if seg.empty:
        return False, None, None
    high, low = seg["high"].max(), seg["low"].min()
    if high == low:
        return False, None, None
    c = float(df.loc[i, "close"])
    retr = (high - c) / (high - low)
    nearest = min(FIB_LEVELS, key=lambda lv: abs(retr - lv))
    ok = abs(retr - nearest) <= FIB_TOL
    return ok, retr, nearest

def ich_state(df: pd.DataFrame, i: int, side: str):
    """وضعیت قیمت نسبت به ابر ایچیموکو + فیلتر اختیاری چیکو"""
    c  = float(df.loc[i, "close"])
    sa = float(df.loc[i, "span_a"])
    sb = float(df.loc[i, "span_b"])
    col = df.loc[i, "cloud"]

    if side == "LONG":
        cond = (c < sa) and (c < sb) and (col == "red")
        if USE_CHIKOU and "chikou" in df.columns:
            cond = cond and (df.loc[i, "chikou"] > df.loc[i, "close"])
        return cond
    else:
        cond = (c > sa) and (c > sb) and (col == "green")
        if USE_CHIKOU and "chikou" in df.columns:
            cond = cond and (df.loc[i, "chikou"] < df.loc[i, "close"])
        return cond

# ===================== ضد تکرار و لاگ =====================
def load_sent_ids() -> set:
    if os.path.exists(SENT_FILE):
        try:
            with open(SENT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_sent_ids(s: set):
    with open(SENT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(s)), f, ensure_ascii=False, indent=2)

def append_log(row: dict):
    header = ["id","type","timestamp_utc","timestamp_tehran","symbol","tf","side","combo","fib_retr","fib_level","span_a","span_b"]
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

# ===================== سیگنال‌ها =====================
def detect_confirmed_signals(df: pd.DataFrame):
    """
    سیگنال قطعی:
    - پرایس‌اکشن الزامی (چکش یا اینگالف)
    - + یکی از ایچیموکو یا فیبوناچی
    هر سه فعال → 🔥
    """
    out = []
    # تا کندل ماقبل آخر (چون تایید اینگالف در i+1 لازم است)
    for i in range(2, len(df) - 1):
        for side in ["LONG", "SHORT"]:
            pa_ok, hamm, eng = pa_flags(df, i, side)
            if not pa_ok:
                continue
            ich = ich_state(df, i, side)
            fib_ok, retr, lvl = fib_zone(df, i)

            if ich or fib_ok:
                parts = ["PriceAction ✅" + (" (Hammer)" if hamm else "") + (" (Engulf)" if eng else "")]
                if ich:    parts.append("Ichimoku ✅")
                if fib_ok: parts.append(f"Fibonacci ≈ {lvl:.3f}")
                combo = " + ".join(parts)
                strong = " 🔥" if (ich and fib_ok and (hamm or eng)) else ""

                # سیگنال برای کندل بعدی صادر می‌شود (پس زمان i+1)
                t_utc = df.loc[i + 1, "time"]
                out.append( ("SIG", t_utc, side, combo + strong, retr, lvl, float(df.loc[i,"span_a"]), float(df.loc[i,"span_b"])) )
    return out

def detect_pre_signal(df: pd.DataFrame):
    """
    هشدار آماده‌باش:
    - در آخرین کندل بسته‌شده (i = len-2) اگر چکش باشد
    - و یکی از فیبو/ایچیموکو همزمان ok باشد
    - هنوز اینگالف تایید نشده (طبیعتاً چون کندل بعدی جاری/نیامده)
    """
    if len(df) < 3:
        return []

    i = len(df) - 2  # آخرین کندل بسته‌شده
    outs = []
    for side in ["LONG","SHORT"]:
        hammer = bool(df.loc[i, "hammer"])
        if not hammer:
            continue
        ich = ich_state(df, i, side)
        fib_ok, retr, lvl = fib_zone(df, i)
        if ich or fib_ok:
            parts = ["PriceAction ✅ (Hammer)"]
            if ich:    parts.append("Ichimoku ✅")
            if fib_ok: parts.append(f"Fibonacci ≈ {lvl:.3f}")
            combo = " + ".join(parts)
            t_utc = df.loc[i, "time"]
            outs.append( ("PRE", t_utc, side, combo, retr, lvl, float(df.loc[i,"span_a"]), float(df.loc[i,"span_b"])) )
    return outs

def format_msg(sig_type, symbol, tf, t_utc, side, combo):
    t_local = to_tehran(t_utc)
    if sig_type == "SIG":
        title = "📣 سیگنال قطعی"
    else:
        title = "⚠️ سیگنال در حال شکل‌گیری"

    return (
        f"{title} ({side})\n"
        f"🔹 نماد: <b>{symbol}</b>\n"
        f"⏱ تایم‌فریم: <b>{tf}</b>\n"
        f"🕒 زمان (تهران): <code>{t_local}</code>\n"
        f"🕒 UTC: <code>{pd.Timestamp(t_utc, tz='UTC').strftime('%Y-%m-%d %H:%M:%S')}</code>\n"
        f"📊 ترکیب فعال: {combo}\n"
        f"📌 منبع: KUCOIN"
    )

# ===================== اجرای یک دور اسکن =====================
def scan_once(sent_ids: set):
    found = 0
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            df = fetch_ohlcv(sym, tf)
            if df.empty: 
                continue

            # آماده‌سازی
            df = add_ichimoku(add_price_action(df))

            # --- هشدار آماده‌باش (PRE)
            for typ, t_utc, side, combo, retr, lvl, sa, sb in detect_pre_signal(df):
                sid = f"{typ}-{pd.Timestamp(t_utc, tz='UTC').strftime('%Y%m%d%H%M')}-{sym.replace('/','')}-{tf}-{side}"
                if sid in sent_ids:
                    continue
                sent_ids.add(sid)
                msg = format_msg(typ, sym, tf, t_utc, side, combo)
                print(msg); tg_send(msg)
                append_log({
                    "id": sid, "type": typ,
                    "timestamp_utc": pd.Timestamp(t_utc, tz='UTC').strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_tehran": to_tehran(t_utc),
                    "symbol": sym, "tf": tf, "side": side, "combo": combo,
                    "fib_retr": round(retr,3) if retr is not None else "", "fib_level": round(lvl,3) if lvl else "",
                    "span_a": sa, "span_b": sb
                })
                found += 1

            # --- سیگنال قطعی (SIG)
            for typ, t_utc, side, combo, retr, lvl, sa, sb in detect_confirmed_signals(df):
                sid = f"{typ}-{pd.Timestamp(t_utc, tz='UTC').strftime('%Y%m%d%H%M')}-{sym.replace('/','')}-{tf}-{side}"
                if sid in sent_ids:
                    continue
                sent_ids.add(sid)
                msg = format_msg(typ, sym, tf, t_utc, side, combo)
                print(msg); tg_send(msg)
                append_log({
                    "id": sid, "type": typ,
                    "timestamp_utc": pd.Timestamp(t_utc, tz='UTC').strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_tehran": to_tehran(t_utc),
                    "symbol": sym, "tf": tf, "side": side, "combo": combo,
                    "fib_retr": round(retr,3) if retr is not None else "", "fib_level": round(lvl,3) if lvl else "",
                    "span_a": sa, "span_b": sb
                })
                found += 1
    return found

# ===================== اجرای خودکار =====================
if __name__ == "__main__":
    print("🤖 ربات تحلیل (KuCoin) فعال شد. منطقه زمانی پیام‌ها: Asia/Tehran | هر ۱۵ دقیقه اسکن.")
    sent = load_sent_ids()
    while True:
        cnt = scan_once(sent)
        if cnt:
            save_sent_ids(sent)
        print(f"✅ دور بررسی تمام شد. سیگنال جدید: {cnt}")
        print("⏳ انتظار برای دور بعد...\n")
        time.sleep(CHECK_INTERVAL)

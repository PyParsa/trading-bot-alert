import os, time, json, csv
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

# ---- timezone (Asia/Tehran) ----
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # اگر لازم شد
TZ_NAME = "Asia/Tehran"
TZ_TEHRAN = ZoneInfo(TZ_NAME)

# ================== تنظیمات ==================
SYMBOLS = ["ETC/USDT","AVAX/USDT","LINK/USDT","SAND/USDT","APE/USDT","UNI/USDT","DOGE/USDT","ALGO/USDT"]
TIMEFRAMES = ["5m", "15m", "1h", "4h"]   # تایم‌فریم‌های معتبر KuCoin
CANDLE_LIMIT = 500
CHECK_INTERVAL = 15 * 60                # هر ۱۵ دقیقه
# (در منطق جدید، محدودیت زمانی حذف شده و ارسال زنده انجام می‌شود)

# --- فیبوناچی (یکبار تعریف) ---
FIB_LEVELS = [0.5, 0.618, 0.786, 0.886]
FIB_TOL    = 0.01                       # ±۱٪ اطراف تراز

LOG_FILE  = "signals_log.csv"
SENT_FILE = "sent_signals.json"         # برای جلوگیری از ارسال تکراری
# =============================================

# --- تلگرام (.env) ---
from dotenv import load_dotenv
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def tg_send(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN یا CHAT_ID تنظیم نشده.")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print("⚠️ خطا در ارسال تلگرام:", e)

# --- KuCoin ---
def make_exchange():
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 15000})

def fetch_ohlcv(symbol, timeframe, limit=CANDLE_LIMIT):
    try:
        ex = make_exchange()
        ex.load_markets()
        if symbol not in ex.symbols:
            print(f"⚠️ {symbol} در KuCoin یافت نشد.")
            return None, pd.DataFrame()
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)  # UTC
        return "kucoin", df
    except Exception as e:
        print(f"❌ KuCoin error {symbol}@{timeframe}: {e}")
        return None, pd.DataFrame()

# --- اندیکاتورها ---
from ta.trend import IchimokuIndicator

def add_ichimoku(df):
    indi = IchimokuIndicator(df["high"], df["low"])
    df["span_a"] = indi.ichimoku_a()
    df["span_b"] = indi.ichimoku_b()
    df["cloud"]  = np.where(df["span_a"] > df["span_b"], "green", "red")
    return df

def add_patterns(df):
    body  = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open","close"]].max(axis=1)
    lower = df[["open","close"]].min(axis=1) - df["low"]
    # چکش ساده (برای هر دو جهت استفاده می‌کنیم)
    df["hammer"] = (lower > 1.5 * body) & (upper < 0.4 * body)
    prev_o, prev_c = df["open"].shift(1), df["close"].shift(1)
    df["bull_engulf"] = (df["close"] > df["open"]) & (prev_c < prev_o) & (df["close"] >= prev_o) & (df["open"] <= prev_c)
    df["bear_engulf"] = (df["close"] < df["open"]) & (prev_c > prev_o) & (df["close"] <= prev_o) & (df["open"] >= prev_c)
    return df

def fib_zone(df, i, lookback=120):
    """بررسی نزدیکی قیمت به سطوح فیبوناچی بازگشتی و مقدار retracement"""
    start = max(0, i - lookback)
    seg = df.iloc[start:i+1]
    if seg.empty:
        return False, None, None
    high, low = seg["high"].max(), seg["low"].min()
    if high == low:
        return False, None, None
    c = float(df.loc[i, "close"])
    retr = (high - c) / (high - low)  # تعریف یکسان برای هر دو جهت
    nearest = min(FIB_LEVELS, key=lambda lv: abs(retr - lv))
    ok = abs(retr - nearest) <= FIB_TOL
    return ok, retr, nearest

# ==== تنظیمات استراتژی ====
USE_CHIKOU = True  # اگر خواستی فیلتر Chikou فعال/غیرفعال باشد

def ich_state(df, i, side):
    """تشخیص وضعیت ایچیموکو نسبت به ابر؛ Chikou اختیاری"""
    c = float(df.loc[i, "close"])
    sa, sb = float(df.loc[i, "span_a"]), float(df.loc[i, "span_b"])
    color = df.loc[i, "cloud"]

    if side == "LONG":
        cond = (c < sa) and (c < sb) and (color == "red")
        if USE_CHIKOU and "chikou" in df.columns:
            cond = cond and (df.loc[i, "chikou"] > df.loc[i, "close"])
        return cond
    else:
        cond = (c > sa) and (c > sb) and (color == "green")
        if USE_CHIKOU and "chikou" in df.columns:
            cond = cond and (df.loc[i, "chikou"] < df.loc[i, "close"])
        return cond

def pa_ok(df, i, side):
    """پرایس‌اکشن: چکش → اینگالف در جهت معامله (اجباری)"""
    if i + 1 >= len(df):  # نیاز به کندل بعدی برای اینگالف
        return False
    if side == "LONG":
        return bool(df.loc[i, "hammer"]) and bool(df.loc[i + 1, "bull_engulf"])
    else:
        return bool(df.loc[i, "hammer"]) and bool(df.loc[i + 1, "bear_engulf"])

# ---------- منطق اصلی صدور سیگنال ----------
def detect_signals(df):
    """
    شرط سیگنال:
    پرایس‌اکشن الزامی است.
    فیبو یا ایچیموکو (حداقل یکی) باید همزمان برقرار باشد.
    هر سه فعال → سیگنال قوی 🔥
    خروجی: [(time_utc, side, reasons_text), ...]
    """
    sigs = []

    # Chikou اختیاری
    if USE_CHIKOU and "chikou" not in df.columns:
        df["chikou"] = df["close"].shift(-26)

    for i in range(2, len(df) - 1):
        for side in ["LONG", "SHORT"]:
            pa = pa_ok(df, i, side)
            if not pa:
                continue  # بدون پرایس‌اکشن اصلاً سیگنال نمی‌دهیم

            ich = ich_state(df, i, side)
            fib_ok, retr, lvl = fib_zone(df, i)

            if ich or fib_ok:
                parts = ["PriceAction ✅"]
                if ich:
                    parts.append("Ichimoku ✅")
                if fib_ok:
                    parts.append(f"Fibonacci ≈ {lvl} (retr={round(retr,3)})")
                combo_text = " + ".join(parts)
                if ich and fib_ok:
                    combo_text += " 🔥"  # سیگنال قوی‌تر

                sigs.append((df.loc[i + 1, "time"], side, combo_text))

    return sigs

# --- ارسال فوری (بدون فیلتر زمانی) ---
def filter_recent_signals(signals):
    return signals

# --- لاگ و ضد تکرار ---
def load_sent_ids():
    if os.path.exists(SENT_FILE):
        try:
            with open(SENT_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_sent_ids(sent_ids: set):
    with open(SENT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(sent_ids)), f, ensure_ascii=False, indent=2)

def append_log(row: dict):
    # ساده‌سازی لاگ: فیلدهای محاسبه‌نشده حذف شدند
    header = ["id","timestamp_utc","timestamp_tehran","symbol","tf","side","reasons"]
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow(row)

def fmt_tehran(ts_utc: datetime) -> str:
    return ts_utc.astimezone(TZ_TEHRAN).strftime("%Y-%m-%d %H:%M:%S")

# --- اجرای یک اسکن ---
def scan_once(sent_ids: set):
    found = []
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            ex_id, df = fetch_ohlcv(sym, tf)
            if df.empty:
                continue
            df = add_ichimoku(df)
            df = add_patterns(df)
            sigs = detect_signals(df)
            if not sigs:
                continue

            # فقط آخرین کندل بسته‌شده را بررسی کن
            last_candle_time = df.iloc[-2]["time"]
            last_signals = [s for s in sigs if s[0] == last_candle_time]
            if not last_signals:
                continue

            for t_utc, side, reasons_text in last_signals:
                sid = f"{t_utc.strftime('%Y%m%d%H%M')}-{sym.replace('/','')}-{tf}-{side}"
                if sid in sent_ids:
                    continue  # تکراری نفرست
                sent_ids.add(sid)

                t_local = fmt_tehran(t_utc)
                msg = (
                    f"📣 سیگنال زنده {side}\n"
                    f"🔹 نماد: <b>{sym}</b>\n"
                    f"⏱ تایم‌فریم: <b>{tf}</b>\n"
                    f"🕒 زمان (تهران): <code>{t_local}</code>\n"
                    f"📌 منبع: KUCOIN\n\n"
                    f"🔎 ترکیب فعال:\n• {reasons_text}"
                )
                print(msg)
                tg_send(msg)

                append_log({
                    "id": sid,
                    "timestamp_utc": t_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_tehran": t_local,
                    "symbol": sym,
                    "tf": tf,
                    "side": side,
                    "reasons": reasons_text,
                })
                found.append((sid, sym, tf, t_utc, side))
    return found

# --- اجرای اتومات ---
if __name__ == "__main__":
    print(f"🤖 ربات تحلیل (KuCoin) فعال شد. منطقه زمانی پیام‌ها: {TZ_NAME} | هر ۱۵ دقیقه اسکن.")
    sent_ids = load_sent_ids()
    while True:
        hits = scan_once(sent_ids)
        if hits:
            save_sent_ids(sent_ids)
        print(f"✅ دور بررسی تمام شد. سیگنال جدید: {len(hits)}")
        print("⏳ انتظار برای دور بعد...\n")
        time.sleep(CHECK_INTERVAL)

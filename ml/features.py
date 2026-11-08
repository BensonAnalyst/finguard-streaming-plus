"""Rules-engine port and model features, both computed in event time.

Every per-customer feature uses only that customer's *earlier* transactions
(rolling windows with closed="left"), so nothing from the future, and never
the label, leaks into a score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulate import CITIES

# ---------------------------------------------------------------------------
# Rules engine: weights copied from kafka_producer/fraud_engine.py.
# Changes needed to make it a fair baseline (documented in README):
#   * time windows use event time, not datetime.now() (needed for batch replay)
#   * HIGH_VALUE: INR 100,000 converted to SGD ~1,500
#   * CARD_TESTING: INR 5-20 is a few cents, so use the intent "SGD 1-20"
#   * IMPOSSIBLE_TRAVEL: the hard-coded Delhi -> London pair generalised to
#     "different country within 20 minutes of the previous transaction"
# ---------------------------------------------------------------------------
RULE_WEIGHTS = {
    "HIGH_VALUE_TRANSACTION": 40,
    "IMPOSSIBLE_TRAVEL": 50,
    "NEW_DEVICE": 20,
    "HIGH_RISK_MERCHANT": 25,
    "BLACKLISTED_MERCHANT": 60,
    "INTERNATIONAL_TRANSACTION": 25,
    "VELOCITY_FRAUD": 45,
    "CARD_TESTING": 30,
}


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _window_stats(df: pd.DataFrame, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Count and amount-sum of each customer's transactions in the `seconds` *before* each one.

    df must be sorted by customer_id, ts. Uses binary search per customer, so
    the current transaction and anything later are never included.
    """
    counts = np.zeros(len(df))
    sums = np.zeros(len(df))
    t_all = df.ts.to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
    a_all = df.amount.to_numpy()
    for idx in df.groupby("customer_id", sort=False).indices.values():
        t, a = t_all[idx], a_all[idx]
        lo = np.searchsorted(t, t - seconds, side="left")
        hi = np.searchsorted(t, t, side="left")  # excludes ties with the current time
        csum = np.concatenate([[0.0], np.cumsum(a)])
        counts[idx] = hi - lo
        sums[idx] = csum[hi] - csum[lo]
    return counts, sums


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Return tx plus rule flags, rules_score and model features."""
    df = tx.sort_values(["customer_id", "ts"], kind="stable").copy()
    g = df.groupby("customer_id", sort=False)

    lat = df.merchant_city.map(lambda c: CITIES[c][0])
    lon = df.merchant_city.map(lambda c: CITIES[c][1])
    home_lat = df.home_city.map(lambda c: CITIES[c][0])
    home_lon = df.home_city.map(lambda c: CITIES[c][1])
    prev_lat, prev_lon = lat.groupby(df.customer_id).shift(), lon.groupby(df.customer_id).shift()
    prev_country = g.merchant_country.shift()
    secs_since_prev = (df.ts - g.ts.shift()).dt.total_seconds()

    df["n_prev_30s"], _ = _window_stats(df, 30)
    df["n_prev_1min"], _ = _window_stats(df, 60)
    df["n_prev_10min"], _ = _window_stats(df, 600)
    df["n_prev_1h"], df["amt_prev_1h"] = _window_stats(df, 3600)
    df["secs_since_prev"] = secs_since_prev.fillna(1e7)

    # --- rules (event-time port) ---
    flags = pd.DataFrame(index=df.index)
    flags["HIGH_VALUE_TRANSACTION"] = df.amount > 1500
    flags["IMPOSSIBLE_TRAVEL"] = (prev_country.notna() & (prev_country != df.merchant_country)
                                  & (secs_since_prev <= 20 * 60))
    flags["NEW_DEVICE"] = df.device_id != df.trusted_device_id
    flags["HIGH_RISK_MERCHANT"] = df.merchant_risk == "HIGH"
    flags["BLACKLISTED_MERCHANT"] = df.is_blacklisted
    flags["INTERNATIONAL_TRANSACTION"] = df.is_international
    flags["VELOCITY_FRAUD"] = df.n_prev_30s >= 5
    flags["CARD_TESTING"] = df.amount.between(1, 20) & (df.n_prev_1min >= 3)
    score = sum(flags[k].astype(int) * w for k, w in RULE_WEIGHTS.items())
    df["rules_score"] = np.minimum(100, score)
    for k in RULE_WEIGHTS:
        df[f"rule_{k.lower()}"] = flags[k].astype(int)

    # --- model features ---
    prior_median = g.amount.transform(lambda s: s.expanding().median().shift())
    df["amt_vs_median"] = df.amount / prior_median.fillna(df.typical_spend)
    df["log_amount"] = np.log1p(df.amount)
    df["new_device"] = flags["NEW_DEVICE"].astype(int)
    seen = df.groupby(["customer_id", "device_id"]).cumcount()  # earlier uses of this device
    df["device_first_use"] = (seen == 0).astype(int) & df["new_device"]
    df["device_uses_before"] = seen
    km_prev = _haversine_km(prev_lat, prev_lon, lat, lon)
    df["km_from_prev"] = km_prev.fillna(0.0)
    hours = (secs_since_prev / 3600).clip(lower=1 / 60)
    df["speed_kmh_from_prev"] = (km_prev / hours).fillna(0.0).clip(upper=5000)
    df["km_from_home"] = _haversine_km(home_lat, home_lon, lat, lon)
    df["hour"] = df.ts.dt.hour + df.ts.dt.minute / 60
    df["hour_vs_peak"] = (df.hour - df.peak_hour).abs()
    df["is_night"] = df.ts.dt.hour.between(0, 5).astype(int)
    df["international"] = df.is_international.astype(int)
    df["merchant_risk_level"] = df.merchant_risk.map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    df["blacklisted"] = df.is_blacklisted.astype(int)
    df["online"] = (df.payment_channel == "ONLINE").astype(int)
    df["category"] = df.merchant_category.astype("category")
    df["segment_cat"] = df.segment.astype("category")
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


MODEL_FEATURES = [
    "log_amount", "amt_vs_median", "amt_prev_1h",
    "n_prev_1min", "n_prev_10min", "n_prev_1h", "secs_since_prev",
    "new_device", "device_first_use", "device_uses_before",
    "km_from_prev", "speed_kmh_from_prev", "km_from_home", "international",
    "hour", "hour_vs_peak", "is_night",
    "merchant_risk_level", "blacklisted", "online", "category", "segment_cat",
]

"""Labelled card-transaction simulator for model evaluation.

Why a separate simulator?
    The streaming producer in kafka_producer/ sets fraud_score with the rules
    engine itself, and its "forced fraud" flag changes only the label, not the
    transaction. Data from that producer can't be used to compare a model with
    the rules: the comparison would be circular, or the fraud would be invisible.

    Here fraud is injected as *behaviour* that changes the transactions, and the
    label is the ground truth of what happened. Neither the rules engine nor the
    model ever sees the label. Legitimate noise is built in on purpose (real
    travel, Johor-Singapore cross-border shoppers, new phones, big legitimate
    purchases, gamers at high-risk merchants), so no method can score 100%.

Setting: Singapore / Malaysia card portfolio, amounts in SGD.

Fraud scenarios (scenario column):
    CARD_TESTING     burst of tiny online charges, often followed by a cash-out purchase
    ACCOUNT_TAKEOVER new device, high-value electronics/jewellery/crypto, often at night
    CARD_CLONE       card used in a foreign city shortly after a genuine home transaction
    LOW_AND_SLOW     normal-looking amounts at normal merchants over several days (hard)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CITIES = {
    # name: (lat, lon, country)
    "Singapore": (1.352, 103.820, "SG"),
    "Johor Bahru": (1.493, 103.741, "MY"),
    "Kuala Lumpur": (3.139, 101.687, "MY"),
    "Penang": (5.414, 100.329, "MY"),
    "Bangkok": (13.756, 100.502, "TH"),
    "Jakarta": (-6.208, 106.846, "ID"),
    "Hong Kong": (22.320, 114.169, "HK"),
    "Tokyo": (35.676, 139.650, "JP"),
    "London": (51.507, -0.128, "GB"),
    "Lagos": (6.524, 3.379, "NG"),
    "Moscow": (55.756, 37.617, "RU"),
}
HOME_CITIES = ["Singapore", "Johor Bahru", "Kuala Lumpur", "Penang"]
HOME_WEIGHTS = [0.60, 0.15, 0.15, 0.10]
TRAVEL_CITIES = ["Bangkok", "Hong Kong", "Tokyo", "London", "Kuala Lumpur", "Singapore", "Penang"]
FRAUD_CITIES = ["Lagos", "Moscow", "Jakarta", "Bangkok", "London"]

# category: (amount multiplier vs customer's typical spend, online?)
CATEGORIES = {
    "Grocery": (0.7, False),
    "Restaurant": (0.8, False),
    "Pharmacy": (0.5, False),
    "Fuel": (0.8, False),
    "Shopping": (1.2, True),
    "Electronics": (3.0, True),
    "Jewellery": (4.0, False),
    "Travel": (5.0, True),
    "Hotel": (3.0, False),
    "Online Gaming": (0.6, True),
    "Crypto Exchange": (2.0, True),
}
EVERYDAY = ["Grocery", "Restaurant", "Pharmacy", "Fuel", "Shopping"]
SEGMENT_SPEND = {"Regular": 40.0, "Gold": 90.0, "Platinum": 250.0, "Corporate": 150.0}


@dataclass
class SimConfig:
    n_customers: int = 3000
    n_merchants: int = 400
    days: int = 28
    txns_per_day: float = 2.5
    compromised_share: float = 0.10
    seed: int = 42
    start: datetime = datetime(2026, 10, 1)


def _merchants(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    cats = list(CATEGORIES)
    rows = []
    for i in range(cfg.n_merchants):
        cat = cats[rng.integers(len(cats))]
        city = (
            HOME_CITIES[rng.choice(4, p=HOME_WEIGHTS)]
            if rng.random() < 0.75
            else list(CITIES)[rng.integers(4, len(CITIES))]
        )
        if cat in ("Online Gaming", "Crypto Exchange"):
            risk = "HIGH"
        elif cat in ("Jewellery", "Electronics"):
            risk = "MEDIUM" if rng.random() < 0.7 else "LOW"
        else:
            risk = "HIGH" if rng.random() < 0.05 else "LOW"
        blacklisted = bool(rng.random() < (0.08 if risk == "HIGH" else 0.005))
        rows.append((f"M{i:04d}", cat, city, CITIES[city][2], risk, blacklisted, CATEGORIES[cat][1]))
    m = pd.DataFrame(rows, columns=["merchant_id", "merchant_category", "merchant_city",
                                    "merchant_country", "merchant_risk", "is_blacklisted", "is_online"])
    # Guarantee every city has at least one everyday and one high-value merchant.
    extra = []
    for city in CITIES:
        for cat in ("Restaurant", "Shopping", "Electronics", "Jewellery"):
            if not ((m.merchant_city == city) & (m.merchant_category == cat)).any():
                extra.append((f"M{len(m) + len(extra):04d}", cat, city, CITIES[city][2],
                              "MEDIUM" if cat in ("Electronics", "Jewellery") else "LOW",
                              False, CATEGORIES[cat][1]))
    if extra:
        m = pd.concat([m, pd.DataFrame(extra, columns=m.columns)], ignore_index=True)
    return m


def _customers(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    segs = list(SEGMENT_SPEND)
    rows = []
    for i in range(cfg.n_customers):
        city = HOME_CITIES[rng.choice(4, p=HOME_WEIGHTS)]
        seg = segs[rng.choice(4, p=[0.55, 0.25, 0.08, 0.12])]
        typical = SEGMENT_SPEND[seg] * float(rng.lognormal(0, 0.35))
        rows.append((f"C{i:05d}", city, CITIES[city][2], seg, typical, f"D{i:05d}",
                     float(rng.uniform(9, 20))))  # peak hour of activity
    return pd.DataFrame(rows, columns=["customer_id", "home_city", "home_country", "segment",
                                       "typical_spend", "trusted_device_id", "peak_hour"])


class _Builder:
    def __init__(self, cfg, rng, customers, merchants):
        self.cfg, self.rng, self.c, self.m = cfg, rng, customers, merchants
        self.rows: list[tuple] = []
        self.by_city_cat = {k: g.index.to_numpy() for k, g in merchants.groupby(["merchant_city", "merchant_category"])}
        self.by_city = {k: g.index.to_numpy() for k, g in merchants.groupby("merchant_city")}
        self.online = merchants.index[merchants.is_online].to_numpy()
        self.online_high = merchants.index[merchants.is_online & (merchants.merchant_risk == "HIGH")].to_numpy()

    def pick(self, city, cats=None):
        if cats is not None:
            pools = [self.by_city_cat[(city, c)] for c in cats if (city, c) in self.by_city_cat]
            if pools:
                pool = np.concatenate(pools)
                return int(pool[self.rng.integers(len(pool))])
        pool = self.by_city[city]
        return int(pool[self.rng.integers(len(pool))])

    def amount(self, cust, midx, scale=1.0, sigma=0.7):
        mult = CATEGORIES[self.m.at[midx, "merchant_category"]][0]
        return round(float(cust.typical_spend * mult * scale * self.rng.lognormal(0, sigma)), 2)

    def add(self, cust, ts, midx, amount, device, label, scenario, channel=None):
        mer = self.m.loc[midx]
        channel = channel or ("ONLINE" if mer.is_online and self.rng.random() < 0.8 else
                   self.rng.choice(["POS", "MOBILE_APP"], p=[0.7, 0.3]))
        self.rows.append((cust.customer_id, ts, mer.merchant_id, amount, device, channel, label, scenario))


def simulate(cfg: SimConfig | None = None) -> pd.DataFrame:
    """Return one row per transaction with is_fraud and scenario ground truth."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    merchants = _merchants(cfg, rng)
    customers = _customers(cfg, rng)
    b = _Builder(cfg, rng, customers, merchants)

    for cust in customers.itertuples(index=False):
        # Legitimate life events
        new_phone_day = int(rng.integers(cfg.days)) if rng.random() < 0.10 else None
        new_phone = f"D{cust.customer_id[1:]}N"
        trip = None
        if rng.random() < 0.08:
            start = int(rng.integers(cfg.days - 3))
            dest = [c for c in TRAVEL_CITIES if c != cust.home_city][rng.integers(len(TRAVEL_CITIES) - 1)]
            trip = (start, start + int(rng.integers(2, 6)), dest)

        for day in range(cfg.days):
            n = rng.poisson(cfg.txns_per_day)
            for _ in range(n):
                hour = float(np.clip(rng.normal(cust.peak_hour, 3.5), 6.5, 23.9))
                ts = cfg.start + timedelta(days=day, hours=hour, seconds=float(rng.integers(60)))
                city = cust.home_city
                if trip and trip[0] <= day <= trip[1]:
                    city = trip[2]
                elif cust.home_city == "Johor Bahru" and rng.random() < 0.20:
                    city = "Singapore"  # cross-border shopping
                elif cust.home_city == "Singapore" and rng.random() < 0.04:
                    city = "Johor Bahru"
                r = rng.random()
                if r < 0.05:
                    midx = int(b.online[rng.integers(len(b.online))])  # online shopping / gaming
                elif r < 0.12:
                    midx = b.pick(city, ["Electronics", "Jewellery", "Hotel", "Travel"])
                else:
                    midx = b.pick(city, EVERYDAY)
                amt = b.amount(cust, midx, scale=5.0 if rng.random() < 0.01 else 1.0)
                device = cust.trusted_device_id
                if new_phone_day is not None and day >= new_phone_day:
                    device = new_phone
                elif rng.random() < 0.01:
                    device = f"X{rng.integers(1e6):06d}"  # borrowed / shared device
                b.add(cust, ts, midx, amt, device, 0, "LEGIT")
                if rng.random() < 0.03:  # food-court style quick second purchase
                    m2 = b.pick(city, ["Restaurant", "Grocery"])
                    b.add(cust, ts + timedelta(seconds=float(rng.integers(20, 240))), m2,
                          b.amount(cust, m2), device, 0, "LEGIT")

        if rng.random() >= cfg.compromised_share:
            continue

        # Fraud event
        scenario = rng.choice(["CARD_TESTING", "ACCOUNT_TAKEOVER", "CARD_CLONE", "LOW_AND_SLOW"],
                              p=[0.30, 0.30, 0.25, 0.15])
        day = int(rng.integers(1, cfg.days - 4))
        t0 = cfg.start + timedelta(days=day, hours=float(rng.uniform(0, 24)))
        fdev = f"F{rng.integers(1e6):06d}"
        if scenario == "CARD_TESTING":
            midx_pool = b.online_high if rng.random() < 0.6 else b.online
            midx = int(midx_pool[rng.integers(len(midx_pool))])
            t = t0
            for _ in range(int(rng.integers(4, 13))):
                t += timedelta(seconds=float(rng.integers(8, 90)))
                b.add(cust, t, midx, round(float(rng.uniform(1, 15)), 2), fdev, 1, scenario)
            if rng.random() < 0.5:  # cash-out
                for _ in range(int(rng.integers(1, 4))):
                    t += timedelta(minutes=float(rng.uniform(2, 30)))
                    m2 = b.pick(cust.home_city if rng.random() < 0.6 else FRAUD_CITIES[rng.integers(5)],
                                ["Electronics", "Jewellery"])
                    b.add(cust, t, m2, b.amount(cust, m2, scale=float(rng.uniform(1.5, 4))), fdev, 1, scenario)
        elif scenario == "ACCOUNT_TAKEOVER":
            if rng.random() < 0.3:  # remote-access malware scam: runs on the victim's own phone
                fdev = cust.trusted_device_id
            if rng.random() < 0.6:
                t0 = t0.replace(hour=int(rng.integers(1, 6)))
            city = cust.home_city if rng.random() < 0.6 else FRAUD_CITIES[rng.integers(5)]
            t = t0
            for _ in range(int(rng.integers(2, 7))):
                t += timedelta(minutes=float(rng.uniform(3, 40)))
                midx = b.pick(city, ["Electronics", "Jewellery", "Crypto Exchange", "Shopping"])
                b.add(cust, t, midx, b.amount(cust, midx, scale=float(rng.uniform(1.5, 5))), fdev, 1, scenario)
        elif scenario == "CARD_CLONE":
            # A genuine home purchase, then the cloned card abroad shortly after.
            m0 = b.pick(cust.home_city, EVERYDAY)
            b.add(cust, t0, m0, b.amount(cust, m0), cust.trusted_device_id, 0, "LEGIT")
            city = FRAUD_CITIES[rng.integers(5)]
            t = t0 + timedelta(minutes=float(rng.uniform(5, 90)))
            for _ in range(int(rng.integers(2, 6))):
                midx = b.pick(city, ["Electronics", "Jewellery", "Shopping", "Restaurant"])
                # Card-present clone: swiped at a shop counter, no customer device involved.
                b.add(cust, t, midx, b.amount(cust, midx, scale=float(rng.uniform(1, 4))),
                      cust.trusted_device_id, 1, scenario, channel="POS")
                t += timedelta(minutes=float(rng.uniform(5, 60)))
        else:  # LOW_AND_SLOW
            if rng.random() < 0.5:  # malware on the victim's own phone
                fdev = cust.trusted_device_id
            t = t0
            for _ in range(int(rng.integers(3, 9))):
                t += timedelta(hours=float(rng.uniform(6, 30)))
                if t.hour < 7:
                    t = t.replace(hour=int(rng.integers(9, 22)))
                midx = b.pick(cust.home_city, EVERYDAY)
                b.add(cust, t, midx, b.amount(cust, midx, scale=float(rng.uniform(0.8, 1.5))), fdev, 1, scenario)

    tx = pd.DataFrame(b.rows, columns=["customer_id", "ts", "merchant_id", "amount", "device_id",
                                       "payment_channel", "is_fraud", "scenario"])
    tx = tx.merge(customers, on="customer_id").merge(merchants, on="merchant_id")
    tx["is_international"] = tx.merchant_country != tx.home_country
    tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
    tx.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(tx))])
    return tx


if __name__ == "__main__":
    df = simulate()
    print(df.shape, f"fraud rate {df.is_fraud.mean():.2%}")
    print(df.scenario.value_counts())

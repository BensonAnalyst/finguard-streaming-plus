"""Local tests for the SPC gold table logic (no Databricks needed).

Run:  pip install pyspark pytest && pytest -q tests/
"""
import importlib.util
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pyspark.sql import SparkSession

SPC_FILE = (
    Path(__file__).resolve().parents[1]
    / "databricks notebooks and pipelines/finguard_project/finguard_streaming/gold/transaction_volume_spc.py"
)


def _load_spc_module():
    # Replace the pipeline decorators with no-ops so the file imports outside a pipeline.
    fake = types.ModuleType("pyspark.pipelines")
    fake.materialized_view = lambda **_: (lambda f: f)
    fake.table = fake.materialized_view
    sys.modules["pyspark.pipelines"] = fake
    import pyspark
    pyspark.pipelines = fake
    spec = importlib.util.spec_from_file_location("spc", SPC_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


spc = _load_spc_module()
T0 = datetime(2026, 11, 8, 9, 0)


@pytest.fixture(scope="session")
def spark():
    s = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    s.conf.set("spark.sql.session.timeZone", "UTC")
    yield s
    s.stop()


def _run(spark, counts, drop_minutes=()):
    rows = [(T0 + timedelta(minutes=i), int(c)) for i, c in enumerate(counts) if i not in drop_minutes]
    df = spark.createDataFrame(rows, "window_start timestamp, transaction_count long")
    out = spc.compute_c_chart(df, baseline_minutes=60, min_baseline_points=30)
    return {r.window_start: r for r in out.collect()}


def _stable(n, mean=300, seed=7):
    return list(np.random.default_rng(seed).poisson(mean, n))


def test_warm_up_has_no_flags(spark):
    res = _run(spark, _stable(40))
    first = res[T0 + timedelta(minutes=5)]
    assert first.spc_status == "WARMING_UP" and first.rules_violated == []


def test_stable_process_is_mostly_in_control(spark):
    res = _run(spark, _stable(200))
    statuses = [r.spc_status for r in res.values() if r.spc_status != "WARMING_UP"]
    # Pure Poisson noise: false-alarm rate should be small.
    assert statuses.count("IN_CONTROL") / len(statuses) > 0.9


def test_spike_flags_we1(spark):
    counts = _stable(90)
    counts[80] = 600  # card-testing burst
    r = _run(spark, counts)[T0 + timedelta(minutes=80)]
    assert r.we1 and r.spc_status == "SPIKE"


def test_silent_minute_is_filled_and_flagged_as_drop(spark):
    res = _run(spark, _stable(90), drop_minutes={70})
    r = res[T0 + timedelta(minutes=70)]
    assert r.transaction_count == 0 and r.spc_status == "DROP"


def test_small_sustained_shift_caught_by_we4_not_we1(spark):
    counts = _stable(70) + list(np.random.default_rng(1).poisson(318, 12))  # ~+1 sigma
    res = _run(spark, counts)
    shifted = [res[T0 + timedelta(minutes=i)] for i in range(70, 82)]
    assert not any(r.we1 for r in shifted)
    assert any(r.we4 or r.we3 for r in shifted)

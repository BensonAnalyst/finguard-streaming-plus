"""Tests for the model-vs-rules evaluation (ml/). Fast: small simulated portfolio."""
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from features import MODEL_FEATURES, build_features  # noqa: E402
from simulate import SimConfig, simulate  # noqa: E402
from train_compare import run  # noqa: E402

SMALL = SimConfig(n_customers=300, days=28, compromised_share=0.2, seed=3)


@pytest.fixture(scope="module")
def tx():
    return simulate(SMALL)


def test_simulator_is_deterministic(tx):
    again = simulate(SMALL)
    pd.testing.assert_frame_equal(tx, again)


def test_all_fraud_scenarios_present(tx):
    assert set(tx.scenario.unique()) >= {"LEGIT", "CARD_TESTING", "ACCOUNT_TAKEOVER", "CARD_CLONE", "LOW_AND_SLOW"}
    assert 0.002 < tx.is_fraud.mean() < 0.05


def test_features_do_not_use_the_future(tx):
    """A transaction's features must not change when later transactions are removed."""
    cutoff = tx.ts.min() + timedelta(days=10)
    full = build_features(tx).set_index("transaction_id")
    past = build_features(tx[tx.ts <= cutoff]).set_index("transaction_id")
    cols = [c for c in MODEL_FEATURES if c not in ("category", "segment_cat")] + ["rules_score"]
    pd.testing.assert_frame_equal(full.loc[past.index, cols], past[cols], check_dtype=False)


def test_label_is_not_a_feature():
    assert not {"is_fraud", "scenario", "fraud_score"} & set(MODEL_FEATURES)


def test_card_testing_burst_triggers_rule(tx):
    f = build_features(tx)
    burst = f[(f.scenario == "CARD_TESTING") & (f.n_prev_1min >= 3) & f.amount.between(1, 20)]
    assert len(burst) > 0 and (burst.rule_card_testing == 1).all()


def test_model_beats_rules_on_held_out_period():
    res, _ = run(SimConfig(n_customers=600, days=28, compromised_share=0.15, seed=11))
    assert res["ranking"]["model_pr_auc"] > res["ranking"]["rules_pr_auc"]
    op = res["operating_points"]
    assert op["xgboost_same_alert_budget"]["recall"] >= op["rules_tuned"]["recall"]

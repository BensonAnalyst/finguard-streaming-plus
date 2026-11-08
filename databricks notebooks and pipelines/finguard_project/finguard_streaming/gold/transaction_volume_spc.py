"""
Gold: Statistical Process Control (SPC) on per-minute transaction volume.

Why SPC for fraud/ops monitoring?
    Rule-based alerts catch *individual* bad transactions. They miss *pattern*
    problems: a card-testing bot ramping up volume, a merchant outage that
    silently drops traffic to zero, or a slow drift that never crosses a
    fixed threshold. A control chart answers one question every minute:
    "Is this minute's volume normal process variation, or a special cause
    that someone should look at?"

Method
    * Chart type: c-chart. Transactions per minute are counts, which are
      approximately Poisson, so sigma = sqrt(c_bar). This is the standard
      chart for count data in manufacturing SPC.
    * Baseline: rolling mean of the previous N minutes (default 60), which
      excludes the current point, so a spike cannot inflate its own limits.
    * Minutes with zero transactions are filled in explicitly. A silent
      minute is a signal (outage or upstream failure), not missing data.
    * Western Electric rules (each evaluated on the same side of the mean):
        WE1  1 point beyond 3 sigma
        WE2  2 of 3 consecutive points beyond 2 sigma
        WE3  4 of 5 consecutive points beyond 1 sigma
        WE4  8 consecutive points on one side of the mean
      WE1 catches spikes and drops; WE2-WE4 catch smaller sustained shifts
      that a fixed threshold never sees.

Tunable via pipeline configuration:
    finguard.spc.baseline_minutes      (default 60)
    finguard.spc.min_baseline_points   (default 30)  warm-up before flagging
"""

from pyspark import pipelines as dp
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

SOURCE_TABLE = "finguard.gold.transaciton_count_by_minute"  # upstream name kept as-is


def _fill_missing_minutes(counts: DataFrame) -> DataFrame:
    """Return one row per minute between first and last window, with 0 for silent minutes."""
    bounds = counts.agg(
        F.min("window_start").alias("lo"), F.max("window_start").alias("hi")
    )
    spine = bounds.select(
        F.explode(
            F.sequence(F.col("lo"), F.col("hi"), F.expr("INTERVAL 1 MINUTE"))
        ).alias("window_start")
    )
    return (
        spine.join(counts.select("window_start", "transaction_count"), "window_start", "left")
        .withColumn("transaction_count", F.coalesce(F.col("transaction_count"), F.lit(0)))
        .withColumn("window_end", F.col("window_start") + F.expr("INTERVAL 1 MINUTE"))
    )


def compute_c_chart(
    counts: DataFrame, baseline_minutes: int = 60, min_baseline_points: int = 30
) -> DataFrame:
    """Add c-chart limits, z-scores and Western Electric rule flags to per-minute counts.

    `counts` needs the columns window_start (timestamp) and transaction_count (long).
    """
    filled = _fill_missing_minutes(counts)

    # Single time series ordered by minute (one row/minute keeps this small).
    w = Window.orderBy("window_start")
    baseline = w.rowsBetween(-baseline_minutes, -1)

    charted = (
        filled.withColumn("baseline_points", F.count("transaction_count").over(baseline))
        .withColumn("c_bar", F.avg("transaction_count").over(baseline))
        .withColumn("sigma", F.sqrt(F.col("c_bar")))
        .withColumn("ucl", F.col("c_bar") + 3 * F.col("sigma"))
        .withColumn("lcl", F.greatest(F.lit(0.0), F.col("c_bar") - 3 * F.col("sigma")))
        .withColumn(
            "z",
            F.when(
                (F.col("baseline_points") >= min_baseline_points) & (F.col("sigma") > 0),
                (F.col("transaction_count") - F.col("c_bar")) / F.col("sigma"),
            ),
        )
    )

    def n_in_last(k: int, cond):
        """How many of the last k points (including the current one) satisfy cond."""
        return F.sum(F.when(cond, 1).otherwise(0)).over(w.rowsBetween(-(k - 1), 0))

    z = F.col("z")
    flagged = (
        charted.withColumn("we1", F.abs(z) > 3)
        .withColumn(
            "we2",
            ((z > 2) & (n_in_last(3, z > 2) >= 2)) | ((z < -2) & (n_in_last(3, z < -2) >= 2)),
        )
        .withColumn(
            "we3",
            ((z > 1) & (n_in_last(5, z > 1) >= 4)) | ((z < -1) & (n_in_last(5, z < -1) >= 4)),
        )
        .withColumn(
            "we4",
            ((z > 0) & (n_in_last(8, z > 0) == 8)) | ((z < 0) & (n_in_last(8, z < 0) == 8)),
        )
    )

    rules = ["we1", "we2", "we3", "we4"]
    # Null z (warm-up) -> no flags rather than null flags.
    for r in rules:
        flagged = flagged.withColumn(r, F.coalesce(F.col(r), F.lit(False)))

    return flagged.withColumn(
        "rules_violated",
        F.array_compact(
            F.array(*[F.when(F.col(r), F.lit(r.upper())) for r in rules])
        ),
    ).withColumn(
        "spc_status",
        F.when(F.col("z").isNull(), F.lit("WARMING_UP"))
        .when(F.col("we1") & (F.col("z") > 0), F.lit("SPIKE"))
        .when(F.col("we1") & (F.col("z") < 0), F.lit("DROP"))
        .when(F.size("rules_violated") > 0, F.lit("SHIFT"))
        .otherwise(F.lit("IN_CONTROL")),
    ).select(
        "window_start", "window_end", "transaction_count",
        "c_bar", "ucl", "lcl", "z",
        "we1", "we2", "we3", "we4", "rules_violated", "spc_status",
    )


@dp.materialized_view(
    name="finguard.gold.transaction_volume_spc",
    comment="c-chart with Western Electric rules on per-minute transaction volume",
)
def transaction_volume_spc() -> DataFrame:
    baseline_minutes = int(spark.conf.get("finguard.spc.baseline_minutes", "60"))
    min_points = int(spark.conf.get("finguard.spc.min_baseline_points", "30"))
    counts = spark.read.table(SOURCE_TABLE)
    return compute_c_chart(counts, baseline_minutes, min_points)


@dp.materialized_view(
    name="finguard.gold.transaction_volume_spc_alerts",
    comment="Only the out-of-control minutes, for dashboards and email alerts",
)
def transaction_volume_spc_alerts() -> DataFrame:
    return spark.read.table("finguard.gold.transaction_volume_spc").filter(
        F.col("spc_status").isin("SPIKE", "DROP", "SHIFT")
    )

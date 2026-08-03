"""Calculate rule-based multi-touch marketing attribution."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_customer_journeys.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"

TOUCHPOINT_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "attribution_touchpoint_credits.csv"
)

CHANNEL_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "attribution_channel_summary.csv"
)

TIME_DECAY_HALF_LIFE_DAYS = 7.0

FIRST_TOUCH = "First Touch"
LAST_TOUCH = "Last Touch"
LINEAR = "Linear"
TIME_DECAY = "Time Decay"
POSITION_BASED = "Position Based"


def load_clean_data() -> pd.DataFrame:
    """Load and validate the cleaned customer journey data."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Clean data was not found. Run "
            "'python src/data_cleaning.py' first."
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        parse_dates=["timestamp"],
    )

    if dataframe.empty:
        raise ValueError("The cleaned dataset is empty.")

    return dataframe.sort_values(
        ["journey_id", "touchpoint_order"]
    ).reset_index(drop=True)


def first_touch_weights(
    journey: pd.DataFrame,
) -> np.ndarray:
    """Give all conversion credit to the first touchpoint."""

    weights = np.zeros(len(journey), dtype=float)
    weights[0] = 1.0

    return weights


def last_touch_weights(
    journey: pd.DataFrame,
) -> np.ndarray:
    """Give all conversion credit to the last touchpoint."""

    weights = np.zeros(len(journey), dtype=float)
    weights[-1] = 1.0

    return weights


def linear_weights(
    journey: pd.DataFrame,
) -> np.ndarray:
    """Divide credit equally across every touchpoint."""

    return np.full(
        len(journey),
        1.0 / len(journey),
        dtype=float,
    )


def time_decay_weights(
    journey: pd.DataFrame,
) -> np.ndarray:
    """Give more credit to touchpoints closer to conversion."""

    conversion_timestamp = journey["timestamp"].max()

    days_before_conversion = (
        conversion_timestamp - journey["timestamp"]
    ).dt.total_seconds() / 86_400

    raw_weights = np.power(
        0.5,
        days_before_conversion.to_numpy()
        / TIME_DECAY_HALF_LIFE_DAYS,
    )

    return raw_weights / raw_weights.sum()


def position_based_weights(
    journey: pd.DataFrame,
) -> np.ndarray:
    """Apply a 40-20-40 position-based attribution model."""

    touchpoint_count = len(journey)

    if touchpoint_count == 1:
        return np.array([1.0])

    if touchpoint_count == 2:
        return np.array([0.5, 0.5])

    weights = np.full(
        touchpoint_count,
        0.20 / (touchpoint_count - 2),
        dtype=float,
    )

    weights[0] = 0.40
    weights[-1] = 0.40

    return weights


MODEL_FUNCTIONS: dict[
    str,
    Callable[[pd.DataFrame], np.ndarray],
] = {
    FIRST_TOUCH: first_touch_weights,
    LAST_TOUCH: last_touch_weights,
    LINEAR: linear_weights,
    TIME_DECAY: time_decay_weights,
    POSITION_BASED: position_based_weights,
}


def calculate_attribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate touchpoint credit for every converted journey."""

    converted_data = dataframe.loc[
        dataframe["converted"] == 1
    ].copy()

    if converted_data.empty:
        raise ValueError(
            "No converted journeys were found."
        )

    results: list[pd.DataFrame] = []

    for journey_id, journey in converted_data.groupby(
        "journey_id",
        sort=False,
    ):
        journey = journey.sort_values(
            "touchpoint_order"
        ).reset_index(drop=True)

        conversion_value = float(
            journey["conversion_value"].sum()
        )

        if conversion_value <= 0:
            raise ValueError(
                f"{journey_id} has no conversion revenue."
            )

        for model_name, model_function in MODEL_FUNCTIONS.items():
            weights = model_function(journey)

            model_result = journey[
                [
                    "customer_id",
                    "journey_id",
                    "touchpoint_id",
                    "touchpoint_order",
                    "timestamp",
                    "channel",
                    "campaign",
                    "device",
                ]
            ].copy()

            model_result["model"] = model_name
            model_result["attribution_weight"] = weights
            model_result["attributed_conversions"] = weights
            model_result["attributed_revenue"] = (
                weights * conversion_value
            )

            results.append(model_result)

    return pd.concat(
        results,
        ignore_index=True,
    )


def validate_results(
    attribution: pd.DataFrame,
    source_data: pd.DataFrame,
) -> None:
    """Confirm attribution totals are mathematically correct."""

    journey_weights = (
        attribution.groupby(
            ["model", "journey_id"]
        )["attribution_weight"]
        .sum()
    )

    if not np.allclose(
        journey_weights.to_numpy(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Some attribution weights do not sum to 1."
        )

    expected_conversions = source_data.loc[
        source_data["converted"] == 1,
        "journey_id",
    ].nunique()

    expected_revenue = source_data.loc[
        source_data["converted"] == 1,
        "conversion_value",
    ].sum()

    model_totals = attribution.groupby("model").agg(
        attributed_conversions=(
            "attributed_conversions",
            "sum",
        ),
        attributed_revenue=(
            "attributed_revenue",
            "sum",
        ),
    )

    if not np.allclose(
        model_totals["attributed_conversions"],
        expected_conversions,
        atol=1e-6,
    ):
        raise ValueError(
            "Attributed conversion totals are incorrect."
        )

    if not np.allclose(
        model_totals["attributed_revenue"],
        expected_revenue,
        atol=0.01,
    ):
        raise ValueError(
            "Attributed revenue totals are incorrect."
        )


def create_channel_summary(
    attribution: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate attribution results by model and channel."""

    summary = (
        attribution.groupby(
            ["model", "channel"],
            as_index=False,
        )
        .agg(
            journeys_touched=(
                "journey_id",
                "nunique",
            ),
            touchpoints=(
                "touchpoint_id",
                "count",
            ),
            attributed_conversions=(
                "attributed_conversions",
                "sum",
            ),
            attributed_revenue=(
                "attributed_revenue",
                "sum",
            ),
        )
    )

    summary["conversion_share"] = (
        summary["attributed_conversions"]
        / summary.groupby("model")[
            "attributed_conversions"
        ].transform("sum")
    )

    summary["revenue_share"] = (
        summary["attributed_revenue"]
        / summary.groupby("model")[
            "attributed_revenue"
        ].transform("sum")
    )

    summary["channel_rank"] = (
        summary.groupby("model")[
            "attributed_revenue"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    return summary.sort_values(
        ["model", "channel_rank"]
    ).reset_index(drop=True)


def save_results(
    attribution: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> None:
    """Save attribution outputs."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    attribution.to_csv(
        TOUCHPOINT_OUTPUT_PATH,
        index=False,
    )

    channel_summary.to_csv(
        CHANNEL_OUTPUT_PATH,
        index=False,
    )


def print_summary(
    attribution: pd.DataFrame,
    channel_summary: pd.DataFrame,
) -> None:
    """Print the highest-ranked channel under each model."""

    print("\nAttribution models completed")
    print("-" * 68)

    top_channels = channel_summary.loc[
        channel_summary["channel_rank"] == 1
    ]

    for _, row in top_channels.iterrows():
        print(
            f"{row['model']:<18}"
            f"Top channel: {row['channel']:<16}"
            f"Revenue: ${row['attributed_revenue']:,.2f}"
        )

    print("-" * 68)
    print(
        f"Converted journeys: "
        f"{attribution['journey_id'].nunique():,}"
    )
    print(f"Touchpoint credits: {TOUCHPOINT_OUTPUT_PATH}")
    print(f"Channel summary:    {CHANNEL_OUTPUT_PATH}")


def main() -> None:
    """Run the attribution-model pipeline."""

    clean_data = load_clean_data()

    attribution = calculate_attribution(
        clean_data
    )

    validate_results(
        attribution,
        clean_data,
    )

    channel_summary = create_channel_summary(
        attribution
    )

    save_results(
        attribution,
        channel_summary,
    )

    print_summary(
        attribution,
        channel_summary,
    )


if __name__ == "__main__":
    main()
"""Clean and prepare customer journey data for attribution analysis."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "customer_journeys.csv"
)

PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

CLEAN_TOUCHPOINTS_PATH = (
    PROCESSED_DIRECTORY
    / "clean_customer_journeys.csv"
)

JOURNEY_SUMMARY_PATH = (
    PROCESSED_DIRECTORY
    / "journey_summary.csv"
)

REQUIRED_COLUMNS = {
    "customer_id",
    "journey_id",
    "touchpoint_id",
    "touchpoint_order",
    "timestamp",
    "channel",
    "campaign",
    "device",
    "converted",
    "is_conversion",
    "conversion_value",
}

TEXT_COLUMNS = [
    "customer_id",
    "journey_id",
    "touchpoint_id",
    "channel",
    "campaign",
    "device",
]


def load_raw_data() -> pd.DataFrame:
    """Load the raw customer journey CSV file."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Run "
            "'python src/data_generator.py' first."
        )

    dataframe = pd.read_csv(RAW_DATA_PATH)

    if dataframe.empty:
        raise ValueError("The raw customer journey dataset is empty.")

    return dataframe


def check_required_columns(dataframe: pd.DataFrame) -> None:
    """Confirm that all required columns exist."""

    missing_columns = REQUIRED_COLUMNS.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def clean_touchpoints(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Clean touchpoint-level customer journey data."""

    cleaned = dataframe.copy()

    cleaned.columns = (
        cleaned.columns
        .str.strip()
        .str.lower()
    )

    check_required_columns(cleaned)

    rows_before_cleaning = len(cleaned)

    cleaned = cleaned.drop_duplicates()

    duplicates_removed = (
        rows_before_cleaning - len(cleaned)
    )

    for column in TEXT_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        errors="raise",
    )

    cleaned["touchpoint_order"] = pd.to_numeric(
        cleaned["touchpoint_order"],
        errors="raise",
    ).astype("int64")

    cleaned["converted"] = pd.to_numeric(
        cleaned["converted"],
        errors="raise",
    ).astype("int64")

    cleaned["is_conversion"] = pd.to_numeric(
        cleaned["is_conversion"],
        errors="raise",
    ).astype("int64")

    cleaned["conversion_value"] = pd.to_numeric(
        cleaned["conversion_value"],
        errors="raise",
    ).astype("float64")

    cleaned = cleaned.sort_values(
        [
            "journey_id",
            "touchpoint_order",
            "timestamp",
        ]
    ).reset_index(drop=True)

    journey_groups = cleaned.groupby(
        "journey_id",
        sort=False,
    )

    cleaned["touchpoint_count"] = (
        journey_groups["touchpoint_id"]
        .transform("count")
    )

    cleaned["first_touch_timestamp"] = (
        journey_groups["timestamp"]
        .transform("min")
    )

    cleaned["last_touch_timestamp"] = (
        journey_groups["timestamp"]
        .transform("max")
    )

    cleaned["hours_since_first_touch"] = (
        (
            cleaned["timestamp"]
            - cleaned["first_touch_timestamp"]
        )
        / pd.Timedelta(hours=1)
    ).round(2)

    cleaned["journey_duration_hours"] = (
        (
            cleaned["last_touch_timestamp"]
            - cleaned["first_touch_timestamp"]
        )
        / pd.Timedelta(hours=1)
    ).round(2)

    cleaned["event_date"] = (
        cleaned["timestamp"]
        .dt.strftime("%Y-%m-%d")
    )

    cleaned["event_month"] = (
        cleaned["timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    return cleaned, duplicates_removed


def validate_clean_data(
    dataframe: pd.DataFrame,
) -> None:
    """Validate customer journey business rules."""

    if dataframe["touchpoint_id"].duplicated().any():
        raise ValueError(
            "Duplicate touchpoint IDs remain after cleaning."
        )

    if dataframe[
        list(REQUIRED_COLUMNS)
    ].isna().any().any():
        raise ValueError(
            "Missing values exist in required columns."
        )

    if not dataframe["converted"].isin([0, 1]).all():
        raise ValueError(
            "The converted column must contain only 0 or 1."
        )

    if not dataframe["is_conversion"].isin([0, 1]).all():
        raise ValueError(
            "The is_conversion column must contain only 0 or 1."
        )

    if (dataframe["conversion_value"] < 0).any():
        raise ValueError(
            "Conversion values cannot be negative."
        )

    conversion_rows = dataframe[
        dataframe["is_conversion"] == 1
    ]

    if not (
        conversion_rows["conversion_value"] > 0
    ).all():
        raise ValueError(
            "Conversion rows must contain positive revenue."
        )

    non_conversion_rows = dataframe[
        dataframe["is_conversion"] == 0
    ]

    if not (
        non_conversion_rows["conversion_value"] == 0
    ).all():
        raise ValueError(
            "Non-conversion rows must contain zero revenue."
        )

    journey_checks = (
        dataframe.groupby("journey_id")
        .agg(
            touchpoint_count=(
                "touchpoint_id",
                "count",
            ),
            minimum_order=(
                "touchpoint_order",
                "min",
            ),
            maximum_order=(
                "touchpoint_order",
                "max",
            ),
            conversions=(
                "is_conversion",
                "sum",
            ),
            converted_minimum=(
                "converted",
                "min",
            ),
            converted_maximum=(
                "converted",
                "max",
            ),
        )
    )

    invalid_orders = (
        (journey_checks["minimum_order"] != 1)
        | (
            journey_checks["maximum_order"]
            != journey_checks["touchpoint_count"]
        )
    )

    if invalid_orders.any():
        raise ValueError(
            "Invalid touchpoint order detected."
        )

    if (journey_checks["conversions"] > 1).any():
        raise ValueError(
            "A journey cannot contain multiple conversions."
        )

    inconsistent_conversion_status = (
        journey_checks["converted_minimum"]
        != journey_checks["converted_maximum"]
    )

    if inconsistent_conversion_status.any():
        raise ValueError(
            "Conversion status is inconsistent within a journey."
        )

    conversion_mismatch = (
        journey_checks["conversions"]
        != journey_checks["converted_maximum"]
    )

    if conversion_mismatch.any():
        raise ValueError(
            "Journey conversion status does not match "
            "the conversion event."
        )


def create_journey_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row for each customer journey."""

    summary = (
        dataframe.groupby(
            "journey_id",
            as_index=False,
            sort=False,
        )
        .agg(
            customer_id=(
                "customer_id",
                "first",
            ),
            first_touch_timestamp=(
                "timestamp",
                "min",
            ),
            last_touch_timestamp=(
                "timestamp",
                "max",
            ),
            first_channel=(
                "channel",
                "first",
            ),
            last_channel=(
                "channel",
                "last",
            ),
            first_campaign=(
                "campaign",
                "first",
            ),
            last_campaign=(
                "campaign",
                "last",
            ),
            touchpoint_count=(
                "touchpoint_id",
                "count",
            ),
            unique_channels=(
                "channel",
                "nunique",
            ),
            converted=(
                "converted",
                "max",
            ),
            conversion_events=(
                "is_conversion",
                "sum",
            ),
            conversion_value=(
                "conversion_value",
                "sum",
            ),
        )
    )

    summary["journey_duration_hours"] = (
        (
            summary["last_touch_timestamp"]
            - summary["first_touch_timestamp"]
        )
        / pd.Timedelta(hours=1)
    ).round(2)

    conversion_timestamps = (
        dataframe.loc[
            dataframe["is_conversion"] == 1,
            [
                "journey_id",
                "timestamp",
            ],
        ]
        .set_index("journey_id")["timestamp"]
    )

    summary["conversion_timestamp"] = (
        summary["journey_id"]
        .map(conversion_timestamps)
    )

    summary["days_to_conversion"] = (
        (
            summary["conversion_timestamp"]
            - summary["first_touch_timestamp"]
        )
        / pd.Timedelta(days=1)
    ).round(2)

    summary["journey_path"] = (
        dataframe.groupby(
            "journey_id",
            sort=False,
        )["channel"]
        .apply(lambda channels: " > ".join(channels))
        .reindex(summary["journey_id"])
        .to_numpy()
    )

    return summary


def save_processed_data(
    touchpoints: pd.DataFrame,
    journey_summary: pd.DataFrame,
) -> None:
    """Save the processed touchpoint and journey datasets."""

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    touchpoints.to_csv(
        CLEAN_TOUCHPOINTS_PATH,
        index=False,
    )

    journey_summary.to_csv(
        JOURNEY_SUMMARY_PATH,
        index=False,
    )


def print_summary(
    touchpoints: pd.DataFrame,
    journey_summary: pd.DataFrame,
    duplicates_removed: int,
) -> None:
    """Print a summary of the cleaning process."""

    total_conversions = int(
        journey_summary["converted"].sum()
    )

    total_revenue = float(
        journey_summary["conversion_value"].sum()
    )

    print("\nData cleaning completed")
    print("-" * 48)
    print(f"Touchpoints:          {len(touchpoints):,}")
    print(f"Journeys:             {len(journey_summary):,}")
    print(f"Duplicates removed:   {duplicates_removed:,}")
    print(f"Conversions:          {total_conversions:,}")
    print(f"Revenue:              ${total_revenue:,.2f}")
    print(f"Clean touchpoints:    {CLEAN_TOUCHPOINTS_PATH}")
    print(f"Journey summary:      {JOURNEY_SUMMARY_PATH}")


def main() -> None:
    """Run the complete data-cleaning pipeline."""

    raw_data = load_raw_data()

    clean_data, duplicates_removed = clean_touchpoints(
        raw_data
    )

    validate_clean_data(clean_data)

    journey_summary = create_journey_summary(
        clean_data
    )

    save_processed_data(
        clean_data,
        journey_summary,
    )

    print_summary(
        clean_data,
        journey_summary,
        duplicates_removed,
    )


if __name__ == "__main__":
    main()
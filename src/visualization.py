"""Create comparison charts for multi-touch attribution models."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
FIGURES_DIRECTORY = REPORTS_DIRECTORY / "figures"

INPUT_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "attribution_channel_summary.csv"
)

COMPARISON_OUTPUT_PATH = (
    REPORTS_DIRECTORY
    / "model_comparison.csv"
)

REVENUE_CHART_PATH = (
    FIGURES_DIRECTORY
    / "channel_revenue_by_model.png"
)

TOP_CHANNEL_CHART_PATH = (
    FIGURES_DIRECTORY
    / "top_channel_by_model.png"
)

SHARE_CHART_PATH = (
    FIGURES_DIRECTORY
    / "channel_revenue_share.png"
)

# Attribution-model names
FIRST_TOUCH = "First Touch"
LAST_TOUCH = "Last Touch"
LINEAR = "Linear"
TIME_DECAY = "Time Decay"
POSITION_BASED = "Position Based"

MODEL_ORDER = [
    FIRST_TOUCH,
    LAST_TOUCH,
    LINEAR,
    TIME_DECAY,
    POSITION_BASED,
]

# Dataset column names
MODEL_COLUMN = "model"
CHANNEL_COLUMN = "channel"
ATTRIBUTED_CONVERSIONS_COLUMN = "attributed_conversions"
ATTRIBUTED_REVENUE_COLUMN = "attributed_revenue"
CONVERSION_SHARE_COLUMN = "conversion_share"
REVENUE_SHARE_COLUMN = "revenue_share"
CHANNEL_RANK_COLUMN = "channel_rank"

# Comparison-table column names
AVERAGE_REVENUE = "Average Revenue"
MINIMUM_REVENUE = "Minimum Revenue"
MAXIMUM_REVENUE = "Maximum Revenue"
MODEL_RANGE = "Model Range"

REQUIRED_COLUMNS = {
    MODEL_COLUMN,
    CHANNEL_COLUMN,
    ATTRIBUTED_CONVERSIONS_COLUMN,
    ATTRIBUTED_REVENUE_COLUMN,
    CONVERSION_SHARE_COLUMN,
    REVENUE_SHARE_COLUMN,
    CHANNEL_RANK_COLUMN,
}


def load_attribution_summary() -> pd.DataFrame:
    """Load and validate channel-level attribution results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Attribution summary not found. Run "
            "'python src/attribution_models.py' first."
        )

    dataframe = pd.read_csv(INPUT_PATH)

    if dataframe.empty:
        raise ValueError(
            "The attribution summary dataset is empty."
        )

    missing_columns = REQUIRED_COLUMNS.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe[MODEL_COLUMN] = pd.Categorical(
        dataframe[MODEL_COLUMN],
        categories=MODEL_ORDER,
        ordered=True,
    )

    unknown_models = dataframe[
        MODEL_COLUMN
    ].isna()

    if unknown_models.any():
        raise ValueError(
            "Unknown attribution model names were detected."
        )

    numeric_columns = [
        ATTRIBUTED_CONVERSIONS_COLUMN,
        ATTRIBUTED_REVENUE_COLUMN,
        CONVERSION_SHARE_COLUMN,
        REVENUE_SHARE_COLUMN,
        CHANNEL_RANK_COLUMN,
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    return dataframe.sort_values(
        [
            MODEL_COLUMN,
            CHANNEL_RANK_COLUMN,
            CHANNEL_COLUMN,
        ]
    ).reset_index(drop=True)


def create_model_comparison(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a wide revenue-comparison table by channel."""

    comparison = dataframe.pivot_table(
        index=CHANNEL_COLUMN,
        columns=MODEL_COLUMN,
        values=ATTRIBUTED_REVENUE_COLUMN,
        aggfunc="sum",
        observed=False,
    ).fillna(0.0)

    comparison = comparison.reindex(
        columns=MODEL_ORDER,
        fill_value=0.0,
    )

    comparison[AVERAGE_REVENUE] = (
        comparison[MODEL_ORDER].mean(axis=1)
    )

    comparison[MINIMUM_REVENUE] = (
        comparison[MODEL_ORDER].min(axis=1)
    )

    comparison[MAXIMUM_REVENUE] = (
        comparison[MODEL_ORDER].max(axis=1)
    )

    comparison[MODEL_RANGE] = (
        comparison[MAXIMUM_REVENUE]
        - comparison[MINIMUM_REVENUE]
    )

    comparison = comparison.sort_values(
        AVERAGE_REVENUE,
        ascending=False,
    )

    comparison.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=True,
    )

    return comparison


def create_revenue_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Plot attributed revenue by channel and model."""

    revenue_pivot = dataframe.pivot_table(
        index=CHANNEL_COLUMN,
        columns=MODEL_COLUMN,
        values=ATTRIBUTED_REVENUE_COLUMN,
        aggfunc="sum",
        observed=False,
    ).fillna(0.0)

    revenue_pivot = revenue_pivot.reindex(
        columns=MODEL_ORDER,
        fill_value=0.0,
    )

    channel_order = (
        revenue_pivot.mean(axis=1)
        .sort_values(ascending=False)
        .index
    )

    revenue_pivot = revenue_pivot.loc[
        channel_order
    ]

    axis = revenue_pivot.plot(
        kind="bar",
        figsize=(15, 8),
    )

    axis.set_title(
        "Attributed Revenue by Channel and Model",
        fontsize=16,
        pad=16,
    )

    axis.set_xlabel("Marketing Channel")
    axis.set_ylabel("Attributed Revenue ($)")

    axis.tick_params(
        axis="x",
        rotation=35,
    )

    axis.legend(
        title="Attribution Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        REVENUE_CHART_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def get_top_channels(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return the highest-revenue channel for each model."""

    ordered_data = dataframe.sort_values(
        [
            MODEL_COLUMN,
            CHANNEL_RANK_COLUMN,
            ATTRIBUTED_REVENUE_COLUMN,
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    top_channels = ordered_data.drop_duplicates(
        subset=[MODEL_COLUMN],
        keep="first",
    ).copy()

    top_channels[MODEL_COLUMN] = pd.Categorical(
        top_channels[MODEL_COLUMN],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return top_channels.sort_values(
        MODEL_COLUMN
    ).reset_index(drop=True)


def create_top_channel_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Plot the top-revenue channel under each model."""

    top_channels = get_top_channels(dataframe)

    labels = (
        top_channels[MODEL_COLUMN]
        .astype("string")
        + "\n"
        + top_channels[CHANNEL_COLUMN].astype("string")
    )

    plt.figure(figsize=(12, 7))

    bars = plt.bar(
        labels,
        top_channels[ATTRIBUTED_REVENUE_COLUMN],
    )

    plt.title(
        "Top Marketing Channel Under Each Attribution Model",
        fontsize=16,
        pad=16,
    )

    plt.xlabel("Attribution Model and Top Channel")
    plt.ylabel("Attributed Revenue ($)")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)

    for bar, revenue in zip(
        bars,
        top_channels[ATTRIBUTED_REVENUE_COLUMN],
        strict=True,
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${revenue:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(
        TOP_CHANNEL_CHART_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_revenue_share_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Plot average channel revenue share across models."""

    average_share = (
        dataframe.groupby(
            CHANNEL_COLUMN,
            as_index=False,
        )[REVENUE_SHARE_COLUMN]
        .mean()
        .sort_values(
            REVENUE_SHARE_COLUMN,
            ascending=True,
        )
    )

    plt.figure(figsize=(11, 7))

    bars = plt.barh(
        average_share[CHANNEL_COLUMN],
        average_share[REVENUE_SHARE_COLUMN],
    )

    plt.title(
        "Average Revenue Share Across Attribution Models",
        fontsize=16,
        pad=16,
    )

    plt.xlabel("Average Revenue Share")
    plt.ylabel("Marketing Channel")
    plt.grid(axis="x", alpha=0.3)

    for bar, share in zip(
        bars,
        average_share[REVENUE_SHARE_COLUMN],
        strict=True,
    ):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {share:.1%}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(
        SHARE_CHART_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def print_summary(
    dataframe: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Print the main attribution-comparison findings."""

    top_channels = get_top_channels(dataframe)

    most_consistent_channel = (
        comparison[MODEL_RANGE].idxmin()
    )

    highest_average_channel = (
        comparison[AVERAGE_REVENUE].idxmax()
    )

    print("\nAttribution visualizations created")
    print("-" * 68)

    for _, row in top_channels.iterrows():
        model_name = str(row[MODEL_COLUMN])
        channel_name = str(row[CHANNEL_COLUMN])
        attributed_revenue = float(
            row[ATTRIBUTED_REVENUE_COLUMN]
        )

        print(
            f"{model_name:<18}"
            f"Top channel: {channel_name:<16}"
            f"Revenue: ${attributed_revenue:,.2f}"
        )

    print("-" * 68)

    print(
        "Highest average revenue channel: "
        f"{highest_average_channel}"
    )

    print(
        "Most consistent channel:         "
        f"{most_consistent_channel}"
    )

    print(
        f"Comparison table:  "
        f"{COMPARISON_OUTPUT_PATH}"
    )

    print(
        f"Revenue chart:     "
        f"{REVENUE_CHART_PATH}"
    )

    print(
        f"Top-channel chart: "
        f"{TOP_CHANNEL_CHART_PATH}"
    )

    print(
        f"Share chart:       "
        f"{SHARE_CHART_PATH}"
    )


def main() -> None:
    """Run the attribution-visualization pipeline."""

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_attribution_summary()

    comparison = create_model_comparison(
        dataframe
    )

    create_revenue_chart(dataframe)
    create_top_channel_chart(dataframe)
    create_revenue_share_chart(dataframe)

    print_summary(
        dataframe,
        comparison,
    )


if __name__ == "__main__":
    main()
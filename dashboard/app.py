"""Interactive dashboard for multi-touch marketing attribution."""

from pathlib import Path
from typing import Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PROCESSED_DIRECTORY: Final = PROJECT_ROOT / "data" / "processed"

JOURNEY_SUMMARY_PATH: Final = (
    PROCESSED_DIRECTORY / "journey_summary.csv"
)

ATTRIBUTION_SUMMARY_PATH: Final = (
    PROCESSED_DIRECTORY
    / "attribution_channel_summary.csv"
)

# Dataset column names
MODEL_COLUMN: Final = "model"
CHANNEL_COLUMN: Final = "channel"
CONVERTED_COLUMN: Final = "converted"
CONVERSION_VALUE_COLUMN: Final = "conversion_value"
TOUCHPOINT_COUNT_COLUMN: Final = "touchpoint_count"
CONVERSION_TIMESTAMP_COLUMN: Final = "conversion_timestamp"
ATTRIBUTED_CONVERSIONS_COLUMN: Final = "attributed_conversions"
ATTRIBUTED_REVENUE_COLUMN: Final = "attributed_revenue"
CONVERSION_SHARE_COLUMN: Final = "conversion_share"
REVENUE_SHARE_COLUMN: Final = "revenue_share"
CHANNEL_RANK_COLUMN: Final = "channel_rank"
CONVERSION_MONTH_COLUMN: Final = "conversion_month"
MONTHLY_REVENUE_COLUMN: Final = "revenue"

# Display labels
ATTRIBUTED_REVENUE_LABEL: Final = "Attributed Revenue"
ATTRIBUTED_CONVERSIONS_LABEL: Final = "Attributed Conversions"
MARKETING_CHANNEL_LABEL: Final = "Marketing Channel"
CONVERSION_SHARE_LABEL: Final = "Conversion Share"
REVENUE_SHARE_LABEL: Final = "Revenue Share"
ATTRIBUTION_MODEL_LABEL: Final = "Attribution Model"
CONVERSION_MONTH_LABEL: Final = "Conversion Month"
RANK_LABEL: Final = "Rank"
CHANNEL_LABEL: Final = "Channel"

# Attribution models
FIRST_TOUCH: Final = "First Touch"
LAST_TOUCH: Final = "Last Touch"
LINEAR: Final = "Linear"
TIME_DECAY: Final = "Time Decay"
POSITION_BASED: Final = "Position Based"

MODEL_ORDER: Final = [
    FIRST_TOUCH,
    LAST_TOUCH,
    LINEAR,
    TIME_DECAY,
    POSITION_BASED,
]

REQUIRED_JOURNEY_COLUMNS: Final = {
    CONVERTED_COLUMN,
    CONVERSION_VALUE_COLUMN,
    TOUCHPOINT_COUNT_COLUMN,
    CONVERSION_TIMESTAMP_COLUMN,
}

REQUIRED_ATTRIBUTION_COLUMNS: Final = {
    MODEL_COLUMN,
    CHANNEL_COLUMN,
    ATTRIBUTED_CONVERSIONS_COLUMN,
    ATTRIBUTED_REVENUE_COLUMN,
    CONVERSION_SHARE_COLUMN,
    REVENUE_SHARE_COLUMN,
    CHANNEL_RANK_COLUMN,
}


st.set_page_config(
    page_title="Marketing Attribution Dashboard",
    page_icon="📊",
    layout="wide",
)


def validate_file(
    file_path: Path,
    description: str,
) -> None:
    """Confirm that a required dashboard file exists."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found at {file_path}. "
            "Run the data pipeline first."
        )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Confirm that a dataset contains all required columns."""

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the journey and attribution summary datasets."""

    validate_file(
        JOURNEY_SUMMARY_PATH,
        "Journey summary",
    )

    validate_file(
        ATTRIBUTION_SUMMARY_PATH,
        "Attribution summary",
    )

    journey_summary = pd.read_csv(
        JOURNEY_SUMMARY_PATH
    )

    attribution_summary = pd.read_csv(
        ATTRIBUTION_SUMMARY_PATH
    )

    validate_columns(
        journey_summary,
        REQUIRED_JOURNEY_COLUMNS,
        "Journey summary",
    )

    validate_columns(
        attribution_summary,
        REQUIRED_ATTRIBUTION_COLUMNS,
        "Attribution summary",
    )

    journey_summary[
        CONVERSION_TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        journey_summary[CONVERSION_TIMESTAMP_COLUMN],
        errors="coerce",
    )

    numeric_journey_columns = [
        CONVERTED_COLUMN,
        CONVERSION_VALUE_COLUMN,
        TOUCHPOINT_COUNT_COLUMN,
    ]

    for column in numeric_journey_columns:
        journey_summary[column] = pd.to_numeric(
            journey_summary[column],
            errors="raise",
        )

    numeric_attribution_columns = [
        ATTRIBUTED_CONVERSIONS_COLUMN,
        ATTRIBUTED_REVENUE_COLUMN,
        CONVERSION_SHARE_COLUMN,
        REVENUE_SHARE_COLUMN,
        CHANNEL_RANK_COLUMN,
    ]

    for column in numeric_attribution_columns:
        attribution_summary[column] = pd.to_numeric(
            attribution_summary[column],
            errors="raise",
        )

    return journey_summary, attribution_summary


def format_currency(value: float) -> str:
    """Format a numeric value as US currency."""

    return f"${value:,.2f}"


def create_revenue_chart(
    model_data: pd.DataFrame,
    selected_model: str,
) -> go.Figure:
    """Create an attributed-revenue chart by channel."""

    chart_data = model_data.sort_values(
        ATTRIBUTED_REVENUE_COLUMN,
        ascending=True,
    )

    figure = px.bar(
        chart_data,
        x=ATTRIBUTED_REVENUE_COLUMN,
        y=CHANNEL_COLUMN,
        orientation="h",
        text_auto=".3s",
        title=(
            "Attributed Revenue by Channel "
            f"under {selected_model}"
        ),
        labels={
            ATTRIBUTED_REVENUE_COLUMN:
                ATTRIBUTED_REVENUE_LABEL,
            CHANNEL_COLUMN:
                MARKETING_CHANNEL_LABEL,
        },
    )

    figure.update_layout(
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis_title=None,
        height=470,
    )

    return figure


def create_conversion_chart(
    model_data: pd.DataFrame,
    selected_model: str,
) -> go.Figure:
    """Create an attributed-conversions chart by channel."""

    chart_data = model_data.sort_values(
        ATTRIBUTED_CONVERSIONS_COLUMN,
        ascending=False,
    )

    figure = px.bar(
        chart_data,
        x=CHANNEL_COLUMN,
        y=ATTRIBUTED_CONVERSIONS_COLUMN,
        text_auto=".3s",
        title=(
            "Attributed Conversions by Channel "
            f"under {selected_model}"
        ),
        labels={
            ATTRIBUTED_CONVERSIONS_COLUMN:
                ATTRIBUTED_CONVERSIONS_LABEL,
            CHANNEL_COLUMN:
                MARKETING_CHANNEL_LABEL,
        },
    )

    figure.update_layout(
        xaxis_title=None,
        height=470,
    )

    return figure


def create_monthly_trend_chart(
    journey_summary: pd.DataFrame,
) -> go.Figure:
    """Create the monthly conversion-revenue trend."""

    converted_journeys = journey_summary.loc[
        (
            journey_summary[CONVERTED_COLUMN] == 1
        )
        & journey_summary[
            CONVERSION_TIMESTAMP_COLUMN
        ].notna()
    ].copy()

    converted_journeys[
        CONVERSION_MONTH_COLUMN
    ] = (
        converted_journeys[
            CONVERSION_TIMESTAMP_COLUMN
        ]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_summary = (
        converted_journeys.groupby(
            CONVERSION_MONTH_COLUMN,
            as_index=False,
        )
        .agg(
            conversions=(
                CONVERTED_COLUMN,
                "sum",
            ),
            revenue=(
                CONVERSION_VALUE_COLUMN,
                "sum",
            ),
        )
    )

    figure = px.line(
        monthly_summary,
        x=CONVERSION_MONTH_COLUMN,
        y=MONTHLY_REVENUE_COLUMN,
        markers=True,
        title="Monthly Conversion Revenue",
        labels={
            CONVERSION_MONTH_COLUMN:
                CONVERSION_MONTH_LABEL,
            MONTHLY_REVENUE_COLUMN:
                "Revenue",
        },
    )

    figure.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        height=430,
    )

    return figure


def create_model_comparison_chart(
    attribution_summary: pd.DataFrame,
    selected_channels: list[str],
) -> go.Figure:
    """Compare attributed revenue across all models."""

    comparison_data = attribution_summary.loc[
        attribution_summary[CHANNEL_COLUMN].isin(
            selected_channels
        )
    ].copy()

    comparison_data[MODEL_COLUMN] = pd.Categorical(
        comparison_data[MODEL_COLUMN],
        categories=MODEL_ORDER,
        ordered=True,
    )

    comparison_data = comparison_data.sort_values(
        MODEL_COLUMN
    )

    figure = px.bar(
        comparison_data,
        x=MODEL_COLUMN,
        y=ATTRIBUTED_REVENUE_COLUMN,
        color=CHANNEL_COLUMN,
        barmode="group",
        title="Channel Revenue Comparison Across Models",
        labels={
            MODEL_COLUMN:
                ATTRIBUTION_MODEL_LABEL,
            ATTRIBUTED_REVENUE_COLUMN:
                ATTRIBUTED_REVENUE_LABEL,
            CHANNEL_COLUMN:
                MARKETING_CHANNEL_LABEL,
        },
    )

    figure.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        height=520,
    )

    return figure


def create_display_table(
    model_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a formatted channel-performance table."""

    display_columns = [
        CHANNEL_RANK_COLUMN,
        CHANNEL_COLUMN,
        ATTRIBUTED_CONVERSIONS_COLUMN,
        ATTRIBUTED_REVENUE_COLUMN,
        CONVERSION_SHARE_COLUMN,
        REVENUE_SHARE_COLUMN,
    ]

    display_data = model_data[
        display_columns
    ].copy()

    display_data = display_data.rename(
        columns={
            CHANNEL_RANK_COLUMN:
                RANK_LABEL,
            CHANNEL_COLUMN:
                CHANNEL_LABEL,
            ATTRIBUTED_CONVERSIONS_COLUMN:
                ATTRIBUTED_CONVERSIONS_LABEL,
            ATTRIBUTED_REVENUE_COLUMN:
                ATTRIBUTED_REVENUE_LABEL,
            CONVERSION_SHARE_COLUMN:
                CONVERSION_SHARE_LABEL,
            REVENUE_SHARE_COLUMN:
                REVENUE_SHARE_LABEL,
        }
    )

    display_data[
        ATTRIBUTED_CONVERSIONS_LABEL
    ] = (
        display_data[
            ATTRIBUTED_CONVERSIONS_LABEL
        ].round(2)
    )

    display_data[
        ATTRIBUTED_REVENUE_LABEL
    ] = (
        display_data[
            ATTRIBUTED_REVENUE_LABEL
        ].map(format_currency)
    )

    display_data[
        CONVERSION_SHARE_LABEL
    ] = (
        display_data[
            CONVERSION_SHARE_LABEL
        ].map(
            lambda value: f"{value:.2%}"
        )
    )

    display_data[
        REVENUE_SHARE_LABEL
    ] = (
        display_data[
            REVENUE_SHARE_LABEL
        ].map(
            lambda value: f"{value:.2%}"
        )
    )

    return display_data.sort_values(
        RANK_LABEL
    ).reset_index(drop=True)


def main() -> None:
    """Render the interactive attribution dashboard."""

    st.title("Multi-Touch Marketing Attribution")

    st.caption(
        "Compare how different attribution models assign "
        "conversion and revenue credit across marketing channels."
    )

    try:
        journey_summary, attribution_summary = (
            load_dashboard_data()
        )
    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        st.error(str(error))
        st.stop()

    available_models = [
        model
        for model in MODEL_ORDER
        if model
        in attribution_summary[MODEL_COLUMN].unique()
    ]

    available_channels = sorted(
        attribution_summary[
            CHANNEL_COLUMN
        ].unique()
    )

    st.sidebar.header("Dashboard Filters")

    default_model_index = (
        available_models.index(LINEAR)
        if LINEAR in available_models
        else 0
    )

    selected_model = st.sidebar.selectbox(
        "Attribution model",
        options=available_models,
        index=default_model_index,
    )

    selected_channels = st.sidebar.multiselect(
        "Marketing channels",
        options=available_channels,
        default=available_channels,
    )

    if not selected_channels:
        st.warning(
            "Select at least one marketing channel "
            "from the sidebar."
        )
        st.stop()

    model_data = attribution_summary.loc[
        (
            attribution_summary[MODEL_COLUMN]
            == selected_model
        )
        & attribution_summary[CHANNEL_COLUMN].isin(
            selected_channels
        )
    ].copy()

    if model_data.empty:
        st.warning(
            "No attribution data matches the selected filters."
        )
        st.stop()

    total_journeys = len(journey_summary)

    total_conversions = int(
        journey_summary[CONVERTED_COLUMN].sum()
    )

    conversion_rate = (
        total_conversions / total_journeys
        if total_journeys
        else 0.0
    )

    total_revenue = float(
        journey_summary[
            CONVERSION_VALUE_COLUMN
        ].sum()
    )

    average_touchpoints = float(
        journey_summary[
            TOUCHPOINT_COUNT_COLUMN
        ].mean()
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Total Journeys",
        f"{total_journeys:,}",
    )

    metric_columns[1].metric(
        "Conversions",
        f"{total_conversions:,}",
    )

    metric_columns[2].metric(
        "Conversion Rate",
        f"{conversion_rate:.2%}",
    )

    metric_columns[3].metric(
        "Total Revenue",
        format_currency(total_revenue),
    )

    metric_columns[4].metric(
        "Avg. Touchpoints",
        f"{average_touchpoints:.2f}",
    )

    st.divider()

    top_channel_row = model_data.sort_values(
        ATTRIBUTED_REVENUE_COLUMN,
        ascending=False,
    ).iloc[0]

    top_channel = str(
        top_channel_row[CHANNEL_COLUMN]
    )

    top_channel_revenue = float(
        top_channel_row[
            ATTRIBUTED_REVENUE_COLUMN
        ]
    )

    st.subheader(
        f"{selected_model} Overview"
    )

    st.info(
        f"Under the {selected_model} model, "
        f"{top_channel} receives the highest revenue credit "
        f"at {format_currency(top_channel_revenue)}."
    )

    left_chart, right_chart = st.columns(2)

    with left_chart:
        revenue_figure = create_revenue_chart(
            model_data,
            selected_model,
        )

        st.plotly_chart(
            revenue_figure,
            width="stretch",
        )

    with right_chart:
        conversion_figure = create_conversion_chart(
            model_data,
            selected_model,
        )

        st.plotly_chart(
            conversion_figure,
            width="stretch",
        )

    st.subheader(
        "Attribution Model Comparison"
    )

    comparison_figure = (
        create_model_comparison_chart(
            attribution_summary,
            selected_channels,
        )
    )

    st.plotly_chart(
        comparison_figure,
        width="stretch",
    )

    st.subheader("Conversion Trend")

    trend_figure = create_monthly_trend_chart(
        journey_summary
    )

    st.plotly_chart(
        trend_figure,
        width="stretch",
    )

    st.subheader(
        "Channel Performance Table"
    )

    display_table = create_display_table(
        model_data
    )

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
    )

    download_data = model_data.to_csv(
        index=False
    ).encode("utf-8")

    download_file_name = (
        selected_model
        .lower()
        .replace(" ", "_")
        + "_attribution.csv"
    )

    st.download_button(
        label="Download filtered attribution data",
        data=download_data,
        file_name=download_file_name,
        mime="text/csv",
    )

    st.caption(
        "The current project uses synthetic customer journey data "
        "to demonstrate attribution methodology."
    )


if __name__ == "__main__":
    main()
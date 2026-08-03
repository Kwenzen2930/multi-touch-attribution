"""Generate synthetic customer journey data for attribution analysis."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_CUSTOMERS = 5_000
START_DATE = pd.Timestamp("2025-01-01")

# Marketing channel constants
PAID_SEARCH = "Paid Search"
ORGANIC_SEARCH = "Organic Search"
PAID_SOCIAL = "Paid Social"
ORGANIC_SOCIAL = "Organic Social"
EMAIL = "Email"
DISPLAY = "Display"
REFERRAL = "Referral"
DIRECT = "Direct"

CHANNELS = [
    PAID_SEARCH,
    ORGANIC_SEARCH,
    PAID_SOCIAL,
    ORGANIC_SOCIAL,
    EMAIL,
    DISPLAY,
    REFERRAL,
    DIRECT,
]

CHANNEL_PROBABILITIES = [
    0.18,
    0.16,
    0.15,
    0.09,
    0.14,
    0.08,
    0.08,
    0.12,
]

CHANNEL_EFFECTIVENESS = {
    PAID_SEARCH: 0.35,
    ORGANIC_SEARCH: 0.28,
    PAID_SOCIAL: 0.20,
    ORGANIC_SOCIAL: 0.12,
    EMAIL: 0.38,
    DISPLAY: 0.10,
    REFERRAL: 0.32,
    DIRECT: 0.40,
}

CAMPAIGNS = {
    PAID_SEARCH: [
        "Brand Search",
        "Product Search",
        "Competitor Search",
    ],
    ORGANIC_SEARCH: [
        "SEO Blog",
        "Product Pages",
        "Educational Content",
    ],
    PAID_SOCIAL: [
        "Instagram Ads",
        "Facebook Ads",
        "LinkedIn Ads",
    ],
    ORGANIC_SOCIAL: [
        "Instagram Organic",
        "TikTok Organic",
        "LinkedIn Organic",
    ],
    EMAIL: [
        "Welcome Series",
        "Newsletter",
        "Promotional Email",
    ],
    DISPLAY: [
        "Retargeting",
        "Prospecting",
        "Banner Campaign",
    ],
    REFERRAL: [
        "Affiliate",
        "Partner Website",
        "Customer Referral",
    ],
    DIRECT: [
        "Direct Visit",
    ],
}

MOBILE = "Mobile"
DESKTOP = "Desktop"
TABLET = "Tablet"

DEVICES = [
    MOBILE,
    DESKTOP,
    TABLET,
]

DEVICE_PROBABILITIES = [
    0.58,
    0.36,
    0.06,
]


def choose_channel(
    rng: np.random.Generator,
    previous_channel: str | None,
) -> str:
    """Select a marketing channel while reducing immediate repetition."""

    probabilities = np.array(
        CHANNEL_PROBABILITIES,
        dtype=float,
    )

    if previous_channel is not None:
        previous_index = CHANNELS.index(previous_channel)

        probabilities[previous_index] *= 0.25
        probabilities /= probabilities.sum()

    return str(
        rng.choice(
            CHANNELS,
            p=probabilities,
        )
    )


def calculate_conversion_probability(
    channels: list[str],
) -> float:
    """Calculate conversion probability from a customer's journey."""

    effectiveness_scores = [
        CHANNEL_EFFECTIVENESS[channel]
        for channel in channels
    ]

    probability = (
        0.03
        + (0.025 * len(channels))
        + (0.14 * max(effectiveness_scores))
        + (0.08 * np.mean(effectiveness_scores))
    )

    if EMAIL in channels:
        probability += 0.04

    if REFERRAL in channels:
        probability += 0.03

    high_intent_channels = {
        DIRECT,
        EMAIL,
        PAID_SEARCH,
    }

    if channels[-1] in high_intent_channels:
        probability += 0.05

    return float(
        np.clip(
            probability,
            0.03,
            0.70,
        )
    )


def generate_customer_journeys(
    number_of_customers: int = NUMBER_OF_CUSTOMERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate one dataset row for each marketing touchpoint."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for customer_number in range(
        1,
        number_of_customers + 1,
    ):
        customer_id = f"CUST-{customer_number:05d}"
        journey_id = f"JOURNEY-{customer_number:05d}"

        number_of_touchpoints = int(
            np.clip(
                rng.poisson(lam=2.5) + 1,
                1,
                8,
            )
        )

        journey_start = START_DATE + pd.Timedelta(
            days=int(rng.integers(0, 180)),
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )

        channels: list[str] = []
        previous_channel: str | None = None

        for _ in range(number_of_touchpoints):
            channel = choose_channel(
                rng,
                previous_channel,
            )

            channels.append(channel)
            previous_channel = channel

        conversion_probability = (
            calculate_conversion_probability(channels)
        )

        converted = bool(
            rng.random() < conversion_probability
        )

        conversion_value = (
            round(
                float(
                    rng.lognormal(
                        mean=4.8,
                        sigma=0.55,
                    )
                ),
                2,
            )
            if converted
            else 0.0
        )

        current_timestamp = journey_start

        for touchpoint_order, channel in enumerate(
            channels,
            start=1,
        ):
            if touchpoint_order > 1:
                current_timestamp += pd.Timedelta(
                    hours=int(rng.integers(2, 96)),
                    minutes=int(rng.integers(0, 60)),
                )

            is_final_touchpoint = (
                touchpoint_order == number_of_touchpoints
            )

            is_conversion = int(
                converted and is_final_touchpoint
            )

            campaign = str(
                rng.choice(CAMPAIGNS[channel])
            )

            device = str(
                rng.choice(
                    DEVICES,
                    p=DEVICE_PROBABILITIES,
                )
            )

            records.append(
                {
                    "customer_id": customer_id,
                    "journey_id": journey_id,
                    "touchpoint_id": (
                        f"{journey_id}-T{touchpoint_order:02d}"
                    ),
                    "touchpoint_order": touchpoint_order,
                    "timestamp": current_timestamp,
                    "channel": channel,
                    "campaign": campaign,
                    "device": device,
                    "converted": int(converted),
                    "is_conversion": is_conversion,
                    "conversion_value": (
                        conversion_value
                        if is_conversion
                        else 0.0
                    ),
                }
            )

    dataframe = pd.DataFrame(records)

    return dataframe.sort_values(
        by=[
            "journey_id",
            "touchpoint_order",
        ]
    ).reset_index(drop=True)


def validate_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the structure and business rules of the dataset."""

    required_columns = {
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

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("The generated dataset is empty.")

    if dataframe["touchpoint_id"].duplicated().any():
        raise ValueError(
            "Duplicate touchpoint IDs were detected."
        )

    if dataframe["journey_id"].isna().any():
        raise ValueError(
            "Missing journey IDs were detected."
        )

    if dataframe["channel"].isna().any():
        raise ValueError(
            "Missing marketing channels were detected."
        )

    if not dataframe["channel"].isin(CHANNELS).all():
        raise ValueError(
            "Unknown marketing channels were detected."
        )

    conversion_rows = dataframe[
        dataframe["is_conversion"] == 1
    ]

    if not (
        conversion_rows["conversion_value"] > 0
    ).all():
        raise ValueError(
            "Every conversion must have a positive value."
        )

    non_conversion_rows = dataframe[
        dataframe["is_conversion"] == 0
    ]

    if not (
        non_conversion_rows["conversion_value"] == 0
    ).all():
        raise ValueError(
            "Non-conversion rows must have zero conversion value."
        )

    conversions_per_journey = (
        dataframe.groupby("journey_id")["is_conversion"]
        .sum()
    )

    if (conversions_per_journey > 1).any():
        raise ValueError(
            "A customer journey cannot contain multiple conversions."
        )


def save_dataset(
    dataframe: pd.DataFrame,
) -> Path:
    """Save the generated dataset as a CSV file."""

    project_root = Path(__file__).resolve().parents[1]

    output_directory = (
        project_root
        / "data"
        / "raw"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "customer_journeys.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def print_summary(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Print a summary of the generated customer journey dataset."""

    total_journeys = dataframe[
        "journey_id"
    ].nunique()

    total_touchpoints = len(dataframe)

    total_conversions = int(
        dataframe["is_conversion"].sum()
    )

    total_revenue = float(
        dataframe["conversion_value"].sum()
    )

    conversion_rate = (
        total_conversions / total_journeys
    )

    average_touchpoints = (
        total_touchpoints / total_journeys
    )

    print("\nCustomer journey dataset created")
    print("-" * 44)
    print(f"Journeys:             {total_journeys:,}")
    print(f"Touchpoints:          {total_touchpoints:,}")
    print(f"Average touchpoints:  {average_touchpoints:.2f}")
    print(f"Conversions:          {total_conversions:,}")
    print(f"Conversion rate:      {conversion_rate:.2%}")
    print(f"Revenue:              ${total_revenue:,.2f}")
    print(f"Saved to:             {output_path}")


def main() -> None:
    """Run the complete customer journey generation pipeline."""

    dataframe = generate_customer_journeys()

    validate_dataset(dataframe)

    output_path = save_dataset(dataframe)

    print_summary(
        dataframe,
        output_path,
    )


if __name__ == "__main__":
    main()
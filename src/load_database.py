"""Load processed marketing attribution datasets into SQLite."""

from pathlib import Path
from typing import Final

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PROCESSED_DIRECTORY: Final = PROJECT_ROOT / "data" / "processed"
DATABASE_PATH: Final = PROCESSED_DIRECTORY / "attribution.db"

TABLE_FILES: Final = {
    "clean_customer_journeys": (
        PROCESSED_DIRECTORY / "clean_customer_journeys.csv"
    ),
    "journey_summary": (
        PROCESSED_DIRECTORY / "journey_summary.csv"
    ),
    "attribution_touchpoint_credits": (
        PROCESSED_DIRECTORY
        / "attribution_touchpoint_credits.csv"
    ),
    "attribution_channel_summary": (
        PROCESSED_DIRECTORY
        / "attribution_channel_summary.csv"
    ),
}

DATE_COLUMNS: Final = {
    "clean_customer_journeys": [
        "timestamp",
        "first_touch_timestamp",
        "last_touch_timestamp",
    ],
    "journey_summary": [
        "first_touch_timestamp",
        "last_touch_timestamp",
        "conversion_timestamp",
    ],
    "attribution_touchpoint_credits": [
        "timestamp",
    ],
}


def validate_input_files() -> None:
    """Confirm that every required processed CSV exists."""

    missing_files = [
        str(file_path)
        for file_path in TABLE_FILES.values()
        if not file_path.exists()
    ]

    if missing_files:
        missing_list = "\n".join(missing_files)

        raise FileNotFoundError(
            "The following processed files are missing:\n"
            f"{missing_list}\n\n"
            "Run the generator, cleaning, and attribution "
            "scripts first."
        )


def create_database_engine() -> Engine:
    """Create the SQLite database connection."""

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return create_engine(
        f"sqlite:///{DATABASE_PATH}"
    )


def read_dataset(
    table_name: str,
    file_path: Path,
) -> pd.DataFrame:
    """Read and prepare one processed CSV dataset."""

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            f"The source file for {table_name} is empty."
        )

    for column in DATE_COLUMNS.get(table_name, []):
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def load_tables(
    engine: Engine,
) -> dict[str, int]:
    """Load all processed datasets into SQLite tables."""

    table_counts: dict[str, int] = {}

    for table_name, file_path in TABLE_FILES.items():
        dataframe = read_dataset(
            table_name,
            file_path,
        )

        dataframe.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=250,
            method="multi",
        )

        table_counts[table_name] = len(dataframe)

    return table_counts


def create_indexes(
    engine: Engine,
) -> None:
    """Create indexes for common analysis fields."""

    index_statements = [
        """
        CREATE INDEX IF NOT EXISTS
        idx_touchpoints_journey_id
        ON clean_customer_journeys (journey_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS
        idx_touchpoints_channel
        ON clean_customer_journeys (channel)
        """,
        """
        CREATE INDEX IF NOT EXISTS
        idx_touchpoints_timestamp
        ON clean_customer_journeys (timestamp)
        """,
        """
        CREATE INDEX IF NOT EXISTS
        idx_journey_summary_converted
        ON journey_summary (converted)
        """,
        """
        CREATE INDEX IF NOT EXISTS
        idx_credits_model_channel
        ON attribution_touchpoint_credits (
            model,
            channel
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS
        idx_channel_summary_model_rank
        ON attribution_channel_summary (
            model,
            channel_rank
        )
        """,
    ]

    with engine.begin() as connection:
        for statement in index_statements:
            connection.execute(text(statement))


def verify_tables(
    engine: Engine,
    expected_counts: dict[str, int],
) -> None:
    """Confirm that each SQLite table has the expected rows."""

    with engine.connect() as connection:
        for table_name, expected_count in expected_counts.items():
            query = text(
                f"SELECT COUNT(*) FROM {table_name}"
            )

            actual_count = int(
                connection.execute(query).scalar_one()
            )

            if actual_count != expected_count:
                raise ValueError(
                    f"{table_name} contains {actual_count:,} "
                    f"rows; expected {expected_count:,}."
                )


def print_summary(
    table_counts: dict[str, int],
) -> None:
    """Print the completed database summary."""

    print("\nSQLite attribution database created")
    print("-" * 58)

    for table_name, row_count in table_counts.items():
        print(
            f"{table_name:<38}"
            f"{row_count:>10,} rows"
        )

    print("-" * 58)
    print(f"Database: {DATABASE_PATH}")


def main() -> None:
    """Run the complete SQLite loading process."""

    validate_input_files()

    engine = create_database_engine()

    table_counts = load_tables(engine)

    create_indexes(engine)

    verify_tables(
        engine,
        table_counts,
    )

    print_summary(table_counts)


if __name__ == "__main__":
    main()
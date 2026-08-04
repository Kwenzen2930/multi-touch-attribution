# Multi-Touch Marketing Attribution

A data analytics project that compares multiple attribution models to determine which marketing channels drive conversions and revenue.

## Live Dashboard

[View the interactive Multi-Touch Attribution Dashboard](https://multi-touch-attribution-kelvin.streamlit.app/)

## Project Overview

This project analyzes synthetic customer journeys across eight marketing channels:

* Paid Search
* Organic Search
* Paid Social
* Organic Social
* Email
* Display
* Referral
* Direct

It compares five attribution models:

* First Touch
* Last Touch
* Linear
* Time Decay
* Position Based

The project includes data generation, cleaning, attribution modeling, SQLite analysis, visualizations, and an interactive Streamlit dashboard.

## Key Results

The generated dataset contains:

* 5,000 customer journeys
* 17,632 marketing touchpoints
* 1,223 conversions
* 24.46% conversion rate
* $169,570.39 total conversion revenue

| Attribution Model | Top Channel | Attributed Conversions | Attributed Revenue |
| ----------------- | ----------- | ---------------------: | -----------------: |
| First Touch       | Paid Search |                 224.00 |         $31,603.91 |
| Last Touch        | Email       |                 215.00 |         $29,902.12 |
| Linear            | Paid Search |                 209.73 |         $28,415.89 |
| Position Based    | Paid Search |                 212.52 |         $28,877.80 |
| Time Decay        | Paid Search |                 207.40 |         $27,935.41 |

The results demonstrate that channel performance changes depending on how conversion credit is assigned.

## Dashboard Features

* Attribution-model selection
* Marketing-channel filters
* Conversion and revenue metrics
* Revenue comparison by channel
* Conversion comparison by channel
* Cross-model performance comparison
* Monthly conversion revenue trend
* Downloadable filtered attribution data

## Technology Stack

* Python
* Pandas
* NumPy
* SQLAlchemy
* SQLite
* Streamlit
* Plotly
* Matplotlib
* scikit-learn

## Project Structure

```text
multi-touch-attribution/
├── dashboard/
│   ├── app.py
│   └── data/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── sql/
│   └── analysis.sql
├── src/
│   ├── attribution_models.py
│   ├── data_cleaning.py
│   ├── data_generator.py
│   ├── load_database.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Kwenzen2930/multi-touch-attribution.git
cd multi-touch-attribution
```

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

## Attribution Models

### First Touch

Assigns all conversion credit to the first channel in the customer journey.

### Last Touch

Assigns all conversion credit to the final channel before conversion.

### Linear

Distributes conversion credit equally across every touchpoint.

### Time Decay

Assigns more credit to touchpoints that occur closer to the conversion.

### Position Based

Assigns more credit to the first and last touchpoints while distributing the remaining credit across the middle interactions.

## Data Notice

This project uses synthetic customer journey data created for analytical and portfolio demonstration purposes.

## Author

Kelvin Wenzen

[GitHub](https://github.com/Kwenzen2930)

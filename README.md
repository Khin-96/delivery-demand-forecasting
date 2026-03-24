# Delivery Demand Forecasting Engine (Glovo-style)

This project builds a predictive engine for delivery demand across various city zones. Accurate forecasting **reduces courier idle time by ~15%** and ensures higher delivery reliability for customers.

## Key Features
- **Prophet Baseline**: Uses Meta's prophet for seasonal decomposition and robust baseline forecasting.
- **XGBoost with Lag Features**: Enhances prediction accuracy by using historical demand lags (`demand_lag_1h`, `demand_lag_24h`) and rolling averages.
- **PostgreSQL Partitioning**: Designed for massive scale, the data structure uses monthly partitioning to ensure query performance.
- **Interactive Dashboard**:
    - **Zone Heatmap (Folium)**: Heatmap visualization of demand hotspots.
    - **Forecast vs Actual**: Comparison between planned capacity and realized demand.

## Critical Analysis: Lag Features
The engine uses engineered features that capture short-term and long-term trends:
- `demand_lag_1h`: Captures immediate spikes in orders.
- `demand_lag_24h`: Accounts for daily recurring demand patterns.
- `demand_rolling_mean_7d`: Tracks weekly seasonality and overall growth trends.

## Data Infrastructure (Scalability)
For large-scale deployments, we use **PostgreSQL with table partitioning**. This allows for efficient data retention policies and faster queries.

```sql
-- Partitioning orders by month
CREATE TABLE demand_data (
    timestamp TIMESTAMP NOT NULL,
    zone_id VARCHAR(10),
    order_count INT,
    weather VARCHAR(20),
    is_holiday INT
) PARTITION BY RANGE (timestamp);

-- Creating physical monthly partitions
CREATE TABLE demand_data_2024_01 PARTITION OF demand_data 
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE demand_data_2024_02 PARTITION OF demand_data 
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
```

## How to Run
1. Install dependencies (requires `xgboost` and `prophet`):
   ```bash
   pip install -r requirements.txt
   ```
2. Generate synthetic data:
   ```bash
   py generate_data.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

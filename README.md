# Spatio-Temporal Demand Forecasting & Resource Optimization

An enterprise-grade forecasting engine that integrates **Machine Learning (XGBoost)** with **Explainable AI (SHAP)** and **Supply-Demand Optimization** to minimize courier idle time and operational inefficiency.

## Advanced Infrastructure
- **XGBoost & Lag Engineering**: High-fidelity modeling using historical demand lags (`demand_lag_24h`) and weekly rolling features.
- **Explainable AI (SHAP)**: Identifies the primary drivers of demand (Weather vs. Seasonality vs. Day of Week) to allow for human-in-the-loop validation of forecasts.
- **PostgreSQL Partitioning**: Architectural design for monthly data isolation, ensuring query performance at petabyte scale.
- **Folium Spatio-Temporal Mapping**: Visualizing demand hotspots across city zones in real-time.

## Data Infrastructure (Scalability)
```sql
-- Partitioning orders by month
CREATE TABLE demand_data (
    timestamp TIMESTAMP NOT NULL,
    zone_id VARCHAR(10),
    order_count INT,
    weather VARCHAR(20),
    is_holiday INT
) PARTITION BY RANGE (timestamp);
```

## How to Run
1. Install dependencies:
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

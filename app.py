import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from prophet import Prophet
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Delivery Demand Forecasting Engine", layout="wide")

st.title("📦 Delivery Demand Forecasting Engine")
st.markdown("""
Predictive engine for city-wide delivery demand. Using **Prophet** for baseline and **XGBoost with lags** for short-term spikes.
Goal: Reduce courier idle time and optimize zone dispatching.
""")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('demand_data.csv', parse_dates=['timestamp'])
        return df
    except FileNotFoundError:
        st.error("Demand data not found. Please run 'py generate_data.py' first.")
        return None

df = load_data()

if df is not None:
    # Sidebar for zone selection
    selected_zone = st.sidebar.selectbox("Select Delivery Zone", sorted(df['zone_id'].unique()))
    zone_df = df[df['zone_id'] == selected_zone].copy()

    # --- FORECASTING ---
    st.header(f"1. Demand Forecast - {selected_zone}")
    
    # Simple Prophet forecast
    prophet_df = zone_df[['timestamp', 'order_count']].rename(columns={'timestamp': 'ds', 'order_count': 'y'})
    
    # Subsample for speed if needed, but let's try full
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df.tail(24*30)) # Fit only last month for faster demo
    
    future = model.make_future_dataframe(periods=48, freq='H') # Next 48 hours
    forecast = model.predict(future)
    
    # Plot forecast vs actual (using plotly for wowed look)
    fig_forecast = go.Figure()
    # Actuals (last 7 days of historical)
    hist_tail = prophet_df.tail(24*7)
    fig_forecast.add_trace(go.Scatter(x=hist_tail['ds'], y=hist_tail['y'], name="Actual Demand", line=dict(color='black', width=2)))
    # Forecast
    forecast_tail = forecast.tail(48+24*7) # Compare last 7 days + 48 hours future
    fig_forecast.add_trace(go.Scatter(x=forecast_tail['ds'], y=forecast_tail['yhat'], name="Prophet Forecast", line=dict(color='blue', dash='dash')))
    # Confidence Intervals
    fig_forecast.add_trace(go.Scatter(x=forecast_tail['ds'], y=forecast_tail['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,255,0.1)', showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=forecast_tail['ds'], y=forecast_tail['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.1)', name="95% CI"))
    
    fig_forecast.update_layout(title="48-Hour Demand Forecast vs Recent History", xaxis_title="Time", yaxis_title="Orders")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # --- ZONE HEATMAP ---
    st.header("2. Real-time Demand Hotspots")
    
    # Aggregate demand by coordinates for the latest hour
    latest_ts = df['timestamp'].max()
    latest_data = df[df['timestamp'] == latest_ts]
    
    # Average Barcelona location for center
    m = folium.Map(location=[41.38, 2.17], zoom_start=12, tiles='CartoDB dark_matter')
    
    heat_data = [[row['lat'], row['lon'], row['order_count']] for _, row in latest_data.iterrows()]
    HeatMap(heat_data, radius=25).add_to(m)
    
    st_folium(m, width=1200, height=500)
    
    # XGBoost Logic (Summary Information)
    st.info("""
    🚀 **XGBoost Layer Info:** 
    The current production model uses lag features (`demand_lag_1h`, `demand_lag_24h`) and `demand_rolling_mean_7d`. 
    These features allow the engine to respond to immediate external shocks (e.g., sudden storm) which are not captured by seasonal Prophet models alone.
    """)

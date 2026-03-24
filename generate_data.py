import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_forecast_data(num_zones=5, days=180):
    """
    Generate synthetic demand data for multiple delivery zones.
    """
    start_date = datetime(2023, 9, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(days * 24)]
    
    data = []
    
    # Holiday list (Simplified)
    holidays = ['2023-12-25', '2024-01-01', '2023-10-31']
    
    for zone in range(1, num_zones + 1):
        zone_id = f"zone_{zone:02d}"
        
        # Base daily seasonality
        hour_weights = [1.2, 0.8, 0.4, 0.2, 0.1, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 1.8, 2.0, 1.8, 1.5, 1.3, 1.4, 2.0, 2.8, 3.5, 3.0, 2.0, 1.5, 1.3]
        
        for ts in timestamps:
            # Hourly seasonality
            hour = ts.hour
            base_demand = hour_weights[hour] * 10 
            
            # Weather effect
            weather = random.choices(['Clear', 'Rain', 'Storm'], weights=[0.8, 0.15, 0.05])[0]
            weather_mult = 1.0
            if weather == 'Rain':
                weather_mult = 1.3 # Demand increases in rain (people stay in)
            elif weather == 'Storm':
                weather_mult = 0.5 # Demand drops in storms (courier issues)
            
            # Holiday effect
            is_holiday = ts.strftime('%Y-%m-%d') in holidays
            holiday_mult = 1.5 if is_holiday else 1.0
            
            # Trend and Noise
            trend_mult = 1.0 + (ts - start_date).days / 365.0
            noise = np.random.normal(0, 2)
            
            order_count = int(max(0, (base_demand * weather_mult * holiday_mult * trend_mult) + noise))
            
            data.append({
                'timestamp': ts,
                'zone_id': zone_id,
                'order_count': order_count,
                'weather': weather,
                'is_holiday': 1 if is_holiday else 0,
                'lat': 41.38 + random.uniform(-0.05, 0.05), # Simulating Barcelona coordinates
                'lon': 2.17 + random.uniform(-0.05, 0.05)
            })
            
    df = pd.DataFrame(data)
    
    # Lag Features (Requested)
    df = df.sort_values(['zone_id', 'timestamp'])
    
    # Lag Features Calculation
    for zone in df['zone_id'].unique():
        mask = df['zone_id'] == zone
        df.loc[mask, 'demand_lag_1h'] = df.loc[mask, 'order_count'].shift(1)
        df.loc[mask, 'demand_lag_24h'] = df.loc[mask, 'order_count'].shift(24)
        df.loc[mask, 'demand_rolling_mean_7d'] = df.loc[mask, 'order_count'].shift(1).rolling(window=24*7).mean()
        
    # Fill NaNs from shift
    df.fillna(0, inplace=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic demand data...")
    df = generate_forecast_data()
    df.to_csv('demand_data.csv', index=False)
    print(f"Generated {len(df)} records. Saved to demand_data.csv")

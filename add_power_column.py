"""Script to add synthetic power column to weather dataset for wind turbine simulation."""
import pandas as pd
import numpy as np

def calculate_turbine_power(wind_speed, rated_power=2000, cut_in=3.5, rated_speed=12, cut_out=25):
    """
    Calculate wind turbine power output based on wind speed using a simplified power curve.
    
    Args:
        wind_speed: Wind speed in m/s
        rated_power: Rated power of turbine in kW
        cut_in: Cut-in wind speed in m/s
        rated_speed: Wind speed at rated power in m/s
        cut_out: Cut-out wind speed in m/s
    
    Returns:
        Power output in kW
    """
    if wind_speed < cut_in or wind_speed > cut_out:
        return 0
    elif wind_speed >= rated_speed:
        return rated_power
    else:
        # Cubic relationship between cut-in and rated speed
        power_ratio = ((wind_speed - cut_in) / (rated_speed - cut_in)) ** 3
        return power_ratio * rated_power

# Read the dataset
df = pd.read_csv('data/wind_dataset.csv')

# Convert wind column to numeric, handling any string values
df['WIND'] = pd.to_numeric(df['WIND'], errors='coerce')

# Calculate power output based on wind speed
df['POWER_KW'] = df['WIND'].apply(lambda x: calculate_turbine_power(x) if pd.notna(x) else 0)

# Add some realistic noise and temperature/density effects
np.random.seed(42)
# Use 15°C as default temperature when T.MAX is NaN
temperature_for_effect = df['T.MAX'].fillna(15)
temperature_effect = 1 + (temperature_for_effect - 15) * 0.005  # Small temperature effect
noise = np.random.normal(1, 0.1, len(df))  # 10% noise
df['POWER_KW'] = df['POWER_KW'] * temperature_effect * noise

# Ensure no negative power
df['POWER_KW'] = df['POWER_KW'].clip(lower=0)

# Round to 2 decimal places
df['POWER_KW'] = df['POWER_KW'].round(2)

# Save the updated dataset
df.to_csv('data/wind_dataset.csv', index=False)
print(f"Updated dataset with {len(df)} rows and added POWER_KW column")
print(f"Power statistics: min={df['POWER_KW'].min():.1f}, max={df['POWER_KW'].max():.1f}, mean={df['POWER_KW'].mean():.1f}")
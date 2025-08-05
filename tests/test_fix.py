#!/usr/bin/env python3
"""Quick test of the fixes"""
import pandas as pd
from datetime import datetime, timedelta
from data_sources import DataCollector

# Test with enhanced currents
collector = DataCollector(use_enhanced_currents=True)
start = datetime.now()
end = start + timedelta(days=3)

print("Testing data collection...")
all_data = collector.collect_all_data(start, end)

# Check for timezone issues
if 'pacioos_enhanced' in all_data:
    df = all_data['pacioos_enhanced']
    print(f"\nPacIOOS Enhanced: {len(df)} records")
    if len(df) > 0:
        sample = df.iloc[0]['datetime']
        print(f"Sample datetime: {sample}")
        print(f"Has timezone: {hasattr(sample, 'tz') and sample.tz is not None}")

# Check wind data
if 'openmeteo' in all_data:
    df = all_data['openmeteo']
    print(f"\nOpenMeteo: {len(df)} records")
    if len(df) > 0:
        sample = df.iloc[0]['datetime']
        print(f"Sample datetime: {sample}")
        print(f"Has timezone: {hasattr(sample, 'tz') and sample.tz is not None}")

# Run the full analysis
print("\nRunning find_optimal_conditions...")
optimal = collector.find_optimal_conditions(start, end)
print(f"Found {len(optimal)} optimal conditions")

if not optimal.empty:
    print("\nSample optimal conditions:")
    for _, row in optimal.head(3).iterrows():
        print(f"  {row['datetime']}: {row['wind_speed']:.1f}kt@{row['wind_dir']:.0f}° vs "
              f"{row['current_speed']:.2f}kt@{row['current_dir']:.0f}°")
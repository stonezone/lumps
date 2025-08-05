#!/usr/bin/env python3
"""Debug script to trace current flow analysis"""
import logging
from datetime import datetime, timedelta
from data_sources import DataCollector

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test with enhanced currents for Aug 6
collector = DataCollector(use_enhanced_currents=True)
start = datetime(2025, 8, 6)
end = start + timedelta(days=1)

print("=== Testing Enhanced Current Data Collection ===")
print(f"Period: {start} to {end}")

# Collect all data
all_data = collector.collect_all_data(start, end)

print("\n=== Available Data Sources ===")
for source, data in all_data.items():
    if not data.empty:
        print(f"{source}: {len(data)} records")
        if 'current_dir' in data.columns:
            # Show current direction distribution
            print(f"  Current directions range: {data['current_dir'].min():.1f}° to {data['current_dir'].max():.1f}°")
            # Count eastward currents
            eastward = data[(data['current_dir'] >= 60) & (data['current_dir'] <= 120)]
            print(f"  True eastward (60-120°): {len(eastward)} records")

# Check what happens in find_optimal_conditions
print("\n=== Running find_optimal_conditions ===")
optimal = collector.find_optimal_conditions(start, end)

if not optimal.empty:
    print(f"\nFound {len(optimal)} 'optimal' conditions")
    print("\nSample records:")
    for idx, row in optimal.head(5).iterrows():
        print(f"  {row['datetime']}: Current {row['current_dir']:.0f}° @ {row['current_speed']:.2f}kt")
        print(f"    Wind: {row['wind_dir']:.0f}° @ {row['wind_speed']:.1f}kt")
        print(f"    Enhancement: {row['enhancement']}")
else:
    print("\nNo optimal conditions found")

# Debug the eastward flow filter
print("\n=== Testing Eastward Flow Filter ===")
if 'pacioos_enhanced' in all_data:
    current_data = all_data['pacioos_enhanced']
    eastward_periods = collector.tidal_analyzer.find_eastward_flow_periods(current_data)
    print(f"Eastward periods found: {len(eastward_periods)}")
    if not eastward_periods.empty:
        print("Sample eastward periods:")
        for idx, row in eastward_periods.head(5).iterrows():
            print(f"  {row['datetime']}: {row['current_dir']:.0f}° @ {row['current_speed']:.2f}kt")

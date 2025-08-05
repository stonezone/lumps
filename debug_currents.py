#!/usr/bin/env python3
"""Debug script to check current directions and wind matching"""

import pandas as pd
from datetime import datetime, timedelta
from data_sources import DataCollector

# Initialize
collector = DataCollector(use_enhanced_currents=True)
start_date = datetime.now()
end_date = start_date + timedelta(days=3)

# Collect all data
print("Collecting data...")
all_data = collector.collect_all_data(start_date, end_date)

# Check current data
if 'pacioos_enhanced' in all_data:
    current_data = all_data['pacioos_enhanced']
    print(f"\nPacIOOS Enhanced data: {len(current_data)} records")
    
    # Show current direction distribution
    print("\nCurrent Direction Distribution:")
    bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    labels = ['N', 'NNE', 'ENE', 'E', 'ESE', 'SE', 'S', 'SSW', 'SW', 'W', 'WNW', 'NW']
    current_data['dir_bin'] = pd.cut(current_data['current_dir'], bins=bins, labels=labels)
    
    for label in labels:
        subset = current_data[current_data['dir_bin'] == label]
        if len(subset) > 0:
            avg_speed = subset['current_speed'].mean()
            print(f"  {label:>4} ({bins[labels.index(label)]:3d}-{bins[labels.index(label)+1]:3d}°): "
                  f"{len(subset):3d} records, avg {avg_speed:.2f} kt")
    
    # Show eastward currents specifically
    eastward_mask = (current_data['current_dir'] >= 30) & (current_data['current_dir'] <= 150)
    eastward = current_data[eastward_mask]
    print(f"\nEastward currents (30-150°): {len(eastward)} records")
    if len(eastward) > 0:
        print("Sample eastward currents:")
        for _, row in eastward.head(5).iterrows():
            print(f"  {row['datetime']}: {row['current_dir']:.0f}° @ {row['current_speed']:.2f} kt")

# Check wind data
print("\n" + "="*60)
if 'openmeteo' in all_data:
    wind_data = all_data['openmeteo']
    print(f"\nOpenMeteo wind data: {len(wind_data)} records")
    print("Sample winds:")
    for _, row in wind_data.head(5).iterrows():
        print(f"  {row['datetime']}: {row['wind_dir']:.0f}° @ {row['wind_speed']:.1f} kt")

# Now test the actual find_optimal_conditions
print("\n" + "="*60)
print("Running find_optimal_conditions...")
optimal = collector.find_optimal_conditions(start_date, end_date)
print(f"Result: {len(optimal)} optimal conditions found")

if optimal.empty:
    print("\nDEBUG: Checking why no optimal conditions found...")
    # Get eastward periods
    if 'pacioos_enhanced' in all_data:
        eastward_periods = collector.tidal_analyzer.find_eastward_flow_periods(current_data)
        print(f"Eastward periods: {len(eastward_periods)}")
        
        if not eastward_periods.empty:
            print("\nChecking wind matching for first few eastward periods:")
            for i, (_, current_row) in enumerate(eastward_periods.head(3).iterrows()):
                dt = current_row['datetime']
                print(f"\nCurrent at {dt}: {current_row['current_dir']:.0f}° @ {current_row['current_speed']:.2f} kt")
                
                # Check for matching wind
                for source_name, wind_data in [('openmeteo', all_data.get('openmeteo', pd.DataFrame()))]:
                    if not wind_data.empty:
                        time_diff = abs(wind_data['datetime'] - dt)
                        if time_diff.min() <= timedelta(hours=2):
                            closest_wind = wind_data.loc[time_diff.idxmin()]
                            print(f"  Closest wind: {closest_wind['wind_dir']:.0f}° @ {closest_wind['wind_speed']:.1f} kt")
                            print(f"  Time diff: {time_diff.min()}")
                            
                            # Test interaction
                            interaction = collector.tidal_analyzer.analyze_current_wind_interaction(
                                current_row['current_dir'], 
                                closest_wind['wind_dir'],
                                current_row['current_speed'],
                                dt
                            )
                            print(f"  Interaction result: {interaction}")
else:
    print("\nOptimal conditions found:")
    for _, row in optimal.head(3).iterrows():
        print(f"  {row['datetime']}: Wind {row['wind_speed']:.1f}kt@{row['wind_dir']:.0f}°, "
              f"Current {row['current_speed']:.2f}kt@{row['current_dir']:.0f}°")

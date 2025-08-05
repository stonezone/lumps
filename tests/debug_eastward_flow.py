#!/usr/bin/env python3
"""
Debug eastward flow detection after spatial aggregation
"""

import sys
sys.path.append('.')
from data_sources import DataCollector
from datetime import datetime, timedelta
import pandas as pd

def debug_eastward_flow():
    collector = DataCollector()
    
    # Get current data
    start_date = datetime.now()
    end_date = start_date + timedelta(days=2)
    
    print("Debugging eastward flow detection...")
    print("=" * 60)
    
    # Get PacIOOS data directly
    if 'pacioos' in collector.sources:
        pacioos_data = collector.sources['pacioos'].fetch_data(start_date, end_date)
        
        if not pacioos_data.empty:
            print(f"PacIOOS aggregated data: {len(pacioos_data)} records")
            print(f"Current direction range: {pacioos_data['current_dir'].min():.0f}° to {pacioos_data['current_dir'].max():.0f}°")
            print(f"Current speed range: {pacioos_data['current_speed'].min():.3f} to {pacioos_data['current_speed'].max():.3f} kt")
            
            # Check eastward flow criteria (60-120°)
            eastward_mask = (pacioos_data['current_dir'] >= 60) & (pacioos_data['current_dir'] <= 120)
            eastward_count = eastward_mask.sum()
            
            print(f"\nEastward flow analysis (60-120°):")
            print(f"Records with eastward flow: {eastward_count} out of {len(pacioos_data)}")
            
            if eastward_count > 0:
                eastward_data = pacioos_data[eastward_mask]
                print("Sample eastward flow records:")
                for _, row in eastward_data.head(5).iterrows():
                    print(f"  {row['datetime']} - {row['current_speed']:.3f}kt @ {row['current_dir']:.0f}°")
            else:
                print("No eastward flow found!")
                print("\nCurrent direction distribution:")
                for _, row in pacioos_data.head(10).iterrows():
                    print(f"  {row['datetime']} - {row['current_speed']:.3f}kt @ {row['current_dir']:.0f}°")
                    
    # Also test tidal analyzer directly
    print(f"\nTesting tidal analyzer directly:")
    eastward_periods = collector.tidal_analyzer.find_eastward_flow_periods(pacioos_data)
    print(f"Tidal analyzer found: {len(eastward_periods)} eastward periods")

if __name__ == '__main__':
    debug_eastward_flow()
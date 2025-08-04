#!/usr/bin/env python3
"""
Debug PacIOOS spatial grid structure to understand duplicate records
"""

import sys
sys.path.append('.')
from data_sources import PacIOOSCurrentSource
from datetime import datetime, timedelta
import pandas as pd

def analyze_pacioos_grid():
    source = PacIOOSCurrentSource()
    
    # Get a small sample of data to analyze structure
    start_date = datetime.now()
    end_date = start_date + timedelta(hours=6)  # Just 6 hours for analysis
    
    print("Analyzing PacIOOS spatial grid structure...")
    print("=" * 60)
    
    data = source.fetch_data(start_date, end_date)
    
    if data.empty:
        print("No PacIOOS data available")
        return
    
    print(f"Total records: {len(data)}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    
    # Analyze spatial distribution
    print(f"\nSpatial Distribution:")
    print(f"Unique latitudes: {data['latitude'].nunique()}")
    print(f"Latitude range: {data['latitude'].min():.4f} to {data['latitude'].max():.4f}")
    print(f"Unique longitudes: {data['longitude'].nunique()}")  
    print(f"Longitude range: {data['longitude'].min():.4f} to {data['longitude'].max():.4f}")
    
    # Analyze temporal distribution
    print(f"\nTemporal Distribution:")
    print(f"Unique timestamps: {data['datetime'].nunique()}")
    print(f"Records per timestamp: {len(data) / data['datetime'].nunique():.1f}")
    
    # Show sample of spatial points for one timestamp
    sample_time = data['datetime'].iloc[0]
    sample_data = data[data['datetime'] == sample_time]
    print(f"\nSample spatial points for {sample_time}:")
    print(f"Number of spatial points: {len(sample_data)}")
    print("Lat/Lon combinations:")
    for _, row in sample_data.head(10).iterrows():
        print(f"  {row['latitude']:.4f}, {row['longitude']:.4f} - Current: {row['current_speed']:.3f}kt@{row['current_dir']:.0f}°")
    
    # Analyze current speed/direction variation across space
    print(f"\nCurrent Data Quality:")
    print(f"Speed range: {data['current_speed'].min():.3f} to {data['current_speed'].max():.3f} kt")
    print(f"Direction range: {data['current_dir'].min():.0f} to {data['current_dir'].max():.0f}°")
    print(f"Records with 0.0kt speed: {(data['current_speed'] == 0.0).sum()}")
    print(f"Records with exactly 90° direction: {(data['current_dir'] == 90.0).sum()}")

if __name__ == '__main__':
    analyze_pacioos_grid()
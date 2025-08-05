#!/usr/bin/env python3
"""
Analyze current directions to determine optimal range for North Shore
"""

import sys
sys.path.append('.')
from data_sources import DataCollector
from datetime import datetime, timedelta
import numpy as np

def analyze_directions():
    collector = DataCollector()
    
    start_date = datetime.now()
    end_date = start_date + timedelta(days=2)
    
    # Get PacIOOS real data
    pacioos_data = collector.sources['pacioos'].fetch_data(start_date, end_date)
    
    print("Current Direction Analysis for North Shore")
    print("=" * 50)
    print(f"Data: {len(pacioos_data)} records")
    
    directions = pacioos_data['current_dir'].values
    speeds = pacioos_data['current_speed'].values
    
    # Analyze direction distribution
    ranges = [
        ("North (315-45°)", lambda d: d >= 315 or d <= 45),
        ("Northeast (45-135°)", lambda d: 45 <= d <= 135), 
        ("Southeast (135-225°)", lambda d: 135 <= d <= 225),
        ("Southwest (225-315°)", lambda d: 225 <= d <= 315),
        ("Eastward (60-120°)", lambda d: 60 <= d <= 120),
        ("Any Eastern (45-135°)", lambda d: 45 <= d <= 135),
    ]
    
    for name, condition in ranges:
        mask = np.array([condition(d) for d in directions])
        count = mask.sum()
        if count > 0:
            avg_speed = speeds[mask].mean()
            print(f"{name}: {count} records, avg speed: {avg_speed:.3f}kt")
        else:
            print(f"{name}: 0 records")
    
    print(f"\nActual directions: {sorted(directions)}")
    
    # Compare with tidal simulation
    print(f"\nComparing with tidal simulation:")
    tidal_data = collector.tidal_analyzer.generate_tidal_simulation(start_date, end_date)
    if not tidal_data.empty:
        tidal_directions = tidal_data['current_dir'].values
        tidal_eastward = ((tidal_directions >= 60) & (tidal_directions <= 120)).sum()
        print(f"Tidal simulation: {tidal_eastward} eastward periods out of {len(tidal_data)}")

if __name__ == '__main__':
    analyze_directions()
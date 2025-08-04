#!/usr/bin/env python3
"""
Debug script to check timezone handling across data sources
"""

import sys
sys.path.append('.')
from data_sources import DataCollector
from datetime import datetime, timedelta

def check_data_source_timezones():
    collector = DataCollector()
    start_date = datetime.now()
    end_date = start_date + timedelta(days=1)
    
    print("Checking timezone handling across data sources...")
    print("=" * 60)
    
    for name, source in collector.sources.items():
        print(f"\n{name.upper()} Data Source:")
        try:
            data = source.fetch_data(start_date, end_date)
            if not data.empty and 'datetime' in data.columns:
                sample_dt = data['datetime'].iloc[0]
                print(f"  Sample datetime: {sample_dt}")
                print(f"  Type: {type(sample_dt)}")
                print(f"  Has timezone: {hasattr(sample_dt, 'tz') and sample_dt.tz is not None}")
                if hasattr(sample_dt, 'tz'):
                    print(f"  Timezone: {sample_dt.tz}")
                print(f"  Records: {len(data)}")
            else:
                print(f"  No data or no datetime column")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Also check tidal simulation
    print(f"\nTIDAL SIMULATION:")
    try:
        tidal_data = collector.tidal_analyzer.generate_tidal_simulation(start_date, end_date)
        if not tidal_data.empty:
            sample_dt = tidal_data['datetime'].iloc[0]
            print(f"  Sample datetime: {sample_dt}")
            print(f"  Type: {type(sample_dt)}")
            print(f"  Has timezone: {hasattr(sample_dt, 'tz') and sample_dt.tz is not None}")
            if hasattr(sample_dt, 'tz'):
                print(f"  Timezone: {sample_dt.tz}")
            print(f"  Records: {len(tidal_data)}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == '__main__':
    check_data_source_timezones()
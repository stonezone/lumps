#!/usr/bin/env python3
"""
Test that tidal simulation fallback still works with timezone fixes
"""

import sys
sys.path.append('.')
from data_sources import DataCollector
from datetime import datetime, timedelta

def test_simulation_fallback():
    """Test simulation when PacIOOS fails"""
    collector = DataCollector()
    
    # Use a date far in the future where PacIOOS will fail
    start_date = datetime(2025, 12, 1)
    end_date = start_date + timedelta(days=7)
    
    print("Testing tidal simulation fallback...")
    print(f"Using future dates where PacIOOS will fail: {start_date} to {end_date}")
    
    try:
        conditions = collector.find_optimal_conditions(start_date, end_date)
        print(f"Success! Found {len(conditions)} conditions using simulation fallback")
        
        if not conditions.empty:
            sample = conditions.iloc[0]
            print(f"Sample condition: {sample['datetime']} - Current: {sample['current_speed']:.3f}kt@{sample['current_dir']:.0f}°")
            print(f"Data source: {sample['source']}")
            
            # Check if it used simulation (should have 'Tidal_Simulation' in source)
            if 'Tidal_Simulation' in sample['source']:
                print("✅ Successfully fell back to tidal simulation")
            else:
                print("❌ Expected tidal simulation but got different source")
        
    except Exception as e:
        print(f"❌ Simulation fallback failed: {e}")

if __name__ == '__main__':
    test_simulation_fallback()
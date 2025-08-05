#!/usr/bin/env python3
"""
Diagnostic tool for LUMPS data sources
Tests connectivity and data availability for all sources
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_sources import DataCollector
from datetime import datetime, timedelta
import requests

def test_data_source(name, source):
    """Test a single data source"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('-'*60)
    
    # Test availability
    try:
        available = source.is_available()
        print(f"✓ Availability check: {'PASS' if available else 'FAIL'}")
        if not available:
            print("  Note: Source marked as unavailable")
    except Exception as e:
        print(f"✗ Availability check failed: {e}")
        available = False
    
    # Test data fetch if available
    if available:
        try:
            start = datetime.now()
            end = start + timedelta(days=1)
            data = source.fetch_data(start, end)
            
            if data.empty:
                print(f"⚠ Data fetch returned empty DataFrame")
            else:
                print(f"✓ Data fetch successful: {len(data)} records")
                print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
                
                # Show sample fields
                cols = [c for c in data.columns if c != 'datetime']
                print(f"  Fields: {', '.join(cols[:5])}")
                
        except Exception as e:
            print(f"✗ Data fetch failed: {e}")
    
    return available

def main():
    print("\n" + "="*60)
    print("LUMPS Data Source Diagnostics")
    print("="*60)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize collector
    collector = DataCollector(use_enhanced_currents=True)
    
    # Track results
    results = {}
    
    # Test each source
    for name, source in collector.sources.items():
        results[name] = test_data_source(name, source)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('-'*60)
    
    working = [n for n, r in results.items() if r]
    failed = [n for n, r in results.items() if not r]
    
    print(f"✓ Working sources ({len(working)}): {', '.join(working)}")
    if failed:
        print(f"✗ Failed sources ({len(failed)}): {', '.join(failed)}")
    
    # Test specific APIs directly
    print(f"\n{'='*60}")
    print("DIRECT API TESTS")
    print('-'*60)
    
    # Test NOAA CWF
    print("\nNOAA Coastal Waters Forecast:")
    try:
        resp = requests.get(
            "https://forecast.weather.gov/product.php",
            params={'issuedby': 'HFO', 'product': 'CWF', 'site': 'hfo'},
            timeout=5
        )
        if resp.status_code == 200 and 'Oahu Leeward Waters' in resp.text:
            print("  ✓ Direct API test: PASS")
        else:
            print(f"  ✗ Status {resp.status_code}")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
    
    # Test PacIOOS ERDDAP
    print("\nPacIOOS ERDDAP:")
    try:
        resp = requests.get(
            "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig.das",
            timeout=5
        )
        if resp.status_code == 200:
            print("  ✓ Direct API test: PASS")
        else:
            print(f"  ✗ Status {resp.status_code}")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
    
    print("\nDiagnostics complete!")
    
if __name__ == "__main__":
    main()

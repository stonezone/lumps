#!/usr/bin/env python3
"""
Test script to debug PacIOOS ERDDAP access
"""

import requests
from datetime import datetime, timedelta

def test_pacioos_range():
    # Test with the exact range from LUMPS
    start_date = datetime.now()
    end_date = start_date + timedelta(days=7)
    
    base_url = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig.csv'
    start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Try the exact query format from LUMPS
    lat_range = "[(21.6):(21.7)]"
    lon_range = "[(-158.2):(-157.9)]"
    time_range = f"[({start_str}):({end_str})]"
    depth_surface = "[0]"
    
    u_query = f"u{time_range}{depth_surface}{lat_range}{lon_range}"
    u_url = f"{base_url}?{u_query}"
    
    print('Testing PacIOOS ERDDAP with LUMPS format...')
    print(f'Time range: {start_str} to {end_str}')
    print(f'U Query URL: {u_url}')
    
    try:
        response = requests.get(u_url, timeout=30)
        print(f'Status: {response.status_code}')
        print(f'Response length: {len(response.text)}')
        
        if response.status_code != 200:
            print(f'Error response: {response.text[:500]}')
        else:
            print(f'Success! First 500 chars: {response.text[:500]}')
            # Count data rows
            lines = response.text.strip().split('\n')
            data_lines = [line for line in lines[2:] if line.strip()]  # Skip header
            print(f'Data records found: {len(data_lines)}')
            
    except Exception as e:
        print(f'Error: {e}')

def test_simple_point():
    # Test with single point and short time range
    start_date = datetime.now()
    end_date = start_date + timedelta(hours=12)
    
    base_url = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig.csv'
    start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Single point query
    query = f'u[({start_str}):({end_str})][0][(21.65):(21.65)][(-158.1):(-158.1)]'
    query_url = f'{base_url}?{query}'
    
    print('\nTesting simple point query...')
    print(f'Query URL: {query_url}')
    
    try:
        response = requests.get(query_url, timeout=30)
        print(f'Status: {response.status_code}')
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            print(f'Data records: {len(lines) - 2}')  # Exclude header
            print(f'Sample data: {lines[2] if len(lines) > 2 else "No data"}')
        else:
            print(f'Error: {response.text[:200]}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_pacioos_range()
    test_simple_point()
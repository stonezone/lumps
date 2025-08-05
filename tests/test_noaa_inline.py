"""
Quick test to understand NOAA Marine HTML structure
Run this in Python REPL: python -c "exec(open('test_noaa_inline.py').read())"
"""

import sys
try:
    import requests
    from bs4 import BeautifulSoup
    import re
    
    url = 'https://marine.weather.gov/MapClick.php?w3=sfcwind&w3u=1&w14=wwh&AheadHour=0&FcstType=digital&textField1=21.8298&textField2=-157.759&site=all&unit=0&dd=&bw=&marine=1'
    
    print(f"Fetching: {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")
        
        # Look specifically at table 4 (index 3) which the issue mentions
        if len(tables) > 3:
            table = tables[3]
            text = table.get_text()
            
            # Show raw text
            print("\n=== Table 4 Raw Text (first 1500 chars) ===")
            print(text[:1500])
            
            # Try to find the concatenated patterns
            print("\n=== Attempting to parse concatenated data ===")
            
            # Remove all whitespace to make parsing easier
            clean_text = text.replace(' ', '').replace('\n', '').replace('\t', '')
            
            # Look for the Date pattern
            if 'Date' in clean_text:
                idx = clean_text.index('Date')
                print(f"\nFound 'Date' at position {idx}")
                # Extract the next 50 characters after 'Date'
                date_section = clean_text[idx:idx+50]
                print(f"Date section: {date_section}")
                
            # Look for Hour(HST) pattern  
            if 'Hour(HST)' in clean_text:
                idx = clean_text.index('Hour(HST)')
                print(f"\nFound 'Hour(HST)' at position {idx}")
                # Extract hours (should be consecutive 2-digit numbers)
                hour_section = clean_text[idx+9:idx+60]  # Skip 'Hour(HST)'
                print(f"Hour section: {hour_section}")
                
            # Look for SurfaceWind(mph) pattern
            if 'SurfaceWind(mph)' in clean_text:
                idx = clean_text.index('SurfaceWind(mph)')
                print(f"\nFound 'SurfaceWind(mph)' at position {idx}")
                wind_section = clean_text[idx+16:idx+70]  # Skip label
                print(f"Wind section: {wind_section}")
                
            # Look for WindDir pattern
            if 'WindDir' in clean_text:
                idx = clean_text.index('WindDir')
                print(f"\nFound 'WindDir' at position {idx}")
                dir_section = clean_text[idx+7:idx+100]  # Skip 'WindDir'
                print(f"Direction section: {dir_section}")
                
            # Look for wave height
            for wave_pattern in ['WindWaveHeight', 'WaveHeight', 'WindWave']:
                if wave_pattern in clean_text:
                    idx = clean_text.index(wave_pattern)
                    print(f"\nFound '{wave_pattern}' at position {idx}")
                    wave_section = clean_text[idx+len(wave_pattern):idx+len(wave_pattern)+50]
                    print(f"Wave section: {wave_section}")
                    break
    else:
        print(f"Error: HTTP {response.status_code}")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Try running: pip install beautifulsoup4 requests")
except Exception as e:
    print(f"Error: {e}")

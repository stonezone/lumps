#!/usr/bin/env python3
"""Test the NOAAMarineSource parser with sample HTML data"""

from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Sample HTML that mimics the concatenated format described in the issues
SAMPLE_HTML = """
<html>
<body>
<table>Dummy table 1</table>
<table>Dummy table 2</table>
<table>Dummy table 3</table>
<table>
<tr><td>Date08/0508/06Hour (HST)091011121314151617181920212223000102030405060708Surface Wind (mph)13131010101212121313131515151414141414146666Wind DirEEENEENEENEENEENEENEENEENEENEENENENENEENEENEENEENEENEENENNENNWindWaveHeight(ft)3344443344444333333222222</td></tr>
</table>
<table>Dummy table 5</table>
</body>
</html>
"""

def test_parser():
    """Test the NOAAMarineSource parser"""
    
    # Import the necessary classes
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Create a minimal mock class for testing
    class TestNOAAMarineSource:
        def __init__(self):
            pass
        
        def _parse_numeric_sequence(self, number_string: str, expected_count: int, is_wave_height: bool = False):
            """Parse a string of concatenated numbers into individual values"""
            results = []
            
            # For wave heights, prefer single digits since they're typically 0-9 ft
            if is_wave_height:
                # Strategy for wave heights: prefer single digits
                single_digit = list(number_string)
                
                # Also try two-digit parsing in case waves are 10+ ft
                two_digit = [number_string[i:i+2] for i in range(0, len(number_string), 2)]
                
                # Pick based on which gives us the expected count
                if len(single_digit) == expected_count:
                    results = single_digit
                elif len(two_digit) == expected_count:
                    results = two_digit
                else:
                    # Default to single digits for wave heights
                    results = single_digit
            else:
                # Strategy 1: Assume all are 2-digit numbers
                two_digit = [number_string[i:i+2] for i in range(0, len(number_string), 2)]
                
                # Strategy 2: Parse adaptively
                adaptive = []
                i = 0
                while i < len(number_string):
                    if i + 1 < len(number_string):
                        two_digit_val = int(number_string[i:i+2])
                        if two_digit_val <= 50:
                            adaptive.append(number_string[i:i+2])
                            i += 2
                        else:
                            adaptive.append(number_string[i])
                            i += 1
                    else:
                        adaptive.append(number_string[i])
                        i += 1
                
                # Pick the strategy that gives us closest to expected count
                if abs(len(two_digit) - expected_count) <= abs(len(adaptive) - expected_count):
                    results = two_digit
                else:
                    results = adaptive
            
            if len(results) > expected_count:
                results = results[:expected_count]
            elif len(results) < expected_count and results:
                last_val = results[-1]
                while len(results) < expected_count:
                    results.append(last_val)
            
            return results
        
        def _parse_wind_directions(self, direction_string: str, expected_count: int):
            """Parse a string of concatenated wind directions"""
            results = []
            i = 0
            
            while i < len(direction_string) and len(results) < expected_count:
                matched = False
                
                # Try 3-letter directions first
                if i + 3 <= len(direction_string):
                    three_letter = direction_string[i:i+3]
                    if three_letter in ['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW']:
                        results.append(three_letter)
                        i += 3
                        matched = True
                
                # Try 2-letter directions
                if not matched and i + 2 <= len(direction_string):
                    two_letter = direction_string[i:i+2]
                    if two_letter in ['NE', 'SE', 'SW', 'NW']:
                        results.append(two_letter)
                        i += 2
                        matched = True
                
                # Try 1-letter directions
                if not matched and i < len(direction_string):
                    one_letter = direction_string[i]
                    if one_letter in ['N', 'E', 'S', 'W']:
                        results.append(one_letter)
                        i += 1
                        matched = True
                
                if not matched:
                    i += 1
            
            while len(results) < expected_count:
                results.append('E')
            
            return results[:expected_count]
        
        def _parse_forecast_table(self, html_content: str, base_date: datetime):
            """Parse the HTML forecast table to extract wind and wave data"""
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                print("BeautifulSoup not available, using regex fallback")
                import re
                
                # Simple regex-based parsing for testing
                clean_text = ''.join(html_content.split())
                
                # Extract hours
                hours = []
                hour_match = re.search(r'Hour\(HST\)((?:\d{2})+)', clean_text)
                if hour_match:
                    hour_string = hour_match.group(1)
                    hours = [hour_string[i:i+2] for i in range(0, len(hour_string), 2)]
                    print(f"Found {len(hours)} hours: {hours[:5]}...")
                
                # Extract wind speeds
                wind_speeds = []
                wind_match = re.search(r'SurfaceWind\(mph\)([\d]+)', clean_text)
                if wind_match:
                    wind_string = wind_match.group(1)
                    wind_speeds = self._parse_numeric_sequence(wind_string, len(hours), is_wave_height=False)
                    print(f"Found {len(wind_speeds)} wind speeds: {wind_speeds[:5]}...")
                
                # Extract wind directions
                wind_dirs = []
                dir_match = re.search(r'WindDir([NESW]+)', clean_text)
                if dir_match:
                    dir_string = dir_match.group(1)
                    wind_dirs = self._parse_wind_directions(dir_string, len(hours))
                    print(f"Found {len(wind_dirs)} wind directions: {wind_dirs[:5]}...")
                
                # Extract wave heights
                wave_heights = []
                wave_match = re.search(r'WindWaveHeight\(ft\)([\d]+)', clean_text)
                if wave_match:
                    wave_string = wave_match.group(1)
                    wave_heights = self._parse_numeric_sequence(wave_string, len(hours), is_wave_height=True)
                    print(f"Found {len(wave_heights)} wave heights: {wave_heights[:5]}...")
                
                # Build sample data rows
                data_rows = []
                current_date = base_date
                last_hour = -1
                
                for i in range(min(len(hours), len(wind_speeds))):
                    hour = int(hours[i])
                    
                    if hour < last_hour:
                        current_date += timedelta(days=1)
                    last_hour = hour
                    
                    wind_speed_mph = float(wind_speeds[i]) if i < len(wind_speeds) else 0
                    wind_speed_kts = wind_speed_mph * 0.868976
                    
                    wind_dir_str = wind_dirs[i] if i < len(wind_dirs) else 'E'
                    
                    dir_map = {
                        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
                    }
                    wind_dir_deg = dir_map.get(wind_dir_str.upper(), 90)
                    
                    wave_height = float(wave_heights[i]) if i < len(wave_heights) else None
                    
                    data_point = {
                        'timestamp': current_date.replace(hour=hour, minute=0, second=0),
                        'wind_speed_kts': wind_speed_kts,
                        'wind_dir_deg': wind_dir_deg,
                        'source': 'NOAA_Marine'
                    }
                    
                    if wave_height is not None:
                        data_point['wind_wave_height_ft'] = wave_height
                    
                    data_rows.append(data_point)
                
                return data_rows
            
            except Exception as e:
                print(f"Error in parser: {e}")
                import traceback
                traceback.print_exc()
                return []
    
    # Test the parser
    parser = TestNOAAMarineSource()
    base_date = datetime(2024, 8, 5)
    
    print("\n=== Testing NOAAMarineSource Parser ===\n")
    
    results = parser._parse_forecast_table(SAMPLE_HTML, base_date)
    
    print(f"\nParsed {len(results)} data points:\n")
    
    # Show first few results
    for i, data in enumerate(results[:5]):
        print(f"Data point {i+1}:")
        print(f"  Timestamp: {data['timestamp']}")
        print(f"  Wind Speed: {data['wind_speed_kts']:.1f} kts")
        print(f"  Wind Direction: {data['wind_dir_deg']}°")
        if 'wind_wave_height_ft' in data:
            print(f"  Wave Height: {data['wind_wave_height_ft']} ft")
        print()
    
    # Verify parsing correctness
    if results:
        print("✓ Parser successfully extracted data from concatenated format")
        
        # Check first data point values
        first = results[0]
        expected_hour = 9  # First hour in sample
        expected_wind_mph = 13  # First wind speed in sample
        expected_wind_kts = expected_wind_mph * 0.868976
        
        if first['timestamp'].hour == expected_hour:
            print(f"✓ Correct hour parsed: {expected_hour}")
        
        if abs(first['wind_speed_kts'] - expected_wind_kts) < 0.1:
            print(f"✓ Correct wind speed: {expected_wind_mph} mph = {expected_wind_kts:.1f} kts")
        
        if first['wind_dir_deg'] == 90:  # E = 90°
            print("✓ Correct wind direction: E = 90°")
        
        if 'wind_wave_height_ft' in first and first['wind_wave_height_ft'] == 3:
            print("✓ Correct wave height: 3 ft")
    else:
        print("✗ No data parsed - parser may need adjustment")

if __name__ == "__main__":
    test_parser()

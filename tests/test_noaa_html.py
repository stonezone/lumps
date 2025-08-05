#!/usr/bin/env python3
"""Test script to examine NOAA Marine Weather HTML structure"""

import requests
from bs4 import BeautifulSoup
import re

def examine_noaa_html():
    """Fetch and examine the NOAA Marine HTML structure"""
    url = 'https://marine.weather.gov/MapClick.php?w3=sfcwind&w3u=1&w14=wwh&AheadHour=0&FcstType=digital&textField1=21.8298&textField2=-157.759&site=all&unit=0&dd=&bw=&marine=1'
    
    print(f"Fetching URL: {url}")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table')
    
    print(f"\nFound {len(tables)} tables on the page\n")
    
    # Look for the forecast table
    for i, table in enumerate(tables):
        table_text = table.get_text()
        
        # Check if this table has forecast data
        if 'Hour (HST)' in table_text or 'Surface Wind' in table_text:
            print(f"=== Table {i} (likely forecast table) ===")
            print(f"Raw text (first 2000 chars):\n{table_text[:2000]}\n")
            
            # Try to extract the concatenated data
            # The data appears to be in format: LabelDataDataData...
            
            # Look for patterns
            if 'Date' in table_text:
                # Extract dates
                date_match = re.search(r'Date([\d/]+)', table_text)
                if date_match:
                    dates_str = date_match.group(1)
                    dates = re.findall(r'\d{2}/\d{2}', dates_str)
                    print(f"Dates found: {dates}")
            
            if 'Hour \(HST\)' in table_text or 'Hour (HST)' in table_text:
                # Extract hours - they should be consecutive 2-digit numbers
                hour_match = re.search(r'Hour \(HST\)([\d]+)', table_text.replace(' ', ''))
                if hour_match:
                    hours_str = hour_match.group(1)
                    # Split into 2-digit chunks
                    hours = [hours_str[i:i+2] for i in range(0, len(hours_str), 2)]
                    print(f"Hours found: {hours[:10]}...")  # Show first 10
            
            if 'Surface Wind' in table_text:
                # Extract wind speeds
                wind_match = re.search(r'Surface Wind \(mph\)([\d]+)', table_text.replace(' ', ''))
                if wind_match:
                    wind_str = wind_match.group(1)
                    # These could be 1 or 2 digit numbers
                    # We need to be smarter about parsing
                    print(f"Wind data string (first 50 chars): {wind_str[:50]}...")
            
            if 'Wind Dir' in table_text:
                # Extract wind directions
                dir_match = re.search(r'Wind Dir([NESW]+)', table_text.replace(' ', ''))
                if dir_match:
                    dir_str = dir_match.group(1)
                    # Parse direction codes (1-3 letters each)
                    print(f"Direction string (first 50 chars): {dir_str[:50]}...")
            
            if 'Wind Wave' in table_text or 'Wave Height' in table_text:
                # Extract wave heights
                wave_match = re.search(r'(?:Wind Wave Height|Wave Height)([\d]+)', table_text.replace(' ', ''))
                if wave_match:
                    wave_str = wave_match.group(1)
                    print(f"Wave height string (first 50 chars): {wave_str[:50]}...")
            
            print("\n--- Table HTML Structure ---")
            # Look at the actual HTML structure
            rows = table.find_all('tr')
            print(f"Number of rows: {len(rows)}")
            
            for j, row in enumerate(rows[:10]):  # Look at first 10 rows
                cells = row.find_all(['td', 'th'])
                print(f"Row {j}: {len(cells)} cells")
                if cells:
                    # Show first few cells
                    for k, cell in enumerate(cells[:5]):
                        cell_text = cell.get_text().strip()
                        if len(cell_text) > 100:
                            cell_text = cell_text[:100] + "..."
                        print(f"  Cell {k}: '{cell_text}'")
            
            print("\n")

if __name__ == "__main__":
    examine_noaa_html()

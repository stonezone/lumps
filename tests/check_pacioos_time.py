#!/usr/bin/env python3
"""
Check the actual time range available in PacIOOS ROMS dataset
"""

import requests
import json
from datetime import datetime

def check_pacioos_time_range():
    try:
        # Get dataset info
        info_url = "https://pae-paha.pacioos.hawaii.edu/erddap/info/roms_hiig/index.json"
        response = requests.get(info_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Find time variable info
            for row in data['table']['rows']:
                if len(row) >= 5 and row[1] == 'time':
                    if row[2] == 'actual_range':
                        time_range = row[4].split(' to ')
                        print(f"Time range: {time_range[0]} to {time_range[1]}")
                        
                        # Parse the times
                        start_time = datetime.fromisoformat(time_range[0].replace('Z', '+00:00'))
                        end_time = datetime.fromisoformat(time_range[1].replace('Z', '+00:00'))
                        
                        print(f"Start: {start_time}")
                        print(f"End: {end_time}")
                        print(f"Duration: {(end_time - start_time).days} days")
                        
                        # Check if current time is within range
                        now = datetime.now()
                        print(f"Current time: {now}")
                        print(f"Within range: {start_time <= now <= end_time}")
                        
                        return start_time, end_time
        else:
            print(f"Failed to get dataset info: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return None, None

if __name__ == '__main__':
    check_pacioos_time_range()
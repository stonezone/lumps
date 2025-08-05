# NOAA Marine Weather Parser Issues

## Overview
The NOAAMarineSource class has been integrated into the LUMPS system to fetch wind and wind wave height forecast data. While the integration architecture is complete and functional, the HTML parser needs refinement to correctly extract data from the NOAA Marine Weather forecast pages.

## Current Status
- ✅ NOAAMarineSource class created and integrated (data_sources.py lines 1826-1965)
- ✅ URL generation working correctly for forecast windows
- ✅ Integrated as primary wind source with fallback to NDBC/CWF/OpenMeteo
- ✅ Wind wave height added to scoring system (analysis.py)
- ❌ HTML parsing not extracting forecast data correctly

## The Problem

### Target URLs
The parser needs to extract data from URLs like:
```
https://marine.weather.gov/MapClick.php?w3=sfcwind&w3u=1&w14=wwh&AheadHour=0&FcstType=digital&textField1=21.8298&textField2=-157.759&site=all&unit=0&dd=&bw=&marine=1
```

Parameters:
- `w3=sfcwind` - Surface wind data
- `w3u=1` - Wind units in mph
- `w14=wwh` - Wind wave height
- `AheadHour` - Forecast offset (0, 48, 96, etc. for 2-day chunks)
- `textField1/2` - Lat/Lon coordinates

### Expected Data Structure
The HTML page contains a forecast table with:
1. **Date row**: Format like "08/05", "08/06"
2. **Hour row**: "Hour (HST)" with values like "09", "10", "11", etc.
3. **Surface Wind (mph)** row: Wind speed values
4. **Wind Dir** row: Direction codes (ENE, E, ESE, etc.)
5. **Wind Wave Height** row: Wave heights in feet

### Current Parser Location
File: `data_sources.py`
Method: `NOAAMarineSource._parse_forecast_table()` (lines 1864-1961)

### Actual HTML Structure (from investigation)
When examining the HTML, we found:
- Multiple tables exist on the page (7 total)
- Table 4 contains the actual forecast data
- The data is in a single concatenated format within table cells
- Example table content:
```
Date08/0508/06Hour (HST)091011121314151617181920212223000102030405060708Surface Wind (mph)13131010101212121313131515151414141414146666Wind DirEEENEENEENEENEENEENEENEENEENEENENENENEENEENEENEENEENEENENNENN
```

### Current Parser Issues
1. **Table Detection**: The parser looks for tables with "Hour (HST)" and "Surface Wind" text, but the actual table structure has these concatenated together
2. **Data Extraction**: The regex patterns assume separated lines, but the data is run together
3. **Wind Wave Height**: Not being found in the current parsing approach

## Required Fix

### Approach 1: Direct Table Parsing
Parse the table by finding the correct table index (likely table 4) and extracting the concatenated text, then using position-based extraction since the data appears to be in fixed positions.

### Approach 2: Better Pattern Matching
Use more robust regex patterns that can handle the concatenated format:
- Find the "Date" section and extract date values
- Find the "Hour (HST)" section and extract hour values
- Find "Surface Wind (mph)" and extract the following numbers
- Find "Wind Dir" and extract direction codes
- Find "Wind Wave" or wave-related text and extract heights

### Test Data for Validation
Run this command to see actual HTML structure:
```python
import requests
from bs4 import BeautifulSoup

url = 'https://marine.weather.gov/MapClick.php?w3=sfcwind&w3u=1&w14=wwh&AheadHour=0&FcstType=digital&textField1=21.8298&textField2=-157.759&site=all&unit=0&dd=&bw=&marine=1'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
tables = soup.find_all('table')
for i, table in enumerate(tables):
    print(f"Table {i}: {table.get_text()[:500]}")
```

## Integration Points
When fixed, the parser should return data in this format:
```python
{
    'timestamp': datetime object,
    'wind_speed_kts': float,  # Converted from mph
    'wind_dir_deg': float,     # Converted from cardinal directions
    'wind_wave_height_ft': float,
    'source': 'NOAA_Marine'
}
```

## Testing
After fixing, test with:
```bash
# Check if NOAA Marine source is available
python lumps.py --check-sources

# Run analysis with verbose output to see if data is being fetched
python lumps.py --days 2 --verbose

# Look for wind wave height scores (Wv) in output
# Should see non-zero values like Wv2, Wv3, etc. when waves are present
```

## Priority
This is a nice-to-have enhancement. The system works fine with existing wind sources (NDBC, NOAA CWF, OpenMeteo). The main benefit of fixing this parser would be getting wind wave height forecasts for better scoring of downwind foiling conditions.
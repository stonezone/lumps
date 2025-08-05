# NOAA Coastal Waters Forecast Integration

## New Data Source Added: NOAA CWF

I've integrated the NOAA Coastal Waters Forecast (CWF) as a high-quality wind data source for LUMPS.

### What It Provides:
- **5-day detailed marine wind forecasts** specifically for Hawaiian waters
- **Oahu Leeward Waters section** - closest to the Turtle Bay to Pipeline run
- **Official NOAA forecast** with wind speed ranges and directions
- **4 data points per forecast period** (6am, 9am, noon, 3pm for day; 6pm, 9pm, midnight, 3am for night)

### Wind Data Priority (Best to Fallback):
1. **NDBC Buoy** (real-time observations) - Most accurate
2. **NOAA Weather Buoy** (recent observations) 
3. **NOAA CWF** (official marine forecast) - NEW!
4. **OpenMeteo** (model forecast)

### Key Features:
- Parses text-based forecast into structured data
- Handles wind ranges (e.g., "15 to 20 knots" → 17.5kt average)
- Converts text directions to degrees (e.g., "East northeast" → 67°)
- Filters to requested date range automatically

### Expected Wind Data From Example:
```
TUESDAY: ENE 15-20kt (perfect for lumps!)
WEDNESDAY: Variable <10kt then NNW 7-10kt (poor)
THURSDAY: E 7-10kt → 10-15kt (building)
FRIDAY: ENE 15-20kt (perfect again!)
SATURDAY: E 10-15kt (good)
```

### Testing:
```bash
# Run with enhanced currents and check for NOAA CWF in logs
python lumps.py --enhanced-currents --start-date 2025-08-05 --days 5 --verbose

# Check logs for NOAA CWF usage
grep "NOAA Coastal Waters Forecast" lumps.log
grep "noaa_cwf" lumps.log
```

This gives LUMPS access to official marine forecasts that are specifically tailored for Hawaiian coastal waters, providing more accurate wind predictions than generic weather models.

---
*Official marine forecasts for better lump predictions!* 🌊

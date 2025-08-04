# LUMPS - North Shore Oahu Downwind Foiling Analysis

Finds optimal conditions for enhanced windswells ("lumps") by identifying when eastward currents oppose ENE trade winds on Oahu's North Shore.

## Overview

**Lump Physics**: Windswells become amplified when ocean currents flow AGAINST the wind direction (similar to Columbia River Gorge). For North Shore downwinding:
- **Foiling Direction**: Turtle Bay → Sunset Beach (westward, WITH the wind)
- **Optimal Current**: Eastward flow (060-120°) OPPOSING the wind
- **Result**: Enhanced lumps as windswells steepen against the opposing current

## Key Findings

Based on oceanographic research and tidal analysis:
- **Target Current Direction**: Eastward flow (060-120°) from tidal variations
- **Trade Winds**: ENE (050-070°) at 15-25+ knots during trade wind season  
- **Optimal Interaction**: Current flowing directly INTO wind creates maximum standing wave enhancement
- **Best Conditions**: Ebb tide at Kahuku Point (73° eastward flow) during ENE trades

## Installation

```bash
cd ~/code/lumps
pip install -r requirements.txt
```

## Quick Start

```bash
# Run 10-day analysis for eastward current flow
python lumps.py

# Custom analysis period  
python lumps.py --start-date 2025-06-16 --days 7

# Only show 6 AM - 7 PM conditions
python lumps.py --time-range 6-19

# Check data source availability
python lumps.py --check-sources

# Verbose output with detailed logging
python lumps.py --verbose --save-report
```

## Data Sources

### NDBC Buoy Stations
- **51201**: Waimea Bay, HI (Primary North Shore reference)
- **51205**: Pauwela, Maui (ENE trade wind reference)
- **51101**: Northwestern Hawaii (Deep water reference)

### NOAA Tide Stations  
- **1612340**: Honolulu, HI (Primary reference)
- **1612668**: Haleiwa, HI (North Shore reference)

### Additional Sources
- PacIOOS wind and current models
- NOAA trade wind forecasts
- Current pattern analysis

## Analysis Parameters

### Trade Wind Configuration
- **Optimal Direction**: 050-070° (ENE)
- **Optimal Speed**: 15-25 knots
- **Peak Conditions**: 18-22 knots

### Current Configuration
- **Primary Direction**: 290° (WNW - westward flow)
- **Typical Speed**: 0.3-0.8 knots
- **Maximum Speed**: 1.0+ knots

### Downwind Conditions
- **Optimal Angle**: 130-180° (current opposing wind)
- **Enhancement**: Maximum wave steepening
- **Time Window**: 6 AM - 7 PM HST (customizable)

## Output Format

```
📅 Day 1: Monday, June 16, 2025
------------------------------------------------------------
08:00 AM | ⭐ EXCELLENT | Wind: 18.5kt@060° | Current: 0.6kt@290° | Interaction: opposing
02:00 PM | 🔥 PRIME | Wind: 21.2kt@060° | Current: 0.7kt@290° | Interaction: opposing
```

### Condition Quality Ratings
- 🔥 **PRIME**: 20+ knot winds (expert conditions)
- ⭐ **EXCELLENT**: 18-20 knot winds (advanced conditions)  
- ✅ **GOOD**: 15-18 knot winds (intermediate conditions)
- 📝 **MARGINAL**: <15 knot winds (beginner conditions)

## Command Line Options

```bash
python analysis.py [options]

Options:
  --start-date YYYY-MM-DD    Start date for analysis (default: today)
  --days N                   Number of days to analyze (default: 7)
  --time-range START-END     Time window in HST (default: 6-19)
  --output-dir DIR           Output directory (default: data)
  --verbose                  Detailed output and recommendations
  --collect-only             Only collect data, skip analysis
  --analyze-only             Only analyze existing data
```

## File Structure

```
~/code/lumps/
├── lumps_data_collector.py   # Main data collection and analysis
├── config.py                 # Configuration and parameters
├── analysis.py           # Command-line interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/                     # Output directory
    ├── ndbc_buoy_data.csv
    ├── noaa_tide_data.csv
    └── optimal_downwind_times.csv
```

## Scientific Background

### Current-Wind Interaction Physics

When the North Equatorial Current flows westward at ~0.5 knots and encounters ENE trade winds creating wind swell, the opposing forces create:

1. **Wave Steepening**: Current flowing into wind swell increases wave face angle
2. **Enhanced "Lumps"**: Steeper, more defined wave formations
3. **Increased Energy**: More power available for downwind foiling
4. **Optimal Bumps**: Consistent, predictable wave trains

### North Shore Oceanography

- **Dominant Current**: North Equatorial Current (westward)
- **Trade Wind Season**: April-October (most consistent)
- **Current Speed**: 0.3-1.0 knots (varies with tide and pressure gradient)
- **Wind Pattern**: ENE trades 15-25 knots (strengthens afternoon)

## Safety Considerations

### Risk Levels by Conditions
- **Beginner**: <18 kt winds, <0.5 kt current
- **Intermediate**: 18-22 kt winds, 0.5-0.7 kt current  
- **Advanced**: 22-25 kt winds, 0.7-1.0 kt current
- **Expert**: 25+ kt winds, 1.0+ kt current

### Safety Recommendations
- Always check current marine weather forecasts
- Consider safety boat support for challenging conditions
- Be aware of increased current strength during spring tides
- Monitor for small craft advisories

## Equipment Recommendations

### By Wind Strength
- **15-18 kt**: 5-7m wing, high-lift foil
- **18-22 kt**: 4-6m wing, balanced foil
- **22-25 kt**: 3-5m wing, high-speed foil

### Downwind-Specific Gear
- Larger wing for catching enhanced lumps
- Higher volume foil for steeper wave faces
- Safety equipment (whistle, phone, backup)

## Troubleshooting

### Common Issues

**No optimal conditions found**
- Check date range (trade wind season: Apr-Oct)
- Expand time window beyond 6 AM - 7 PM
- Verify internet connection for data sources

**Data collection errors**
- Check internet connectivity
- Verify NDBC/NOAA services are operational
- Review log files in `lumps_analysis.log`

**Unexpected results**
- Cross-reference with marine weather forecasts
- Consider seasonal variations in current patterns
- Check for equipment maintenance at buoy stations

## Contributing

To improve the analysis:
1. Add additional data sources (Surfline, local observations)
2. Implement machine learning for pattern recognition
3. Add real-time alerts and notifications
4. Create mobile app interface

## References

- NOAA National Data Buoy Center (NDBC)
- Pacific Islands Ocean Observing System (PacIOOS)
- NOAA Tides and Currents
- Pat Caldwell surf forecasting methodology
- North Pacific oceanographic research

## License

Open source - use and modify as needed for oceanographic research and water sports analysis.

---

*Created for analyzing optimal downwind foiling conditions on Oahu's North Shore*

# LUMPS Project Overview

## Purpose
LUMPS is a North Shore Oahu downwind foiling analysis system that identifies optimal conditions for enhanced windswells ("lumps") used in downwind foiling. The system analyzes when eastward ocean currents oppose ENE trade winds to create amplified windswells.

## Key Physics
- **Lump Formation**: Windswells become amplified when ocean currents flow AGAINST the wind direction (similar to Columbia River Gorge effect)
- **Foiling Direction**: Turtle Bay → Sunset Beach (westward, WITH the wind)
- **Optimal Current**: Eastward flow (060-120°) OPPOSING ENE trade winds (050-070°)
- **Result**: Enhanced lumps as windswells steepen against the opposing current

## Tech Stack
- **Language**: Python 3.x
- **Core Libraries**: 
  - requests (API calls)
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib, seaborn, plotly (visualization)
  - pytz (timezone handling)
- **Data Sources**: 
  - NDBC buoy data
  - NOAA Tides and Currents
  - PacIOOS ocean models
  - OpenMeteo weather data

## Architecture
- **lumps.py**: Main CLI application with argument parsing and workflow orchestration
- **data_sources.py**: Data collection interfaces using abstract base class pattern
- **analysis.py**: Condition analysis and report generation
- **Modular Design**: Abstract base class `DataSource` for extensible data source integration

## Entry Points
- Primary: `python lumps.py` (main analysis tool)
- Test scripts: Various `test_*.py` and `debug_*.py` files for development
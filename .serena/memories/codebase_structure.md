# Codebase Structure

## Root Directory Files
- **lumps.py**: Main CLI application entry point
- **data_sources.py**: Data collection classes with abstract base class pattern
- **analysis.py**: Condition analysis and report generation
- **requirements.txt**: Python dependencies with pinned versions
- **README.md**: User documentation and scientific background
- **CLAUDE.md**: Development instructions and architecture notes

## Core Components

### lumps.py (Main Application)
- `main()`: Primary entry point with workflow orchestration
- `parse_arguments()`: CLI argument parsing with comprehensive options
- `setup_logging()`: Configurable logging to file and console
- `print_banner()`: Application startup banner
- `check_data_sources()`: Data source availability verification

### data_sources.py (Data Collection)
- `DataSource`: Abstract base class for all data sources
- `NOAACurrentSource`: NOAA Tides and Currents API (limited for Hawaii)
- `PacIOOSCurrentSource`: PacIOOS current forecasts with tidal simulation
- `NDFCBuoySource`: NDBC buoy data for wind/wave conditions
- `OpenMeteoWindSource`: Alternative wind data source
- `TidalCurrentAnalyzer`: Eastward flow analysis and current-wind interactions
- `DataCollector`: Coordinates multiple data sources with fallback handling

### analysis.py (Analysis Engine)
- `ConditionAnalyzer`: Condition classification (prime/excellent/good/marginal)
- `ReportGenerator`: Output report formatting and file generation

## Directory Structure
```
lumps/
├── lumps.py              # Main CLI application
├── data_sources.py       # Data collection interfaces
├── analysis.py          # Analysis and reporting
├── requirements.txt     # Dependencies
├── README.md           # User documentation
├── CLAUDE.md           # Development notes
├── data/               # Output directory
│   ├── *.csv          # Daily condition data
│   └── *.txt          # Generated reports
├── archive/           # Historical data/backups
├── debug_*.py         # Development debugging scripts
└── test_*.py          # Testing scripts
```

## Data Flow
1. **lumps.py** orchestrates the workflow
2. **DataCollector** aggregates data from multiple sources
3. **TidalCurrentAnalyzer** identifies eastward current periods
4. **ConditionAnalyzer** classifies conditions by quality
5. **ReportGenerator** formats and saves output reports

## Extension Points
- Add new data sources by inheriting from `DataSource` abstract base class
- Extend analysis by modifying `ConditionAnalyzer` classification logic
- Add new output formats via `ReportGenerator` methods
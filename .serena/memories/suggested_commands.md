# Suggested Commands for LUMPS Project

## Running the Application

### Basic Analysis
```bash
# Basic 10-day analysis from today
python lumps.py

# Custom date range and time window
python lumps.py --start-date 2025-06-16 --days 10 --time-range 6-19

# Check data source availability
python lumps.py --check-sources

# Verbose output with detailed logging
python lumps.py --verbose --save-report
```

### Common Options
- `--start-date YYYY-MM-DD`: Start date for analysis (default: today)
- `--days N`: Number of days to analyze (default: 7)
- `--time-range START-END`: Time window in HST (default: 6-19)
- `--verbose`: Detailed output and recommendations
- `--save-report`: Save detailed report to file
- `--output-dir DIR`: Output directory (default: data)

## Development Commands

### Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### Testing
```bash
# Test PacIOOS data access
python test_pacioos.py

# Test simulation fallback
python test_simulation_fallback.py
```

### Debugging
```bash
# Debug specific components
python debug_eastward_flow.py
python debug_pacioos_grid.py
python debug_timezones.py
python debug_directions.py
python check_pacioos_time.py
```

## System Commands (macOS)
```bash
# File operations
ls -la                    # List files with details
find . -name "*.py"      # Find Python files
grep -r "pattern" .      # Search in files
cd ~/code/lumps          # Navigate to project

# Git operations
git status
git add .
git commit -m "message"
git push origin main
```

## Output Files
- `data/optimal_conditions_YYYYMMDD.csv`: Raw condition data
- `data/lumps_report_YYYYMMDD.txt`: Detailed analysis report
- `lumps.log`: Application logging
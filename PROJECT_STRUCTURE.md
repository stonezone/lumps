# LUMPS Project Structure

## Root Directory Files

### Core Application Files
- `lumps.py` - Main CLI application
- `data_sources.py` - Data collection interfaces and sources
- `analysis.py` - Condition analysis and scoring logic

### Configuration & Documentation
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `CLAUDE.md` - Claude AI development guide
- `.gitignore` - Git ignore configuration

### Web Interface
- `webui.sh` - Web UI server management script
- `web/` - Web interface directory
  - `index.html` - Main web page
  - `manifest.json` - PWA manifest
  - `sw.js` - Service worker for offline capability
  - `data/` - Generated JSON data for web UI

### Data Output
- `data/` - Generated reports and CSV files (gitignored)

### Development & Testing
- `tests/` - All test and debug scripts
  - Test scripts (test_*.py)
  - Debug scripts (debug_*.py)
  - Verification scripts (verify_*.py, check_*.py)
  - `docs/` - Development documentation and fix summaries

## Files Excluded from Git

### Logs
- `*.log` - All log files
- `lumps.log` - Application log
- `webui.log` - Web UI server log

### Data Files
- `data/*.csv` - Generated CSV reports
- `data/*.txt` - Generated text reports
- `data/*.json` - Generated JSON data

### System Files
- `.DS_Store` - macOS system files
- `__pycache__/` - Python cache
- `.serena/` - Serena AI files
- `.claude/` - Claude AI files
- `archive/` - Archived files

### Virtual Environments
- `venv/`, `env/`, `.venv/`, `.env` - Python virtual environments

## How to Use

### Running the Application
```bash
# Basic analysis
python lumps.py

# With options
python lumps.py --start-date 2025-01-15 --days 7 --verbose

# Check data sources
python lumps.py --check-sources
```

### Web Interface
```bash
# Start web UI with fresh data
./webui.sh start

# Stop web UI
./webui.sh stop

# Refresh data without restarting
./webui.sh refresh

# Check status
./webui.sh status
```

### Running Tests
```bash
# Run specific test
python tests/test_pacioos.py

# Run verification scripts
python tests/verify_eastward_fix.py
```

## GitHub Repository Setup

1. Initialize repository:
```bash
git init
git add .
git commit -m "Initial commit: LUMPS downwind foiling analysis system"
```

2. Add remote and push:
```bash
git remote add origin https://github.com/yourusername/lumps.git
git branch -M main
git push -u origin main
```

3. The `.gitignore` is configured to exclude:
   - All log files
   - Generated data files
   - System and IDE files
   - Python cache and virtual environments
   - Test output files

## Dependencies

Install all dependencies with:
```bash
pip install -r requirements.txt
```

Main dependencies:
- requests - API calls
- pandas - Data manipulation
- numpy - Numerical operations
- beautifulsoup4 - HTML parsing
- matplotlib, seaborn, plotly - Visualization
- pytz - Timezone handling

## Project Purpose

LUMPS identifies optimal conditions for downwind foiling on Oahu's North Shore by analyzing when eastward ocean currents (060-120°) oppose ENE trade winds (050-070°), creating enhanced windswells ("lumps") for better foiling conditions.
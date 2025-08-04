# Code Style and Conventions

## Python Style
- **PEP 8 compliant**: Standard Python formatting
- **Function documentation**: Docstrings using triple quotes
- **Type hints**: Used in abstract base class methods
- **Class design**: Abstract base classes for extensibility

## Naming Conventions
- **Classes**: PascalCase (e.g., `DataSource`, `ConditionAnalyzer`)
- **Functions/Methods**: snake_case (e.g., `fetch_data`, `setup_logging`)
- **Variables**: snake_case (e.g., `start_date`, `optimal_conditions`)
- **Constants**: UPPER_CASE (e.g., logging level constants)

## Code Organization
- **Abstract Base Classes**: Used for data source interfaces
- **Separation of Concerns**: 
  - CLI logic in lumps.py
  - Data collection in data_sources.py
  - Analysis logic in analysis.py
- **Error Handling**: Comprehensive try/catch with logging
- **Logging**: Structured logging to both file and console

## Documentation Style
- **Docstrings**: Brief description for functions and classes
- **Comments**: Minimal inline comments, self-documenting code preferred
- **README**: Comprehensive usage and scientific background
- **CLAUDE.md**: Development instructions and architecture notes

## Dependencies
- **Requirements**: Pinned versions in requirements.txt
- **Imports**: Standard library first, then third-party, then local
- **External APIs**: Graceful degradation when services unavailable

## Data Handling
- **Pandas DataFrames**: Primary data structure for analysis
- **Date/Time**: UTC internally, HST for user display
- **CSV Output**: Standard format for data persistence
- **Error Logging**: Detailed logging for debugging data source issues
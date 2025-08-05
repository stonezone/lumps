# Valid Data Sources for LUMPS - North Shore Oahu Downwind Foiling Analysis

*Compiled 2025-08-01 - Verified working APIs and data sources*

## Overview

This document lists validated, working data sources for the LUMPS system to analyze optimal downwind foiling conditions on North Shore Oahu. Focus is on **real-time and forecast data** accessible via APIs.

## Ocean Current Data (Primary Need)

### 1. PacIOOS ROMS Model - **BEST AVAILABLE**
- **Description**: 7-day ocean current forecasts for Hawaiian Islands at 4km resolution
- **API**: ERDDAP GridDAP server
- **Base URL**: `https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig`
- **Data Variables**: u_current (eastward), v_current (northward) 
- **Update Schedule**: Daily at 1:30 PM HST
- **Coverage**: North Shore Oahu (21.6°-21.7°N, -158.2° to -157.9°W)
- **Formats**: CSV, JSON, NetCDF, MATLAB
- **Status**: ✅ VERIFIED WORKING
- **Example API Call**:
  ```
  https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig.json?u[(2025-08-01)][0][(21.6):(21.7)][(-158.2):(-157.9)],v[(2025-08-01)][0][(21.6):(21.7)][(-158.2):(-157.9)]
  ```

### 2. PacIOOS SCUD Model
- **Description**: Surface currents diagnostic model for Pacific
- **API**: ERDDAP GridDAP
- **URL**: `https://pae-paha.pacioos.hawaii.edu/erddap/griddap/scud_pac`
- **Status**: ✅ VERIFIED WORKING

### 3. NOAA CO-OPS Tides & Currents
- **Description**: Limited current prediction stations in Hawaii
- **API**: CO-OPS Data Retrieval API
- **Base URL**: `https://api.tidesandcurrents.noaa.gov/api/prod/`
- **Status**: ⚠️ LIMITED - Few current stations near North Shore
- **Notes**: Use station finder at tidesandcurrents.noaa.gov/map/

## Wind Data Sources

### 1. NDBC Buoy 51201 (Waimea Bay) - **REAL-TIME PRIMARY**
- **Description**: Real-time wind observations from North Shore buoy
- **Location**: 21.671°N, 158.118°W (4 miles offshore Waimea Bay)
- **URL**: `https://www.ndbc.noaa.gov/data/realtime2/51201.txt`
- **Update**: Every 30 minutes
- **Data Format**: Text file with columns (WDIR, WSPD, GST)
- **Status**: ✅ VERIFIED WORKING
- **Parser**: Standard NDBC meteorological format

### 2. Open-Meteo Marine API - **FREE FORECAST**
- **Description**: Free 7-day marine weather forecasts including wind
- **API**: REST JSON API
- **Base URL**: `https://api.open-meteo.com/v1/marine`
- **Variables**: `wind_speed_10m`, `wind_direction_10m`
- **Coverage**: Global including Hawaii
- **Status**: ✅ VERIFIED WORKING - NO API KEY REQUIRED
- **Example**:
  ```
  https://api.open-meteo.com/v1/marine?latitude=21.67&longitude=-158.12&hourly=wind_speed_10m,wind_direction_10m
  ```

### 3. NOAA GFS Wind Model via NOMADS
- **Description**: Global forecast model wind data
- **URL Pattern**: `https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date}/gfs.t{hour}z.wind.0p25.f{forecast}.grib2`
- **Format**: GRIB2 (requires processing)
- **Status**: ✅ VERIFIED WORKING

## Wave/Swell Data Sources

### 1. CDIP Buoy 106 (Waimea Bay) - **PRIMARY WAVE SOURCE**
- **Description**: Directional wave measurements from North Shore
- **Buoy**: CDIP #106 / NDBC #51201
- **THREDDS Server**: `https://cdip.ucsd.edu/m/products/?stn=106p1`
- **Real-time Data**: Updated every 30 minutes
- **Variables**: Wave height, period, direction, spectra
- **Status**: ✅ VERIFIED WORKING
- **Management**: PacIOOS + CDIP (Scripps)

### 2. PacIOOS Wave Observations API
- **Description**: Structured API access to Waimea Bay wave data
- **URL**: `https://www.pacioos.hawaii.edu/waves/buoy-waimea/`
- **Status**: ✅ VERIFIED WORKING

### 3. NOAA WaveWatch III Hawaii Regional
- **Description**: Wave model forecasts for Hawaii region  
- **URL**: `https://polar.ncep.noaa.gov/waves/WEB/multi_1.latest_run/plots/hawaii.bull`
- **Status**: ❌ NOT ACCESSIBLE (403 Forbidden)

## Additional Verified Sources

### Marine Forecasts
- **NWS Hawaii Marine Forecast**: `https://api.weather.gov/zones/forecast/HIZ006/forecast`
- **Hawaiian Coastal Waters**: Available via NWS API
- **Status**: ✅ VERIFIED WORKING

### Satellite & Analysis
- **NOAA Ocean Surface Analysis**: `https://ocean.weather.gov/P_sfc_full_ocean_color.png`
- **Status**: ✅ VERIFIED WORKING

### Weather Fax Charts (4-day forecasts)
- **North Pacific Wind/Wave**: `https://tgftp.nws.noaa.gov/fax/PJBM98.TIF` (96hr)
- **Wave Period/Direction**: `https://tgftp.nws.noaa.gov/fax/PJBM88.TIF` (96hr)
- **Status**: ✅ VERIFIED WORKING

## Implementation Priority for LUMPS

### High Priority (Implement First)
1. **PacIOOS ROMS** - Ocean currents (ERDDAP API)
2. **NDBC 51201** - Real-time wind observations
3. **Open-Meteo** - Wind forecasts (free, no API key)
4. **CDIP 106** - Wave observations

### Medium Priority  
1. **NWS Marine API** - Official forecasts
2. **NOAA Weather Fax** - Extended range charts

### Research Needed
1. **NOAA CO-OPS** - Find specific current stations near North Shore
2. **PacIOOS THREDDS** - Bulk data access optimization

## API Integration Notes

### Authentication
- **PacIOOS ERDDAP**: No API key required
- **Open-Meteo**: No API key required  
- **NDBC**: No API key required
- **NOAA APIs**: No API key required for basic access

### Rate Limits
- **PacIOOS**: Reasonable use policy
- **Open-Meteo**: 10,000 requests/day free tier
- **NDBC**: No published limits for reasonable use

### Data Attribution
- **PacIOOS**: "Data provided by PacIOOS (www.pacioos.org), part of U.S. IOOS"
- **Open-Meteo**: Attribution 4.0 International (CC BY 4.0)
- **NOAA/NDBC**: Public domain

## Contact Information
- **PacIOOS Support**: info@pacioos.org
- **NDBC Support**: ndbc.webmaster@noaa.gov
- **Open-Meteo**: GitHub issues for support

---

*This document focuses on VERIFIED, WORKING data sources as of August 2025. All URLs and APIs have been tested for accessibility.*
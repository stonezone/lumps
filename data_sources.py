#!/usr/bin/env python3
"""
Modular data source interfaces for North Shore current and wind data
Focuses on finding eastward current flow (Sunset Beach → Turtle Bay) into ENE trades
"""

import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import pytz
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO

logger = logging.getLogger(__name__)

def retry_request(url: str, params: Dict, max_retries: int = 3, timeout: int = 30) -> requests.Response:
    """Helper function to retry HTTP requests with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 500, 502, 503, 504]:  # Retry on server errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"HTTP {response.status_code} error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            else:
                logger.error(f"HTTP {response.status_code} error (non-retryable)")
                break
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Request timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error("Request timeout after all retries")
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Request error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Request failed after all retries: {e}")
                raise
    
    # If we get here, all retries failed
    raise requests.exceptions.RequestException(f"All {max_retries} attempts failed")

class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    @abstractmethod
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class NOAACurrentSource(DataSource):
    """NOAA Current Predictions API - Kahuku Point Station"""
    
    def __init__(self, station_id: str = "HAI1112_26"):
        self.station_id = station_id
        self.base_url = "https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/currentPredictionsDownload"
        
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch current predictions from NOAA Kahuku Point station"""
        try:
            # Check if request is too far in the future for NOAA
            max_future_days = 30
            if start_date > datetime.now() + timedelta(days=max_future_days):
                logger.warning(f"NOAA data not available for dates more than {max_future_days} days in future")
                logger.info("NOAA is historical/near-term forecast only - skipping for far future dates")
                return pd.DataFrame()
            
            # NOAA predictions available for recent dates - use actual requested dates
            date_range = min((end_date - start_date).days + 1, 7)  # Max 7 days for API limit
            
            # Use requested start_date instead of hardcoded date
            actual_start = start_date
            
            # Use current predictions API for Kahuku Point
            params = {
                'id': self.station_id,
                'start_date': actual_start.strftime('%Y-%m-%d'),
                'range': str(date_range),
                'interval': 'MAX_SLACK',
                'time_zone': 'LST_LDT',
                'units': '1',  # 1 = knots
                'format': 'txt'
            }
            
            response = retry_request(self.base_url, params, max_retries=3, timeout=30)
            
            if response.status_code == 200 and 'Kahuku Point' in response.text:
                # Parse the text response (it's not JSON)
                lines = response.text.strip().split('\n')
                
                # Find where data starts (after header lines)
                data_start = 0
                for i, line in enumerate(lines):
                    if 'Date' in line and 'Time' in line:
                        data_start = i + 1
                        break
                
                # Parse current predictions
                data = []
                for line in lines[data_start:]:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            date_str = parts[0]
                            time_str = ' '.join(parts[1:3])  # e.g., "12:44 AM"
                            speed = float(parts[3])
                            
                            # Parse metadata to get ebb/flood directions
                            # Mean Flood Dir: 265° (westward), Mean Ebb Dir: 73° (eastward)
                            # Negative speeds = ebb (eastward), Positive = flood (westward)
                            if speed < 0:
                                direction = 73  # Ebb tide - eastward
                            elif speed > 0:
                                direction = 265  # Flood tide - westward  
                            else:
                                direction = 0  # Slack water
                            
                            data.append({
                                'datetime': pd.to_datetime(f"{date_str} {time_str}"),
                                'current_speed': abs(speed),
                                'current_dir': direction,
                                'tide_type': 'ebb' if speed < 0 else 'flood' if speed > 0 else 'slack',
                                'source': 'NOAA_Kahuku'
                            })
                
                if data:
                    df = pd.DataFrame(data)
                    logger.info(f"Fetched {len(df)} current predictions from Kahuku Point")
                    return df
            else:
                logger.error(f"CRITICAL: NOAA API failed for requested dates {actual_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                logger.error(f"Status {response.status_code}, Response: {response.text[:200]}")
            
            logger.error(f"CRITICAL: No current data available from NOAA station {self.station_id} for requested dates")
            return pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NOAA API request failed after retries: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching NOAA data: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if NOAA current predictions API is responding"""
        # NOAA current predictions API is currently failing with HTTP 500 errors
        # Disabling this source to prevent log spam until API is fixed
        logger.debug("NOAA current predictions API disabled due to persistent HTTP 500 errors")
        return False

class PacIOOSCurrentSource(DataSource):
    """PacIOOS current forecasts and observations"""
    
    def __init__(self):
        # PacIOOS endpoints (based on research - may need adjustment)
        self.forecast_url = "http://www.pacioos.hawaii.edu/currents/model-oahu/"
        self.obs_url = "http://www.pacioos.hawaii.edu/currents/obs-oahu/"
    
    def _get_dataset_end_time(self, base_url: str) -> Optional[datetime]:
        """Dynamically get dataset end time from ERDDAP dataset info"""
        try:
            # Get dataset info from ERDDAP
            info_url = base_url.replace('.csv', '.das')
            response = requests.get(info_url, timeout=10)
            
            if response.status_code == 200:
                # Parse DAS format to find time dimension max value
                for line in response.text.split('\n'):
                    if 'time_coverage_end' in line and '"' in line:
                        # Extract timestamp from line like: time_coverage_end "2025-08-07T00:00:00Z";
                        timestamp_str = line.split('"')[1]
                        return pd.to_datetime(timestamp_str).to_pydatetime().replace(tzinfo=None)
                        
                # Fallback: look for time actual_range
                for line in response.text.split('\n'):
                    if 'actual_range' in line and 'time' in line:
                        # Extract the second timestamp (end time)
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                end_timestamp = float(parts[-1].rstrip(';'))
                                return datetime.utcfromtimestamp(end_timestamp)
                            except (ValueError, IndexError):
                                continue
                                
        except Exception as e:
            logger.debug(f"Could not get dynamic dataset end time: {e}")
            
        return None
        
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch PacIOOS current data from ERDDAP - Hawaii ROMS forecast (7-day limit)"""
        try:
            base_url = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig.csv"
            
            # Dynamic dataset boundary checking - get current dataset limits from ERDDAP
            dataset_end = self._get_dataset_end_time(base_url)
            if dataset_end is None:
                logger.warning("Could not determine PacIOOS dataset end time, using 7-day limit from now")
                dataset_end = datetime.now() + timedelta(days=7)
            
            actual_end_date = min(end_date, dataset_end)
            
            if actual_end_date < end_date:
                logger.info(f"Adjusting PacIOOS end date from {end_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')} (dataset boundary limit)")
            
            # Format dates for ERDDAP (ISO format)
            start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = actual_end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Get u and v components separately, then combine
            # North Shore area: Turtle Bay to Sunset Beach
            lat_range = "[(21.6):(21.7)]"
            lon_range = "[(-158.2):(-157.9)]"
            time_range = f"[({start_str}):({end_str})]"
            depth_surface = "[0]"
            
            # Fetch u component (eastward velocity)
            u_query = f"u{time_range}{depth_surface}{lat_range}{lon_range}"
            u_url = f"{base_url}?{u_query}"
            
            logger.info(f"Fetching PacIOOS u-component: {u_url}")
            u_response = requests.get(u_url, timeout=30)
            
            if u_response.status_code != 200:
                logger.warning(f"PacIOOS u-component request failed: {u_response.status_code}")
                logger.warning(f"Response: {u_response.text[:200]}")
                return pd.DataFrame()
            
            # Fetch v component (northward velocity)  
            v_query = f"v{time_range}{depth_surface}{lat_range}{lon_range}"
            v_url = f"{base_url}?{v_query}"
            
            logger.info(f"Fetching PacIOOS v-component: {v_url}")
            v_response = requests.get(v_url, timeout=30)
            
            if v_response.status_code != 200:
                logger.warning(f"PacIOOS v-component request failed: {v_response.status_code}")
                logger.warning(f"Response: {v_response.text[:200]}")
                return pd.DataFrame()
            
            # Parse and combine u/v data
            logger.info("Successfully fetched PacIOOS ERDDAP data - parsing components")
            return self._parse_uv_components(u_response.text, v_response.text)
            
        except Exception as e:
            logger.warning(f"PacIOOS ERDDAP error: {e}")
            
        logger.warning("PacIOOS ERDDAP failed - returning empty DataFrame")
        return pd.DataFrame()
    
    def _parse_uv_components(self, u_csv: str, v_csv: str) -> pd.DataFrame:
        """Parse separate u and v component CSV responses from ERDDAP"""
        try:
            # Parse u component data
            u_data = {}
            for line in u_csv.strip().split('\n')[2:]:  # Skip header and units
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 5:
                        time_key = parts[0]  # Use time as key
                        u_val = float(parts[4]) if parts[4] != 'NaN' else 0.0
                        u_data[time_key] = {
                            'time': parts[0],
                            'lat': float(parts[2]),
                            'lon': float(parts[3]),
                            'u': u_val
                        }
            
            # Parse v component data and combine with u
            combined_data = []
            for line in v_csv.strip().split('\n')[2:]:  # Skip header and units
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 5:
                        time_key = parts[0]
                        v_val = float(parts[4]) if parts[4] != 'NaN' else 0.0
                        
                        if time_key in u_data:
                            u_val = u_data[time_key]['u']
                            
                            # Calculate current speed and direction
                            current_speed = np.sqrt(u_val**2 + v_val**2)
                            # Direction: angle from east (u) towards north (v), then convert to compass bearing
                            current_dir = (90 - np.degrees(np.arctan2(v_val, u_val))) % 360
                            
                            # Convert UTC datetime to tz-naive to match other data sources
                            dt_utc = pd.to_datetime(parts[0])
                            dt_naive = dt_utc.tz_localize(None) if dt_utc.tz is not None else dt_utc
                            
                            combined_data.append({
                                'datetime': dt_naive,
                                'current_speed': current_speed,
                                'current_dir': current_dir,
                                'latitude': float(parts[2]),
                                'longitude': float(parts[3]),
                                'u_component': u_val,
                                'v_component': v_val,
                                'source': 'PacIOOS_ROMS_Hawaii'
                            })
            
            if combined_data:
                df = pd.DataFrame(combined_data)
                logger.info(f"Successfully parsed {len(df)} current records from PacIOOS ROMS (before spatial aggregation)")
                
                # Aggregate spatial data by timestamp to avoid duplicates
                # Average current speed and use median direction per timestamp
                aggregated_data = []
                for timestamp in df['datetime'].unique():
                    timestamp_data = df[df['datetime'] == timestamp]
                    
                    # Calculate spatial average of current speed
                    avg_speed = timestamp_data['current_speed'].mean()
                    
                    # For direction, use the direction from the grid point with strongest current
                    max_speed_idx = timestamp_data['current_speed'].idxmax()
                    representative_dir = timestamp_data.loc[max_speed_idx, 'current_dir']
                    
                    # Use representative lat/lon (center of grid approximately)
                    center_lat = timestamp_data['latitude'].median()
                    center_lon = timestamp_data['longitude'].median()
                    
                    aggregated_data.append({
                        'datetime': timestamp,
                        'current_speed': avg_speed,
                        'current_dir': representative_dir,
                        'latitude': center_lat,
                        'longitude': center_lon,
                        'u_component': timestamp_data['u_component'].mean(),
                        'v_component': timestamp_data['v_component'].mean(),
                        'source': 'PacIOOS_ROMS_Hawaii'
                    })
                
                aggregated_df = pd.DataFrame(aggregated_data)
                logger.info(f"Aggregated to {len(aggregated_df)} unique timestamps from PacIOOS ROMS")
                return aggregated_df
            else:
                logger.warning("No valid current data parsed from PacIOOS components")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error parsing PacIOOS u/v components: {e}")
            return pd.DataFrame()
    
    def _parse_erddap_csv(self, csv_text: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Parse ERDDAP CSV response and convert to current data format"""
        try:
            lines = csv_text.strip().split('\n')
            
            # ERDDAP format: header line, units line, then data
            # Find header (first non-comment line with time,depth,latitude,longitude,u,v)
            header_idx = -1
            for i, line in enumerate(lines):
                if not line.startswith('#') and 'time' in line and 'u,v' in line:
                    header_idx = i
                    break
            
            if header_idx == -1:
                logger.error("No valid ERDDAP header found")
                return pd.DataFrame()
            
            header = lines[header_idx].split(',')
            logger.info(f"ERDDAP CSV header: {header}")
            
            # Data starts after header + units line (skip both)
            data_start = header_idx + 2
            
            if data_start >= len(lines):
                logger.error("No data lines found after ERDDAP header")
                return pd.DataFrame()
            
            data = []
            for line in lines[data_start:]:
                if line.strip():
                    values = line.split(',')
                    if len(values) >= len(header):
                        try:
                            # ERDDAP format: time, depth, latitude, longitude, u, v
                            time_str = values[0]
                            dt = pd.to_datetime(time_str)
                            
                            # Get u (eastward) and v (northward) components
                            u_val = float(values[4]) if values[4] != 'NaN' and values[4] != '' else 0.0
                            v_val = float(values[5]) if values[5] != 'NaN' and values[5] != '' else 0.0
                            
                            # Calculate speed and direction from u/v components
                            current_speed = np.sqrt(u_val**2 + v_val**2)
                            current_dir = np.degrees(np.arctan2(u_val, v_val)) % 360
                            
                            data.append({
                                'datetime': dt,
                                'current_speed': current_speed,
                                'current_dir': current_dir,
                                'source': 'PacIOOS_ERDDAP'
                            })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Error parsing ERDDAP line: {line}, error: {e}")
                            continue
            
            df = pd.DataFrame(data)
            if df.empty:
                logger.error("No valid current data parsed from ERDDAP response")
                return pd.DataFrame()
                
            logger.info(f"Successfully parsed {len(df)} current records from PacIOOS ERDDAP")
            return df
            
        except Exception as e:
            logger.error(f"Critical error parsing ERDDAP CSV: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check PacIOOS availability"""
        # Placeholder - would check actual PacIOOS endpoints
        return True

@dataclass
class SpatialBounds:
    """Geographic bounding box for North Shore Oahu"""
    lat_min: float = 21.59    # Southern boundary  
    lat_max: float = 21.72    # Northern boundary
    lon_min: float = -158.11  # Western boundary (Haleiwa)
    lon_max: float = -157.97  # Eastern boundary (Kahuku Point)
    depth: float = 0.25       # Near-surface layer


class PacIOOSEnhancedCurrentSource(DataSource):
    """Enhanced PacIOOS current source with hourly interpolation and daylight filtering
    
    Based on the proven working approach from fetch_pacioos_currents_v2_fixed.py
    Uses ERDDAP CSV method that has been validated to work reliably.
    """
    
    def __init__(self, use_hourly_interpolation=False, daylight_only=False):
        self.use_hourly_interpolation = use_hourly_interpolation
        self.daylight_only = daylight_only
        self.bounds = SpatialBounds()
        self.base_url = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig"
        self.timezone_offset = -10  # Hawaii Standard Time
        self.sleep_seconds = 1.0

        
    def _get_dataset_end_time(self) -> Optional[datetime]:
        """Dynamically get dataset end time from ERDDAP dataset info"""
        try:
            # Get dataset info from ERDDAP
            info_url = f"{self.base_url}.das"
            response = requests.get(info_url, timeout=10)
            
            if response.status_code == 200:
                # Parse DAS format to find time dimension max value
                for line in response.text.split('\n'):
                    if 'time_coverage_end' in line and '"' in line:
                        # Extract timestamp from line like: time_coverage_end "2025-08-07T00:00:00Z";
                        timestamp_str = line.split('"')[1]
                        return pd.to_datetime(timestamp_str).to_pydatetime().replace(tzinfo=None)
                        
                # Fallback: look for time actual_range
                for line in response.text.split('\n'):
                    if 'actual_range' in line and 'time' in line:
                        # Extract the second timestamp (end time)
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                end_timestamp = float(parts[-1].rstrip(';'))
                                return datetime.utcfromtimestamp(end_timestamp)
                            except (ValueError, IndexError):
                                continue
                                
        except Exception as e:
            logger.debug(f"Could not get dynamic dataset end time: {e}")
            
        return None
        
    def _build_url(self, variable: str, time_start: datetime, time_end: datetime) -> str:
        """Construct an ERDDAP query URL for the given variable and time range"""
        start_iso = time_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = time_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        b = self.bounds
        # Format: dataset.csv?var[(start):stride:(end)][(depth)][(lat_min):(lat_max)][(lon_min):(lon_max)]
        url = (
            f"{self.base_url}.csv?{variable}"
            f"[({start_iso}):1:({end_iso})]"
            f"[({b.depth})]"
            f"[({b.lat_min}):1:({b.lat_max})]"
            f"[({b.lon_min}):1:({b.lon_max})]"
        )
        return url

    def _fetch_component(self, variable: str, time_start: datetime, time_end: datetime) -> pd.DataFrame:
        """Retrieve one velocity component from ERDDAP for a time range"""
        url = self._build_url(variable, time_start, time_end)
        logger.debug(f"Requesting {variable} from {time_start} to {time_end}")
        
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            
            # Parse CSV
            csv_text = resp.text
            df = pd.read_csv(StringIO(csv_text))
            
            if df.empty:
                return pd.DataFrame()
                
            # Remove rows where time equals the string 'UTC' (units row)
            df = df[df['time'] != 'UTC']
            
            # Convert ALL numeric columns properly to avoid string/int errors
            numeric_cols = ['depth', 'latitude', 'longitude', variable]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Rename the velocity column to avoid collision
            if variable in df.columns:
                df = df.rename(columns={variable: f"{variable}_velocity"})
                
            return df
            
        except Exception as exc:
            logger.warning(f"Failed to fetch {variable} data: {exc}")
            return pd.DataFrame()

    def _chunk_time_ranges(self, start: datetime, end: datetime, hours_per_chunk: int = 24) -> List[Tuple[datetime, datetime]]:
        """Split a larger time range into smaller chunks"""
        ranges = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(hours=hours_per_chunk), end)
            ranges.append((current, chunk_end))
            current = chunk_end
        return ranges

    def _merge_uv(self, u: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
        """Merge u and v components on shared coordinates"""
        if u is None or u.empty or v is None or v.empty:
            return pd.DataFrame()
            
        merge_cols = ['time', 'depth', 'latitude', 'longitude']
        try:
            merged = pd.merge(
                u,
                v[merge_cols + ['v_velocity']],
                on=merge_cols,
                how='inner',
            )
            return merged
        except Exception as exc:
            logger.warning(f"Error merging components: {exc}")
            return pd.DataFrame()

    def _compute_speed_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add current speed and direction columns derived from u and v components"""
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Calculate magnitude in m/s
        df.loc[:, 'current_speed_ms'] = np.sqrt(df['u_velocity'] ** 2 + df['v_velocity'] ** 2)
        
        # Calculate direction in degrees (0° = North, 90° = East)
        df.loc[:, 'current_direction_deg'] = np.degrees(np.arctan2(df['u_velocity'], df['v_velocity']))
        df.loc[:, 'current_direction_deg'] = (df['current_direction_deg'] + 360) % 360
        
        # Convert to knots (1 m/s = 1.94384 knots) and standard current_speed field
        df.loc[:, 'current_speed_knots'] = df['current_speed_ms'] * 1.94384
        df.loc[:, 'current_speed'] = df['current_speed_knots']  # Standard field name
        df.loc[:, 'current_dir'] = df['current_direction_deg']  # Standard field name
        
        return df

    def _filter_daylight_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to daylight hours only (6 AM - 6 PM HST)"""
        if not self.daylight_only or df.empty:
            return df
        
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Convert to Hawaii local time  
        df.loc[:, 'local_time'] = df['time'] + pd.Timedelta(hours=self.timezone_offset)
        df.loc[:, 'hour'] = df['local_time'].dt.hour
        
        # Filter to daylight hours
        daylight_df = df[(df['hour'] >= 6) & (df['hour'] <= 18)].copy()
        logger.info(f"Filtered to daylight hours: {len(daylight_df)} of {len(df)} records remain")
        return daylight_df.sort_values(['time', 'latitude', 'longitude'])

    def _interpolate_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate 3-hourly current data to hourly resolution"""
        if not self.use_hourly_interpolation or df.empty:
            return df
        
        # Handle duplicate time labels by keeping only the first occurrence
        df_clean = df.drop_duplicates(subset=['time'], keep='first').copy()
        
        # Set time as index for resampling
        df_indexed = df_clean.set_index('time')
        
        # Interpolate to hourly frequency
        hourly_df = df_indexed.resample('1H').interpolate(method='linear')
        
        # Reset index and recalculate derived fields if needed
        result = hourly_df.reset_index()
        logger.info(f"Interpolated to hourly: {len(result)} records from {len(df_clean)} original")
        return result

    def _process_enhanced_data(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Process merged U/V data with enhanced features"""
        if merged_df.empty:
            return pd.DataFrame()
        
        # Convert string timestamps to pandas datetime (ensure tz-naive)
        merged_df['time'] = pd.to_datetime(merged_df['time']).dt.tz_localize(None)
        
        # Ensure velocity columns are numeric and remove NaN values
        merged_df['u_velocity'] = pd.to_numeric(merged_df['u_velocity'], errors='coerce')
        merged_df['v_velocity'] = pd.to_numeric(merged_df['v_velocity'], errors='coerce')
        merged_df = merged_df.dropna(subset=['u_velocity', 'v_velocity'])
        
        if merged_df.empty:
            return pd.DataFrame()
        
        # Compute speed and direction
        processed_df = self._compute_speed_direction(merged_df)
        
        # Apply daylight filtering if enabled
        if self.daylight_only:
            processed_df = self._filter_daylight_hours(processed_df)
        
        # Apply hourly interpolation if enabled
        if self.use_hourly_interpolation:
            processed_df = self._interpolate_to_hourly(processed_df)
        
        # Convert to standard format for compatibility with existing system
        if not processed_df.empty:
            # Aggregate spatial data by timestamp to match existing format
            aggregated_data = []
            for timestamp in processed_df['time'].unique():
                timestamp_data = processed_df[processed_df['time'] == timestamp]
                
                # Calculate spatial average of current speed
                avg_speed = timestamp_data['current_speed'].mean()
                
                # For direction, use the direction from the grid point with strongest current
                max_speed_idx = timestamp_data['current_speed'].idxmax()
                representative_dir = timestamp_data.loc[max_speed_idx, 'current_dir']
                
                # Use representative lat/lon (center of grid approximately)
                center_lat = timestamp_data['latitude'].median()
                center_lon = timestamp_data['longitude'].median()
                
                aggregated_data.append({
                    'datetime': timestamp if isinstance(timestamp, datetime) else pd.to_datetime(timestamp).tz_localize(None),  # Convert to tz-naive
                    'current_speed': avg_speed,
                    'current_dir': representative_dir,
                    'latitude': center_lat,
                    'longitude': center_lon,
                    'u_component': timestamp_data['u_velocity'].mean(),
                    'v_component': timestamp_data['v_velocity'].mean(),
                    'source': 'PacIOOS_Enhanced'
                })
            
            result_df = pd.DataFrame(aggregated_data)
            logger.info(f"Enhanced processing complete: {len(result_df)} aggregated timestamps")
            return result_df
        
        return pd.DataFrame()

    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch PacIOOS current data using enhanced ERDDAP CSV approach"""
        try:
            logger.info("Starting enhanced PacIOOS current data fetch")
            
            # Check dataset boundaries to avoid 404 errors
            dataset_end = self._get_dataset_end_time()
            if dataset_end is None:
                logger.warning("Could not determine PacIOOS dataset end time, using 7-day limit from now")
                dataset_end = datetime.now() + timedelta(days=7)
            
            actual_end_date = min(end_date, dataset_end)
            
            if actual_end_date < end_date:
                logger.info(f"Adjusting enhanced PacIOOS end date from {end_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')} (dataset boundary limit)")
            
            all_u = []
            all_v = []
            
            # Fetch data in chunks to avoid timeouts
            for t0, t1 in self._chunk_time_ranges(start_date, actual_end_date, 24):
                u_chunk = self._fetch_component('u', t0, t1)
                if not u_chunk.empty:
                    all_u.append(u_chunk)
                time.sleep(self.sleep_seconds)
                
                v_chunk = self._fetch_component('v', t0, t1)
                if not v_chunk.empty:
                    all_v.append(v_chunk)
                time.sleep(self.sleep_seconds)

            if not all_u or not all_v:
                logger.warning("No U or V data retrieved from enhanced PacIOOS source")
                return pd.DataFrame()
                
            logger.info(f"Combining {len(all_u)} U chunks and {len(all_v)} V chunks")
            u_df = pd.concat(all_u, ignore_index=True)
            v_df = pd.concat(all_v, ignore_index=True)
            
            merged = self._merge_uv(u_df, v_df)
            if merged.empty:
                logger.warning("Failed to merge U and V components")
                return pd.DataFrame()
                
            logger.info(f"Merged data points: {len(merged)}")
            
            # Apply enhanced processing
            result = self._process_enhanced_data(merged)
            
            if not result.empty:
                logger.info(f"Enhanced PacIOOS fetch successful: {len(result)} final records")
            else:
                logger.warning("No data remaining after enhanced processing")
                
            return result
            
        except Exception as e:
            logger.warning(f"Enhanced PacIOOS fetch failed: {e}")
            return pd.DataFrame()

    def is_available(self) -> bool:
        """Check PacIOOS ERDDAP server availability"""
        try:
            # Quick test request for a small time window
            test_url = f"{self.base_url}.csv?u[(2025-08-01T00:00:00Z):(2025-08-01T03:00:00Z)][(0.25)][(21.6):(21.7)][(-158.1):(-158.0)]"
            response = requests.get(test_url, timeout=10)
            return response.status_code == 200
        except:
            return False

class NDFCBuoySource(DataSource):
    """NDBC buoy data for wave and wind conditions"""
    
    def __init__(self, station_id: str = "51201"):  # Waimea Bay
        self.station_id = station_id
        self.base_url = "https://www.ndbc.noaa.gov/data/realtime2/"
        
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch NDBC buoy meteorological data"""
        try:
            url = f"{self.base_url}{self.station_id}.txt"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                
                # Find header line
                data_lines = []
                for i, line in enumerate(lines):
                    if line.startswith('#YY'):
                        # Skip header and units lines
                        data_lines = lines[i+2:]
                        break
                
                if not data_lines:
                    return pd.DataFrame()
                
                # Parse NDBC format
                records = []
                for line in data_lines[:100]:  # Limit to recent data
                    parts = line.split()
                    if len(parts) >= 12:
                        try:
                            dt = datetime(
                                int(parts[0]), int(parts[1]), int(parts[2]),
                                int(parts[3]), int(parts[4])
                            )
                            
                            record = {
                                'datetime': dt,
                                'wind_dir': float(parts[5]) if parts[5] != 'MM' else np.nan,
                                'wind_speed': float(parts[6]) if parts[6] != 'MM' else np.nan,
                                'wave_height': float(parts[8]) if parts[8] != 'MM' else np.nan,
                                'wave_period': float(parts[9]) if parts[9] != 'MM' else np.nan,
                                'wave_dir': float(parts[11]) if parts[11] != 'MM' else np.nan,
                                'source': f'NDBC_{self.station_id}'
                            }
                            records.append(record)
                        except (ValueError, IndexError):
                            continue
                
                df = pd.DataFrame(records)
                # Filter to date range
                mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
                return df[mask]
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching NDBC data: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if NDBC station is responding"""
        try:
            url = f"{self.base_url}{self.station_id}.txt"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

class OpenMeteoWindSource(DataSource):
    """Open-Meteo weather API for wind forecasts (free, no API key)"""
    
    def __init__(self, latitude: float = 21.67, longitude: float = -158.12):
        # Default to Waimea Bay / North Shore Oahu coordinates
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch wind forecast data from Open-Meteo Weather API"""
        try:
            # Open-Meteo provides 7-day forecasts from current time
            params = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'hourly': 'windspeed_10m,winddirection_10m',
                'timezone': 'UTC'
            }
            
            logger.info(f"Fetching Open-Meteo wind data for {self.latitude}, {self.longitude}")
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'hourly' in data:
                    hourly = data['hourly']
                    times = hourly.get('time', [])
                    wind_speeds = hourly.get('windspeed_10m', [])
                    wind_dirs = hourly.get('winddirection_10m', [])
                    
                    records = []
                    for i, time_str in enumerate(times):
                        try:
                            dt = pd.to_datetime(time_str)
                            
                            # Filter to requested date range
                            if start_date <= dt <= end_date:
                                wind_speed = wind_speeds[i] if i < len(wind_speeds) and wind_speeds[i] is not None else np.nan
                                wind_dir = wind_dirs[i] if i < len(wind_dirs) and wind_dirs[i] is not None else np.nan
                                
                                # Skip invalid data
                                if pd.notna(wind_speed) and pd.notna(wind_dir):
                                    records.append({
                                        'datetime': dt,
                                        'wind_speed': wind_speed,
                                        'wind_dir': wind_dir,
                                        'source': 'OpenMeteo_Marine'
                                    })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Error parsing Open-Meteo time: {time_str}, error: {e}")
                            continue
                    
                    df = pd.DataFrame(records)
                    if not df.empty:
                        logger.info(f"Successfully fetched {len(df)} wind records from Open-Meteo")
                        return df
                    else:
                        logger.error("No valid wind data in requested date range from Open-Meteo")
                else:
                    logger.error("No hourly data in Open-Meteo response")
            else:
                logger.error(f"Open-Meteo API failed: {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text[:200]}")
            
        except Exception as e:
            logger.error(f"Critical error fetching Open-Meteo data: {e}")
        
        # Return empty DataFrame on failure - no fallback to hardcoded values
        logger.error("Open-Meteo wind data fetch failed - returning empty DataFrame")
        return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if Open-Meteo API is responding"""
        try:
            test_params = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'hourly': 'windspeed_10m',
                'forecast_days': 1
            }
            response = requests.get(self.base_url, params=test_params, timeout=10)
            return response.status_code == 200 and 'hourly' in response.json()
        except:
            return False


class NOAACWFWindSource(DataSource):
    """
    NOAA Coastal Waters Forecast - Marine wind forecast for Hawaiian waters
    Provides 5-day detailed wind forecasts for Oahu Leeward Waters
    """
    
    def __init__(self):
        self.base_url = "https://forecast.weather.gov/product.php"
        self.timezone = pytz.timezone('Pacific/Honolulu')
        
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and parse NOAA Coastal Waters Forecast"""
        try:
            # Fetch the forecast text
            params = {
                'issuedby': 'HFO',
                'product': 'CWF',
                'site': 'hfo'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML to extract the forecast text
            forecast_text = self._extract_forecast_text(response.text)
            
            if not forecast_text:
                logger.warning("No forecast text found in NOAA CWF response")
                return pd.DataFrame()
            
            # Parse Oahu Leeward Waters section
            leeward_forecast = self._extract_leeward_forecast(forecast_text)
            
            if not leeward_forecast:
                logger.warning("No Oahu Leeward Waters forecast found")
                return pd.DataFrame()
            
            # Convert forecast to dataframe
            forecast_data = self._parse_forecast_to_dataframe(leeward_forecast, start_date)
            
            # Filter to requested date range
            if not forecast_data.empty:
                forecast_data = forecast_data[
                    (forecast_data['datetime'] >= start_date) & 
                    (forecast_data['datetime'] <= end_date)
                ]
            
            logger.info(f"Fetched {len(forecast_data)} wind forecasts from NOAA CWF")
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error fetching NOAA CWF data: {e}")
            return pd.DataFrame()
    
    def _extract_forecast_text(self, html_content: str) -> str:
        """Extract the forecast text from HTML"""
        try:
            # Look for the pre-formatted text section
            pre_match = re.search(r'<pre[^>]*>(.*?)</pre>', html_content, re.DOTALL)
            if pre_match:
                return pre_match.group(1)
            return ""
        except Exception as e:
            logger.error(f"Error extracting forecast text: {e}")
            return ""
    
    def _extract_leeward_forecast(self, forecast_text: str) -> str:
        """Extract Oahu Leeward Waters section"""
        try:
            # Find the Oahu Leeward Waters section
            start_marker = "Oahu Leeward Waters-"
            end_markers = ["$$", "Kauai Channel-", "Oahu Windward Waters-", "Kaiwi Channel-"]
            
            start_idx = forecast_text.find(start_marker)
            if start_idx == -1:
                return ""
            
            # Find the end of this section
            end_idx = len(forecast_text)
            for marker in end_markers:
                idx = forecast_text.find(marker, start_idx)
                if idx > start_idx:
                    end_idx = min(end_idx, idx)
            
            return forecast_text[start_idx:end_idx]
            
        except Exception as e:
            logger.error(f"Error extracting leeward forecast: {e}")
            return ""
    
    def _parse_forecast_to_dataframe(self, forecast_text: str, base_date: datetime) -> pd.DataFrame:
        """Parse forecast text into structured dataframe"""
        try:
            
            # Extract issue time
            time_match = re.search(r'(\d{1,2})\s*([AP]M)\s+HST', forecast_text)
            
            # Define period patterns
            period_patterns = [
                (r'\.TONIGHT\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'tonight'),
                (r'\.TUESDAY\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'tuesday'),
                (r'\.TUESDAY NIGHT\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'tuesday_night'),
                (r'\.WEDNESDAY\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'wednesday'),
                (r'\.WEDNESDAY NIGHT\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'wednesday_night'),
                (r'\.THURSDAY\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'thursday'),
                (r'\.THURSDAY NIGHT\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'thursday_night'),
                (r'\.FRIDAY\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'friday'),
                (r'\.SATURDAY\.\.\.(.+?)(?=\.\w+\.\.\.|\Z)', 'saturday'),
            ]
            
            forecast_data = []
            
            for pattern, period_name in period_patterns:
                match = re.search(pattern, forecast_text, re.DOTALL | re.IGNORECASE)
                if match:
                    period_text = match.group(1).strip()
                    
                    # Check if there's a "becoming" pattern for afternoon winds
                    has_becoming = 'becoming' in period_text.lower() and 'afternoon' in period_text.lower()
                    
                    # Parse wind information
                    wind_info = self._parse_wind_info(period_text)
                    
                    if wind_info:
                        # Calculate datetime for this period
                        period_dt = self._get_period_datetime(period_name, base_date)
                        
                        # Add multiple time points for each period
                        if 'night' in period_name.lower():
                            hours = [18, 21, 0, 3]  # Evening through early morning
                        else:
                            hours = [6, 9, 12, 15]  # Morning through afternoon
                        
                        for hour in hours:
                            dt = period_dt.replace(hour=hour % 24)
                            if hour >= 24:
                                dt += timedelta(days=1)
                            
                            # For periods with "becoming...in the afternoon", 
                            # use different winds for morning vs afternoon
                            if has_becoming and hour < 12:
                                # Morning: use the first part (variable winds)
                                # Re-parse to get just the morning part
                                morning_text = period_text.split('becoming')[0]
                                morning_wind = self._parse_wind_info_simple(morning_text)
                                if morning_wind:
                                    forecast_data.append({
                                        'datetime': dt,
                                        'wind_speed': morning_wind['speed'],
                                        'wind_dir': morning_wind['direction'],
                                        'wind_speed_min': morning_wind['speed_min'],
                                        'wind_speed_max': morning_wind['speed_max'],
                                        'source': 'NOAA_CWF',
                                        'quality': 'official_forecast',
                                        'period': period_name
                                    })
                            else:
                                # Afternoon or no "becoming": use the main wind info
                                forecast_data.append({
                                    'datetime': dt,
                                    'wind_speed': wind_info['speed'],
                                    'wind_dir': wind_info['direction'],
                                    'wind_speed_min': wind_info['speed_min'],
                                    'wind_speed_max': wind_info['speed_max'],
                                    'source': 'NOAA_CWF',
                                    'quality': 'official_forecast',
                                    'period': period_name
                                })
            
            return pd.DataFrame(forecast_data)
            
        except Exception as e:
            logger.error(f"Error parsing forecast to dataframe: {e}")
            return pd.DataFrame()
    
    def _parse_wind_info(self, text: str) -> Dict:
        """Parse wind direction and speed from forecast text"""
        try:
            # Check for "becoming" pattern to get afternoon winds
            becoming_match = re.search(
                r'becoming\s+(north northwest|northwest|north|northeast|east northeast|east|southeast|south|southwest|west|variable)\s+(\d+)(?:\s+to\s+(\d+))?\s+knots?',
                text, re.IGNORECASE
            )
            
            # Common wind patterns - FIXED to handle actual NOAA format
            patterns = [
                # Pattern for "Winds variable less than X knots" (must come first)
                r'Winds?\s+variable\s+less\s+than\s+(\d+)\s+knots?',
                # Pattern for directional winds with range
                r'(East northeast|Northeast|East southeast|East|Southeast|North northwest|Northwest|North|South|West|Variable)\s+winds?\s+(?:around\s+)?(\d+)\s+to\s+(\d+)\s+knots?',
                # Pattern for directional winds with single speed
                r'(East northeast|Northeast|East southeast|East|Southeast|North northwest|Northwest|North|South|West|Variable)\s+winds?\s+(?:around\s+)?(\d+)\s+knots?',
            ]
            
            # Parse the main wind pattern
            main_wind = None
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    # Handle "variable less than" pattern
                    if 'variable less than' in text.lower():
                        speed_max = int(groups[0])
                        speed_min = 3  # Assume minimum of 3 knots for "variable less than"
                        speed_avg = (speed_min + speed_max) / 2
                        direction = 0  # Variable winds have no specific direction
                        
                        main_wind = {
                            'direction': direction,
                            'speed': speed_avg,
                            'speed_min': speed_min,
                            'speed_max': speed_max
                        }
                        break
                    
                    # Parse direction for other patterns
                    direction_text = groups[0].lower()
                    direction = self._text_to_degrees(direction_text)
                    
                    # Parse speed
                    if len(groups) == 3:
                        speed_min = int(groups[1])
                        speed_max = int(groups[2])
                    elif len(groups) == 2:
                        speed_min = speed_max = int(groups[1])
                    else:
                        continue
                    
                    speed_avg = (speed_min + speed_max) / 2
                    
                    main_wind = {
                        'direction': direction,
                        'speed': speed_avg,
                        'speed_min': speed_min,
                        'speed_max': speed_max
                    }
                    break
            
            # If there's a "becoming" pattern, use that for afternoon
            # Otherwise return the main wind pattern
            if becoming_match and 'afternoon' in text.lower():
                groups = becoming_match.groups()
                direction_text = groups[0].lower()
                direction = self._text_to_degrees(direction_text)
                
                if groups[2]:  # Range: X to Y knots
                    speed_min = int(groups[1])
                    speed_max = int(groups[2])
                else:  # Single speed: X knots
                    speed_min = speed_max = int(groups[1])
                
                speed_avg = (speed_min + speed_max) / 2
                
                # For now, return the afternoon wind as primary
                # In the future, we might want to return both
                return {
                    'direction': direction,
                    'speed': speed_avg,
                    'speed_min': speed_min,
                    'speed_max': speed_max,
                    'has_morning_variant': main_wind is not None
                }
            
            if main_wind:
                return main_wind
            
            # If no pattern matched, log what we couldn't parse
            logger.debug(f"Could not parse wind info from: {text[:100]}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing wind info: {e}")
            return None

    def _parse_wind_info_simple(self, text: str) -> Dict:
        """Simple wind parser for morning/first part of forecast"""
        try:
            # Pattern for "Winds variable less than X knots"
            match = re.search(r'Winds?\s+variable\s+less\s+than\s+(\d+)\s+knots?', text, re.IGNORECASE)
            if match:
                speed_max = int(match.group(1))
                return {
                    'direction': 0,  # Variable
                    'speed': (3 + speed_max) / 2,  # Average
                    'speed_min': 3,
                    'speed_max': speed_max
                }
            
            # Pattern for regular winds
            match = re.search(
                r'([\w\s]+?)\s+winds?\s+(\d+)(?:\s+to\s+(\d+))?\s+knots?',
                text, re.IGNORECASE
            )
            if match:
                direction_text = match.group(1).strip().lower()
                direction = self._text_to_degrees(direction_text)
                
                if match.group(3):  # Range
                    speed_min = int(match.group(2))
                    speed_max = int(match.group(3))
                else:  # Single speed
                    speed_min = speed_max = int(match.group(2))
                
                return {
                    'direction': direction,
                    'speed': (speed_min + speed_max) / 2,
                    'speed_min': speed_min,
                    'speed_max': speed_max
                }
            
            return None
        except Exception as e:
            logger.error(f"Error in simple wind parsing: {e}")
            return None
    
    def _text_to_degrees(self, direction_text: str) -> int:
        """Convert text wind direction to degrees"""
        directions = {
            'north': 0,
            'north northeast': 22,
            'northeast': 45,
            'east northeast': 67,
            'east': 90,
            'east southeast': 112,
            'southeast': 135,
            'south southeast': 157,
            'south': 180,
            'south southwest': 202,
            'southwest': 225,
            'west southwest': 247,
            'west': 270,
            'west northwest': 292,
            'northwest': 315,
            'north northwest': 337,
            'variable': 0  # Default for variable winds
        }
        
        return directions.get(direction_text.lower(), 0)
    
    def _get_period_datetime(self, period_name: str, base_date: datetime) -> datetime:
        """Convert period name to datetime
        
        This function needs to be dynamic based on what day the forecast was issued.
        We need to figure out what day 'today' is from the base_date.
        """
        # Get the day of week for base_date (0=Monday, 6=Sunday)
        base_day = base_date.weekday()
        
        # Map day names to day of week numbers
        day_numbers = {
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6
        }
        
        # Handle special cases
        if period_name.lower() == 'tonight':
            return base_date  # Tonight is still today's date
        
        # Extract the day name from the period (e.g., "wednesday" from "wednesday_night")
        day_name = period_name.lower().replace('_night', '').replace('night', '').strip()
        
        if day_name in day_numbers:
            target_day = day_numbers[day_name]
            days_ahead = (target_day - base_day) % 7
            
            # If days_ahead is 0, it means it's the same day of week
            # Check if we mean today or next week
            if days_ahead == 0 and period_name.lower() != 'tonight':
                # If the period name matches today's day, use today
                # Otherwise it's next week
                if base_date.strftime('%A').lower() == day_name:
                    days_ahead = 0
                else:
                    days_ahead = 7
            
            return base_date + timedelta(days=days_ahead)
        
        # Fallback to base_date if we can't parse
        logger.warning(f"Could not parse period name: {period_name}")
        return base_date
    
    def is_available(self) -> bool:
        """Check if NOAA CWF is accessible"""
        try:
            # HEAD request returns 500 but GET works, so do a lightweight GET
            params = {
                'issuedby': 'HFO',
                'product': 'CWF', 
                'site': 'hfo'
            }
            response = requests.get(self.base_url, params=params, timeout=5)
            # Check if we got a valid response with forecast content
            return response.status_code == 200 and 'Oahu Leeward Waters' in response.text
        except:
            return False

class TidalCurrentAnalyzer:
    """Analyzes tidal patterns to find eastward current flow periods"""
    
    def __init__(self, location: str = "north_shore_oahu"):
        self.location = location
        
    def generate_tidal_simulation(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Enhanced tidal current simulation using multiple harmonic components
        Based on real Kahuku Point tidal patterns with added variability
        """
        logger.info("Generating enhanced tidal current simulation using multiple harmonic components")
        
        # Create hourly time series
        time_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        tidal_data = []
        for dt in time_range:
            # Convert to hours since epoch for tidal calculations
            hours_since_epoch = (dt - datetime(1970, 1, 1)).total_seconds() / 3600
            
            # Major tidal constituents for realistic variability
            # M2 semi-diurnal (12.42h) - dominant lunar component
            M2_amplitude = 1.2
            M2_period = 12.42066
            M2_phase = np.random.uniform(-0.2, 0.2)  # Small random phase variation
            
            # S2 solar semi-diurnal (12h) - solar component
            S2_amplitude = 0.4
            S2_period = 12.0
            S2_phase = np.pi / 6
            
            # K1 lunar diurnal (23.93h) - diurnal inequality
            K1_amplitude = 0.6
            K1_period = 23.9344
            K1_phase = np.pi / 4
            
            # O1 lunar diurnal (25.82h) - adds diurnal variability
            O1_amplitude = 0.3
            O1_period = 25.8193
            O1_phase = np.pi / 3
            
            # N2 lunar elliptic (12.66h) - lunar monthly variation
            N2_amplitude = 0.2
            N2_period = 12.6581
            N2_phase = np.pi / 8
            
            # Calculate harmonic components
            M2_component = M2_amplitude * np.cos(2 * np.pi * hours_since_epoch / M2_period + M2_phase)
            S2_component = S2_amplitude * np.cos(2 * np.pi * hours_since_epoch / S2_period + S2_phase)
            K1_component = K1_amplitude * np.cos(2 * np.pi * hours_since_epoch / K1_period + K1_phase)
            O1_component = O1_amplitude * np.cos(2 * np.pi * hours_since_epoch / O1_period + O1_phase)
            N2_component = N2_amplitude * np.cos(2 * np.pi * hours_since_epoch / N2_period + N2_phase)
            
            # Combined tidal forcing with realistic variability
            tidal_forcing = M2_component + S2_component + K1_component + O1_component + N2_component
            
            # Add small amount of noise for realism
            noise = np.random.normal(0, 0.05)
            tidal_forcing += noise
            
            # Map tidal forcing to current speed and direction with gradual transitions
            if tidal_forcing < -0.4:  # Strong ebb
                direction = 73 + np.random.uniform(-5, 5)  # Add directional variability
                speed = abs(tidal_forcing) * 0.85
            elif tidal_forcing < -0.1:  # Moderate ebb
                direction = 73 + np.random.uniform(-8, 8)
                speed = abs(tidal_forcing) * 0.75
            elif tidal_forcing > 0.4:  # Strong flood
                direction = 265 + np.random.uniform(-5, 5)
                speed = abs(tidal_forcing) * 0.85
            elif tidal_forcing > 0.1:  # Moderate flood
                direction = 265 + np.random.uniform(-8, 8)
                speed = abs(tidal_forcing) * 0.75
            else:  # Slack water with weak variable currents
                direction = np.random.uniform(0, 360)  # Random weak currents
                speed = abs(tidal_forcing) * 0.3 + np.random.uniform(0, 0.2)
            
            # Ensure direction stays in valid range
            direction = direction % 360
            
            # Determine tide type with more nuanced classification
            if tidal_forcing < -0.3:
                tide_type = 'strong_ebb'
            elif tidal_forcing < -0.1:
                tide_type = 'weak_ebb'
            elif tidal_forcing > 0.3:
                tide_type = 'strong_flood'
            elif tidal_forcing > 0.1:
                tide_type = 'weak_flood'
            else:
                tide_type = 'slack'
            
            tidal_data.append({
                'datetime': dt,
                'current_speed': max(0, speed),  # Ensure non-negative
                'current_dir': direction,
                'tidal_forcing': tidal_forcing,
                'tide_type': tide_type,
                'source': 'Enhanced_Tidal_Simulation'
            })
        
        df = pd.DataFrame(tidal_data)
        logger.info(f"Generated {len(df)} hours of enhanced tidal current simulation")
        
        # Log statistics about flow patterns
        eastward_periods = df[(df['current_dir'] >= 30) & (df['current_dir'] <= 150)]
        northward_periods = df[((df['current_dir'] >= 315) | (df['current_dir'] <= 45))]
        strong_ebb = df[df['tide_type'] == 'strong_ebb']
        
        logger.info(f"Simulation includes:")
        logger.info(f"  {len(eastward_periods)} hours of eastward flow (30-150°)")
        logger.info(f"  {len(northward_periods)} hours of northward flow (315-45°)")
        logger.info(f"  {len(strong_ebb)} hours of strong ebb tide")
        logger.info(f"  Average current speed: {df['current_speed'].mean():.2f} knots")
        
        return df
    
    def find_eastward_flow_periods(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Find periods of eastward flow that can oppose ENE trade winds
        Optimal range: NNE to ENE (010-080°) for coast-parallel opposition
        """
        try:
            if current_data.empty:
                return pd.DataFrame()
            
            # TRUE eastward flow criteria for opposing ENE trades:
            # - Direction: 010-080° (NNE through ENE - coast-parallel flow)
            # - This opposes typical ENE trade winds (050-070°)
            # - Speed: >= 0.1 knots (meaningful current for wave enhancement)
            
            # True eastward flow that opposes ENE trade winds (050-070°)
            # Optimal range: NNE to ENE (010-080°) - parallel to coast, not toward shore
            eastward_mask = (
                (current_data['current_dir'] >= 10) &    # NNE (010°)
                (current_data['current_dir'] <= 80) &    # ENE (080°)
                (current_data['current_speed'] >= 0.1)   # Meaningful current
            )
            
            eastward_periods = current_data[eastward_mask].copy()
            
            if not eastward_periods.empty:
                # Categorize the flow directions within eastward range
                eastward_periods['flow_category'] = pd.cut(
                    eastward_periods['current_dir'],
                    bins=[10, 30, 50, 65, 80],
                    labels=['NNE', 'NE', 'NE-ENE', 'ENE'],
                    include_lowest=True
                )
                
                # Add tidal enhancement
                eastward_periods['tidal_enhancement'] = eastward_periods['datetime'].apply(
                    lambda dt: self._calculate_tidal_phase_enhancement(dt)
                )
                eastward_periods['tidal_phase'] = eastward_periods['datetime'].apply(
                    lambda dt: self._get_tidal_phase_description(dt)
                )
                
                # Log what we found
                flow_counts = eastward_periods['flow_category'].value_counts()
                logger.info(f"Found {len(eastward_periods)} eastward flow periods (010-080°)")
                logger.info(f"Flow breakdown: {flow_counts.to_dict()}")
                logger.info(f"Average current speed: {eastward_periods['current_speed'].mean():.2f} kt")
                
                # Log NE currents specifically since user sees them
                ne_currents = eastward_periods[eastward_periods['flow_category'] == 'NE']
                if not ne_currents.empty:
                    logger.info(f"NE currents (030-060°): {len(ne_currents)} periods found")
                
                eastward_periods['is_true_eastward'] = True
            else:
                logger.warning("NO EASTWARD CURRENTS FOUND (010-080°)")
                # Debug: show what directions ARE in the data
                if not current_data.empty:
                    dir_summary = current_data.groupby(pd.cut(current_data['current_dir'], 
                        bins=[0, 45, 90, 135, 180, 225, 270, 315, 360]))['current_dir'].count()
                    logger.info(f"Current directions in data: {dir_summary.to_dict()}")
                
            return eastward_periods
            
        except Exception as e:
            logger.error(f"Error analyzing eastward flow: {e}")
            return pd.DataFrame()

    def debug_current_directions(self, current_data: pd.DataFrame):
        """Debug function to see actual current directions"""
        if current_data.empty:
            logger.info("DEBUG: No current data to analyze")
            return
            
        logger.info("DEBUG: Current direction distribution")
        logger.info(f"Total records: {len(current_data)}")
        
        # Group by 45-degree sectors
        bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        current_data['sector'] = pd.cut(current_data['current_dir'], bins=bins, labels=labels)
        
        sector_counts = current_data['sector'].value_counts().sort_index()
        for sector, count in sector_counts.items():
            avg_speed = current_data[current_data['sector'] == sector]['current_speed'].mean()
            sector_idx = labels.index(sector)
            logger.info(f"  {sector} ({bins[sector_idx]}-{bins[sector_idx+1]}°): "
                       f"{count} records, avg {avg_speed:.2f} kt")
        
        # Show some actual NE examples if they exist
        ne_currents = current_data[(current_data['current_dir'] >= 30) & 
                                   (current_data['current_dir'] <= 60)]
        if not ne_currents.empty:
            logger.info(f"\nDEBUG: Found {len(ne_currents)} NE currents (030-060°)")
            sample = ne_currents.head(3)
            for _, row in sample.iterrows():
                logger.info(f"  {row['datetime']}: {row['current_dir']:.0f}° @ {row['current_speed']:.2f} kt")
    
    def analyze_current_wind_interaction(self, current_dir: float, wind_dir: float, 
                                   current_speed: float, datetime_obj: datetime = None) -> Dict:
        """
        Analyze current-wind interaction
        ENE winds typically 050-070°
        Best opposition: currents 010-080° (NNE through ENE, coast-parallel)
        """
        try:
            # Calculate angle between current and wind
            angle_diff = abs(current_dir - wind_dir)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Check if this is true eastward flow (ENE to ESE)
            is_eastward = 60 <= current_dir <= 120  # True eastward flow range
            
            # Special handling for ENE currents vs ENE winds (optimal opposition)
            is_ne_current = 50 <= current_dir <= 70  # ENE range
            typical_ene_wind = 50 <= wind_dir <= 70
            
            # Calculate tidal enhancement
            tidal_enhancement = 0.0
            tidal_phase = "unknown"
            if datetime_obj:
                tidal_enhancement = self._calculate_tidal_phase_enhancement(datetime_obj)
                tidal_phase = self._get_tidal_phase_description(datetime_obj)
            
            # Determine interaction type
            if is_eastward:
                if angle_diff <= 20:
                    interaction = "current_into_wind_strong"
                    base_enhancement = "maximum"
                elif angle_diff <= 45:
                    interaction = "current_into_wind_moderate"
                    base_enhancement = "excellent"
                elif angle_diff <= 90:
                    interaction = "current_into_wind_weak"
                    base_enhancement = "good"
                else:
                    interaction = "current_cross_wind"
                    base_enhancement = "moderate"
                    
                # Bonus for NE current vs ENE wind (classic setup)
                if is_ne_current and typical_ene_wind:
                    logger.debug(f"Classic setup: NE current ({current_dir}°) vs ENE wind ({wind_dir}°)")
                    if base_enhancement == "good":
                        base_enhancement = "excellent"
                    elif base_enhancement == "moderate":
                        base_enhancement = "good"
            else:
                interaction = "non_optimal"
                base_enhancement = "minimal"
            
            enhancement_multiplier = 1.0 + (current_speed * 1.5 if is_eastward else 0)
            
            # Determine if conditions are surfable
            # Include various scenarios beyond just perfect opposition
            surfable = False
            if is_eastward:
                if angle_diff <= 90:  # Current opposing or perpendicular to wind
                    surfable = True
                elif base_enhancement in ["maximum", "excellent", "good"]:
                    surfable = True
                    
            return {
                'angle_difference': angle_diff,
                'interaction_type': interaction,
                'wave_enhancement': base_enhancement,
                'enhancement_multiplier': enhancement_multiplier,
                'tidal_enhancement': tidal_enhancement,
                'tidal_phase': tidal_phase,
                'is_eastward_current': is_eastward,
                'is_ne_current': is_ne_current,
                'optimal_conditions': is_eastward and angle_diff <= 90,
                'surfable_conditions': surfable,  # Add this key that the code expects
                'debug_info': f"Current {current_dir}° @ {current_speed:.1f}kt vs Wind {wind_dir}°"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing interaction: {e}")
            return {'optimal_conditions': False, 'is_eastward_current': False}
    
    def _categorize_enhancement(self, enhancement: str) -> str:
        """Categorize wave enhancement for clearer reporting"""
        if enhancement in ['maximum']:
            return 'prime_enhancement'
        elif enhancement in ['excellent', 'good']:
            return 'strong_enhancement'  
        elif enhancement in ['moderate']:
            return 'moderate_enhancement'
        else:
            return 'light_enhancement'
    
    def _calculate_tidal_phase_enhancement(self, dt: datetime) -> float:
        """
        Calculate tidal phase enhancement with focus on rising tide
        Peak enhancement: 2 hours before high tide through 30 min after
        """
        # Get high tide times for this date
        high_tide_times = self._get_daily_high_tide_estimates(dt.date())
        
        # Find time to nearest high tide AND whether we're before or after
        current_hour = dt.hour + dt.minute / 60.0
        
        best_enhancement = 0.0
        for ht in high_tide_times:
            # Calculate hours until this high tide
            hours_until_high = ht - current_hour
            
            # Handle day boundaries
            if hours_until_high < -12:
                hours_until_high += 24
            elif hours_until_high > 12:
                hours_until_high -= 24
            
            # Enhancement based on position in tidal cycle
            if -2.5 <= hours_until_high <= -0.5:  # 2.5 to 0.5 hours BEFORE high
                # Peak enhancement zone - rising tide with maximum current
                enhancement = 1.0  # Maximum enhancement
                logger.debug(f"PRIME TIDAL PHASE: {abs(hours_until_high):.1f}h before high tide")
            elif -0.5 < hours_until_high <= 0.5:  # 30 min before to 30 min after high
                # High tide zone - good depth, slowing current
                enhancement = 0.7
                logger.debug(f"HIGH TIDE ZONE: {abs(hours_until_high):.1f}h from high tide")
            elif -4.0 <= hours_until_high < -2.5:  # 4 to 2.5 hours before high
                # Early rising tide - building enhancement
                enhancement = 0.5
                logger.debug(f"RISING TIDE: {abs(hours_until_high):.1f}h before high tide")
            elif 0.5 < hours_until_high <= 1.0:  # 30-60 min after high
                # Post-high residual enhancement
                enhancement = 0.3
                logger.debug(f"POST-HIGH: {hours_until_high:.1f}h after high tide")
            else:
                # Falling tide or slack low - minimal enhancement
                enhancement = 0.0
            
            best_enhancement = max(best_enhancement, enhancement)
        
        return best_enhancement   # No enhancement during other periods

    def _get_tidal_phase_description(self, dt: datetime) -> str:
        """Get human-readable tidal phase description"""
        enhancement = self._calculate_tidal_phase_enhancement(dt)
        
        if enhancement >= 0.9:
            return "RISING-2hr"  # 2hr before high (BEST)
        elif enhancement >= 0.6:
            return "HIGH TIDE"   # Near peak
        elif enhancement >= 0.4:
            return "RISING"      # Building
        elif enhancement >= 0.2:
            return "POST-HIGH"   # Residual flow
        else:
            return "FALLING/LOW" # Minimal enhancement
    
    def _get_daily_high_tide_estimates(self, date_obj) -> List[float]:
        """
        Calculate approximate high tide times for a given date
        Based on M2 lunar semi-diurnal tide (dominant component)
        Returns list of hours (0-24) when high tides occur
        """
        # M2 constituent period: 12.4206 hours
        # For North Shore Oahu, approximate high tide reference
        # Using January 1, 2025 00:00 as a reference point with known tidal phase
        
        from datetime import date
        reference_date = date(2025, 1, 1)
        reference_high_tide_hour = 6.2  # Approximate high tide time on reference date
        
        # Calculate days since reference
        days_since_ref = (date_obj - reference_date).days
        
        # M2 advances by about 50 minutes per day (24.8 hours / 2 tides per day)
        daily_advance_hours = 0.833  # 50 minutes = 0.833 hours
        
        # Calculate first high tide of the day
        first_high_tide = (reference_high_tide_hour + (days_since_ref * daily_advance_hours)) % 24
        
        # Second high tide is approximately 12.42 hours later
        second_high_tide = (first_high_tide + 12.42) % 24
        
        return [first_high_tide, second_high_tide]
    
    def _combine_enhancements(self, base_enhancement: str, tidal_factor: float) -> str:
        """
        Combine base wave enhancement with tidal phase enhancement
        """
        enhancement_levels = {
            "minimal": 1,
            "moderate": 2, 
            "good": 3,
            "maximum": 4
        }
        
        level_names = ["minimal", "moderate", "good", "maximum"]
        
        # Convert to numeric, add tidal enhancement, convert back
        current_level = enhancement_levels.get(base_enhancement, 2)
        
        # Tidal enhancement can bump up one level if significant
        if tidal_factor >= 0.12:
            enhanced_level = min(current_level + 1, 4)
        elif tidal_factor >= 0.06:
            # Small boost but not enough to change category
            enhanced_level = current_level
        else:
            enhanced_level = current_level
        
        return level_names[enhanced_level - 1]

class NOAAWeatherBuoySource(DataSource):
    """NOAA Weather Buoy data for additional wind coverage"""
    
    def __init__(self, station_id: str = "51201"):  # Waimea Bay
        self.station_id = station_id
        self.base_url = "https://www.ndbc.noaa.gov/data/realtime2/"
        
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch additional NOAA weather buoy data with more historical coverage"""
        try:
            # Try standard current data first
            std_url = f"{self.base_url}{self.station_id}.txt"
            response = requests.get(std_url, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"NOAA weather buoy {self.station_id} not available")
                return pd.DataFrame()
            
            lines = response.text.strip().split('\n')
            
            # Parse NDBC meteorological data format
            data_lines = []
            for i, line in enumerate(lines):
                if line.startswith('#YY') or line.startswith('#yr'):
                    # Skip header and units lines
                    data_lines = lines[i+2:]
                    break
            
            if not data_lines:
                return pd.DataFrame()
            
            # Parse recent records (up to 200 for better coverage)
            records = []
            for line in data_lines[:200]:  # Extended record count
                parts = line.split()
                if len(parts) >= 12:
                    try:
                        dt = datetime(
                            int(parts[0]), int(parts[1]), int(parts[2]),
                            int(parts[3]), int(parts[4])
                        )
                        
                        # Filter to requested date range
                        if start_date <= dt <= end_date:
                            wind_dir = float(parts[5]) if parts[5] != 'MM' else np.nan
                            wind_speed = float(parts[6]) if parts[6] != 'MM' else np.nan
                            
                            # Only include valid wind data
                            if not (pd.isna(wind_dir) or pd.isna(wind_speed)):
                                records.append({
                                    'datetime': dt,
                                    'wind_dir': wind_dir,
                                    'wind_speed': wind_speed,
                                    'air_temp': float(parts[13]) if len(parts) > 13 and parts[13] != 'MM' else np.nan,
                                    'source': f'NOAA_Buoy_{self.station_id}'
                                })
                    except (ValueError, IndexError):
                        continue
            
            df = pd.DataFrame(records)
            if not df.empty:
                logger.info(f"Collected {len(df)} wind records from NOAA weather buoy {self.station_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching NOAA weather buoy data: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if NOAA weather buoy is responding"""
        try:
            url = f"{self.base_url}{self.station_id}.txt"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

class NOAAMarineSource(DataSource):
    """NOAA Marine Weather Forecast for wind and wind wave data"""
    
    def __init__(self):
        self.base_url = "https://marine.weather.gov/MapClick.php"
        # Location: ~14NM NE of Kahuku (North Shore Oahu)
        self.lat = 21.8298
        self.lon = -157.759
        self.max_forecast_days = 6
        
    def _generate_url(self, ahead_hours: int = 0) -> str:
        """Generate NOAA marine forecast URL for specific forecast window"""
        params = {
            'w3': 'sfcwind',      # Surface wind
            'w3u': '1',           # Wind units (mph)
            'w14': 'wwh',         # Wind wave height
            'AheadHour': str(ahead_hours),
            'FcstType': 'digital',
            'textField1': str(self.lat),
            'textField2': str(self.lon),
            'site': 'all',
            'unit': '0',
            'dd': '',
            'bw': '',
            'marine': '1'
        }
        
        from urllib.parse import urlencode
        return f"{self.base_url}?{urlencode(params)}"
    
    def _parse_forecast_table(self, html_content: str, base_date: datetime) -> List[Dict]:
        """Parse the HTML forecast table to extract wind and wave data
        
        The NOAA Marine forecast table has a unique structure where all data
        is concatenated together within table cells, not separated.
        Example: "Date08/0508/06Hour (HST)091011121314..."
        """
        from bs4 import BeautifulSoup
        import re
        
        soup = BeautifulSoup(html_content, 'html.parser')
        data_rows = []
        
        # Find all tables
        tables = soup.find_all('table')
        
        # According to investigation, table 4 (index 3) typically contains the forecast data
        # But we'll also check other tables if needed
        forecast_table = None
        table_indices_to_check = [3, 4, 5, 2, 6, 1, 0]  # Priority order based on typical structure
        
        for idx in table_indices_to_check:
            if idx < len(tables):
                table = tables[idx]
                text = table.get_text()
                # Look for the data table that has Hour and Surface Wind
                if 'Hour (HST)' in text and 'Surface Wind' in text:
                    forecast_table = table
                    logger.debug(f"Found forecast table at index {idx}")
                    break
        
        if not forecast_table:
            logger.warning("Could not find forecast table in HTML")
            return data_rows
        
        # Get the table text and clean it
        table_text = forecast_table.get_text()
        # Remove all whitespace to work with concatenated format
        clean_text = ''.join(table_text.split())
        
        # Extract dates using regex
        dates = []
        date_pattern = r'Date((?:\d{2}/\d{2})+)'
        date_match = re.search(date_pattern, clean_text)
        if date_match:
            date_string = date_match.group(1)
            dates = re.findall(r'(\d{2}/\d{2})', date_string)
            logger.debug(f"Found dates: {dates}")
        
        # Extract hours - they appear after "Hour(HST)" as consecutive 2-digit numbers
        hours = []
        hour_pattern = r'Hour\(HST\)((?:\d{2})+)'
        hour_match = re.search(hour_pattern, clean_text)
        if hour_match:
            hour_string = hour_match.group(1)
            # Split into 2-digit chunks
            hours = [hour_string[i:i+2] for i in range(0, len(hour_string), 2)]
            logger.debug(f"Found {len(hours)} hours")
        
        # Extract wind speeds - they appear after "SurfaceWind(mph)"
        wind_speeds = []
        wind_pattern = r'SurfaceWind\(mph\)([\d]+)'
        wind_match = re.search(wind_pattern, clean_text)
        if wind_match:
            wind_string = wind_match.group(1)
            # Wind speeds can be 1 or 2 digits
            # We need to parse intelligently based on hour count
            wind_speeds = self._parse_numeric_sequence(wind_string, len(hours), is_wave_height=False)
            logger.debug(f"Found {len(wind_speeds)} wind speeds")
        
        # Extract wind directions - they appear after "WindDir"
        wind_dirs = []
        dir_pattern = r'WindDir([NESW]+)'
        dir_match = re.search(dir_pattern, clean_text)
        if dir_match:
            dir_string = dir_match.group(1)
            # Parse direction codes (1-3 letters each, e.g., E, NE, ENE, NNE)
            wind_dirs = self._parse_wind_directions(dir_string, len(hours))
            logger.debug(f"Found {len(wind_dirs)} wind directions")
        
        # Extract wave heights if available
        wave_heights = []
        # Try different wave patterns
        wave_patterns = [
            r'WindWaveHeight\(ft\)([\d]+)',
            r'WindWaveHeight([\d]+)',
            r'WaveHeight\(ft\)([\d]+)',
            r'WaveHeight([\d]+)',
            r'WindWave([\d]+)'
        ]
        
        for pattern in wave_patterns:
            wave_match = re.search(pattern, clean_text)
            if wave_match:
                wave_string = wave_match.group(1)
                wave_heights = self._parse_numeric_sequence(wave_string, len(hours), is_wave_height=True)
                logger.debug(f"Found {len(wave_heights)} wave heights with pattern {pattern}")
                break
        
        # Build data rows
        if not hours or not wind_speeds:
            logger.warning("Missing required data (hours or wind speeds)")
            return data_rows
        
        # Figure out how many hours per date
        hours_per_date = len(hours) // len(dates) if dates else 24
        
        current_date = base_date
        date_index = 0
        hours_in_current_date = 0
        last_hour = -1
        
        for i in range(min(len(hours), len(wind_speeds))):
            try:
                hour = int(hours[i])
                
                # Check for date change based on hour reset or hours per date
                if hour < last_hour or hours_in_current_date >= hours_per_date:
                    current_date += timedelta(days=1)
                    date_index += 1
                    hours_in_current_date = 0
                
                last_hour = hour
                hours_in_current_date += 1
                
                # Parse wind speed (convert mph to knots)
                wind_speed_mph = float(wind_speeds[i]) if i < len(wind_speeds) else 0
                wind_speed_kts = wind_speed_mph * 0.868976
                
                # Parse wind direction
                wind_dir_str = wind_dirs[i] if i < len(wind_dirs) else 'E'
                
                # Convert wind direction to degrees
                dir_map = {
                    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
                }
                wind_dir_deg = dir_map.get(wind_dir_str.upper(), 90)
                
                # Parse wave height
                wave_height = float(wave_heights[i]) if i < len(wave_heights) else None
                
                # Only add data point if we have valid wind data
                if wind_speed_kts >= 0:
                    data_point = {
                        'timestamp': current_date.replace(hour=hour, minute=0, second=0),
                        'wind_speed_kts': wind_speed_kts,
                        'wind_dir_deg': wind_dir_deg,
                        'source': 'NOAA_Marine'
                    }
                    
                    # Only add wave height if we have it
                    if wave_height is not None:
                        data_point['wind_wave_height_ft'] = wave_height
                    
                    data_rows.append(data_point)
                
            except Exception as e:
                logger.debug(f"Error parsing data at index {i}: {e}")
                continue
        
        logger.info(f"Parsed {len(data_rows)} data points from NOAA Marine forecast")
        return data_rows
    
    def _parse_numeric_sequence(self, number_string: str, expected_count: int, is_wave_height: bool = False) -> List[str]:
        """Parse a string of concatenated numbers into individual values
        
        This handles cases where numbers can be 1 or 2 digits.
        Strategy: Try different parsing approaches and pick the one that
        gives us closest to the expected count.
        
        Args:
            number_string: String of concatenated numbers
            expected_count: Expected number of values
            is_wave_height: If True, prefer single-digit parsing (wave heights are typically 0-9 ft)
        """
        results = []
        
        # For wave heights, prefer single digits since they're typically 0-9 ft
        if is_wave_height:
            # Strategy for wave heights: prefer single digits
            single_digit = list(number_string)
            
            # Also try two-digit parsing in case waves are 10+ ft
            two_digit = [number_string[i:i+2] for i in range(0, len(number_string), 2)]
            
            # Pick based on which gives us the expected count
            if len(single_digit) == expected_count:
                results = single_digit
            elif len(two_digit) == expected_count:
                results = two_digit
            else:
                # Default to single digits for wave heights
                results = single_digit
        else:
            # Strategy 1: Assume all are 2-digit numbers
            two_digit = [number_string[i:i+2] for i in range(0, len(number_string), 2)]
            
            # Strategy 2: Parse adaptively - look for patterns
            # If we see numbers > 31, they're likely concatenated single digits
            adaptive = []
            i = 0
            while i < len(number_string):
                if i + 1 < len(number_string):
                    two_digit_val = int(number_string[i:i+2])
                    # Wind speeds typically < 50 mph
                    if two_digit_val <= 50:
                        adaptive.append(number_string[i:i+2])
                        i += 2
                    else:
                        # Probably two single digits
                        adaptive.append(number_string[i])
                        i += 1
                else:
                    adaptive.append(number_string[i])
                    i += 1
            
            # Pick the strategy that gives us closest to expected count
            if abs(len(two_digit) - expected_count) <= abs(len(adaptive) - expected_count):
                results = two_digit
            else:
                results = adaptive
        
        # If we have too many values, truncate; if too few, pad with last value
        if len(results) > expected_count:
            results = results[:expected_count]
        elif len(results) < expected_count and results:
            last_val = results[-1]
            while len(results) < expected_count:
                results.append(last_val)
        
        return results
    
    def _parse_wind_directions(self, direction_string: str, expected_count: int) -> List[str]:
        """Parse a string of concatenated wind directions
        
        Wind directions can be 1-3 letters (N, NE, ENE, NNE, etc.)
        This is tricky because they're variable length.
        """
        results = []
        i = 0
        
        while i < len(direction_string) and len(results) < expected_count:
            # Try to match the longest valid direction first
            matched = False
            
            # Try 3-letter directions first (NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW)
            if i + 3 <= len(direction_string):
                three_letter = direction_string[i:i+3]
                if three_letter in ['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW']:
                    results.append(three_letter)
                    i += 3
                    matched = True
            
            # Try 2-letter directions (NE, SE, SW, NW)
            if not matched and i + 2 <= len(direction_string):
                two_letter = direction_string[i:i+2]
                if two_letter in ['NE', 'SE', 'SW', 'NW']:
                    results.append(two_letter)
                    i += 2
                    matched = True
            
            # Try 1-letter directions (N, E, S, W)
            if not matched and i < len(direction_string):
                one_letter = direction_string[i]
                if one_letter in ['N', 'E', 'S', 'W']:
                    results.append(one_letter)
                    i += 1
                    matched = True
            
            # If no match, skip this character (shouldn't happen with valid data)
            if not matched:
                i += 1
        
        # Pad with 'E' if we don't have enough directions
        while len(results) < expected_count:
            results.append('E')
        
        return results[:expected_count]
    
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch NOAA marine forecast data for the specified date range"""
        from datetime import timedelta
        import requests
        
        all_data = []
        
        # Calculate how many days we need
        total_days = (end_date - start_date).days + 1
        if total_days > self.max_forecast_days:
            logger.warning(f"Requested {total_days} days but NOAA marine forecast only provides {self.max_forecast_days} days")
            total_days = min(total_days, self.max_forecast_days)
        
        # Fetch data in ~2-day chunks (48 hours)
        current_ahead_hours = 0
        while current_ahead_hours < total_days * 24:
            try:
                url = self._generate_url(current_ahead_hours)
                logger.debug(f"Fetching NOAA marine forecast from: {url}")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Calculate base date for this chunk
                chunk_base_date = datetime.now() + timedelta(hours=current_ahead_hours)
                
                # Parse the HTML response
                chunk_data = self._parse_forecast_table(response.text, chunk_base_date)
                all_data.extend(chunk_data)
                
                # Move to next 48-hour window
                current_ahead_hours += 48
                
            except Exception as e:
                logger.error(f"Error fetching NOAA marine forecast: {e}")
                break
        
        if not all_data:
            logger.warning("No NOAA marine forecast data retrieved")
            return pd.DataFrame()
        
        # Convert to DataFrame and filter by date range
        df = pd.DataFrame(all_data)
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} hours of NOAA marine forecast data")
        return df
    
    def is_available(self) -> bool:
        """Check if NOAA marine forecast service is available"""
        import requests
        try:
            url = self._generate_url(0)
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

class DataCollector:
    """Main data collection coordinator using multiple sources"""
    
    def __init__(self, use_enhanced_currents=False, hourly_interpolation=False, daylight_only=False):
        # Initialize current sources with enhanced option
        if use_enhanced_currents:
            self.sources = {
                'pacioos_enhanced': PacIOOSEnhancedCurrentSource(
                    use_hourly_interpolation=hourly_interpolation,
                    daylight_only=daylight_only
                ),
                'pacioos': PacIOOSCurrentSource(),  # Fallback
                'noaa': NOAACurrentSource(),
                'noaa_marine': NOAAMarineSource(),  # Primary wind & wave source
                'ndbc': NDFCBuoySource(),  # Fallback wind source
                'openmeteo': OpenMeteoWindSource(),
                'noaa_weather': NOAAWeatherBuoySource(),
                'noaa_cwf': NOAACWFWindSource()  # Coastal Waters Forecast
            }
        else:
            self.sources = {
                'pacioos': PacIOOSCurrentSource(),
                'noaa': NOAACurrentSource(),
                'noaa_marine': NOAAMarineSource(),  # Primary wind & wave source
                'ndbc': NDFCBuoySource(),  # Fallback wind source
                'openmeteo': OpenMeteoWindSource(),
                'noaa_weather': NOAAWeatherBuoySource(),
                'noaa_cwf': NOAACWFWindSource()  # Coastal Waters Forecast
            }
        self.tidal_analyzer = TidalCurrentAnalyzer()
        
    def get_available_sources(self) -> List[str]:
        """Check which data sources are currently available"""
        available = []
        for name, source in self.sources.items():
            if source.is_available():
                available.append(name)
                logger.info(f"Data source '{name}' is available")
            else:
                logger.warning(f"Data source '{name}' is not available")
        return available
    
    def collect_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Collect data from all available sources"""
        all_data = {}
        
        for name, source in self.sources.items():
            try:
                logger.info(f"Collecting data from {name}")
                data = source.fetch_data(start_date, end_date)
                if not data.empty:
                    all_data[name] = data
                    logger.info(f"Collected {len(data)} records from {name}")
                else:
                    logger.warning(f"No data returned from {name}")
            except Exception as e:
                logger.error(f"Error collecting from {name}: {e}")
        
        return all_data
    
    def find_optimal_conditions(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Find optimal conditions: eastward current flow into ENE trade winds
        This creates the standing wave effect you want for downwind foiling
        """
        # Collect all data
        all_data = self.collect_all_data(start_date, end_date)
        
        # Debug: Show what data sources are available
        logger.debug(f"Available data sources: {list(all_data.keys())}")
        
        if not all_data:
            logger.error("CRITICAL SYSTEM ERROR: No data sources are available")
            logger.error("ASSISTANCE NEEDED: Please check internet connection and data source APIs")
            logger.error("System cannot proceed without real oceanographic data - EXITING")  
            return pd.DataFrame()
        
        # Smart current data selection with quality weighting
        current_data = None
        current_source_quality = None
        
        # Priority order: PacIOOS Enhanced > PacIOOS > NOAA > Simulation
        if 'pacioos_enhanced' in all_data and not all_data['pacioos_enhanced'].empty:
            current_data = all_data['pacioos_enhanced']
            current_source_quality = 'model_forecast'
            logger.info("Using PacIOOS Enhanced ROMS model forecast for current data (highest quality)")
        elif 'pacioos' in all_data and not all_data['pacioos'].empty:
            current_data = all_data['pacioos']
            current_source_quality = 'model_forecast'
            logger.info("Using PacIOOS ROMS model forecast for current data (highest quality)")
        elif 'noaa' in all_data and not all_data['noaa'].empty:
            current_data = all_data['noaa']
            current_source_quality = 'tidal_predictions'
            logger.info("Using NOAA tidal predictions for current data (good quality)")
        else:
            logger.warning("No real current data available from APIs")
            current_source_quality = 'simulation'
        
                # Get wind data from multiple sources
        noaa_marine_data = all_data.get('noaa_marine', pd.DataFrame())  # Primary source with wave height
        wind_data = all_data.get('ndbc', pd.DataFrame())
        noaa_weather_data = all_data.get('noaa_weather', pd.DataFrame())
        openmeteo_wind_data = all_data.get('openmeteo', pd.DataFrame())
        noaa_cwf_data = all_data.get('noaa_cwf', pd.DataFrame())
        
                # Smart wind data selection with quality prioritization
        wind_source_priority = []
        if not noaa_marine_data.empty:
            wind_source_priority.append(('noaa_marine', noaa_marine_data, 'official_forecast'))
        if not wind_data.empty:
            wind_source_priority.append(('ndbc', wind_data, 'observed'))
        if not noaa_weather_data.empty:
            wind_source_priority.append(('noaa_weather', noaa_weather_data, 'observed'))
        if not noaa_cwf_data.empty:
            wind_source_priority.append(('noaa_cwf', noaa_cwf_data, 'official_forecast'))
        if not openmeteo_wind_data.empty:
            wind_source_priority.append(('openmeteo', openmeteo_wind_data, 'forecast'))
        
        if not wind_source_priority:
            logger.error("CRITICAL SYSTEM ERROR: No wind data available from any source")
            logger.error("ASSISTANCE NEEDED: Both NDBC buoy and Open-Meteo wind services are unavailable")
            logger.error("Wind data is ESSENTIAL for current-wind interaction analysis - EXITING")
            return pd.DataFrame()
        else:
            logger.info(f"Available wind sources: {[f'{name}({quality})' for name, _, quality in wind_source_priority]}")
        
        if current_data is None or current_data.empty:
            logger.error("CRITICAL: No real current data available from any API source")
            logger.error("LUMPS requires real oceanographic data - simulation disabled per user request")
            logger.error("Please check API connectivity or try different date ranges")
            return pd.DataFrame()  # Return empty - no simulation fallback
        
        # Debug current directions before analysis
        self.tidal_analyzer.debug_current_directions(current_data)
        
        # Find eastward flow periods
        eastward_periods = self.tidal_analyzer.find_eastward_flow_periods(current_data)
        
        if eastward_periods.empty:
            logger.warning("No eastward flow periods found")
            return pd.DataFrame()
        
        # Combine with wind data if available
        optimal_conditions = []
        
        for _, current_row in eastward_periods.iterrows():
            dt = current_row['datetime']
            
            # Ensure dt is timezone-naive for comparison
            if hasattr(dt, 'tz') and dt.tz is not None:
                dt = dt.tz_localize(None)
            
                        # Find matching wind data with quality prioritization
            wind_dir = None
            wind_speed = None
            wind_wave_height = None
            wind_source = None
            wind_quality = None
            
                                    # Priority 1: NOAA Marine forecast data (includes wind wave height, highest priority)
            if not noaa_marine_data.empty:
                try:
                    # Check if data has 'timestamp' or 'datetime' column
                    time_col = 'timestamp' if 'timestamp' in noaa_marine_data.columns else 'datetime'
                    time_diff = abs(noaa_marine_data[time_col] - dt)
                    if time_diff.min() <= timedelta(hours=1):
                        closest_wind = noaa_marine_data.loc[time_diff.idxmin()]
                        wind_dir = closest_wind.get('wind_dir_deg', closest_wind.get('wind_dir', 90))
                        wind_speed = closest_wind.get('wind_speed_kts', closest_wind.get('wind_speed', 0))
                        wind_wave_height = closest_wind.get('wind_wave_height_ft', 0.0)
                        wind_source = 'NOAA_Marine'
                        wind_quality = 'official_forecast'
                        logger.debug(f"Using NOAA Marine forecast data for {dt}")
                except (TypeError, KeyError) as e:
                    logger.debug(f"Error with NOAA Marine data: {e}")
            
                                    # Priority 2: NDBC buoy data (real observations, highest quality)
            if (wind_dir is None or pd.isna(wind_dir)) and not wind_data.empty:
                try:
                    time_diff = abs(wind_data['datetime'] - dt)
                    if time_diff.min() <= timedelta(hours=1):
                        closest_wind = wind_data.loc[time_diff.idxmin()]
                        wind_dir = closest_wind['wind_dir']
                        wind_speed = closest_wind['wind_speed']
                        wind_source = 'NDBC_Buoy'
                        wind_quality = 'observed'
                        logger.debug(f"Using NDBC wind data for {dt}")
                except TypeError as e:
                    logger.debug(f"Timezone mismatch with NDBC data: {e}")
            
                        # Priority 3: NOAA Weather Buoy data (real observations, high quality)
            if (wind_dir is None or pd.isna(wind_dir)) and not noaa_weather_data.empty:
                time_diff = abs(noaa_weather_data['datetime'] - dt)
                if time_diff.min() <= timedelta(hours=1):
                    closest_wind = noaa_weather_data.loc[time_diff.idxmin()]
                    wind_dir = closest_wind['wind_dir']
                    wind_speed = closest_wind['wind_speed']
                    wind_source = 'NOAA_Weather_Buoy'
                    wind_quality = 'observed'
                    logger.debug(f"Using NOAA weather buoy data for {dt}")
            
                        # Priority 4: NOAA CWF data (official marine forecast, excellent quality)
            if (wind_dir is None or pd.isna(wind_dir)) and not noaa_cwf_data.empty:
                time_diff = abs(noaa_cwf_data['datetime'] - dt)
                if time_diff.min() <= timedelta(hours=3):  # Allow 3-hour window for official forecasts
                    closest_wind = noaa_cwf_data.loc[time_diff.idxmin()]
                    wind_dir = closest_wind['wind_dir']
                    wind_speed = closest_wind['wind_speed']
                    wind_source = 'NOAA_CWF'
                    wind_quality = 'official_forecast'
                    logger.debug(f"Using NOAA Coastal Waters Forecast data for {dt}")
            
                        # Priority 5: Open-Meteo forecast data (model forecast, good quality)
            if (wind_dir is None or pd.isna(wind_dir)) and not openmeteo_wind_data.empty:
                time_diff = abs(openmeteo_wind_data['datetime'] - dt)
                if time_diff.min() <= timedelta(hours=2):  # Allow 2-hour window for forecasts
                    closest_wind = openmeteo_wind_data.loc[time_diff.idxmin()]
                    wind_dir = closest_wind['wind_dir']
                    wind_speed = closest_wind['wind_speed']
                    wind_source = 'OpenMeteo_Forecast'
                    wind_quality = 'forecast'
                    logger.debug(f"Using Open-Meteo wind data for {dt}")
            
            # If both sources failed, report critical error for this time point
            if wind_dir is None or pd.isna(wind_dir) or wind_speed is None or pd.isna(wind_speed):
                logger.warning(f"CRITICAL: No wind data available for {dt} - skipping this analysis point")
                continue
            
            # Analyze interaction (now includes tidal phase enhancement)
            interaction = self.tidal_analyzer.analyze_current_wind_interaction(
                current_row['current_dir'], wind_dir, current_row['current_speed'], dt
            )
            
            # Debug the interaction results
            logger.debug(f"Interaction analysis for {dt}:")
            logger.debug(f"  Current: {current_row['current_dir']:.0f}° @ {current_row['current_speed']:.2f}kt")
            logger.debug(f"  Wind: {wind_dir:.0f}° @ {wind_speed:.1f}kt")
            logger.debug(f"  Angle diff: {interaction.get('angle_difference', 'N/A'):.0f}°")
            logger.debug(f"  Surfable: {interaction.get('surfable_conditions', False)}")
            logger.debug(f"  Optimal: {interaction.get('optimal_conditions', False)}")
            
            # Include ALL surfable conditions - not just optimal standing waves
            if interaction.get('surfable_conditions', False):
                                optimal_conditions.append({
                    'datetime': dt,
                    'current_speed': current_row['current_speed'],
                    'current_dir': current_row['current_dir'],
                    'current_source': current_row.get('source', 'Unknown'),
                    'current_quality': current_source_quality,
                    'wind_speed': wind_speed,
                    'wind_dir': wind_dir,
                    'wind_wave_height_ft': wind_wave_height if wind_wave_height is not None else 0.0,
                    'wind_source': wind_source,
                    'wind_quality': wind_quality,
                    'interaction': interaction,
                    'enhancement': interaction['wave_enhancement'],
                    'flow_direction': 'sunset_to_turtle_bay'
                })
        
        result_df = pd.DataFrame(optimal_conditions)
        
        if not result_df.empty:
            logger.info(f"SUCCESS: Found {len(result_df)} optimal conditions for eastward current into ENE trades")
        else:
            logger.warning("NO OPTIMAL CONDITIONS FOUND in the requested time period")
            logger.warning("This may be normal - eastward current flow opposing ENE trades is relatively rare")
            logger.info("Try extending the analysis period or checking during different tidal phases")
        
        return result_df
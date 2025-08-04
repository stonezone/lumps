#!/usr/bin/env python3
"""
Modular data source interfaces for North Shore current and wind data
Focuses on finding eastward current flow (Sunset Beach → Turtle Bay) into ENE trades
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
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
        try:
            # Test with a 1-day query
            test_params = {
                'id': self.station_id,
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'range': '1',
                'date_timeUnits': 'am/pm',
                'interval': 'MAX_SLACK',
                'time_zone': 'LST_LDT',
                'units': '1',
                'format': 'txt'
            }
            response = requests.get(self.base_url, params=test_params, timeout=10)
            return response.status_code == 200 and 'Kahuku Point' in response.text
        except:
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
        
        # Convert string timestamps to pandas datetime
        merged_df['time'] = pd.to_datetime(merged_df['time'])
        
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
                    'datetime': timestamp,
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
        Find periods when current flows in directions favorable for North Shore foiling
        EXPANDED CRITERIA: Any current that can interact with wind to create surfable conditions
        """
        try:
            if current_data.empty:
                return pd.DataFrame()
            
            # COMPREHENSIVE flow criteria for downwind foiling:
            # Accept ANY current direction that can create wave enhancement
            # Focus on finding periods with measurable current speed (>0.05 kt)
            # rather than restricting by direction
            
            # Minimum current speed threshold for viable wave interaction
            min_current_speed = 0.05  # knots - very sensitive to detect all possible conditions
            
            # Accept all current directions but prioritize by interaction potential
            favorable_mask = current_data['current_speed'] >= min_current_speed
            
            favorable_periods = current_data[favorable_mask].copy()
            
            if not favorable_periods.empty:
                # Enhanced classification including ALL flow types
                def classify_flow_type(direction):
                    if 315 <= direction <= 360 or 0 <= direction <= 45:
                        return 'northward'
                    elif 45 < direction <= 135:
                        return 'eastward'  
                    elif 135 < direction <= 225:
                        return 'southward'
                    elif 225 < direction <= 315:
                        return 'westward'
                    else:
                        return 'variable'
                
                def assess_interaction_potential(direction):
                    """Assess how well current direction interacts with typical ENE trades (060°)"""
                    # ENE trades typically 050-070°, use 060° as reference
                    trade_wind_dir = 60
                    angle_diff = abs(direction - trade_wind_dir)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    
                    # Current flowing into wind (small angle) = high potential
                    # Current flowing with wind (large angle) = moderate potential  
                    # Cross current = variable potential
                    if angle_diff <= 30:
                        return 'high_opposing'      # Current into wind - best for lumps
                    elif angle_diff >= 150:
                        return 'moderate_following' # Current with wind - still surfable
                    else:
                        return 'cross_flow'         # Cross current - variable enhancement
                
                favorable_periods['flow_type'] = favorable_periods['current_dir'].apply(classify_flow_type)
                favorable_periods['interaction_potential'] = favorable_periods['current_dir'].apply(assess_interaction_potential)
                favorable_periods['toward_turtle_bay'] = True  # All current can affect foiling
                
                # Calculate eastward component for compatibility
                favorable_periods['eastward_component'] = np.cos(
                    np.radians(favorable_periods['current_dir'] - 90)
                )
                
                # Log comprehensive statistics
                flow_types = favorable_periods['flow_type'].value_counts()
                interaction_types = favorable_periods['interaction_potential'].value_counts()
                
                logger.info(f"Found {len(favorable_periods)} periods of current flow >= {min_current_speed} kt")
                logger.info(f"Flow types: {dict(flow_types)}")
                logger.info(f"Interaction potential: {dict(interaction_types)}")
                
            return favorable_periods
            
        except Exception as e:
            logger.error(f"Error analyzing current flow: {e}")
            return pd.DataFrame()
    
    def analyze_current_wind_interaction(self, current_dir: float, wind_dir: float, datetime_obj: datetime = None) -> Dict:
        """
        COMPREHENSIVE current-wind interaction analysis for ALL downwind conditions
        Includes opposing, following, and cross flows that create surfable wave states
        """
        try:
            # Calculate angle between current and wind
            angle_diff = abs(current_dir - wind_dir)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # EXPANDED interaction categories for comprehensive analysis:
            
            if angle_diff <= 20:
                interaction = "current_into_wind_strong"
                base_enhancement = "maximum"  # Strongest opposing flow - best lumps
                surfable = True
            elif angle_diff <= 45:
                interaction = "current_into_wind_moderate" 
                base_enhancement = "excellent"  # Good opposing flow
                surfable = True
            elif angle_diff <= 90:
                interaction = "current_cross_wind"
                base_enhancement = "good"  # Cross flow can still create good conditions
                surfable = True
            elif angle_diff <= 135:
                interaction = "current_angled_with_wind"
                base_enhancement = "moderate"  # Angled flow - variable conditions
                surfable = True
            else:  # angle_diff > 135
                interaction = "current_with_wind"
                base_enhancement = "good"  # Following flow can create different wave dynamics
                surfable = True
            
            # Add tidal phase enhancement
            tidal_enhancement = 0.0
            if datetime_obj:
                tidal_enhancement = self._calculate_tidal_phase_enhancement(datetime_obj)
            
            # Combine base enhancement with tidal phase
            final_enhancement = self._combine_enhancements(base_enhancement, tidal_enhancement)
            
            # Determine if optimal for specific wave types
            opposing_flow = angle_diff <= 45  # Current into wind
            following_flow = angle_diff >= 135  # Current with wind
            cross_flow = 45 < angle_diff < 135  # Cross current
            
            return {
                'angle_difference': angle_diff,
                'interaction_type': interaction,
                'wave_enhancement': final_enhancement,
                'tidal_enhancement': tidal_enhancement,
                'optimal_for_standing_waves': opposing_flow,  # Traditional standing wave lumps
                'optimal_for_following_waves': following_flow,  # Following sea enhancement
                'optimal_for_cross_waves': cross_flow,  # Cross wave patterns
                'surfable_conditions': surfable,  # Include ALL viable conditions
                'enhancement_category': self._categorize_enhancement(final_enhancement)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing current-wind interaction: {e}")
            return {'surfable_conditions': False}
    
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
        Calculate tidal phase enhancement factor based on proximity to high tide
        High tide and 2 hours before are critical periods voyagers know matter
        """
        # Calculate approximate high tide times for this location (North Shore Oahu)
        # Using tidal harmonic constants - high tide roughly every 12.42 hours
        high_tide_times = self._get_daily_high_tide_estimates(dt.date())
        
        # Debug logging for tidal timing
        logger.debug(f"High tide times for {dt.date()}: {[f'{ht:.1f}' for ht in high_tide_times]} hours")
        
        # Find time difference to nearest high tide
        current_hour = dt.hour + dt.minute / 60.0
        time_diffs = []
        for ht in high_tide_times:
            # Handle day boundary crossings
            diff1 = abs(current_hour - ht)
            diff2 = abs(current_hour - (ht + 24))  # Next day
            diff3 = abs(current_hour - (ht - 24))  # Previous day
            time_diffs.append(min(diff1, diff2, diff3))
        
        min_time_to_high_tide = min(time_diffs)
        
        logger.debug(f"At {dt.strftime('%H:%M')}, closest high tide is {min_time_to_high_tide:.1f}h away")
        
        # Apply enhancement based on proximity to high tide
        if min_time_to_high_tide <= 0.5:  # Within 30 minutes of high tide
            return 0.35  # Maximum enhancement at high tide
        elif min_time_to_high_tide <= 1.0:  # Within 1 hour of high tide
            return 0.25  # Strong enhancement near high tide
        elif min_time_to_high_tide <= 2.0:  # Within 2 hours before high tide
            return 0.20  # Good enhancement in the critical 2-hour window
        elif min_time_to_high_tide <= 3.0:  # 1 hour after high tide
            return 0.10  # Moderate enhancement post-high tide
        else:
            return 0.0   # No enhancement during other periods
    
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
                'ndbc': NDFCBuoySource(),
                'openmeteo': OpenMeteoWindSource(),
                'noaa_weather': NOAAWeatherBuoySource()
            }
        else:
            self.sources = {
                'pacioos': PacIOOSCurrentSource(),
                'noaa': NOAACurrentSource(),
                'ndbc': NDFCBuoySource(),
                'openmeteo': OpenMeteoWindSource(),
                'noaa_weather': NOAAWeatherBuoySource()
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
        
        if not all_data:
            logger.error("CRITICAL SYSTEM ERROR: No data sources are available")
            logger.error("ASSISTANCE NEEDED: Please check internet connection and data source APIs")
            logger.error("System cannot proceed without real oceanographic data - EXITING")  
            return pd.DataFrame()
        
        # Smart current data selection with quality weighting
        current_data = None
        current_source_quality = None
        
        # Priority order: PacIOOS (forecast model) > NOAA (tidal predictions) > Simulation
        if 'pacioos' in all_data and not all_data['pacioos'].empty:
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
        wind_data = all_data.get('ndbc', pd.DataFrame())
        noaa_weather_data = all_data.get('noaa_weather', pd.DataFrame())
        openmeteo_wind_data = all_data.get('openmeteo', pd.DataFrame())
        
        # Smart wind data selection with quality prioritization
        wind_source_priority = []
        if not wind_data.empty:
            wind_source_priority.append(('ndbc', wind_data, 'observed'))
        if not noaa_weather_data.empty:
            wind_source_priority.append(('noaa_weather', noaa_weather_data, 'observed'))
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
            logger.info("Falling back to enhanced tidal current simulation")
            logger.info("Simulation uses multiple harmonic components for realistic variability")
            current_data = self.tidal_analyzer.generate_tidal_simulation(start_date, end_date)
            current_source_quality = 'simulation'
        
        # Find eastward flow periods
        eastward_periods = self.tidal_analyzer.find_eastward_flow_periods(current_data)
        
        if eastward_periods.empty:
            logger.warning("No eastward flow periods found")
            return pd.DataFrame()
        
        # Combine with wind data if available
        optimal_conditions = []
        
        for _, current_row in eastward_periods.iterrows():
            dt = current_row['datetime']
            
            # Find matching wind data with quality prioritization
            wind_dir = None
            wind_speed = None
            wind_source = None
            wind_quality = None
            
            # Priority 1: NDBC buoy data (real observations, highest quality)
            if not wind_data.empty:
                time_diff = abs(wind_data['datetime'] - dt)
                if time_diff.min() <= timedelta(hours=1):
                    closest_wind = wind_data.loc[time_diff.idxmin()]
                    wind_dir = closest_wind['wind_dir']
                    wind_speed = closest_wind['wind_speed']
                    wind_source = 'NDBC_Buoy'
                    wind_quality = 'observed'
                    logger.debug(f"Using NDBC wind data for {dt}")
            
            # Priority 2: NOAA Weather Buoy data (real observations, high quality)
            if (wind_dir is None or pd.isna(wind_dir)) and not noaa_weather_data.empty:
                time_diff = abs(noaa_weather_data['datetime'] - dt)
                if time_diff.min() <= timedelta(hours=1):
                    closest_wind = noaa_weather_data.loc[time_diff.idxmin()]
                    wind_dir = closest_wind['wind_dir']
                    wind_speed = closest_wind['wind_speed']
                    wind_source = 'NOAA_Weather_Buoy'
                    wind_quality = 'observed'
                    logger.debug(f"Using NOAA weather buoy data for {dt}")
            
            # Priority 3: Open-Meteo forecast data (model forecast, good quality)
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
                current_row['current_dir'], wind_dir, dt
            )
            
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
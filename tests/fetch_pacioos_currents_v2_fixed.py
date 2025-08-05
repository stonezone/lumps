#!/usr/bin/env python3
"""
Robust PacIOOS Hawaii Current Data Extractor - Fixed Version

This module provides a class for downloading and processing near–surface
current forecasts for the north shore of Oʻahu (Haleʻiwa to Kahuku Point).

Features:
 - Queries the PacIOOS ERDDAP server for both eastward (`u`) and northward
   (`v`) velocity components.
 - Splits large time ranges into smaller chunks to avoid server timeouts.
 - Applies local timezone conversion (Hawaii is UTC‑10) and filters
   daylight hours (06:00–18:00 local).
 - Computes current speed (m/s and knots) and direction (degrees and
   compass points).
 - Saves the processed data to a CSV file.

The default configuration targets the north shore region between Haleʻiwa
and Kahuku Point, but you can override the spatial bounds, depth, and
forecast length as needed.

Example usage::

    from fetch_pacioos_currents_v2_fixed import PacIOOSCurrentExtractor
    extractor = PacIOOSCurrentExtractor()
    data = extractor.get_forecast_data(days=7)
    extractor.save_data(data, 'north_shore_currents.csv')

Author: Manus AI Assistant
Date: August 2025
Version: 2.0 (Fixed)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


@dataclass
class SpatialBounds:
    """Simple container for geographic bounding box."""
    lat_min: float = 21.59
    lat_max: float = 21.72
    lon_min: float = -158.11
    lon_max: float = -157.97
    depth: float = 0.25  # near‑surface layer


class PacIOOSCurrentExtractor:
    """Extract near–shore current forecasts from the PacIOOS ERDDAP server."""

    def __init__(
        self,
        bounds: SpatialBounds | None = None,
        base_url: str = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/roms_hiig",
        timezone_offset_hours: int = -10,
        sleep_seconds: float = 1.0,
    ) -> None:
        """
        Initialize the current extractor.
        
        Args:
            bounds: Geographic bounding box for data extraction
            base_url: ERDDAP server base URL
            timezone_offset_hours: Hawaii timezone offset from UTC (-10)
            sleep_seconds: Delay between requests to be respectful to server
        """
        self.base_url = base_url.rstrip('.')
        self.bounds = bounds or SpatialBounds()
        self.timezone_offset = timezone_offset_hours
        self.sleep_seconds = max(0.0, sleep_seconds)

    def _build_url(
        self,
        variable: str,
        time_start: datetime,
        time_end: datetime,
    ) -> str:
        """Construct an ERDDAP query URL for the given variable and time range."""
        start_iso = time_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = time_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        b = self.bounds
        # Format: dataset.csv?var[(start):stride:(end)][(depth)][(lat_min):(lat_max)][(lon_min):(lon_max)]
        # FIXED: Removed the problematic "&download=1" parameter
        url = (
            f"{self.base_url}.csv?{variable}"
            f"[({start_iso}):1:({end_iso})]"
            f"[({b.depth})]"
            f"[({b.lat_min}):1:({b.lat_max})]"
            f"[({b.lon_min}):1:({b.lon_max})]"
        )
        return url

    def _fetch_component(
        self,
        variable: str,
        time_start: datetime,
        time_end: datetime,
    ) -> pd.DataFrame | None:
        """Retrieve one velocity component from ERDDAP for a time range."""
        url = self._build_url(variable, time_start, time_end)
        print(f"Requesting {variable} from {time_start} to {time_end}")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            # Parse CSV
            csv_text = resp.text
            # ERDDAP includes a header row describing units; drop it
            df = pd.read_csv(StringIO(csv_text))
            if df.empty:
                return None
            # Remove rows where time equals the string 'UTC' (units row)
            df = df[df['time'] != 'UTC']
            # FIXED: Convert ALL numeric columns properly to avoid string/int errors
            numeric_cols = ['depth', 'latitude', 'longitude', variable]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Rename the velocity column to avoid collision
            if variable in df.columns:
                df = df.rename(columns={variable: f"{variable}_velocity"})
            return df
        except Exception as exc:
            print(f"Failed to fetch {variable} data: {exc}")
            return None

    def _chunk_time_ranges(
        self,
        start: datetime,
        end: datetime,
        hours_per_chunk: int = 24,
    ) -> List[Tuple[datetime, datetime]]:
        """Split a larger time range into smaller chunks."""
        ranges: List[Tuple[datetime, datetime]] = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(hours=hours_per_chunk), end)
            ranges.append((current, chunk_end))
            current = chunk_end
        return ranges

    def _merge_uv(self, u: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame | None:
        """Merge u and v components on shared coordinates."""
        if u is None or v is None:
            return None
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
            print(f"Error merging components: {exc}")
            return None

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute local time, daylight filter, and derived speed/direction."""
        if df is None or df.empty:
            return df
        # Convert string timestamps to pandas datetime
        df['time'] = pd.to_datetime(df['time'])
        # Compute local time in Hawaii (UTC‑10)
        df['local_time'] = df['time'] + pd.Timedelta(hours=self.timezone_offset)
        df['hour'] = df['local_time'].dt.hour
        # Keep only daylight hours (06:00–18:00)
        df = df[(df['hour'] >= 6) & (df['hour'] <= 18)].copy()
        if df.empty:
            return df
        
        # FIXED: Ensure velocity columns are numeric and remove NaN values
        df['u_velocity'] = pd.to_numeric(df['u_velocity'], errors='coerce')
        df['v_velocity'] = pd.to_numeric(df['v_velocity'], errors='coerce')
        df = df.dropna(subset=['u_velocity', 'v_velocity'])
        
        if df.empty:
            return df
        
        # Compute speed and direction
        df = self._compute_speed_direction(df)
        return df.sort_values(['time', 'latitude', 'longitude'])

    def _compute_speed_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add current speed and direction columns derived from u and v components.

        This method computes:
        - `current_speed_ms`: magnitude of the current vector in m/s.
        - `current_direction_deg`: direction in degrees (0° = North, 90° = East).
        - `current_speed_knots`: magnitude in knots.
        - `current_direction_compass`: compass direction (e.g., NNE, SSE).

        Args:
            df (pd.DataFrame): DataFrame containing `u_velocity` and `v_velocity`.

        Returns:
            pd.DataFrame: The input DataFrame with additional columns.
        """
        if df is None or df.empty:
            return df
        
        # Calculate magnitude in m/s
        df['current_speed_ms'] = np.sqrt(df['u_velocity'] ** 2 + df['v_velocity'] ** 2)
        
        # Calculate direction in degrees (0° = North, 90° = East)
        df['current_direction_deg'] = np.degrees(np.arctan2(df['u_velocity'], df['v_velocity']))
        df['current_direction_deg'] = (df['current_direction_deg'] + 360) % 360
        
        # Convert to knots (1 m/s = 1.94384 knots)
        df['current_speed_knots'] = df['current_speed_ms'] * 1.94384
        
        # Add human‑readable compass direction
        directions = [
            'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',
        ]
        def deg_to_compass(deg: float) -> str:
            idx = int(round(deg / 22.5)) % 16
            return directions[idx]
        
        df['current_direction_compass'] = df['current_direction_deg'].apply(deg_to_compass)
        return df

    def get_forecast_data(
        self,
        days: int = 7,
        hours_per_request: int = 24,
    ) -> pd.DataFrame | None:
        """
        Download and process forecast current data for a number of days.

        Args:
            days (int): How many days ahead to request.
            hours_per_request (int): Number of hours per server request. Smaller
                values mitigate timeouts; 24 is a reasonable default.

        Returns:
            pandas.DataFrame: Processed current forecast data, or None on failure.
        """
        start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=days)
        print(f"PacIOOS Current Extractor - North Shore Oahu")
        print(f"Area: {self.bounds.lat_min}°N to {self.bounds.lat_max}°N, {self.bounds.lon_min}°W to {self.bounds.lon_max}°W")
        print(f"Depth: {self.bounds.depth}m (surface)")
        print(f"Forecast period: {days} days")
        print(f"Forecast start: {start_time} UTC")
        print(f"Forecast end:   {end_time} UTC")
        print("=" * 60)

        all_u: List[pd.DataFrame] = []
        all_v: List[pd.DataFrame] = []
        
        # Fetch data in chunks
        for t0, t1 in self._chunk_time_ranges(start_time, end_time, hours_per_request):
            u_chunk = self._fetch_component('u', t0, t1)
            if u_chunk is not None and not u_chunk.empty:
                all_u.append(u_chunk)
            time.sleep(self.sleep_seconds)
            
            v_chunk = self._fetch_component('v', t0, t1)
            if v_chunk is not None and not v_chunk.empty:
                all_v.append(v_chunk)
            time.sleep(self.sleep_seconds)

        if not all_u or not all_v:
            print("No data retrieved; check server availability or parameters.")
            return None
            
        print(f"\nCombining {len(all_u)} U chunks and {len(all_v)} V chunks...")
        u_df = pd.concat(all_u, ignore_index=True)
        v_df = pd.concat(all_v, ignore_index=True)
        
        print(f"Total U data points: {len(u_df)}")
        print(f"Total V data points: {len(v_df)}")
        
        merged = self._merge_uv(u_df, v_df)
        if merged is None or merged.empty:
            print("Failed to merge U and V components")
            return None
            
        print(f"Merged data points: {len(merged)}")
        
        processed = self._process(merged)
        if processed is None or processed.empty:
            print("No data remaining after processing")
            return None
            
        print(f"Final processed data points: {len(processed)}")
        print(f"Time range: {processed['local_time'].min()} to {processed['local_time'].max()} (Hawaii time)")
        
        return processed

    def save_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> Optional[str]:
        """Save the processed DataFrame to a CSV file."""
        if df is None or df.empty:
            print("No data to save.")
            return None
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"north_shore_currents_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Data written to {filename}")
        return filename

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print summary statistics of the forecast data."""
        if df is None or df.empty:
            print("No data available for summary")
            return
        
        print("\n" + "="*60)
        print("FORECAST SUMMARY")
        print("="*60)
        
        print(f"Total data points: {len(df)}")
        print(f"Time range: {df['local_time'].min()} to {df['local_time'].max()} (Hawaii time)")
        print(f"Unique locations: {len(df[['latitude', 'longitude']].drop_duplicates())}")
        
        if 'current_speed_knots' in df.columns:
            speed_stats = df['current_speed_knots'].describe()
            print(f"\nCurrent Speed (knots):")
            print(f"  Mean: {speed_stats['mean']:.3f}")
            print(f"  Min:  {speed_stats['min']:.3f}")
            print(f"  Max:  {speed_stats['max']:.3f}")
        
        if 'current_direction_compass' in df.columns:
            direction_counts = df['current_direction_compass'].value_counts().head(5)
            print(f"\nMost common current directions:")
            for direction, count in direction_counts.items():
                print(f"  {direction}: {count} occurrences")


def main() -> None:
    """Entry point for command line use."""
    print("PacIOOS Hawaii Current Data Extractor v2.0 (Fixed)")
    print("=" * 60)
    
    extractor = PacIOOSCurrentExtractor()
    data = extractor.get_forecast_data(days=7)
    
    if data is not None and not data.empty:
        # Print summary
        extractor.print_summary(data)
        
        # Save data
        filename = extractor.save_data(data)
        
        print(f"\n" + "="*60)
        print("SUCCESS: Current forecast data retrieved and processed!")
        print("="*60)
        print(f"File saved: {filename}")
        print(f"Columns: {list(data.columns)}")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        display_cols = ['time', 'local_time', 'latitude', 'longitude', 
                       'current_speed_knots', 'current_direction_deg', 'current_direction_compass']
        sample = data[display_cols].head(3)
        print(sample.to_string(index=False))
        
    else:
        print("\nFAILED: Could not retrieve forecast data")
        print("Check your internet connection and try again.")


if __name__ == '__main__':
    main()


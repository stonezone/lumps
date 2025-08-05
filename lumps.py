#!/usr/bin/env python3
"""
LUMPS - North Shore Oahu Downwind Foiling Analysis
Main application for finding eastward current flow into ENE trade winds

Usage:
    python lumps.py --start-date 2025-06-16 --days 7 --time-range 6-19
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from data_sources import DataCollector
from analysis import ConditionAnalyzer, ReportGenerator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lumps.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    LUMPS - North Shore Current Analysis              ║
    ║                                                                      ║
    ║  Foilers ride: Turtle Bay → Sunset Beach (westward WITH wind)       ║
    ║  Best lumps: Eastward current OPPOSING ENE trade winds              ║
    ║                                                                      ║
    ║  Data Sources: NOAA | PacIOOS | NDBC | Tidal Analysis              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Find optimal eastward current flow into ENE trade winds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lumps.py                                    # 10-day analysis from today
  python lumps.py --days 5                          # 5-day analysis  
  python lumps.py --start-date 2025-06-20           # Custom start date
  python lumps.py --time-range 8-18                 # 8 AM to 6 PM window
  python lumps.py --verbose                         # Detailed logging
  python lumps.py --check-sources                   # Test data source availability
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for analysis (YYYY-MM-DD), defaults to today'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to analyze (default: 7, matches PacIOOS forecast range)'
    )
    
    parser.add_argument(
        '--time-range',
        type=str,
        default='6-19',
        help='Time range in HST hours (default: 6-19 for 6AM-7PM)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for data files (default: data)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--check-sources',
        action='store_true',
        help='Check data source availability and exit'
    )
    
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detailed report to file'
    )
    
    parser.add_argument(
        '--enhanced-currents',
        action='store_true',
        help='Use enhanced PacIOOS currents with improved processing'
    )
    
    parser.add_argument(
        '--hourly-interpolation',
        action='store_true', 
        help='Interpolate current data to hourly resolution (requires --enhanced-currents)'
    )
    
    parser.add_argument(
        '--daylight-only',
        action='store_true',
        help='Filter current data to daylight hours only 6 AM - 6 PM HST (requires --enhanced-currents)'
    )
    
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='Output web-friendly JSON data to web/data/current.json'
    )
    
    return parser.parse_args()

def check_data_sources(collector: DataCollector):
    """Check and report data source availability"""
    print("🔍 Checking data source availability...")
    print("-" * 50)
    
    available_sources = collector.get_available_sources()
    
    for name, source in collector.sources.items():
        status = "✅ Available" if name in available_sources else "❌ Unavailable"
        print(f"{name.upper():10} : {status}")
    
    print(f"\nTotal available sources: {len(available_sources)}/{len(collector.sources)}")
    
    if not available_sources:
        print("\n⚠️  WARNING: No data sources are available!")
        print("This may be due to network issues or API limitations.")
    
    return len(available_sources) > 0

import json
from pathlib import Path

def save_web_data(conditions_df, start_date, end_date, output_dir='web/data'):
    """Save web-friendly JSON data for the LUMPS web interface"""
    if conditions_df.empty:
        return
    
    # Ensure web data directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get current conditions (most recent entry)
    current_entry = conditions_df.iloc[0] if not conditions_df.empty else None
    
    def get_quality_emoji(quality):
        """Get emoji for quality level matching CLI output"""
        quality_map = {
            'EXCELLENT': '⭐',
            'GOOD': '✅', 
            'MODERATE': '🌀',
            'MARGINAL': '😐',
            'POOR': '❌'
        }
        return quality_map.get(quality, '🌀')
    
    def format_cli_entry(row):
        """Format a single entry like CLI output"""
        dt = row.get('datetime')
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        
        time_str = dt.strftime('%I:%M %p') if dt else 'N/A'
        
        # Parse interaction data for tidal info
        interaction = row.get('interaction', {})
        if isinstance(interaction, str):
            try:
                import ast
                interaction = ast.literal_eval(interaction)
            except (ValueError, SyntaxError):
                interaction = {}
        
        # Get quality and emoji
        quality = row.get('quality', 'MODERATE').upper()
        emoji = get_quality_emoji(quality)
        
        # Wind and current info
        wind_speed = row.get('wind_speed', 0)
        wind_dir = row.get('wind_dir', 0)
        current_speed = row.get('current_speed', 0)
        current_dir = row.get('current_dir', 0)
        
        # Tide info from interaction data
        tidal_phase = interaction.get('tidal_phase', 'N/A')
        
        # Scoring info (if available)
        scoring_debug = row.get('scoring_debug', 'N/A')
        total_score = row.get('total_score', 0)
        
        return {
            'time': time_str,
            'quality_emoji': emoji,
            'quality': quality,
            'wind_display': f"🌐{wind_speed:.1f}kt@{wind_dir:03.0f}°",
            'current_display': f"🌊{current_speed:.1f}kt@{current_dir:03.0f}°",
            'tide_display': tidal_phase,
            'score_display': scoring_debug if scoring_debug != 'N/A' else f"Score: {total_score}/15",
            'cli_format': f"{time_str} | {emoji} {quality} | Wind: 🌐{wind_speed:.1f}kt@{wind_dir:03.0f}° | Current: 🌊{current_speed:.1f}kt@{current_dir:03.0f}° | Tide: {tidal_phase} | {scoring_debug if scoring_debug != 'N/A' else f'Score: {total_score}/15'}"
        }
    
    # Calculate score for current conditions
    def calculate_web_score(row):
        if row is None:
            return 0
        
        score = 5  # Base score
        
        # Wind contribution (0-3 points)
        wind_speed = row.get('wind_speed', 0)
        if wind_speed >= 18:
            score += 3
        elif wind_speed >= 15:
            score += 2
        elif wind_speed >= 12:
            score += 1
        
        # Current contribution (0-2 points)
        current_speed = row.get('current_speed', 0)
        if current_speed >= 0.3:
            score += 2
        elif current_speed >= 0.2:
            score += 1
        
        return min(score, 10)
    
    def get_quality_from_score(score):
        if score >= 8:
            return 'EXCELLENT'
        elif score >= 6.5:
            return 'GOOD'
        elif score >= 5:
            return 'MODERATE'
        else:
            return 'POOR'
    
    # Group data by day for CLI-style breakdown
    daily_data = {}
    
    for _, row in conditions_df.iterrows():
        dt = row.get('datetime')
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        
        if dt:
            date_key = dt.strftime('%Y-%m-%d')
            day_name = dt.strftime('%A, %B %d, %Y')
            
            if date_key not in daily_data:
                daily_data[date_key] = {
                    'date': date_key,
                    'day_name': day_name,
                    'conditions': []
                }
            
            daily_data[date_key]['conditions'].append(format_cli_entry(row))
    
    # Convert to list sorted by date
    daily_breakdown = []
    for date_key in sorted(daily_data.keys()):
        day_data = daily_data[date_key]
        # Sort conditions by time within each day
        day_data['conditions'].sort(key=lambda x: x['time'])
        daily_breakdown.append(day_data)
    
    # Build web data structure
    current_score = calculate_web_score(current_entry)
    
    web_data = {
        'generated': datetime.now().isoformat(),
        'current': {
            'score': current_score,
            'quality': get_quality_from_score(current_score),
            'wind': {
                'speed': float(current_entry.get('wind_speed', 0)) if current_entry is not None else 0,
                'direction': float(current_entry.get('wind_dir', 0)) if current_entry is not None else 0,
                'source': current_entry.get('wind_source', 'Unknown') if current_entry is not None else 'Unknown'
            },
            'current': {
                'speed': float(current_entry.get('current_speed', 0)) if current_entry is not None else 0,
                'direction': float(current_entry.get('current_dir', 0)) if current_entry is not None else 0,
                'source': current_entry.get('current_source', 'Unknown') if current_entry is not None else 'Unknown'
            },
            'enhancement': current_entry.get('enhancement', 'none') if current_entry is not None else 'none'
        },
        'daily_breakdown': daily_breakdown,
        'timeline': []
    }
    
    # Add timeline data with full details
    for _, row in conditions_df.iterrows():
        datetime_str = str(row.get('datetime', '')) if row.get('datetime') is not None else ''
        
        # Parse interaction data if it exists
        interaction = row.get('interaction', {})
        if isinstance(interaction, str):
            # Parse the string representation of the dictionary
            try:
                import ast
                interaction = ast.literal_eval(interaction)
            except (ValueError, SyntaxError):
                interaction = {}
        
        web_data['timeline'].append({
            'datetime': datetime_str,
            'current_speed': float(row.get('current_speed', 0)),
            'current_dir': float(row.get('current_dir', 0)),
            'current_source': row.get('current_source', 'Unknown'),
            'wind_speed': float(row.get('wind_speed', 0)),
            'wind_dir': float(row.get('wind_dir', 0)),
            'wind_source': row.get('wind_source', 'Unknown'),
            'enhancement': row.get('enhancement', 'none'),
            # Add new detailed data
            'tidal_phase': row.get('tidal_phase', 'N/A'),
            'tidal_enhancement': float(row.get('tidal_enhancement', 0)),
            'scoring_debug': row.get('scoring_debug', 'N/A'),
            'total_score': int(row.get('total_score', 0)),
            'quality': row.get('quality', 'unknown'),
            'skill_level': row.get('skill_level', 'unknown'),
            'is_true_eastward': row.get('is_true_eastward', False),
            'recommendation': row.get('recommendation', ''),
            'angle_difference': float(interaction.get('angle_difference', 0)),
            'interaction_type': interaction.get('interaction_type', 'unknown')
        })
    
    # Save to JSON file
    json_file = Path(output_dir) / 'current.json'
    with open(json_file, 'w') as f:
        json.dump(web_data, f, indent=2)
    
    print(f"💾 Web data saved to {json_file}")

def main():
    """Main application entry point"""
    args = parse_arguments()
    
    # Setup
    logger = setup_logging(args.verbose)
    print_banner()
    
    # Parse dates and time range
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.start_date}. Use YYYY-MM-DD")
            return 1
    else:
        start_date = datetime.now()
    
    end_date = start_date + timedelta(days=args.days)
    
    try:
        start_hour, end_hour = map(int, args.time_range.split('-'))
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23 and start_hour < end_hour):
            raise ValueError("Invalid time range")
    except ValueError:
        logger.error(f"Invalid time range: {args.time_range}. Use format: START-END (e.g., 6-19)")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Initialize components
    collector = DataCollector(
        use_enhanced_currents=args.enhanced_currents,
        hourly_interpolation=args.hourly_interpolation,
        daylight_only=args.daylight_only
    )
    analyzer = ConditionAnalyzer()
    reporter = ReportGenerator()
    
    logger.info(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Time window: {start_hour:02d}:00 - {end_hour:02d}:00 HST")
    logger.info(f"Target: Eastward current (030-150°) flowing INTO ENE trades (050-070°)")
    
    # Check data sources if requested
    if args.check_sources:
        has_sources = check_data_sources(collector)
        return 0 if has_sources else 1
    
    try:
        # Check data source availability
        available_sources = collector.get_available_sources()
        if not available_sources:
            logger.error("No data sources available. Use --check-sources to diagnose.")
            return 1
        
        logger.info(f"Using data sources: {', '.join(available_sources)}")
        
        # Find optimal conditions
        logger.info("🔍 Searching for eastward current flow periods...")
        optimal_conditions = collector.find_optimal_conditions(start_date, end_date)
        
        if optimal_conditions.empty:
            logger.warning("No eastward current flow periods found in data")
            print("\n❌ NO EASTWARD CURRENT CONDITIONS FOUND")
            print("\nThis could mean:")
            print("• Tidal currents are primarily north-south oscillating in this area")
            print("• Dominant westward North Equatorial Current overwhelms eastward flow")
            print("• Need longer analysis period to capture tidal variations")
            print("• Different locations may have better eastward tidal flow")
            print("\nTry expanding the analysis period or checking different dates.")
            return 0
        
        # Filter to specified time window
        filtered_conditions = analyzer.filter_time_window(
            optimal_conditions, start_hour, end_hour
        )
        
        if filtered_conditions.empty:
            logger.warning(f"No conditions found in time window {start_hour}-{end_hour}")
            print(f"\n❌ No eastward current found in {start_hour:02d}:00-{end_hour:02d}:00 HST window")
            print("Try expanding the time range with --time-range 0-23")
            return 0
        
        # Analyze conditions
        logger.info("📊 Analyzing optimal condition periods...")
        analysis = analyzer.analyze_optimal_periods(filtered_conditions)
        
        # Generate and display report
        report = reporter.generate_summary_report(analysis, start_date, end_date)
        print(report)
        
        # Save report if requested
        if args.save_report:
            report_file = f"{args.output_dir}/lumps_report_{start_date.strftime('%Y%m%d')}.txt"
            saved_file = reporter.save_report(report, report_file)
            if saved_file:
                print(f"\n📄 Detailed report saved to: {saved_file}")
        
        # Save data
        data_file = f"{args.output_dir}/optimal_conditions_{start_date.strftime('%Y%m%d')}.csv"
        filtered_conditions.to_csv(data_file, index=False)
        logger.info(f"Data saved to {data_file}")
        
        # Save web JSON data if requested
        if args.output_json:
            save_web_data(filtered_conditions, start_date, end_date)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
LUMPS - North Shore Oahu Downwind Foiling Analysis
Main application for finding eastward current flow into ENE trade winds

Usage:
    python lumps.py --start-date 2025-06-16 --days 10 --time-range 6-19
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
        default=10,
        help='Number of days to analyze (default: 10)'
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
    collector = DataCollector()
    analyzer = ConditionAnalyzer()
    reporter = ReportGenerator()
    
    logger.info(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Time window: {start_hour:02d}:00 - {end_hour:02d}:00 HST")
    logger.info(f"Target: Eastward current (060-120°) flowing INTO ENE trades (050-070°)")
    
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
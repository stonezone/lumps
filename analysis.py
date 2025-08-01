#!/usr/bin/env python3
"""
Analysis engine for finding optimal downwind foiling conditions
Eastward current flow (Sunset Beach → Turtle Bay) into ENE trade winds
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ConditionAnalyzer:
    """Analyzes optimal downwind foiling conditions"""
    
    def __init__(self):
        self.trade_wind_range = (50, 70)  # ENE direction range
        self.optimal_wind_speed = (15, 25)  # Optimal wind speed range
        self.eastward_current_range = (60, 120)  # Eastward flow range
        
    def classify_conditions(self, wind_speed: float, current_speed: float, 
                          interaction_type: str) -> Dict:
        """Classify condition quality based on wind, current, and interaction"""
        
        # Base classification on wind speed
        if wind_speed >= 22:
            base_quality = "prime"
            skill_level = "expert"
        elif wind_speed >= 18:
            base_quality = "excellent" 
            skill_level = "advanced"
        elif wind_speed >= 15:
            base_quality = "good"
            skill_level = "intermediate"
        else:
            base_quality = "marginal"
            skill_level = "beginner"
        
        # Enhance based on current strength and interaction
        if current_speed >= 0.5 and interaction_type == "current_into_wind":
            if base_quality == "good":
                base_quality = "excellent"
            elif base_quality == "marginal":
                base_quality = "good"
        
        return {
            'quality': base_quality,
            'skill_level': skill_level,
            'wind_speed': wind_speed,
            'current_speed': current_speed,
            'recommendation': self._get_recommendation(base_quality, skill_level)
        }
    
    def _get_recommendation(self, quality: str, skill_level: str) -> str:
        """Generate text recommendation based on conditions"""
        recommendations = {
            'prime': f"🔥 PRIME conditions - {skill_level} level - Maximum wave enhancement",
            'excellent': f"⭐ EXCELLENT conditions - {skill_level} level - Strong wave enhancement", 
            'good': f"✅ GOOD conditions - {skill_level} level - Moderate wave enhancement",
            'marginal': f"📝 MARGINAL conditions - {skill_level} level - Light wave enhancement"
        }
        return recommendations.get(quality, "Conditions analyzed")
    
    def filter_time_window(self, data: pd.DataFrame, start_hour: int = 6, 
                          end_hour: int = 19) -> pd.DataFrame:
        """Filter data to specified time window (default 6 AM - 7 PM HST)"""
        if data.empty:
            return data
            
        mask = (data['datetime'].dt.hour >= start_hour) & (data['datetime'].dt.hour <= end_hour)
        return data[mask]
    
    def analyze_optimal_periods(self, conditions_data: pd.DataFrame) -> Dict:
        """Analyze and summarize optimal condition periods"""
        
        if conditions_data.empty:
            return {
                'total_periods': 0,
                'by_quality': {},
                'peak_times': [],
                'daily_summary': {},
                'best_overall': None
            }
        
        # Add quality classifications
        analyzed_conditions = []
        for _, row in conditions_data.iterrows():
            interaction = row['interaction']
            classification = self.classify_conditions(
                row['wind_speed'], 
                row['current_speed'],
                interaction['interaction_type']
            )
            
            analyzed_conditions.append({
                'datetime': row['datetime'],
                'wind_speed': row['wind_speed'],
                'wind_dir': row['wind_dir'],
                'current_speed': row['current_speed'],
                'current_dir': row['current_dir'],
                'quality': classification['quality'],
                'skill_level': classification['skill_level'],
                'recommendation': classification['recommendation'],
                'enhancement': row['enhancement'],
                'flow_direction': row['flow_direction']
            })
        
        df = pd.DataFrame(analyzed_conditions)
        
        # Count by quality
        quality_counts = df['quality'].value_counts().to_dict()
        
        # Find peak times (prime or excellent conditions)
        peak_mask = df['quality'].isin(['prime', 'excellent'])
        peak_times = df[peak_mask].head(5).to_dict('records')
        
        # Daily summary
        df['date'] = df['datetime'].dt.date
        daily_summary = {}
        for date, day_group in df.groupby('date'):
            daily_summary[str(date)] = {
                'total_periods': len(day_group),
                'best_quality': day_group['quality'].value_counts().index[0] if not day_group.empty else 'none',
                'peak_wind': day_group['wind_speed'].max(),
                'avg_current': day_group['current_speed'].mean()
            }
        
        # Best overall condition
        prime_conditions = df[df['quality'] == 'prime']
        if not prime_conditions.empty:
            best_idx = prime_conditions['wind_speed'].idxmax()
            best_overall = df.loc[best_idx].to_dict()
        else:
            excellent_conditions = df[df['quality'] == 'excellent']
            if not excellent_conditions.empty:
                best_idx = excellent_conditions['wind_speed'].idxmax()
                best_overall = df.loc[best_idx].to_dict()
            else:
                best_overall = None
        
        return {
            'total_periods': len(df),
            'by_quality': quality_counts,
            'peak_times': peak_times,
            'daily_summary': daily_summary,
            'best_overall': best_overall,
            'all_conditions': df.to_dict('records')
        }

class ReportGenerator:
    """Generates formatted reports for optimal conditions"""
    
    def __init__(self):
        self.quality_emojis = {
            'prime': '🔥',
            'excellent': '⭐', 
            'good': '✅',
            'marginal': '📝'
        }
    
    def generate_summary_report(self, analysis: Dict, start_date: datetime, 
                               end_date: datetime) -> str:
        """Generate a comprehensive summary report"""
        
        if analysis['total_periods'] == 0:
            return self._generate_no_conditions_report(start_date, end_date)
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("🌊 OPTIMAL LUMP CONDITIONS - North Shore Oahu")
        report_lines.append("Foilers ride: Turtle Bay → Sunset Beach (westward WITH wind)")
        report_lines.append("Best lumps: Eastward current OPPOSING ENE trade winds")
        report_lines.append(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        report_lines.append("=" * 80)
        
        # Daily breakdown
        conditions = analysis['all_conditions']
        if conditions:
            current_date = None
            day_count = 0
            
            for condition in conditions:
                dt = condition['datetime']
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt)
                
                # Print day header if new day
                if current_date != dt.date():
                    current_date = dt.date()
                    day_count += 1
                    report_lines.append(f"\n📅 Day {day_count}: {dt.strftime('%A, %B %d, %Y')}")
                    report_lines.append("-" * 60)
                
                # Format condition entry
                emoji = self.quality_emojis.get(condition['quality'], '❓')
                quality_label = condition['quality'].upper()
                
                report_lines.append(
                    f"{dt.strftime('%I:%M %p')} | "
                    f"{emoji} {quality_label} | "
                    f"Wind: {condition['wind_speed']:.1f}kt@{condition['wind_dir']:03.0f}° | "
                    f"Current: {condition['current_speed']:.1f}kt@{condition['current_dir']:03.0f}° | "
                    f"Enhancement: {condition['enhancement']}"
                )
        
        # Summary statistics
        report_lines.append(f"\n{'=' * 80}")
        report_lines.append("📊 SUMMARY STATISTICS")
        report_lines.append("=" * 80)
        report_lines.append(f"Total optimal periods found: {analysis['total_periods']}")
        
        for quality, count in analysis['by_quality'].items():
            emoji = self.quality_emojis.get(quality, '❓')
            report_lines.append(f"{emoji} {quality.capitalize()} conditions: {count}")
        
        # Peak recommendations
        if analysis['peak_times']:
            report_lines.append(f"\n🎯 TOP RECOMMENDATIONS:")
            for i, peak in enumerate(analysis['peak_times'][:3], 1):
                dt = peak['datetime']
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt)
                report_lines.append(
                    f"   {i}. {dt.strftime('%a %m/%d at %I:%M %p')} - "
                    f"{peak['wind_speed']:.1f}kt winds, {peak['current_speed']:.1f}kt eastward current"
                )
        
        # Best overall condition
        if analysis['best_overall']:
            best = analysis['best_overall']
            dt = best['datetime']
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt)
            report_lines.append(f"\n🏆 PEAK CONDITION:")
            report_lines.append(f"    {dt.strftime('%A, %B %d at %I:%M %p')}")
            report_lines.append(f"    Wind: {best['wind_speed']:.1f}kt @ {best['wind_dir']:.0f}°")
            report_lines.append(f"    Current: {best['current_speed']:.1f}kt @ {best['current_dir']:.0f}°")
            report_lines.append(f"    {best['recommendation']}")
        
        report_lines.append(f"\n{'=' * 80}")
        report_lines.append("Analysis complete - Data shows eastward tidal current periods")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _generate_no_conditions_report(self, start_date: datetime, end_date: datetime) -> str:
        """Generate report when no optimal conditions found"""
        return f"""
{'=' * 80}
❌ NO OPTIMAL EASTWARD CURRENT CONDITIONS FOUND
Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
{'=' * 80}

No periods found where:
- Current flows eastward (060-120°) from Sunset Beach toward Turtle Bay
- Current flows INTO ENE trade winds (050-070°) 
- Wind speeds are adequate for foiling (15+ knots)

POSSIBLE REASONS:
1. Tidal currents in this area may be primarily north-south oscillating
2. Dominant westward flow from North Equatorial Current
3. Limited eastward flow periods during analyzed timeframe
4. Data sources may not capture localized current variations

RECOMMENDATIONS:
1. Expand analysis to longer time periods (monthly)
2. Research local tidal current charts for North Shore
3. Consider wind-driven surface currents during strong trades
4. Investigate if there are specific moon phases that create eastward flow
5. Look for areas with different tidal dynamics (bays, channels)

{'=' * 80}
"""

    def save_report(self, report: str, filename: str = "lumps_analysis_report.txt") -> str:
        """Save report to file"""
        try:
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""
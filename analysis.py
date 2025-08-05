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
        self.eastward_current_range = (10, 80)  # Coast-parallel eastward flow range
        
    def classify_conditions(self, wind_speed: float, current_speed: float, 
                          interaction_type: str, angle_difference: float,
                          is_eastward: bool, tidal_enhancement: float = 0.0,
                          wind_wave_height: float = 0.0) -> Dict:
        """
        Classify conditions with wind + current + tide + wave scoring
        Total possible: 20 points (5 wind + 5 current + 5 tide + 5 wave)
        
        CRITICAL: Wind is the PRIMARY driver of lumps
        - <12kt wind = minimal lumps regardless of current (marginal only)
        - 12-15kt wind = light lumps (moderate max)
        - 15kt+ wind = proper conditions for scoring
        """
        
        # Wind component (0-5 points)
        if wind_speed >= 22:
            wind_score = 5
        elif wind_speed >= 18:
            wind_score = 4
        elif wind_speed >= 15:
            wind_score = 3
        elif wind_speed >= 12:
            wind_score = 2
        elif wind_speed >= 8:
            wind_score = 1
        else:
            wind_score = 0
        
        # Current component (0-5 points) - ONLY for true eastward flow!
        current_score = 0
        if is_eastward and interaction_type.startswith("current_into_wind"):
            if current_speed >= 0.8:
                current_score = 5
            elif current_speed >= 0.5:
                current_score = 4
            elif current_speed >= 0.3:
                current_score = 3
            elif current_speed >= 0.2:
                current_score = 2
            else:
                current_score = 1
        elif not is_eastward:
            current_score = 0  # No points for non-eastward flow
        
        # Tidal component (0-5 points)
        tidal_score = int(tidal_enhancement * 5)  # Convert 0-1 scale to 0-5 points
        
        # Wind wave height component (0-5 points)
        # Critical for downwind foiling - bigger lumps = better conditions
        if wind_wave_height >= 5:
            wave_score = 5
        elif wind_wave_height >= 4:
            wave_score = 4
        elif wind_wave_height >= 3:
            wave_score = 3
        elif wind_wave_height >= 2:
            wave_score = 2
        elif wind_wave_height >= 1:
            wave_score = 1
        else:
            wave_score = 0
        
        # Total score determines quality (now out of 20)
        total_score = wind_score + current_score + tidal_score + wave_score
        
        # CRITICAL: Wind is the primary requirement for lumps
        # Without sufficient wind (15kt+), downgrade ratings regardless of current/tide
        if wind_speed < 12:
            # Minimal wind = minimal lumps, cap quality at marginal
            quality = "marginal"
            skill_level = "challenging"
        elif wind_speed < 15:
            # Light wind, cap quality at moderate regardless of other factors
            if total_score >= 9:
                quality = "moderate"
                skill_level = "intermediate"
            else:
                quality = "marginal"
                skill_level = "challenging"
        else:
            # Good wind (15kt+), use full scoring system
            if total_score >= 17:
                quality = "prime"
                skill_level = "expert"
            elif total_score >= 13:
                quality = "excellent"
                skill_level = "advanced"
            elif total_score >= 9:
                quality = "good"
                skill_level = "intermediate"
            elif total_score >= 5:
                quality = "moderate"
                skill_level = "beginner"
            else:
                quality = "marginal"
                skill_level = "challenging"
        
        # Enhanced debug info
        debug = f"W{wind_score}+C{current_score}+T{tidal_score}+Wv{wave_score}={total_score}/20"
        
        return {
            'quality': quality,
            'skill_level': skill_level,
            'wind_speed': wind_speed,
            'current_speed': current_speed,
            'wind_wave_height': wind_wave_height,
            'tidal_score': tidal_score,
            'wave_score': wave_score,
            'total_score': total_score,
            'scoring_debug': debug,
            'is_true_eastward': is_eastward,
            'tidal_enhancement': tidal_enhancement,
            'recommendation': self._get_recommendation(quality, skill_level, is_eastward, tidal_score)
        }
    
    def _get_recommendation(self, quality: str, skill_level: str, is_eastward: bool = False, tidal_score: int = 0) -> str:
        """Generate text recommendation based on conditions"""
        eastward_text = " - TRUE eastward current" if is_eastward else ""
        tidal_text = f" - Tidal bonus: {tidal_score}/5" if tidal_score > 0 else ""
        
        recommendations = {
            'prime': f"🔥 PRIME conditions - {skill_level} level - Maximum wave enhancement{eastward_text}{tidal_text}",
            'excellent': f"⭐ EXCELLENT conditions - {skill_level} level - Strong wave enhancement{eastward_text}{tidal_text}", 
            'good': f"✅ GOOD conditions - {skill_level} level - Moderate wave enhancement{eastward_text}{tidal_text}",
            'moderate': f"🌀 MODERATE conditions - {skill_level} level - Light wave enhancement{eastward_text}{tidal_text}",
            'marginal': f"😐 MARGINAL conditions - {skill_level} level - Minimal enhancement{eastward_text}{tidal_text}"
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
            
            # Get tidal enhancement and eastward status from interaction data
            tidal_enhancement = interaction.get('tidal_enhancement', 0.0)
            is_eastward = interaction.get('is_eastward_current', False)
            angle_difference = interaction.get('angle_difference', 0.0)
            
                        # Get wind wave height if available
            wind_wave_height = row.get('wind_wave_height_ft', 0.0)
            
            classification = self.classify_conditions(
                row['wind_speed'], 
                row['current_speed'],
                interaction['interaction_type'],
                angle_difference,
                is_eastward,
                tidal_enhancement,
                wind_wave_height
            )
            
            analyzed_conditions.append({
                'datetime': row['datetime'],
                'wind_speed': row['wind_speed'],
                'wind_dir': row['wind_dir'],
                'wind_source': row.get('wind_source', 'Unknown'),
                'wind_quality': row.get('wind_quality', 'Unknown'),
                'current_speed': row['current_speed'],
                'current_dir': row['current_dir'],
                'current_source': row.get('current_source', 'Unknown'),
                'current_quality': row.get('current_quality', 'Unknown'),
                'quality': classification['quality'],
                'skill_level': classification['skill_level'],
                'recommendation': classification['recommendation'],
                'enhancement': row['enhancement'],
                'flow_direction': row['flow_direction'],
                # Add new tidal and scoring data
                'tidal_phase': interaction.get('tidal_phase', 'N/A'),
                'tidal_enhancement': tidal_enhancement,
                'scoring_debug': classification.get('scoring_debug', 'N/A'),
                'total_score': classification.get('total_score', 0),
                'is_true_eastward': is_eastward
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
            'moderate': '🌀',
            'light': '💨',
            'marginal': '😐'
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
        report_lines.append("")
        report_lines.append("📊 DATA SOURCE LEGEND:")
        report_lines.append("Wind: 📡=Observed (NDBC buoy) | 📈=Official forecast (NOAA CWF) | 🌐=Model forecast (OpenMeteo)")
        report_lines.append("Current: 🌊=Model forecast (PacIOOS) | 🌙=Tidal predictions (NOAA) | 🔬=Simulation")
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
                
                # Add data source quality indicators
                wind_quality_icon = "📡" if condition.get('wind_quality') == 'observed' else "📈" if condition.get('wind_quality') == 'official_forecast' else "🌐" if condition.get('wind_quality') == 'forecast' else "❓"
                current_quality_icon = "🌊" if condition.get('current_quality') == 'model_forecast' else "🌙" if condition.get('current_quality') == 'tidal_predictions' else "🔬" if condition.get('current_quality') == 'simulation' else "❓"
                
                # Get tidal and scoring info
                tidal_phase = condition.get('tidal_phase', 'N/A')
                scoring_debug = condition.get('scoring_debug', 'N/A')
                
                report_lines.append(
                    f"{dt.strftime('%I:%M %p')} | "
                    f"{emoji} {quality_label} | "
                    f"Wind: {wind_quality_icon}{condition['wind_speed']:.1f}kt@{condition['wind_dir']:03.0f}° | "
                    f"Current: {current_quality_icon}{condition['current_speed']:.1f}kt@{condition['current_dir']:03.0f}° | "
                    f"Tide: {tidal_phase} | "
                    f"Score: {scoring_debug}"
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
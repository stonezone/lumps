# Wind Scoring Fix Summary

## Issue Identified
The system was rating 8.7kt wind conditions as "EXCELLENT" because:
- Wind: 1 point (8.7kt)
- Current: 4 points (good opposing current)
- Tide: 5 points (prime tidal phase)
- Total: 10/15 = "excellent" ❌

This is incorrect because **wind is the PRIMARY driver of lumps**. Without sufficient wind (15kt+), there are no significant windswells to enhance, regardless of current conditions.

## Fix Applied
Added wind speed validation to the scoring system:

1. **< 12kt wind**: Quality capped at "marginal"
   - Minimal windswells = minimal lumps
   - Current enhancement irrelevant without base windswells

2. **12-15kt wind**: Quality capped at "moderate"  
   - Light windswells only
   - Some enhancement possible but limited

3. **15kt+ wind**: Full scoring system applies
   - Proper windswells for downwind foiling
   - Current and tide enhance existing windswells

## Result
- 8.7kt wind will now correctly show as "marginal" regardless of current/tide
- Only conditions with 15kt+ wind can achieve "good" or better ratings
- Scoring now reflects reality: no wind = no lumps

## Testing
Run the analysis again and verify:
```bash
python lumps.py --enhanced-currents --start-date 2025-08-06 --days 3 --verbose
```

The 8.7kt conditions should now show as "marginal" instead of "excellent".

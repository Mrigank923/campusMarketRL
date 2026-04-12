"""Enhanced trend model with seasonal awareness and LLM influence."""

from __future__ import annotations

import random
from typing import Optional

try:
    from campus_market_env.enums import TrendTypeEnum
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from enums import TrendTypeEnum


# Indian seasonal mapping
INDIAN_SEASONS = {
    "January": ("Jan-Mar", "winter", [1, 2, 3]),
    "February": ("Jan-Mar", "winter", [1, 2, 3]),
    "March": ("Jan-Mar", "spring", [1, 2, 3]),
    "April": ("Apr-Jun", "summer", [4, 5, 6]),
    "May": ("Apr-Jun", "summer", [4, 5, 6]),
    "June": ("Apr-Jun", "summer", [4, 5, 6]),
    "July": ("Jul-Sep", "monsoon", [7, 8, 9]),
    "August": ("Jul-Sep", "monsoon", [7, 8, 9]),
    "September": ("Jul-Sep", "monsoon", [7, 8, 9]),
    "October": ("Oct-Dec", "autumn", [10, 11, 12]),
    "November": ("Oct-Dec", "autumn", [10, 11, 12]),
    "December": ("Oct-Dec", "autumn", [10, 11, 12]),
}

# Map seasons to base trend types
SEASON_TO_BASE_TREND = {
    "winter": TrendTypeEnum.NORMAL,      # Cold, indoor activities
    "spring": TrendTypeEnum.FESTIVAL,    # Holi festival, fresh start
    "summer": TrendTypeEnum.EXAM,        # Final exams, heat
    "monsoon": TrendTypeEnum.NORMAL,     # Rainy, stable
    "autumn": TrendTypeEnum.HOLIDAY,     # Diwali coming, festive prep
}

# Month numbers for reference
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",
    4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September",
    10: "October", 11: "November", 12: "December",
}

MONTH_TO_NUMBER = {v: k for k, v in MONTH_NAMES.items()}


def get_season_for_month(month: int) -> str:
    """Return the season name for a given month (1-12)."""
    month_name = MONTH_NAMES.get(month, "January")
    _, season, _ = INDIAN_SEASONS[month_name]
    return season


def get_base_seasonal_trend(month: int) -> TrendTypeEnum:
    """Return the base trend for a given month in India."""
    season = get_season_for_month(month)
    return SEASON_TO_BASE_TREND.get(season, TrendTypeEnum.NORMAL)


def get_seasonal_trend(
    start_month: int,
    day: int,
    seed: int | None = None,
) -> TrendTypeEnum:
    """
    Get trend for a given day, starting from a specific month.
    
    This accounts for:
    - Seasonal progression (crossing months/seasons)
    - Deterministic randomness within season
    - Smooth transitions at month boundaries
    
    Args:
        start_month: Starting month (1-12)
        day: Day number in simulation (1-90)
        seed: Random seed for deterministic behavior
    
    Returns:
        TrendTypeEnum for the current day
    """
    # Calculate current month based on day offset
    days_per_month = 30  # Approximate
    month_offset = (day - 1) // days_per_month
    current_month = ((start_month - 1 + month_offset) % 12) + 1
    
    # Get base trend for current month
    base_trend = get_base_seasonal_trend(current_month)
    
    # Add some deterministic variation within the season
    rng = random.Random(((seed or 0) * 131) + (day * 17) + (current_month * 43))
    draw = rng.random()
    
    # Occasionally vary from base trend (20% of the time)
    if draw < 0.2:
        # Alternate between FESTIVAL/HOLIDAY for variety
        alternatives = [TrendTypeEnum.FESTIVAL, TrendTypeEnum.HOLIDAY, TrendTypeEnum.EXAM]
        return random.choice(alternatives)
    
    return base_trend


def adjust_trend_for_llm_performance(
    base_trend: TrendTypeEnum,
    customer_satisfaction: float,
    market_sentiment: float,
    llm_marketing_spend: float,
    llm_prices_high: bool,
    inventory_status: str,  # "low", "balanced", "high"
) -> TrendTypeEnum:
    """
    Adjust the trend based on LLM's performance.
    
    The LLM can shift the trend within reasonable bounds:
    - Strong performance (high satisfaction + marketing) → warmer trend
    - Poor performance (low satisfaction + high prices) → colder trend
    - Balanced → trend stays same
    
    Args:
        base_trend: The seasonal base trend
        customer_satisfaction: Current satisfaction (0.0-1.0)
        market_sentiment: Current market sentiment (0.0-1.0)
        llm_marketing_spend: Total marketing budget spent
        llm_prices_high: Whether prices are too high
        inventory_status: "low", "balanced", or "high"
    
    Returns:
        Adjusted trend, potentially shifted from base
    """
    
    # Calculate trend shift score
    shift_score = 0.0
    
    # Positive factors (make trend warmer/better)
    if customer_satisfaction > 0.85 and llm_marketing_spend > 2000:
        shift_score += 0.3  # Strong performance
    elif customer_satisfaction > 0.7 and llm_marketing_spend > 1000:
        shift_score += 0.15  # Good performance
    
    if market_sentiment > 0.7:
        shift_score += 0.1  # Market responding well
    
    # Negative factors (make trend colder/worse)
    if llm_prices_high and inventory_status == "high":
        shift_score -= 0.2  # Poor pricing strategy
    
    if customer_satisfaction < 0.3:
        shift_score -= 0.15  # Customers unhappy
    
    # Apply shift based on score
    if shift_score > 0.25:
        # Shift trend warmer
        return shift_trend_up(base_trend)
    elif shift_score < -0.15:
        # Shift trend colder
        return shift_trend_down(base_trend)
    else:
        # Keep trend same
        return base_trend


def shift_trend_up(trend: TrendTypeEnum) -> TrendTypeEnum:
    """Shift trend to be more positive/warmer."""
    shift_map = {
        TrendTypeEnum.NORMAL: TrendTypeEnum.FESTIVAL,
        TrendTypeEnum.HOLIDAY: TrendTypeEnum.FESTIVAL,
        TrendTypeEnum.EXAM: TrendTypeEnum.NORMAL,
        TrendTypeEnum.FESTIVAL: TrendTypeEnum.FESTIVAL,  # Already at top
    }
    return shift_map.get(trend, trend)


def shift_trend_down(trend: TrendTypeEnum) -> TrendTypeEnum:
    """Shift trend to be more negative/colder."""
    shift_map = {
        TrendTypeEnum.FESTIVAL: TrendTypeEnum.NORMAL,
        TrendTypeEnum.NORMAL: TrendTypeEnum.EXAM,
        TrendTypeEnum.HOLIDAY: TrendTypeEnum.EXAM,
        TrendTypeEnum.EXAM: TrendTypeEnum.EXAM,  # Already at bottom
    }
    return shift_map.get(trend, trend)


def get_trend_multiplier(trend: TrendTypeEnum) -> float:
    """Return the demand multiplier for a given trend."""
    multipliers = {
        TrendTypeEnum.FESTIVAL: 1.35,   # Best: festivals boost sales
        TrendTypeEnum.NORMAL: 1.0,      # Baseline
        TrendTypeEnum.EXAM: 0.85,       # Exams reduce casual spending
        TrendTypeEnum.HOLIDAY: 1.15,    # Good: holidays encourage shopping
    }
    return multipliers.get(trend, 1.0)


def get_trend_description(trend: TrendTypeEnum) -> str:
    """Get human-readable description of trend."""
    descriptions = {
        TrendTypeEnum.FESTIVAL: "🎉 Festival time - customers excited",
        TrendTypeEnum.NORMAL: "📊 Normal market conditions",
        TrendTypeEnum.EXAM: "📚 Exam season - students focused on studies",
        TrendTypeEnum.HOLIDAY: "🏖️ Holiday season - festive shopping",
    }
    return descriptions.get(trend, "Unknown trend")

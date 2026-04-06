"""Trend and seasonal event modeling for the campus market."""

from __future__ import annotations

import random

from campus_market_env.config import (
    TREND_MULTIPLIER_EXAM,
    TREND_MULTIPLIER_FESTIVAL,
    TREND_MULTIPLIER_HOLIDAY,
    TREND_MULTIPLIER_NORMAL,
)
from campus_market_env.enums import TrendTypeEnum


def get_trend(day: int, quarter: int, seed: int | None = None) -> TrendTypeEnum:
    """Return a deterministic seasonal trend for the given day and quarter."""

    rng = random.Random(((seed or 0) * 131) + (day * 17) + (quarter * 43))
    draw = rng.random()

    if quarter == 1:
        if draw < 0.68:
            return TrendTypeEnum.NORMAL
        if draw < 0.9:
            return TrendTypeEnum.FESTIVAL
        return TrendTypeEnum.EXAM
    if quarter == 2:
        if draw < 0.5:
            return TrendTypeEnum.EXAM
        if draw < 0.85:
            return TrendTypeEnum.NORMAL
        return TrendTypeEnum.FESTIVAL
    if quarter == 3:
        if draw < 0.78:
            return TrendTypeEnum.NORMAL
        if draw < 0.9:
            return TrendTypeEnum.HOLIDAY
        return TrendTypeEnum.EXAM

    if draw < 0.52:
        return TrendTypeEnum.FESTIVAL
    if draw < 0.85:
        return TrendTypeEnum.NORMAL
    return TrendTypeEnum.HOLIDAY


def get_trend_multiplier(trend: TrendTypeEnum) -> float:
    """Return the demand multiplier for a given trend."""

    if trend == TrendTypeEnum.FESTIVAL:
        return TREND_MULTIPLIER_FESTIVAL
    if trend == TrendTypeEnum.EXAM:
        return TREND_MULTIPLIER_EXAM
    if trend == TrendTypeEnum.HOLIDAY:
        return TREND_MULTIPLIER_HOLIDAY
    return TREND_MULTIPLIER_NORMAL

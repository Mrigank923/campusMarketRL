"""Student demand generation for the campus market environment."""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field

from campus_market_env.config import (
    HIGH_BUDGET_RANGE,
    HIGH_CLUSTER_SIZE_RANGE,
    HOLIDAY_MAX_STUDENT_CLUSTERS,
    LOW_BUDGET_RANGE,
    LOW_CLUSTER_SIZE_RANGE,
    MAX_STUDENT_CLUSTERS,
    MID_BUDGET_RANGE,
    MID_CLUSTER_SIZE_RANGE,
    MIN_STUDENT_CLUSTERS,
)
from campus_market_env.enums import ShopTypeEnum, TrendTypeEnum


class StudentCluster(BaseModel):
    """Represents a student segment that may visit the campus market."""

    model_config = ConfigDict(extra="forbid")

    size: int = Field(ge=1)
    avg_budget: float = Field(ge=0.0)
    preference: ShopTypeEnum
    price_sensitivity: float = Field(ge=0.0, le=1.0)
    visit_probability: float = Field(ge=0.0, le=1.0)


def _trend_budget_multiplier(trend: TrendTypeEnum) -> float:
    if trend == TrendTypeEnum.FESTIVAL:
        return 1.18
    if trend == TrendTypeEnum.EXAM:
        return 0.95
    if trend == TrendTypeEnum.HOLIDAY:
        return 1.05
    return 1.0


def _trend_visit_shift(trend: TrendTypeEnum) -> float:
    if trend == TrendTypeEnum.FESTIVAL:
        return 0.12
    if trend == TrendTypeEnum.EXAM:
        return -0.12
    if trend == TrendTypeEnum.HOLIDAY:
        return -0.08
    return 0.0


def generate_student_clusters(
    seed: int,
    day: int,
    trend: TrendTypeEnum,
) -> list[StudentCluster]:
    """Generate deterministic student demand clusters for a given day."""

    rng = random.Random((seed * 97) + (day * 31))
    max_clusters = HOLIDAY_MAX_STUDENT_CLUSTERS if trend == TrendTypeEnum.HOLIDAY else MAX_STUDENT_CLUSTERS
    cluster_count = rng.randint(MIN_STUDENT_CLUSTERS, max_clusters)
    budget_multiplier = _trend_budget_multiplier(trend)
    visit_shift = _trend_visit_shift(trend)
    preferences = list(ShopTypeEnum)

    clusters: list[StudentCluster] = []
    for cluster_index in range(cluster_count):
        band_selector = (cluster_index + day + rng.randint(0, 2)) % 3
        if band_selector == 0:
            budget_range = LOW_BUDGET_RANGE
            size_range = LOW_CLUSTER_SIZE_RANGE
            price_sensitivity = rng.uniform(0.65, 0.95)
            visit_probability = rng.uniform(0.38, 0.6)
        elif band_selector == 1:
            budget_range = MID_BUDGET_RANGE
            size_range = MID_CLUSTER_SIZE_RANGE
            price_sensitivity = rng.uniform(0.35, 0.7)
            visit_probability = rng.uniform(0.3, 0.55)
        else:
            budget_range = HIGH_BUDGET_RANGE
            size_range = HIGH_CLUSTER_SIZE_RANGE
            price_sensitivity = rng.uniform(0.1, 0.45)
            visit_probability = rng.uniform(0.22, 0.48)

        size = rng.randint(*size_range)
        if trend == TrendTypeEnum.FESTIVAL:
            size = int(round(size * 1.12))
        elif trend == TrendTypeEnum.EXAM:
            size = int(round(size * 0.9))
        elif trend == TrendTypeEnum.HOLIDAY:
            size = int(round(size * 0.8))

        preference = preferences[(cluster_index + rng.randint(0, len(preferences) - 1)) % len(preferences)]
        avg_budget = rng.uniform(*budget_range) * budget_multiplier
        adjusted_visit_probability = max(0.05, min(0.95, visit_probability + visit_shift))

        clusters.append(
            StudentCluster(
                size=max(8, size),
                avg_budget=round(avg_budget, 2),
                preference=preference,
                price_sensitivity=round(price_sensitivity, 4),
                visit_probability=round(adjusted_visit_probability, 4),
            ),
        )

    return clusters

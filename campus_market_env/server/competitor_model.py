"""Competitor generation and pressure estimation."""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field

from campus_market_env.config import (
    COMPETITOR_COUNT,
    COMPETITOR_INVENTORY_RANGE,
    COMPETITOR_MARKETING_RANGE,
    COMPETITOR_PRICING_RANGE,
)
from campus_market_env.enums import ShopTypeEnum


class CompetitorShop(BaseModel):
    """Represents a nearby competitor shop."""

    model_config = ConfigDict(extra="forbid")

    shop_type: ShopTypeEnum
    pricing_factor: float = Field(ge=0.8, le=1.2)
    marketing_power: float = Field(ge=0.0, le=1.0)
    inventory_level: float = Field(ge=0.0, le=1.0)


class CompetitorConfig(BaseModel):
    """Configuration for deterministic competitor generation."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    focal_shop_type: ShopTypeEnum
    count: int = Field(default=COMPETITOR_COUNT, ge=2, le=8)


def generate_competitors(config: CompetitorConfig) -> list[CompetitorShop]:
    """Generate a deterministic competitor set."""

    rng = random.Random(config.seed)
    shop_types = list(ShopTypeEnum)
    competitors: list[CompetitorShop] = []

    for index in range(config.count):
        if index == 0:
            shop_type = config.focal_shop_type
        else:
            shop_type = shop_types[rng.randint(0, len(shop_types) - 1)]
        competitors.append(
            CompetitorShop(
                shop_type=shop_type,
                pricing_factor=round(rng.uniform(*COMPETITOR_PRICING_RANGE), 4),
                marketing_power=round(rng.uniform(*COMPETITOR_MARKETING_RANGE), 4),
                inventory_level=round(rng.uniform(*COMPETITOR_INVENTORY_RANGE), 4),
            ),
        )

    return competitors


def compute_competitor_pressure(
    main_shop: ShopTypeEnum,
    competitors: list[CompetitorShop],
) -> float:
    """Compute normalized competitor pressure from nearby shops."""

    if not competitors:
        return 0.0

    total_score = 0.0
    max_score = 0.0
    for competitor in competitors:
        same_type_weight = 1.0 if competitor.shop_type == main_shop else 0.45
        pricing_pressure = max(0.0, 1.05 - competitor.pricing_factor) / 0.25
        marketing_pressure = competitor.marketing_power
        inventory_pressure = competitor.inventory_level

        raw_score = (
            (0.45 * same_type_weight)
            + (0.25 * pricing_pressure)
            + (0.2 * marketing_pressure)
            + (0.1 * inventory_pressure)
        )
        total_score += raw_score
        max_score += 1.0

    return round(max(0.0, min(total_score / max_score, 1.0)), 4)

"""Pure simulation helpers for the campus market environment."""

from __future__ import annotations

import random
from statistics import fmean

from pydantic import BaseModel, ConfigDict, Field

from campus_market_env.config import (
    AUTO_RESTOCK_TARGET_LEVEL,
    AUTO_RESTOCK_UNIT_COST,
    ACTIVE_TRAFFIC_MULTIPLIER,
    BASE_PRICE,
    CLOSING_TRAFFIC_MULTIPLIER,
    COMPETITOR_DISCOUNT_PRESSURE_DELTA,
    DEFAULT_AWARENESS,
    DEFAULT_BUDGET,
    DEFAULT_CUSTOMER_SATISFACTION,
    DEFAULT_INVENTORY_LEVEL,
    DEFAULT_MARKET_SENTIMENT,
    EVENT_COMPETITOR_DISCOUNT_PROBABILITY,
    EVENT_INFLATION_PROBABILITY,
    EVENT_SUPPLY_SHORTAGE_PROBABILITY,
    EXCESS_MARKETING_PENALTY_DIVISOR,
    EXCESS_MARKETING_THRESHOLD,
    INFLATION_PRICE_MULTIPLIER,
    INVENTORY_CAPACITY_UNITS,
    INVENTORY_THRESHOLD,
    MEMORY_WINDOW_DAYS,
    MONTH_LENGTH_DAYS,
    MORNING_TRAFFIC_MULTIPLIER,
    OVERPRICING_PENALTY_MULTIPLIER,
    OVERPRICING_THRESHOLD,
    OVERSTOCK_LEVEL,
    OVERSTOCK_PENALTY_MULTIPLIER,
    QUARTER_LENGTH_DAYS,
    REWARD_CLAMP_MAX,
    REWARD_CLAMP_MIN,
    REWARD_SMOOTHING_WEIGHT,
    STOCKOUT_REWARD_PENALTY,
    SUPPLY_SHORTAGE_INVENTORY_DELTA,
)
from campus_market_env.models import (
    CampusMarketAction,
    CampusMarketObservation,
    CampusMarketState,
)
from campus_market_env.server.competitor_model import (
    CompetitorConfig,
    compute_competitor_pressure as calculate_competitor_pressure,
    generate_competitors,
)
from campus_market_env.server.student_model import StudentCluster, generate_student_clusters
from campus_market_env.server.trend_model import get_trend, get_trend_multiplier
from campus_market_env.enums import PhaseEnum, ShopTypeEnum, TrendTypeEnum

InfoValue = str | int | float | bool

MARKET_SENTIMENT_TREND_WEIGHT = 0.25
MARKET_SENTIMENT_SATISFACTION_WEIGHT = 0.35
MARKET_SENTIMENT_COMPETITOR_WEIGHT = 0.22

AWARENESS_DECAY_FACTOR = 0.93
MARKETING_AWARENESS_DIVISOR = 3_500.0
MAX_MARKETING_AWARENESS_LIFT = 0.18
REVENUE_AWARENESS_DIVISOR = 2_500.0
REVENUE_AWARENESS_WEIGHT = 0.12
SATISFACTION_AWARENESS_WEIGHT = 0.15
TREND_AWARENESS_WEIGHT = 0.12
COMPETITOR_AWARENESS_WEIGHT = 0.08

DEFAULT_PRICE_SENSITIVITY = 0.5
MIN_VISIT_PROBABILITY = 0.01
MAX_VISIT_PROBABILITY = 0.99
FOCUS_MATCH_VISIT_MULTIPLIER = 1.12
FOCUS_MISMATCH_VISIT_MULTIPLIER = 0.93

CONVERSION_BASELINE = 0.24
CONVERSION_TREND_WEIGHT = 0.12
CONVERSION_SATISFACTION_WEIGHT = 0.32
CONVERSION_PRICE_PENALTY_BASE = 0.28
CONVERSION_PRICE_PENALTY_SENSITIVITY_WEIGHT = 0.35
CONVERSION_DISCOUNT_WEIGHT = 0.12

SATISFACTION_DECAY_FACTOR = 0.94
SATISFACTION_CONVERSION_BASELINE = 0.3
SATISFACTION_CONVERSION_WEIGHT = 0.18
SATISFACTION_INVENTORY_BASELINE = 0.5
SATISFACTION_INVENTORY_WEIGHT = 0.1
SATISFACTION_STOCKOUT_PENALTY = 0.28

BASE_PRICE_ADJUSTMENT_WEIGHT = 0.35
COMPETITOR_REWARD_WEIGHT = 2.5
MEMORY_REVENUE_REWARD_WEIGHT = 0.004
MEMORY_SATISFACTION_REWARD_WEIGHT = 6.0


class RandomEventImpact(BaseModel):
    """Impact of deterministic random events for the current day."""

    model_config = ConfigDict(extra="forbid")

    event_name: str = "none"
    base_price_multiplier: float = Field(default=1.0, ge=0.1)
    inventory_delta: float = 0.0
    competitor_pressure_delta: float = 0.0


class StepComputation(BaseModel):
    """Pure engine output for a single environment step."""

    model_config = ConfigDict(extra="forbid")

    observation: CampusMarketObservation
    reward: float
    debug: dict[str, InfoValue]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def average_or_default(values: list[float], default: float) -> float:
    return default if not values else float(fmean(values))


def derive_seed(base_seed: int, offset: int) -> int:
    return (base_seed * 1_009) + (offset * 9_973)


def get_quarter(day: int) -> int:
    return (((day - 1) // QUARTER_LENGTH_DAYS) % 4) + 1


def reset_monthly_budget(
    previous_budget: float,
    day: int,
    phase: str,
) -> float:
    if day > 1 and phase == PhaseEnum.MORNING.value and ((day - 1) % MONTH_LENGTH_DAYS == 0):
        return DEFAULT_BUDGET
    return previous_budget


def compute_market_sentiment(
    trend: TrendTypeEnum,
    competitor_pressure: float,
    recent_satisfaction: float,
) -> float:
    sentiment = (
        DEFAULT_MARKET_SENTIMENT
        + ((get_trend_multiplier(trend) - 1.0) * MARKET_SENTIMENT_TREND_WEIGHT)
        + ((recent_satisfaction - 0.5) * MARKET_SENTIMENT_SATISFACTION_WEIGHT)
        - (competitor_pressure * MARKET_SENTIMENT_COMPETITOR_WEIGHT)
    )
    return round(clamp(sentiment, 0.0, 1.0), 4)


def compute_awareness(
    previous_awareness: float,
    marketing_spend: float,
    recent_revenue: float,
    recent_satisfaction: float,
    competitor_pressure: float,
    trend: TrendTypeEnum,
) -> float:
    marketing_lift = min(marketing_spend / MARKETING_AWARENESS_DIVISOR, MAX_MARKETING_AWARENESS_LIFT)
    revenue_signal = min(recent_revenue / REVENUE_AWARENESS_DIVISOR, 1.0) * REVENUE_AWARENESS_WEIGHT
    satisfaction_signal = recent_satisfaction * SATISFACTION_AWARENESS_WEIGHT
    seasonal_signal = (get_trend_multiplier(trend) - 1.0) * TREND_AWARENESS_WEIGHT
    next_awareness = (
        (previous_awareness * AWARENESS_DECAY_FACTOR)
        + marketing_lift
        + revenue_signal
        + satisfaction_signal
        + seasonal_signal
        - (competitor_pressure * COMPETITOR_AWARENESS_WEIGHT)
    )
    return round(clamp(next_awareness, 0.05, 1.0), 4)


def compute_cluster_price_sensitivity(student_clusters: list[StudentCluster]) -> float:
    total_students = sum(cluster.size for cluster in student_clusters)
    if total_students <= 0:
        return DEFAULT_PRICE_SENSITIVITY
    weighted_sensitivity = sum(cluster.size * cluster.price_sensitivity for cluster in student_clusters)
    return round(clamp(weighted_sensitivity / total_students, 0.0, 1.0), 4)


def adjust_clusters_for_phase(
    student_clusters: list[StudentCluster],
    phase: str,
) -> list[StudentCluster]:
    phase_multiplier = MORNING_TRAFFIC_MULTIPLIER
    if phase == PhaseEnum.ACTIVE.value:
        phase_multiplier = ACTIVE_TRAFFIC_MULTIPLIER
    elif phase == PhaseEnum.CLOSING.value:
        phase_multiplier = CLOSING_TRAFFIC_MULTIPLIER

    adjusted_clusters: list[StudentCluster] = []
    for cluster in student_clusters:
        adjusted_clusters.append(
            cluster.model_copy(
                update={
                    "visit_probability": round(
                        clamp(
                            cluster.visit_probability * phase_multiplier,
                            MIN_VISIT_PROBABILITY,
                            MAX_VISIT_PROBABILITY,
                        ),
                        4,
                    ),
                },
            ),
        )
    return adjusted_clusters


def align_clusters_with_focus(
    student_clusters: list[StudentCluster],
    product_focus: ShopTypeEnum,
) -> list[StudentCluster]:
    aligned_clusters: list[StudentCluster] = []
    for cluster in student_clusters:
        preference_multiplier = (
            FOCUS_MATCH_VISIT_MULTIPLIER
            if cluster.preference == product_focus
            else FOCUS_MISMATCH_VISIT_MULTIPLIER
        )
        aligned_clusters.append(
            cluster.model_copy(
                update={
                    "visit_probability": round(
                        clamp(
                            cluster.visit_probability * preference_multiplier,
                            MIN_VISIT_PROBABILITY,
                            MAX_VISIT_PROBABILITY,
                        ),
                        4,
                    ),
                },
            ),
        )
    return aligned_clusters


def compute_traffic(
    student_clusters: list[StudentCluster],
    awareness: float,
    competitor_pressure: float,
    trend: TrendTypeEnum,
) -> int:
    base_traffic = sum(cluster.size * cluster.visit_probability for cluster in student_clusters)
    traffic = base_traffic * awareness * (1.0 - competitor_pressure) * get_trend_multiplier(trend)
    return max(0, int(round(traffic)))


def compute_conversion(
    price_adjustment: float,
    price_sensitivity: float,
    satisfaction: float,
    trend: TrendTypeEnum,
) -> float:
    trend_effect = (get_trend_multiplier(trend) - 1.0) * CONVERSION_TREND_WEIGHT
    price_penalty = price_adjustment * (
        CONVERSION_PRICE_PENALTY_BASE
        + (price_sensitivity * CONVERSION_PRICE_PENALTY_SENSITIVITY_WEIGHT)
    )
    discount_bonus = max(-price_adjustment, 0.0) * CONVERSION_DISCOUNT_WEIGHT
    conversion = (
        CONVERSION_BASELINE
        + trend_effect
        + (satisfaction * CONVERSION_SATISFACTION_WEIGHT)
        - price_penalty
        + discount_bonus
    )
    return round(clamp(conversion, 0.05, 0.8), 4)


def compute_revenue(
    traffic: int,
    conversion: float,
    base_price: float = BASE_PRICE,
) -> float:
    return round(max(0.0, traffic * conversion * base_price), 2)


def estimate_sales(traffic: int, conversion: float) -> int:
    return max(0, int(round(traffic * conversion)))


def compute_auto_restock_cost(
    current_inventory: float,
    sales: int,
    restock_amount: int,
) -> float:
    current_units = int(round(clamp(current_inventory, 0.0, 1.0) * INVENTORY_CAPACITY_UNITS))
    post_sales_units = max(0, current_units + max(0, restock_amount) - max(0, sales))
    if (post_sales_units / INVENTORY_CAPACITY_UNITS) >= INVENTORY_THRESHOLD:
        return 0.0

    target_units = int(round(AUTO_RESTOCK_TARGET_LEVEL * INVENTORY_CAPACITY_UNITS))
    auto_restock_units = max(0, target_units - post_sales_units)
    return round(auto_restock_units * AUTO_RESTOCK_UNIT_COST, 2)


def update_inventory(
    current_inventory: float,
    sales: int,
    restock_amount: int,
) -> tuple[float, bool]:
    current_units = int(round(clamp(current_inventory, 0.0, 1.0) * INVENTORY_CAPACITY_UNITS))
    available_units = min(INVENTORY_CAPACITY_UNITS, current_units + max(0, restock_amount))
    realized_sales = min(available_units, max(0, sales))
    stockout_flag = realized_sales < max(0, sales)
    remaining_units = max(0, available_units - realized_sales)

    if (remaining_units / INVENTORY_CAPACITY_UNITS) < INVENTORY_THRESHOLD:
        auto_restock_target = int(round(AUTO_RESTOCK_TARGET_LEVEL * INVENTORY_CAPACITY_UNITS))
        remaining_units = min(INVENTORY_CAPACITY_UNITS, max(remaining_units, auto_restock_target))

    return round(remaining_units / INVENTORY_CAPACITY_UNITS, 4), stockout_flag


def compute_satisfaction(
    conversion: float,
    inventory_level: float,
    stockout_flag: bool,
    previous_satisfaction: float,
) -> float:
    conversion_lift = (
        (conversion - SATISFACTION_CONVERSION_BASELINE) * SATISFACTION_CONVERSION_WEIGHT
    )
    inventory_signal = (
        (inventory_level - SATISFACTION_INVENTORY_BASELINE) * SATISFACTION_INVENTORY_WEIGHT
    )
    stockout_penalty = SATISFACTION_STOCKOUT_PENALTY if stockout_flag else 0.0
    satisfaction = (
        (previous_satisfaction * SATISFACTION_DECAY_FACTOR)
        + conversion_lift
        + inventory_signal
        - stockout_penalty
    )
    return round(clamp(satisfaction, 0.0, 1.0), 4)


def apply_random_events(
    state: CampusMarketState,
    seed: int | None = None,
) -> RandomEventImpact:
    if state.current_phase != PhaseEnum.MORNING.value:
        return RandomEventImpact()

    rng = random.Random((seed or 0) + (state.current_day * 211))
    draw = rng.random()
    if draw < EVENT_INFLATION_PROBABILITY:
        return RandomEventImpact(
            event_name="inflation",
            base_price_multiplier=INFLATION_PRICE_MULTIPLIER,
        )
    if draw < (EVENT_INFLATION_PROBABILITY + EVENT_SUPPLY_SHORTAGE_PROBABILITY):
        return RandomEventImpact(
            event_name="supply_shortage",
            inventory_delta=SUPPLY_SHORTAGE_INVENTORY_DELTA,
        )
    if draw < (
        EVENT_INFLATION_PROBABILITY
        + EVENT_SUPPLY_SHORTAGE_PROBABILITY
        + EVENT_COMPETITOR_DISCOUNT_PROBABILITY
    ):
        return RandomEventImpact(
            event_name="competitor_discount",
            competitor_pressure_delta=COMPETITOR_DISCOUNT_PRESSURE_DELTA,
        )
    return RandomEventImpact()


def compute_reward(
    revenue: float,
    satisfaction: float,
    stockout_flag: bool,
    inventory_level: float,
    competitor_pressure: float,
    action: CampusMarketAction,
) -> float:
    overstock_penalty = 0.0
    if inventory_level > OVERSTOCK_LEVEL:
        overstock_penalty = (inventory_level - OVERSTOCK_LEVEL) * OVERSTOCK_PENALTY_MULTIPLIER

    overpricing_penalty = 0.0
    if action.price_adjustment > OVERPRICING_THRESHOLD:
        overpricing_penalty = (
            (action.price_adjustment - OVERPRICING_THRESHOLD) * OVERPRICING_PENALTY_MULTIPLIER
        )

    excess_marketing_penalty = 0.0
    if action.marketing_spend > EXCESS_MARKETING_THRESHOLD:
        excess_marketing_penalty = (
            (action.marketing_spend - EXCESS_MARKETING_THRESHOLD) / EXCESS_MARKETING_PENALTY_DIVISOR
        )

    reward = (
        (revenue * 0.01)
        + (satisfaction * 10.0)
        - (float(stockout_flag) * STOCKOUT_REWARD_PENALTY)
        - overstock_penalty
        - overpricing_penalty
        - excess_marketing_penalty
        - (competitor_pressure * COMPETITOR_REWARD_WEIGHT)
    )
    return round(clamp(reward, REWARD_CLAMP_MIN, REWARD_CLAMP_MAX), 4)


def smooth_reward(
    base_reward: float,
    revenue_memory: list[float],
    satisfaction_memory: list[float],
) -> float:
    memory_revenue_signal = average_or_default(revenue_memory, 0.0) * MEMORY_REVENUE_REWARD_WEIGHT
    memory_satisfaction_signal = average_or_default(
        satisfaction_memory,
        DEFAULT_CUSTOMER_SATISFACTION,
    ) * MEMORY_SATISFACTION_REWARD_WEIGHT
    smoothed = (
        (base_reward * REWARD_SMOOTHING_WEIGHT)
        + ((memory_revenue_signal + memory_satisfaction_signal) * (1.0 - REWARD_SMOOTHING_WEIGHT))
    )
    return round(clamp(smoothed, REWARD_CLAMP_MIN, REWARD_CLAMP_MAX), 4)


def build_initial_observation(
    state: CampusMarketState,
    base_seed: int,
) -> CampusMarketObservation:
    quarter = get_quarter(state.current_day)
    trend = get_trend(state.current_day, quarter, seed=derive_seed(base_seed, 1))
    student_clusters = generate_student_clusters(
        seed=derive_seed(base_seed, 2),
        day=state.current_day,
        trend=trend,
    )
    adjusted_clusters = adjust_clusters_for_phase(student_clusters, state.current_phase)
    competitors = generate_competitors(
        CompetitorConfig(
            seed=derive_seed(base_seed, 3),
            focal_shop_type=ShopTypeEnum.CAFE,
        ),
    )
    competitor_pressure = calculate_competitor_pressure(ShopTypeEnum.CAFE, competitors)
    awareness = compute_awareness(
        previous_awareness=DEFAULT_AWARENESS,
        marketing_spend=0.0,
        recent_revenue=0.0,
        recent_satisfaction=DEFAULT_CUSTOMER_SATISFACTION,
        competitor_pressure=competitor_pressure,
        trend=trend,
    )
    traffic = compute_traffic(
        student_clusters=adjusted_clusters,
        awareness=awareness,
        competitor_pressure=competitor_pressure,
        trend=trend,
    )
    conversion = compute_conversion(
        price_adjustment=0.0,
        price_sensitivity=compute_cluster_price_sensitivity(adjusted_clusters),
        satisfaction=DEFAULT_CUSTOMER_SATISFACTION,
        trend=trend,
    )
    revenue = compute_revenue(traffic=traffic, conversion=conversion, base_price=BASE_PRICE)
    market_sentiment = compute_market_sentiment(
        trend=trend,
        competitor_pressure=competitor_pressure,
        recent_satisfaction=DEFAULT_CUSTOMER_SATISFACTION,
    )
    return CampusMarketObservation(
        day=state.current_day,
        phase=state.current_phase,
        shop_traffic=traffic,
        conversion_rate=conversion,
        revenue=revenue,
        customer_satisfaction=DEFAULT_CUSTOMER_SATISFACTION,
        satisfaction=DEFAULT_CUSTOMER_SATISFACTION,
        inventory_level=DEFAULT_INVENTORY_LEVEL,
        monthly_budget=DEFAULT_BUDGET,
        awareness=awareness,
        market_sentiment=market_sentiment,
        competitor_pressure=competitor_pressure,
        trend_factor=get_trend_multiplier(trend),
    )


def compute_step(
    state: CampusMarketState,
    action: CampusMarketAction,
    previous_observation: CampusMarketObservation,
    base_seed: int,
) -> StepComputation:
    product_focus = ShopTypeEnum(action.product_focus)
    quarter = get_quarter(state.current_day)
    trend = get_trend(
        day=state.current_day,
        quarter=quarter,
        seed=derive_seed(base_seed, 1),
    )
    student_clusters = generate_student_clusters(
        seed=derive_seed(base_seed, 2),
        day=state.current_day,
        trend=trend,
    )
    focused_clusters = align_clusters_with_focus(student_clusters, product_focus)
    phase_clusters = adjust_clusters_for_phase(focused_clusters, state.current_phase)
    competitors = generate_competitors(
        CompetitorConfig(
            seed=derive_seed(base_seed, 3),
            focal_shop_type=product_focus,
        ),
    )
    competitor_pressure = calculate_competitor_pressure(product_focus, competitors)

    recent_revenue = average_or_default(state.last_7_days_revenue, 0.0)
    recent_satisfaction = average_or_default(
        state.last_7_days_satisfaction,
        previous_observation.customer_satisfaction,
    )
    awareness = compute_awareness(
        previous_awareness=previous_observation.awareness,
        marketing_spend=action.marketing_spend,
        recent_revenue=recent_revenue,
        recent_satisfaction=recent_satisfaction,
        competitor_pressure=competitor_pressure,
        trend=trend,
    )
    price_sensitivity = compute_cluster_price_sensitivity(phase_clusters)
    traffic = compute_traffic(
        student_clusters=phase_clusters,
        awareness=awareness,
        competitor_pressure=competitor_pressure,
        trend=trend,
    )
    conversion = compute_conversion(
        price_adjustment=action.price_adjustment,
        price_sensitivity=price_sensitivity,
        satisfaction=previous_observation.customer_satisfaction,
        trend=trend,
    )
    effective_base_price = BASE_PRICE * (1.0 + (action.price_adjustment * BASE_PRICE_ADJUSTMENT_WEIGHT))
    revenue = compute_revenue(
        traffic=traffic,
        conversion=conversion,
        base_price=effective_base_price,
    )
    sales = estimate_sales(traffic=traffic, conversion=conversion)
    auto_restock_cost = compute_auto_restock_cost(
        current_inventory=previous_observation.inventory_level,
        sales=sales,
        restock_amount=action.restock_amount,
    )
    inventory_level, stockout_flag = update_inventory(
        current_inventory=previous_observation.inventory_level,
        sales=sales,
        restock_amount=action.restock_amount,
    )
    satisfaction = compute_satisfaction(
        conversion=conversion,
        inventory_level=inventory_level,
        stockout_flag=stockout_flag,
        previous_satisfaction=previous_observation.customer_satisfaction,
    )
    event = apply_random_events(state=state, seed=derive_seed(base_seed, 4))

    adjusted_competitor_pressure = clamp(
        competitor_pressure + event.competitor_pressure_delta,
        0.0,
        1.0,
    )
    adjusted_inventory_level = clamp(
        inventory_level + event.inventory_delta,
        0.0,
        1.0,
    )
    if event.inventory_delta < 0.0 and adjusted_inventory_level < inventory_level:
        satisfaction = compute_satisfaction(
            conversion=conversion,
            inventory_level=adjusted_inventory_level,
            stockout_flag=stockout_flag or adjusted_inventory_level <= 0.01,
            previous_satisfaction=satisfaction,
        )
        inventory_level = adjusted_inventory_level
        stockout_flag = stockout_flag or inventory_level <= 0.01
    else:
        inventory_level = adjusted_inventory_level

    effective_base_price *= event.base_price_multiplier
    if event.base_price_multiplier != 1.0:
        revenue = compute_revenue(
            traffic=traffic,
            conversion=conversion,
            base_price=effective_base_price,
        )

    budget_start = reset_monthly_budget(
        previous_budget=previous_observation.monthly_budget,
        day=state.current_day,
        phase=state.current_phase,
    )
    monthly_budget = round(
        max(0.0, budget_start - action.marketing_spend - auto_restock_cost),
        2,
    )
    market_sentiment = compute_market_sentiment(
        trend=trend,
        competitor_pressure=adjusted_competitor_pressure,
        recent_satisfaction=recent_satisfaction,
    )

    base_reward = compute_reward(
        revenue=revenue,
        satisfaction=satisfaction,
        stockout_flag=stockout_flag,
        inventory_level=inventory_level,
        competitor_pressure=adjusted_competitor_pressure,
        action=action,
    )
    reward = smooth_reward(
        base_reward=base_reward,
        revenue_memory=(state.last_7_days_revenue + [revenue])[-MEMORY_WINDOW_DAYS:],
        satisfaction_memory=(state.last_7_days_satisfaction + [satisfaction])[-MEMORY_WINDOW_DAYS:],
    )

    observation = CampusMarketObservation(
        day=state.current_day,
        phase=state.current_phase,
        shop_traffic=traffic,
        conversion_rate=conversion,
        revenue=revenue,
        customer_satisfaction=satisfaction,
        satisfaction=satisfaction,
        inventory_level=inventory_level,
        monthly_budget=monthly_budget,
        awareness=awareness,
        market_sentiment=market_sentiment,
        competitor_pressure=adjusted_competitor_pressure,
        trend_factor=get_trend_multiplier(trend),
    )

    debug: dict[str, InfoValue] = {
        "trend": trend.value,
        "quarter": quarter,
        "cluster_count": len(student_clusters),
        "price_sensitivity": price_sensitivity,
        "sales": sales,
        "stockout_flag": stockout_flag,
        "event": event.event_name,
        "event_price_multiplier": round(event.base_price_multiplier, 4),
        "event_inventory_delta": round(event.inventory_delta, 4),
        "auto_restock_cost": auto_restock_cost,
        "effective_base_price": round(effective_base_price, 2),
        "base_reward": base_reward,
    }
    return StepComputation(observation=observation, reward=reward, debug=debug)

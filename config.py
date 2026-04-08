"""Package configuration constants."""

from typing import Final

MAX_DAYS_PER_EPISODE: Final[int] = 90
PHASES_PER_DAY: Final[int] = 3
INVENTORY_THRESHOLD: Final[float] = 0.2
DEFAULT_BUDGET: Final[float] = 10_000.0
DEFAULT_AWARENESS: Final[float] = 0.42
DEFAULT_INVENTORY_LEVEL: Final[float] = 0.72
DEFAULT_CUSTOMER_SATISFACTION: Final[float] = 0.58
DEFAULT_MARKET_SENTIMENT: Final[float] = 0.5
BASE_PRICE: Final[float] = 100.0
MEMORY_WINDOW_DAYS: Final[int] = 7
INVENTORY_CAPACITY_UNITS: Final[int] = 400
AUTO_RESTOCK_TARGET_LEVEL: Final[float] = 0.45
AUTO_RESTOCK_UNIT_COST: Final[float] = 1.8
MONTH_LENGTH_DAYS: Final[int] = 30
QUARTER_LENGTH_DAYS: Final[int] = 23

MIN_STUDENT_CLUSTERS: Final[int] = 3
MAX_STUDENT_CLUSTERS: Final[int] = 6
HOLIDAY_MAX_STUDENT_CLUSTERS: Final[int] = 4
LOW_BUDGET_RANGE: Final[tuple[float, float]] = (70.0, 120.0)
MID_BUDGET_RANGE: Final[tuple[float, float]] = (121.0, 220.0)
HIGH_BUDGET_RANGE: Final[tuple[float, float]] = (221.0, 360.0)
LOW_CLUSTER_SIZE_RANGE: Final[tuple[int, int]] = (18, 45)
MID_CLUSTER_SIZE_RANGE: Final[tuple[int, int]] = (30, 70)
HIGH_CLUSTER_SIZE_RANGE: Final[tuple[int, int]] = (20, 55)

COMPETITOR_COUNT: Final[int] = 4
COMPETITOR_MARKETING_RANGE: Final[tuple[float, float]] = (0.2, 1.0)
COMPETITOR_INVENTORY_RANGE: Final[tuple[float, float]] = (0.45, 1.0)
COMPETITOR_PRICING_RANGE: Final[tuple[float, float]] = (0.8, 1.2)

MORNING_TRAFFIC_MULTIPLIER: Final[float] = 0.78
ACTIVE_TRAFFIC_MULTIPLIER: Final[float] = 1.0
CLOSING_TRAFFIC_MULTIPLIER: Final[float] = 0.68

TREND_MULTIPLIER_NORMAL: Final[float] = 1.0
TREND_MULTIPLIER_FESTIVAL: Final[float] = 1.3
TREND_MULTIPLIER_EXAM: Final[float] = 0.7
TREND_MULTIPLIER_HOLIDAY: Final[float] = 0.5

EVENT_INFLATION_PROBABILITY: Final[float] = 0.03
EVENT_SUPPLY_SHORTAGE_PROBABILITY: Final[float] = 0.03
EVENT_COMPETITOR_DISCOUNT_PROBABILITY: Final[float] = 0.03
INFLATION_PRICE_MULTIPLIER: Final[float] = 1.08
SUPPLY_SHORTAGE_INVENTORY_DELTA: Final[float] = -0.12
COMPETITOR_DISCOUNT_PRESSURE_DELTA: Final[float] = 0.15

OVERSTOCK_LEVEL: Final[float] = 0.8
OVERSTOCK_PENALTY_MULTIPLIER: Final[float] = 15.0
OVERPRICING_THRESHOLD: Final[float] = 0.5
OVERPRICING_PENALTY_MULTIPLIER: Final[float] = 12.0
EXCESS_MARKETING_THRESHOLD: Final[float] = 1_200.0
EXCESS_MARKETING_PENALTY_DIVISOR: Final[float] = 300.0
STOCKOUT_REWARD_PENALTY: Final[float] = 20.0
REWARD_CLAMP_MIN: Final[float] = -50.0
REWARD_CLAMP_MAX: Final[float] = 50.0
REWARD_SMOOTHING_WEIGHT: Final[float] = 0.82

GYM_MARKETING_SPEND_MAX: Final[float] = 2_000.0
GYM_RESTOCK_AMOUNT_MAX: Final[int] = 200
GYM_PRODUCT_FOCUS_COUNT: Final[int] = 4
GYM_REVENUE_MAX: Final[float] = 50_000.0
GYM_OBSERVATION_VECTOR_SIZE: Final[int] = 11

OBSERVATION_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "day",
    "phase_index",
    "shop_traffic",
    "conversion_rate",
    "revenue",
    "customer_satisfaction",
    "inventory_level",
    "monthly_budget",
    "awareness",
    "market_sentiment",
    "competitor_pressure",
)

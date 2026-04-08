

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tasks import ensure_source_package
ensure_source_package()

from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv
from campus_market_env.enums import ShopTypeEnum
from structured_stdout import emit_end, emit_start


TASK_NAME = "medium_adaptive_pricing"
TASK_SEED = 137
TASK_HORIZON_DAYS = 60
STEPS_PER_DAY = 3
TASK_MAX_STEPS = TASK_HORIZON_DAYS * STEPS_PER_DAY  # 180

REVENUE_TARGET = 400_000.0
MIN_AVG_SATISFACTION = 0.48
MAX_STOCKOUT_FRACTION = 0.15
MIN_AVG_REWARD = 3.0


_FOCUS_ROTATION: list[str] = [
    ShopTypeEnum.FOOD.value,
    ShopTypeEnum.CAFE.value,
    ShopTypeEnum.TECH.value,
    ShopTypeEnum.STATIONARY.value,
]


@dataclass
class TaskResult:
    task_name: str = TASK_NAME
    total_steps: int = 0
    cumulative_reward: float = 0.0
    cumulative_revenue: float = 0.0
    avg_satisfaction: float = 0.0
    avg_reward: float = 0.0
    stockout_count: int = 0
    stockout_fraction: float = 0.0
    step_rewards: list[float] = field(default_factory=list)
    step_revenues: list[float] = field(default_factory=list)
    step_satisfactions: list[float] = field(default_factory=list)


def _adaptive_action(obs, step_idx: int) -> CampusMarketAction:

    trend = obs.trend_factor
    pressure = obs.competitor_pressure
    satisfaction = obs.customer_satisfaction
    inventory = obs.inventory_level

    
    if trend < 0.8:
        price_adj = -0.15
    elif trend > 1.1:
        price_adj = 0.10 if satisfaction > 0.50 else 0.0
    else:
        price_adj = -0.05 if pressure > 0.55 else 0.03

    if trend > 1.1:
        marketing = min(500.0, obs.monthly_budget * 0.08)
    elif satisfaction < 0.45:
        marketing = min(350.0, obs.monthly_budget * 0.06)
    else:
        marketing = min(200.0, obs.monthly_budget * 0.03)

    if inventory < 0.35:
        restock = 40
    elif inventory < 0.55:
        restock = 20
    else:
        restock = 5

    quarter_idx = (step_idx // 69) % len(_FOCUS_ROTATION)
    focus = _FOCUS_ROTATION[quarter_idx]

    return CampusMarketAction(
        price_adjustment=round(max(-1.0, min(1.0, price_adj)), 2),
        marketing_spend=round(marketing, 2),
        restock_amount=restock,
        product_focus=focus,
    )


def run(env: CampusMarketEnv | None = None) -> TaskResult:
    """Execute the medium task and return raw metrics."""

    if env is None:
        env = CampusMarketEnv(seed=TASK_SEED)

    obs = env.reset(seed=TASK_SEED)

    result = TaskResult()
    satisfaction_sum = 0.0
    stockout_steps = 0

    for step_idx in range(TASK_MAX_STEPS):
        action = _adaptive_action(obs, step_idx)
        obs = env.step(action)

        result.step_rewards.append(obs.reward)
        result.step_revenues.append(obs.revenue)
        result.step_satisfactions.append(obs.customer_satisfaction)
        result.cumulative_reward += obs.reward
        result.cumulative_revenue += obs.revenue
        satisfaction_sum += obs.customer_satisfaction

        if obs.info.get("stockout_flag", False):
            stockout_steps += 1

        result.total_steps = step_idx + 1

        if obs.done:
            break

    result.avg_satisfaction = (
        satisfaction_sum / result.total_steps if result.total_steps > 0 else 0.0
    )
    result.avg_reward = (
        result.cumulative_reward / result.total_steps if result.total_steps > 0 else 0.0
    )
    result.stockout_count = stockout_steps
    result.stockout_fraction = (
        stockout_steps / result.total_steps if result.total_steps > 0 else 0.0
    )

    return result


GRADING_CRITERIA = {
    "revenue_target": REVENUE_TARGET,
    "min_avg_satisfaction": MIN_AVG_SATISFACTION,
    "max_stockout_fraction": MAX_STOCKOUT_FRACTION,
    "min_avg_reward": MIN_AVG_REWARD,
}


if __name__ == "__main__":
    emit_start(task=TASK_NAME, difficulty="medium", seed=TASK_SEED)
    res = run()
    emit_end(
        task=res.task_name,
        steps=res.total_steps,
        cumulative_reward=round(res.cumulative_reward, 2),
        cumulative_revenue=round(res.cumulative_revenue, 2),
        avg_satisfaction=round(res.avg_satisfaction, 4),
        avg_reward=round(res.avg_reward, 4),
        stockout_fraction=round(res.stockout_fraction, 4),
    )



from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is importable when running as a script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tasks import ensure_source_package
ensure_source_package()

from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv
from campus_market_env.enums import ShopTypeEnum
from structured_stdout import emit_end, emit_start


TASK_NAME = "easy_steady_state"
TASK_SEED = 42
TASK_HORIZON_DAYS = 30
STEPS_PER_DAY = 3
TASK_MAX_STEPS = TASK_HORIZON_DAYS * STEPS_PER_DAY  # 90

REVENUE_TARGET = 150_000.0
MIN_AVG_SATISFACTION = 0.40
MAX_STOCKOUT_FRACTION = 0.25


@dataclass
class TaskResult:

    task_name: str = TASK_NAME
    total_steps: int = 0
    cumulative_reward: float = 0.0
    cumulative_revenue: float = 0.0
    avg_satisfaction: float = 0.0
    stockout_count: int = 0
    stockout_fraction: float = 0.0
    step_rewards: list[float] = field(default_factory=list)
    step_revenues: list[float] = field(default_factory=list)
    step_satisfactions: list[float] = field(default_factory=list)


def _heuristic_action(obs) -> CampusMarketAction:
    price_adj = -0.05 if obs.customer_satisfaction < 0.45 else 0.02
    marketing = min(200.0, obs.monthly_budget * 0.04)
    restock = 15 if obs.inventory_level < 0.5 else 5
    return CampusMarketAction(
        price_adjustment=price_adj,
        marketing_spend=marketing,
        restock_amount=restock,
        product_focus=ShopTypeEnum.FOOD.value,
    )


def run(env: CampusMarketEnv | None = None) -> TaskResult:
    if env is None:
        env = CampusMarketEnv(seed=TASK_SEED)
    obs = env.reset(seed=TASK_SEED)
    result = TaskResult()
    satisfaction_sum = 0.0
    stockout_steps = 0

    for step_idx in range(TASK_MAX_STEPS):
        action = _heuristic_action(obs)
        obs = env.step(action)

        result.step_rewards.append(obs.reward)
        result.step_revenues.append(obs.revenue)
        result.step_satisfactions.append(obs.customer_satisfaction)
        result.cumulative_reward += obs.reward
        result.cumulative_revenue += obs.revenue
        satisfaction_sum += obs.customer_satisfaction

        info = obs.info
        if info.get("stockout_flag", False):
            stockout_steps += 1

        result.total_steps = step_idx + 1

        if obs.done:
            break

    result.avg_satisfaction = (
        satisfaction_sum / result.total_steps if result.total_steps > 0 else 0.0
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
}


if __name__ == "__main__":
    emit_start(task=TASK_NAME, difficulty="easy", seed=TASK_SEED)
    res = run()
    emit_end(
        task=res.task_name,
        steps=res.total_steps,
        cumulative_reward=round(res.cumulative_reward, 2),
        cumulative_revenue=round(res.cumulative_revenue, 2),
        avg_satisfaction=round(res.avg_satisfaction, 4),
        stockout_fraction=round(res.stockout_fraction, 4),
    )

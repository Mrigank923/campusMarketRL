from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv
from campus_market_env.enums import ShopTypeEnum

TASK_NAME = "hard_full_horizon"
TASK_SEED = 7
TASK_HORIZON_DAYS = 90
STEPS_PER_DAY = 3
TASK_MAX_STEPS = TASK_HORIZON_DAYS * STEPS_PER_DAY  # 270

REVENUE_TARGET = 900_000.0
MIN_AVG_SATISFACTION = 0.55
MAX_STOCKOUT_FRACTION = 0.08
MIN_AVG_REWARD = 5.0
MIN_FINAL_BUDGET = 500.0
MIN_FINAL_AWARENESS = 0.50


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
    final_budget: float = 0.0
    final_awareness: float = 0.0
    step_rewards: list[float] = field(default_factory=list)
    step_revenues: list[float] = field(default_factory=list)
    step_satisfactions: list[float] = field(default_factory=list)


def _best_focus(trend_factor: float, step_idx: int) -> str:
    if trend_factor > 1.1:
        return ShopTypeEnum.TECH.value if step_idx % 2 == 0 else ShopTypeEnum.CAFE.value
    if trend_factor < 0.8:
        return ShopTypeEnum.FOOD.value
    cycle = [
        ShopTypeEnum.FOOD.value,
        ShopTypeEnum.CAFE.value,
        ShopTypeEnum.STATIONARY.value,
        ShopTypeEnum.TECH.value,
    ]
    return cycle[(step_idx // 9) % len(cycle)]  # rotate every 3 days


def _sophisticated_action(obs, step_idx: int) -> CampusMarketAction:
    trend = obs.trend_factor
    pressure = obs.competitor_pressure
    satisfaction = obs.customer_satisfaction
    inventory = obs.inventory_level
    budget = obs.monthly_budget
    awareness = obs.awareness

    if satisfaction < 0.45:
        price_adj = -0.20  
    elif trend > 1.1 and satisfaction > 0.55:
        price_adj = 0.12  
    elif pressure > 0.60:
        price_adj = -0.10  
    elif trend < 0.75:
        price_adj = -0.12  
    else:
        price_adj = 0.03  

    budget_fraction = 0.05
    if awareness < 0.45:
        budget_fraction = 0.10  
        budget_fraction = 0.07  
    elif budget < 2000:
        budget_fraction = 0.02  

    marketing = min(600.0, budget * budget_fraction)
    marketing = max(50.0, marketing)


    if inventory < 0.25:
        restock = 60  
    elif inventory < 0.40:
        restock = 35
    elif inventory < 0.60:
        restock = 15
    else:
        restock = 3  

    focus = _best_focus(trend, step_idx)

    return CampusMarketAction(
        price_adjustment=round(max(-1.0, min(1.0, price_adj)), 2),
        marketing_spend=round(marketing, 2),
        restock_amount=restock,
        product_focus=focus,
    )


def run(env: CampusMarketEnv | None = None) -> TaskResult:

    if env is None:
        env = CampusMarketEnv(seed=TASK_SEED)

    obs = env.reset(seed=TASK_SEED)

    result = TaskResult()
    satisfaction_sum = 0.0
    stockout_steps = 0

    for step_idx in range(TASK_MAX_STEPS):
        action = _sophisticated_action(obs, step_idx)
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
    result.final_budget = obs.monthly_budget
    result.final_awareness = obs.awareness

    return result


GRADING_CRITERIA = {
    "revenue_target": REVENUE_TARGET,
    "min_avg_satisfaction": MIN_AVG_SATISFACTION,
    "max_stockout_fraction": MAX_STOCKOUT_FRACTION,
    "min_avg_reward": MIN_AVG_REWARD,
    "min_final_budget": MIN_FINAL_BUDGET,
    "min_final_awareness": MIN_FINAL_AWARENESS,
}


if __name__ == "__main__":
    res = run()
    print(f"Task: {res.task_name}")
    print(f"Steps completed:    {res.total_steps}")
    print(f"Cumulative reward:  {res.cumulative_reward:.2f}")
    print(f"Cumulative revenue: {res.cumulative_revenue:.2f}")
    print(f"Avg satisfaction:   {res.avg_satisfaction:.4f}")
    print(f"Avg reward/step:    {res.avg_reward:.4f}")
    print(f"Stockout fraction:  {res.stockout_fraction:.2%}")
    print(f"Final budget:       {res.final_budget:.2f}")
    print(f"Final awareness:    {res.final_awareness:.4f}")

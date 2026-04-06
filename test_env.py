"""Local smoke test for the campus market environment."""

from __future__ import annotations

import numpy as np

from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv
from campus_market_env.enums import ShopTypeEnum

try:
    from campus_market_env.gym_env import CampusMarketGymEnv
except ModuleNotFoundError:
    CampusMarketGymEnv = None


def main() -> None:
    rng = np.random.default_rng(7)
    shop_types = [shop_type.value for shop_type in ShopTypeEnum]

    env = CampusMarketEnv(seed=7)
    observation = env.reset(seed=7)
    print("reset:", observation.model_dump())

    for step_index in range(10):
        action = CampusMarketAction(
            price_adjustment=float(rng.uniform(-0.3, 0.4)),
            marketing_spend=float(rng.uniform(0.0, 800.0)),
            restock_amount=int(rng.integers(0, 80)),
            product_focus=str(rng.choice(shop_types)),
        )
        observation = env.step(action)
        print(f"step {step_index} observation:", observation.model_dump())
        print(f"step {step_index} reward:", observation.reward)
        print(f"step {step_index} done:", observation.done)
        print(f"step {step_index} info:", observation.info)
        if observation.done:
            break

    if CampusMarketGymEnv is None:
        print("gym wrapper skipped: gymnasium is not installed in this environment")
        return

    gym_env = CampusMarketGymEnv(seed=7)
    gym_observation, gym_info = gym_env.reset(seed=7)
    print("gym reset vector:", gym_observation.tolist())
    print("gym reset info keys:", sorted(gym_info.keys()))
    print("market state keys:", sorted(env.market_state.model_dump().keys()))


if __name__ == "__main__":
    main()

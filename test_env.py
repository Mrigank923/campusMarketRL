"""Local smoke test for the campus market environment."""

from __future__ import annotations

import numpy as np

from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv
from campus_market_env.enums import ShopTypeEnum
from structured_stdout import emit_end, emit_start, emit_step

try:
    from campus_market_env.gym_env import CampusMarketGymEnv
except ModuleNotFoundError:
    CampusMarketGymEnv = None


def main() -> None:
    rng = np.random.default_rng(7)
    shop_types = [shop_type.value for shop_type in ShopTypeEnum]
    emit_start(script="test_env", seed=7)

    env = CampusMarketEnv(seed=7)
    observation = env.reset(seed=7)
    emit_step(kind="reset", observation=observation.model_dump(mode="json"))

    for step_index in range(10):
        action = CampusMarketAction(
            price_adjustment=float(rng.uniform(-0.3, 0.4)),
            marketing_spend=float(rng.uniform(0.0, 800.0)),
            restock_amount=int(rng.integers(0, 80)),
            product_focus=str(rng.choice(shop_types)),
        )
        observation = env.step(action)
        emit_step(
            kind="env_step",
            step=step_index,
            action=action.model_dump(mode="json"),
            observation=observation.model_dump(mode="json"),
            reward=round(observation.reward, 2),
            done=observation.done,
            info=observation.info,
        )
        if observation.done:
            break

    if CampusMarketGymEnv is None:
        emit_end(status="gym_skipped", reason="gymnasium_not_installed")
        return

    gym_env = CampusMarketGymEnv(seed=7)
    gym_observation, gym_info = gym_env.reset(seed=7)
    emit_step(
        kind="gym_reset",
        vector=gym_observation.tolist(),
        info_keys=sorted(gym_info.keys()),
        market_state_keys=sorted(env.market_state.model_dump().keys()),
    )
    emit_end(status="ok", steps=step_index + 1 if "step_index" in locals() else 0)


if __name__ == "__main__":
    main()

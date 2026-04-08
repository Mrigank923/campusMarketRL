"""Gymnasium-compatible wrapper for the campus market environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .config import (
    DEFAULT_BUDGET,
    GYM_MARKETING_SPEND_MAX,
    GYM_OBSERVATION_VECTOR_SIZE,
    GYM_PRODUCT_FOCUS_COUNT,
    GYM_RESTOCK_AMOUNT_MAX,
    GYM_REVENUE_MAX,
    MAX_DAYS_PER_EPISODE,
    OBSERVATION_FEATURE_NAMES,
)
from .models import CampusMarketAction, CampusMarketObservation
from .server.environment import CampusMarketEnv
from .enums import PhaseEnum, ShopTypeEnum

ObservationVector: TypeAlias = NDArray[np.float32]
RawActionMapping: TypeAlias = Mapping[str, int | float | str | NDArray[np.float32]]

PHASE_TO_INDEX: dict[str, float] = {
    PhaseEnum.MORNING.value: 0.0,
    PhaseEnum.ACTIVE.value: 1.0,
    PhaseEnum.CLOSING.value: 2.0,
}
PRODUCT_FOCUS_ORDER: tuple[ShopTypeEnum, ...] = (
    ShopTypeEnum.CAFE,
    ShopTypeEnum.STATIONARY,
    ShopTypeEnum.FOOD,
    ShopTypeEnum.TECH,
)


def observation_to_vector(observation: CampusMarketObservation) -> ObservationVector:
    """Project a structured observation into a fixed Gymnasium feature vector."""

    return np.asarray(
        [
            float(observation.day),
            PHASE_TO_INDEX[observation.phase],
            float(observation.shop_traffic),
            float(observation.conversion_rate),
            float(observation.revenue),
            float(observation.customer_satisfaction),
            float(observation.inventory_level),
            float(observation.monthly_budget),
            float(observation.awareness),
            float(observation.market_sentiment),
            float(observation.competitor_pressure),
        ],
        dtype=np.float32,
    )


class CampusMarketGymEnv(gym.Env[ObservationVector, CampusMarketAction | RawActionMapping]):
    """Gymnasium adapter around the OpenEnv-compatible campus market environment."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._env = CampusMarketEnv(seed=seed)
        self.action_space = spaces.Dict(
            {
                "price_adjustment": spaces.Box(
                    low=np.asarray([-1.0], dtype=np.float32),
                    high=np.asarray([1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "marketing_spend": spaces.Box(
                    low=np.asarray([0.0], dtype=np.float32),
                    high=np.asarray([GYM_MARKETING_SPEND_MAX], dtype=np.float32),
                    dtype=np.float32,
                ),
                "restock_amount": spaces.Box(
                    low=np.asarray([0], dtype=np.int32),
                    high=np.asarray([GYM_RESTOCK_AMOUNT_MAX], dtype=np.int32),
                    dtype=np.int32,
                ),
                "product_focus": spaces.Discrete(GYM_PRODUCT_FOCUS_COUNT),
            },
        )
        self.observation_space = spaces.Box(
            low=np.asarray(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.asarray(
                [
                    float(MAX_DAYS_PER_EPISODE),
                    2.0,
                    10_000.0,
                    1.0,
                    GYM_REVENUE_MAX,
                    1.0,
                    1.0,
                    DEFAULT_BUDGET,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            shape=(GYM_OBSERVATION_VECTOR_SIZE,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, int | float | str] | None = None,
    ) -> tuple[ObservationVector, dict[str, object]]:
        del options
        observation = self._env.reset(seed=seed)
        return observation_to_vector(observation), {
            "observation": observation.model_dump(mode="json"),
            "observation_features": OBSERVATION_FEATURE_NAMES,
            "state": self._env.state.model_dump(mode="json"),
            "market_state": self._env.market_state.model_dump(mode="json"),
        }

    def step(
        self,
        action: CampusMarketAction | RawActionMapping,
    ) -> tuple[ObservationVector, float, bool, bool, dict[str, object]]:
        validated_action = self._coerce_action(action)
        observation = self._env.step(validated_action)
        return observation_to_vector(observation), observation.reward, observation.done, False, {
            "observation": observation.model_dump(mode="json"),
            "observation_features": OBSERVATION_FEATURE_NAMES,
            "state": self._env.state.model_dump(mode="json"),
            "market_state": self._env.market_state.model_dump(mode="json"),
            **observation.info,
        }

    def render(self) -> None:
        """No local renderer is provided for the Gymnasium wrapper."""

    def close(self) -> None:
        """Release environment resources."""

    def _coerce_action(self, action: CampusMarketAction | RawActionMapping) -> CampusMarketAction:
        if isinstance(action, CampusMarketAction):
            return action

        product_focus_value = self._extract_scalar(action["product_focus"])
        product_focus_index = int(round(product_focus_value))
        clamped_index = max(0, min(product_focus_index, len(PRODUCT_FOCUS_ORDER) - 1))
        return CampusMarketAction(
            price_adjustment=float(self._extract_scalar(action["price_adjustment"])),
            marketing_spend=float(max(0.0, self._extract_scalar(action["marketing_spend"]))),
            restock_amount=int(max(0.0, self._extract_scalar(action["restock_amount"]))),
            product_focus=PRODUCT_FOCUS_ORDER[clamped_index].value,
        )

    @staticmethod
    def _extract_scalar(value: int | float | str | NDArray[np.float32]) -> float:
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        if isinstance(value, str):
            raise TypeError("String values are not valid in Gymnasium raw actions.")
        return float(value)

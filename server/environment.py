"""Core environment implementation for the campus market simulation."""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..models import (
    CampusMarketAction,
    CampusMarketObservation,
    CampusMarketSessionState,
    CampusMarketState,
)
from .engine import build_initial_observation, compute_step
from .state_manager import create_initial_state, transition_after_step

InfoValue = str | int | float | bool
InfoDict = dict[str, InfoValue]


class CampusMarketEnv(
    Environment[CampusMarketAction, CampusMarketObservation, CampusMarketState]
):
    """Deterministic campus market environment with session and market state."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._seed = 0 if seed is None else seed
        self._rng = random.Random(self._seed)
        self._state = CampusMarketSessionState(episode_id=str(uuid4()), step_count=0)
        self._market_state = create_initial_state(episode_id=self._state.episode_id)
        self._last_observation: CampusMarketObservation | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        shop_type: str = "cafe",
        **kwargs: Any,
    ) -> CampusMarketObservation:
        """Reset the simulation and return the opening observation.
        
        Args:
            seed: Random seed for reproducibility
            episode_id: Unique episode identifier
            shop_type: Type of shop to simulate ("cafe", "food", "tech", "stationary", etc.)
            **kwargs: Additional arguments (ignored)
        """
        del kwargs

        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)

        self._state = CampusMarketSessionState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._market_state = create_initial_state(episode_id=self._state.episode_id)
        # Set shop type for this episode
        self._market_state.shop_type = shop_type
        initial_observation = build_initial_observation(
            state=self._market_state,
            base_seed=self._next_step_seed(),
        )
        self._last_observation = self._build_observation(
            observation=initial_observation,
            reward=0.0,
            done=False,
            info=self._build_info(
                current_state=self._market_state,
                next_state=self._market_state,
            ),
        )
        return self._last_observation

    def step(
        self,
        action: CampusMarketAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CampusMarketObservation:
        """Execute one market action and return the next observation."""
        del timeout_s, kwargs

        current_state = self.market_state
        if current_state.done:
            raise RuntimeError("Environment episode is complete. Call reset() before step().")
        if self._last_observation is None:
            raise RuntimeError("Environment observation state is unavailable. Call reset().")

        self._state = CampusMarketSessionState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + 1,
        )
        validated_action = CampusMarketAction.model_validate(action)
        step_result = compute_step(
            state=current_state,
            action=validated_action,
            previous_observation=self._last_observation,
            base_seed=self._next_step_seed(),
        )
        next_state = transition_after_step(
            state=current_state,
            revenue=step_result.observation.revenue,
            satisfaction=step_result.observation.customer_satisfaction,
        )
        self._market_state = next_state
        self._last_observation = self._build_observation(
            observation=step_result.observation,
            reward=step_result.reward,
            done=next_state.done,
            info=self._build_info(
                current_state=current_state,
                next_state=next_state,
                extra=step_result.debug,
            ),
        )
        return self._last_observation

    @property
    def state(self) -> CampusMarketState:
        return self._market_state.model_copy(
            update={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "total_steps": self._state.step_count,
            },
        )

    @property
    def market_state(self) -> CampusMarketState:
        return self._market_state

    def _next_step_seed(self) -> int:
        return self._rng.randint(1, 1_000_000_000)

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="campus_market_env",
            description="Campus market reinforcement-learning environment.",
            version="0.1.0",
        )

    def _build_info(
        self,
        current_state: CampusMarketState,
        next_state: CampusMarketState,
        extra: InfoDict | None = None,
    ) -> InfoDict:
        info: InfoDict = {
            "executed_day": current_state.current_day,
            "executed_phase": current_state.current_phase,
            "next_day": next_state.current_day,
            "next_phase": next_state.current_phase,
            "total_steps": next_state.total_steps,
            "done": next_state.done,
        }
        if extra:
            info.update(extra)
        return info

    def _build_observation(
        self,
        observation: CampusMarketObservation,
        reward: float,
        done: bool,
        info: InfoDict,
    ) -> CampusMarketObservation:
        return observation.model_copy(
            update={
                "reward": reward,
                "done": done,
                "info": {
                    "episode_id": self._state.episode_id,
                    **info,
                },
                "metadata": {
                    "step": self._state.step_count,
                    "seed": self._seed,
                    "day": observation.day,
                },
            },
        )

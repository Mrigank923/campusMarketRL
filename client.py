"""OpenEnv client for the campus market environment server."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import CampusMarketAction, CampusMarketObservation, CampusMarketState


class CampusMarketEnvClient(
    EnvClient[CampusMarketAction, CampusMarketObservation, CampusMarketState]
):
    """OpenEnv WebSocket client for the campus market environment server."""

    def _step_payload(self, action: CampusMarketAction) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CampusMarketObservation]:
        obs_data = dict(payload.get("observation", {}))
        if "reward" not in obs_data and "reward" in payload:
            obs_data["reward"] = payload.get("reward")
        if "done" not in obs_data and "done" in payload:
            obs_data["done"] = payload.get("done")

        observation = CampusMarketObservation.model_validate(obs_data)
        reward = payload.get("reward", observation.reward)
        done = payload.get("done", observation.done)

        return StepResult(
            observation=observation,
            reward=float(reward) if reward is not None else None,
            done=bool(done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> CampusMarketState:
        return CampusMarketState.model_validate(payload)

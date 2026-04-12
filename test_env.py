"""Local smoke test for the campus market environment.

Usage:
  python test_env.py                    # Random actions (default)
  python test_env.py --llm              # LLM-powered actions
  python test_env.py --heuristic        # Heuristic fallback
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Optional

import numpy as np
from openai import OpenAI
from pydantic import ValidationError

from campus_market_env.models import CampusMarketAction
from campus_market_env.server.environment import CampusMarketEnv
from campus_market_env.enums import ShopTypeEnum
from structured_stdout import emit_end, emit_start, emit_step

try:
    from campus_market_env.gym_env import CampusMarketGymEnv
except ModuleNotFoundError:
    CampusMarketGymEnv = None


# ============================================================================
# LLM Support
# ============================================================================

def load_api_config() -> tuple[Optional[str], str, str]:
    """Load API configuration from environment."""
    env_file = ".env"
    if os.path.exists(env_file):
        for line in open(env_file).readlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    return api_key, api_base_url, model_name


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a campus shop. Maximize long-term reward.
    Return only valid JSON with: price_adjustment, marketing_spend, restock_amount.
    """
).strip()


def get_llm_action(client: OpenAI, observation: dict, step: int) -> Optional[CampusMarketAction]:
    """Get action from LLM."""
    try:
        prompt = f"""Current state (step {step}):
- Day: {observation['day']}/90
- Satisfaction: {observation['customer_satisfaction']:.2%}
- Inventory: {observation['inventory_level']:.1%}
- Awareness: {observation['awareness']:.2%}
- Budget: ${observation['monthly_budget']:.0f}
- Trend: {observation['trend_factor']:.2f}x

Return JSON only: {{"price_adjustment": float, "marketing_spend": float, "restock_amount": int}}"""

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        raw = completion.choices[0].message.content or ""
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
            return CampusMarketAction(
                price_adjustment=float(parsed.get("price_adjustment", 0.0)),
                marketing_spend=float(parsed.get("marketing_spend", 0.0)),
                restock_amount=int(parsed.get("restock_amount", 0)),
            )
    except Exception as e:
        print(f"  LLM error: {e}")
    return None


def get_heuristic_action(observation: dict) -> CampusMarketAction:
    """Get action from heuristic fallback."""
    restock = 30 if observation["inventory_level"] < 0.45 else 10
    price_adj = -0.1 if observation["customer_satisfaction"] < 0.5 else 0.0
    marketing = 200.0 if observation["awareness"] < 0.5 else 100.0

    return CampusMarketAction(
        price_adjustment=price_adj,
        marketing_spend=marketing,
        restock_amount=restock,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test campus market environment")
    parser.add_argument("--llm", action="store_true", help="Use LLM for decisions")
    parser.add_argument("--heuristic", action="store_true", help="Use heuristic for decisions")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    emit_start(script="test_env", seed=7)

    # Initialize LLM if requested
    client = None
    if args.llm:
        api_key, api_base, model = load_api_config()
        if api_key:
            client = OpenAI(base_url=api_base, api_key=api_key)
            print(f"[LLM] Using {model}", flush=True)
        else:
            print("[LLM] No API key found, falling back to heuristic", flush=True)
            args.heuristic = True

    env = CampusMarketEnv(seed=7)
    observation = env.reset(seed=7)
    emit_step(kind="reset", observation=observation.model_dump(mode="json"))

    for step_index in range(10):
        obs_dict = observation.model_dump(mode="json")

        if args.llm and client:
            action = get_llm_action(client, obs_dict, step_index + 1)
            if action is None:
                action = get_heuristic_action(obs_dict)
        elif args.heuristic:
            action = get_heuristic_action(obs_dict)
        else:
            action = CampusMarketAction(
                price_adjustment=float(rng.uniform(-0.3, 0.4)),
                marketing_spend=float(rng.uniform(0.0, 800.0)),
                restock_amount=int(rng.integers(0, 80)),
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

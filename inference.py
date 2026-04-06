"""Baseline OpenAI-driven evaluator for the campus market environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from campus_market_env.client import CampusMarketEnvClient
from campus_market_env.models import CampusMarketAction, CampusMarketStepResult
from campus_market_env.enums import ShopTypeEnum

SAFE_DEFAULT_ACTION: Final[dict[str, float | int | str]] = {
    "price_adjustment": 0.0,
    "marketing_spend": 0.0,
    "restock_amount": 10,
    "product_focus": ShopTypeEnum.FOOD.value,
}
PRODUCT_FOCUS_MAP: Final[dict[str, str]] = {
    "CAFE": ShopTypeEnum.CAFE.value,
    "FOOD": ShopTypeEnum.FOOD.value,
    "TECH": ShopTypeEnum.TECH.value,
    "STATIONARY": ShopTypeEnum.STATIONARY.value,
}


class LLMActionResponse(BaseModel):
    """Structured action returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    price_adjustment: float
    marketing_spend: float
    restock_amount: int
    product_focus: Literal["CAFE", "FOOD", "TECH", "STATIONARY"]


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def safe_default_action() -> CampusMarketAction:
    return CampusMarketAction.model_validate(SAFE_DEFAULT_ACTION)


def build_prompt(result: CampusMarketStepResult) -> str:
    return f"""
You are controlling a campus shop in an RL environment.

Goal:
Maximize long-term revenue and customer satisfaction.

Observation:
{result.observation.model_dump_json()}

Return action in JSON:
{{
  "price_adjustment": float (-1 to 1),
  "marketing_spend": float,
  "restock_amount": int,
  "product_focus": "CAFE|FOOD|TECH|STATIONARY"
}}
""".strip()


def choose_action(
    client: OpenAI,
    result: CampusMarketStepResult,
    model_name: str,
) -> CampusMarketAction:
    prompt = build_prompt(result)
    try:
        response = client.responses.parse(
            model=model_name,
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": "You are a deterministic RL policy. Output only a valid JSON action.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=LLMActionResponse,
        )
        parsed = response.output_parsed
        if parsed is None:
            return safe_default_action()
        return CampusMarketAction(
            price_adjustment=parsed.price_adjustment,
            marketing_spend=parsed.marketing_spend,
            restock_amount=parsed.restock_amount,
            product_focus=PRODUCT_FOCUS_MAP[parsed.product_focus],
        )
    except (ValidationError, KeyError, ValueError, TypeError):
        return safe_default_action()


def main() -> None:
    load_env_file(Path(".env"))
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("CAMPUS_MARKET_ENV_BASE_URL", "http://localhost:7860/api")
    llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    env = CampusMarketEnvClient(base_url=base_url)

    for task_id in range(3):
        result = env.reset(seed=task_id)
        total_reward = 0.0

        while not result.done:
            action = choose_action(llm_client, result, model_name)
            result = env.step(action)
            total_reward += result.reward

        print(f"Task {task_id}: {total_reward:.4f}")


if __name__ == "__main__":
    main()

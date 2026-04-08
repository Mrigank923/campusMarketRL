"""Sample-style inference script for the Campus Market environment."""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Final, Optional

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from campus_market_env.client import CampusMarketEnvClient
from campus_market_env.config import MAX_DAYS_PER_EPISODE, PHASES_PER_DAY, REWARD_CLAMP_MAX
from campus_market_env.enums import ShopTypeEnum
from campus_market_env.models import CampusMarketAction, CampusMarketObservation


class LLMActionResponse(BaseModel):
    """Structured action returned by the model."""

    model_config = ConfigDict(extra="forbid")

    price_adjustment: float = Field(ge=-1.0, le=1.0)
    marketing_spend: float = Field(ge=0.0)
    restock_amount: int = Field(ge=0)
    product_focus: str

    @field_validator("product_focus")
    @classmethod
    def validate_product_focus(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_PRODUCT_FOCUS:
            raise ValueError(f"product_focus must be one of {VALID_PRODUCT_FOCUS}")
        return normalized


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


load_env_file(Path(".env"))

# Required by the submission format / sample script.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Environment-specific configuration.
ENV_BASE_URL = os.getenv("CAMPUS_MARKET_ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "campus_market_inference")
BENCHMARK = os.getenv("BENCHMARK", "campus_market_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", str(MAX_DAYS_PER_EPISODE * PHASES_PER_DAY)))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

VALID_PRODUCT_FOCUS: Final[tuple[str, ...]] = tuple(shop.value for shop in ShopTypeEnum)
MAX_TOTAL_REWARD: Final[float] = max(1.0, MAX_STEPS * REWARD_CLAMP_MAX)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a campus shop in a reinforcement-learning environment.

    Your goal is to maximize long-term reward while keeping revenue, satisfaction,
    inventory, and awareness healthy across the episode.

    Return exactly one JSON object with this schema:
    {
      "price_adjustment": float,   // between -1.0 and 1.0
      "marketing_spend": float,    // >= 0
      "restock_amount": int,       // >= 0
      "product_focus": "cafe" | "food" | "tech" | "stationary"
    }

    Do not include markdown, explanations, or extra text.
    """
).strip()


def sanitize_inline(value: str | None) -> str:
    if value is None or value == "":
        return "null"
    return " ".join(value.splitlines()).strip() or "null"


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={sanitize_inline(task)} env={sanitize_inline(env)} model={sanitize_inline(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={sanitize_inline(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={sanitize_inline(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def safe_default_action(observation: CampusMarketObservation) -> CampusMarketAction:
    """Fallback heuristic used when the model response is unavailable or invalid."""

    if observation.inventory_level < 0.25:
        restock_amount = 60
    elif observation.inventory_level < 0.45:
        restock_amount = 30
    elif observation.inventory_level < 0.65:
        restock_amount = 12
    else:
        restock_amount = 4

    if observation.customer_satisfaction < 0.45:
        price_adjustment = -0.15
    elif observation.trend_factor > 1.1 and observation.customer_satisfaction > 0.55:
        price_adjustment = 0.10
    elif observation.competitor_pressure > 0.60:
        price_adjustment = -0.08
    else:
        price_adjustment = 0.02

    if observation.awareness < 0.45:
        marketing_spend = min(500.0, observation.monthly_budget * 0.08)
    elif observation.customer_satisfaction < 0.50:
        marketing_spend = min(300.0, observation.monthly_budget * 0.05)
    else:
        marketing_spend = min(180.0, observation.monthly_budget * 0.03)

    if observation.trend_factor > 1.1:
        product_focus = ShopTypeEnum.TECH.value
    elif observation.trend_factor < 0.8:
        product_focus = ShopTypeEnum.FOOD.value
    elif observation.phase == "morning":
        product_focus = ShopTypeEnum.CAFE.value
    else:
        product_focus = ShopTypeEnum.STATIONARY.value

    return CampusMarketAction(
        price_adjustment=round(price_adjustment, 2),
        marketing_spend=round(marketing_spend, 2),
        restock_amount=restock_amount,
        product_focus=product_focus,
    )


def build_user_prompt(
    step: int,
    observation: CampusMarketObservation,
    history: list[str],
) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation:
        {observation.model_dump_json()}

        Previous steps:
        {history_block}

        Choose the next action that best improves long-term score.
        Return JSON only.
        """
    ).strip()


def parse_action_response(raw_text: str) -> LLMActionResponse:
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("empty model response")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("model response did not contain a JSON object") from None
        parsed = json.loads(cleaned[start : end + 1])

    return LLMActionResponse.model_validate(parsed)


def choose_action(
    client: OpenAI | None,
    observation: CampusMarketObservation,
    step: int,
    history: list[str],
) -> tuple[CampusMarketAction, Optional[str]]:
    """Query the model for an action, or fall back to a safe heuristic."""

    if client is None:
        return safe_default_action(observation), "missing HF_TOKEN/API_KEY"

    prompt = build_user_prompt(step=step, observation=observation, history=history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = completion.choices[0].message.content or ""
        parsed = parse_action_response(content)
        action = CampusMarketAction(
            price_adjustment=parsed.price_adjustment,
            marketing_spend=parsed.marketing_spend,
            restock_amount=parsed.restock_amount,
            product_focus=parsed.product_focus,
        )
        return action, None
    except (ValidationError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return safe_default_action(observation), str(exc)
    except Exception as exc:
        return safe_default_action(observation), str(exc)


def action_to_log_string(action: CampusMarketAction) -> str:
    return action.model_dump_json()


async def create_env() -> CampusMarketEnvClient:
    if LOCAL_IMAGE_NAME:
        return await CampusMarketEnvClient.from_docker_image(LOCAL_IMAGE_NAME)

    env = CampusMarketEnvClient(base_url=ENV_BASE_URL)
    await env.connect()
    return env


async def main() -> None:
    env = await create_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, action_error = choose_action(
                client=client,
                observation=observation,
                step=step,
                history=history,
            )

            reward = 0.0
            done = True
            step_error: Optional[str] = None

            try:
                result = await env.step(action)
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                observation = result.observation
            except Exception as exc:
                step_error = str(exc) if action_error is None else f"{action_error}; {exc}"

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=action_to_log_string(action),
                reward=reward,
                done=done,
                error=step_error,
            )

            history.append(
                " | ".join(
                    [
                        f"step={step}",
                        f"action={action_to_log_string(action)}",
                        f"reward={reward:.2f}",
                        f"day={observation.day}",
                        f"phase={observation.phase}",
                        f"satisfaction={observation.customer_satisfaction:.3f}",
                        f"inventory={observation.inventory_level:.3f}",
                        f"budget={observation.monthly_budget:.2f}",
                    ]
                )
            )

            if step_error is not None or done:
                break

        score = max(0.0, min(sum(rewards) / MAX_TOTAL_REWARD, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

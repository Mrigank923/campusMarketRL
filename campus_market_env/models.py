"""Pydantic models for actions, observations, and environment state."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from campus_market_env.enums import PhaseEnum, ShopTypeEnum

InfoValue: TypeAlias = str | int | float | bool


def _enum_values(enum_type: type[PhaseEnum] | type[ShopTypeEnum]) -> set[str]:
    return {member.value for member in enum_type}


class CampusMarketAction(BaseModel):
    """Agent action for the campus market simulation."""

    model_config = ConfigDict(extra="forbid")

    price_adjustment: float = Field(ge=-1.0, le=1.0)
    marketing_spend: float = Field(ge=0.0)
    restock_amount: int = Field(ge=0)
    product_focus: str

    @field_validator("product_focus")
    @classmethod
    def validate_product_focus(cls, value: str) -> str:
        if value not in _enum_values(ShopTypeEnum):
            raise ValueError(
                f"product_focus must be one of: {sorted(_enum_values(ShopTypeEnum))}",
            )
        return value


class CampusMarketObservation(BaseModel):
    """Environment observation returned to the agent."""

    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=1)
    phase: str
    shop_traffic: int = Field(ge=0)
    conversion_rate: float = Field(ge=0.0, le=1.0)
    revenue: float = Field(ge=0.0)
    customer_satisfaction: float = Field(ge=0.0, le=1.0)
    satisfaction: float | None = Field(default=None, ge=0.0, le=1.0)
    inventory_level: float = Field(ge=0.0, le=1.0)
    monthly_budget: float = Field(ge=0.0)
    awareness: float = Field(ge=0.0, le=1.0)
    market_sentiment: float = Field(ge=0.0, le=1.0)
    competitor_pressure: float = Field(ge=0.0, le=1.0)
    trend_factor: float = Field(ge=0.0)
    reward: float = 0.0
    done: bool = False
    info: dict[str, InfoValue] = Field(default_factory=dict)
    metadata: dict[str, InfoValue] = Field(default_factory=dict)

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, value: str) -> str:
        if value not in _enum_values(PhaseEnum):
            raise ValueError(f"phase must be one of: {sorted(_enum_values(PhaseEnum))}")
        return value

    @model_validator(mode="after")
    def sync_satisfaction(self) -> "CampusMarketObservation":
        if self.satisfaction is None:
            self.satisfaction = self.customer_satisfaction
        elif abs(self.satisfaction - self.customer_satisfaction) > 1e-6:
            raise ValueError("satisfaction must match customer_satisfaction.")
        return self


class CampusMarketState(BaseModel):
    """Minimal environment state exposed through the OpenEnv interface."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    current_day: int = Field(ge=1)
    current_phase: str
    total_steps: int = Field(ge=0)
    done: bool
    last_7_days_revenue: list[float] = Field(default_factory=list)
    last_7_days_satisfaction: list[float] = Field(default_factory=list)
    current_day_revenue: float = Field(default=0.0, ge=0.0)
    current_day_satisfaction_total: float = Field(default=0.0, ge=0.0)
    current_day_observation_count: int = Field(default=0, ge=0)

    @field_validator("current_phase")
    @classmethod
    def validate_current_phase(cls, value: str) -> str:
        if value not in _enum_values(PhaseEnum):
            raise ValueError(
                f"current_phase must be one of: {sorted(_enum_values(PhaseEnum))}",
            )
        return value


class CampusMarketSessionState(BaseModel):
    """Process-local session state for the running environment server."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    step_count: int = Field(default=0, ge=0)


class CampusMarketStepResult(BaseModel):
    """Transport-friendly step/reset result for clients and API responses."""

    model_config = ConfigDict(extra="forbid")

    observation: CampusMarketObservation
    reward: float = 0.0
    done: bool = False
    info: dict[str, InfoValue] = Field(default_factory=dict)

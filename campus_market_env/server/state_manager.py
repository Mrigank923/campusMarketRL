"""State transition helpers for the campus market environment."""

from __future__ import annotations

from campus_market_env.config import MAX_DAYS_PER_EPISODE, MEMORY_WINDOW_DAYS, PHASES_PER_DAY
from campus_market_env.models import CampusMarketState
from campus_market_env.enums import PhaseEnum

PHASE_SEQUENCE: tuple[str, ...] = (
    PhaseEnum.MORNING.value,
    PhaseEnum.ACTIVE.value,
    PhaseEnum.CLOSING.value,
)


def create_initial_state(episode_id: str) -> CampusMarketState:
    return CampusMarketState(
        episode_id=episode_id,
        current_day=1,
        current_phase=PhaseEnum.MORNING.value,
        total_steps=0,
        done=False,
        last_7_days_revenue=[],
        last_7_days_satisfaction=[],
        current_day_revenue=0.0,
        current_day_satisfaction_total=0.0,
        current_day_observation_count=0,
    )


def advance_phase(state: CampusMarketState) -> CampusMarketState:
    current_index = PHASE_SEQUENCE.index(state.current_phase)
    if current_index == PHASES_PER_DAY - 1:
        return advance_day(state)

    next_phase = PHASE_SEQUENCE[current_index + 1]
    return state.model_copy(update={"current_phase": next_phase})


def advance_day(state: CampusMarketState) -> CampusMarketState:
    next_day = state.current_day + 1
    return state.model_copy(
        update={
            "current_day": next_day,
            "current_phase": PhaseEnum.MORNING.value,
        },
    )


def is_done(state: CampusMarketState) -> bool:
    return state.total_steps >= (MAX_DAYS_PER_EPISODE * PHASES_PER_DAY)


def _trim_memory(values: list[float]) -> list[float]:
    return values[-MEMORY_WINDOW_DAYS:]


def transition_after_step(
    state: CampusMarketState,
    revenue: float,
    satisfaction: float,
) -> CampusMarketState:
    updated = state.model_copy(
        update={
            "total_steps": state.total_steps + 1,
            "current_day_revenue": round(state.current_day_revenue + revenue, 2),
            "current_day_satisfaction_total": round(
                state.current_day_satisfaction_total + satisfaction,
                4,
            ),
            "current_day_observation_count": state.current_day_observation_count + 1,
        },
    )

    if updated.current_phase == PhaseEnum.CLOSING.value:
        average_satisfaction = (
            updated.current_day_satisfaction_total / updated.current_day_observation_count
            if updated.current_day_observation_count > 0
            else 0.0
        )
        updated = updated.model_copy(
            update={
                "last_7_days_revenue": _trim_memory(
                    updated.last_7_days_revenue + [round(updated.current_day_revenue, 2)],
                ),
                "last_7_days_satisfaction": _trim_memory(
                    updated.last_7_days_satisfaction + [round(average_satisfaction, 4)],
                ),
                "current_day_revenue": 0.0,
                "current_day_satisfaction_total": 0.0,
                "current_day_observation_count": 0,
            },
        )

    if is_done(updated):
        return updated.model_copy(update={"done": True})

    return advance_phase(updated).model_copy(update={"done": False})

# tasks/definitions.py
# Pre-computed campus market task observations for benchmark evaluation.

TASKS = {
    "easy_steady_state": {
        "description": "Maintain steady-state revenue and satisfaction over 30 days",
        "expected_metric": "revenue",
        "steps": 30,  # 10 days * 3 phases
        "criteria": {
            "cumulative_revenue": 75000.0,
            "avg_satisfaction": 0.55,
            "max_stockout_fraction": 0.10,
        },
    },

    "medium_adaptive_pricing": {
        "description": "Adapt pricing to seasonal trends and competitor pressure over 60 days",
        "expected_metric": "revenue + adaptation",
        "steps": 60,  # 20 days * 3 phases
        "criteria": {
            "cumulative_revenue": 180000.0,
            "avg_satisfaction": 0.58,
            "max_stockout_fraction": 0.08,
            "avg_reward": 3.5,
        },
    },

    "hard_full_horizon": {
        "description": "Manage full 90-day episode with budget and awareness constraints",
        "expected_metric": "holistic performance",
        "steps": 90,  # 30 days * 3 phases
        "criteria": {
            "cumulative_revenue": 400000.0,
            "avg_satisfaction": 0.60,
            "max_stockout_fraction": 0.06,
            "avg_reward": 4.0,
            "final_budget": 2000.0,
            "final_awareness": 0.65,
        },
    },

    "adverse_hostile_market": {
        "description": (
            "Survive a hostile 40-day episode with stochastic demand shocks, "
            "supply-chain disruptions, aggressive competitor surges, and "
            "unpredictable trend reversals. The environment actively tries "
            "to destabilise the agent."
        ),
        "expected_metric": "resilience + recovery",
        "steps": 120,  # 40 days × 3 phases
        "criteria": {
            "cumulative_revenue": 320000.0,
            "avg_satisfaction": 0.52,
            "satisfaction_variance": 0.04,      # must keep variance ≤ this
            "max_stockout_fraction": 0.15,
            "recovery_ratio": 0.60,             # fraction of shocks recovered within 3 steps
            "avg_reward": 2.0,
            "final_budget": 800.0,
            "final_awareness": 0.45,
            "competitor_survival_score": 0.50,   # avg(1 - competitor_pressure) over episode
        },
    },
}

TASK_NAMES = list(TASKS.keys())

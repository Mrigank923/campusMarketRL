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
}

TASK_NAMES = list(TASKS.keys())

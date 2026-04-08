---
title: Campus Market RL
sdk: docker
app_port: 7860
---

<div align="center">

# Campus Market RL

### OpenEnv-Compatible Reinforcement Learning Environment for Campus Retail Strategy

**Price smart. Stock wisely. Win the quad.**

[![Built on OpenEnv](https://img.shields.io/badge/Built%20on-OpenEnv-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![FastAPI + Gymnasium + Docker](https://img.shields.io/badge/Runtime-FastAPI%20%2B%20Gymnasium%20%2B%20Docker-009688)](https://fastapi.tiangolo.com/)
[![Inference: OpenAI-Compatible](https://img.shields.io/badge/Inference-OpenAI%20Compatible-orange)](https://github.com/openai/openai-python)
[![Hugging Face Space Ready](https://img.shields.io/badge/HuggingFace-Docker%20Space-yellow)](https://huggingface.co/)

[**Open the Repository**](https://github.com/Mrigank923/campusMarketRL)

</div>

---

## What Is This?

A **Gymnasium-compatible, OpenEnv-compatible RL environment** where a single agent operates a campus market across a 90-day episode. At every step, the agent controls four levers: **pricing**, **marketing spend**, **restocking**, and **product focus**.

The environment simulates realistic retail pressure from three directions at once:

- **Student demand clusters** with different budgets, preferences, and price sensitivity
- **Nearby competitors** that create pricing and marketing pressure
- **Seasonal campus trends** such as festival, exam, and holiday periods

Each action affects traffic, conversion, revenue, inventory, satisfaction, budget, and long-term reward. The runtime is deterministic under a seed, exposes a clean HTTP/WebSocket API through OpenEnv, includes a Gymnasium wrapper for RL workflows, and ships with benchmark tasks plus a grading script for evaluation.

This repo focuses on the environment package, server, wrappers, benchmark tasks, and deployment files. It does **not** include a separate frontend app or a multi-stage training pipeline.

> Built around an OpenEnv-compatible environment package, with local server entrypoints, benchmark tasks, and deployment files included in the repo.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [The Four Action Controls](#the-four-action-controls)
- [Reward Function](#reward-function)
- [Seasonal Trends and Random Events](#seasonal-trends-and-random-events)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Interfaces](#interfaces)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Key Design Decisions](#key-design-decisions)
- [Research Inspirations](#research-inspirations)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Optional: `HF_TOKEN` if you want to run `inference.py` with a hosted model

### Installation

```bash
# Clone
git clone https://github.com/Mrigank923/campusMarketRL.git
cd campusMarketRL

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional Gymnasium support
pip install -e .[gym]

# Optional inference configuration
cp .env.example .env
```

### Run the Environment Server

```bash
# Option A: convenience entrypoint
python main.py

# Option B: package entrypoint
server --port 7860
```

Open:

- `http://localhost:7860/docs`

### Deployment Files

The repo includes deployment-related files:

- `Dockerfile`
- `server/Dockerfile`
- `openenv.yaml`
- `validate-submission.sh`

### Headless Mode

```bash
# Local smoke test
python test_env.py

# Run benchmark tasks and generate a grading report
python tasks/grader.py

# Run the sample LLM-driven inference loop
python inference.py
```

---

## Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                                                                            │
 │   CLIENT LAYER                                                             │
 │                                                                            │
 │   Raw HTTP          OpenEnv Client         Gymnasium Wrapper      LLM Loop  │
 │   curl / requests   CampusMarketEnvClient  CampusMarketGymEnv     inference │
 │                                                                            │
 │   POST /reset       WebSocket session      fixed-size vector      OpenAI-   │
 │   POST /step        compatible client      observations           compatible │
 │   GET /state                                                             │
 │                                                                            │
 ├────────────────────────────────────────────────────────────────────────────┤
 │                                                                            │
 │   SERVICE LAYER  (FastAPI + OpenEnv HTTP server)                           │
 │                                                                            │
 │   main.py / server.app                                                     │
 │   /health   /reset   /step   /state   /schema   /ws                        │
 │   OpenEnv-generated HTTP API and interactive docs                          │
 │                                                                            │
 ├────────────────────────────────────────────────────────────────────────────┤
 │                                                                            │
 │   CORE ENVIRONMENT                                                         │
 │                                                                            │
 │   CampusMarketEnv                                                          │
 │   reset() -> create_initial_state() -> build_initial_observation()         │
 │   step(action) -> compute_step() -> transition_after_step()                │
 │                                                                            │
 ├────────────────────────────────────────────────────────────────────────────┤
 │                                                                            │
 │   SIMULATION ENGINE                                                        │
 │                                                                            │
 │   engine.py            state_manager.py         models.py / enums.py       │
 │   reward shaping       day + phase advances     typed actions + state      │
 │   traffic model        rolling 7-day memory     Pydantic validation        │
 │   inventory logic                                                           │
 │                                                                            │
 ├────────────────────────────────────────────────────────────────────────────┤
 │                                                                            │
 │   MARKET SIGNALS                                                           │
 │                                                                            │
 │   student_model.py      competitor_model.py      trend_model.py            │
 │   student clusters      competitor pressure      seasonal demand           │
 │   budgets + prefs       pricing + inventory      festival/exam/holiday     │
 │                                                                            │
 ├────────────────────────────────────────────────────────────────────────────┤
 │                                                                            │
 │   EVALUATION                                                               │
 │                                                                            │
 │   tasks/task_easy.py   tasks/task_medium.py   tasks/task_hard.py           │
 │   tasks/grader.py -> weighted overall grade + grading_report.txt           │
 │                                                                            │
 └────────────────────────────────────────────────────────────────────────────┘
```

### Environment Step Flow

```text
 Environment State
       │
       ▼
 Read observation {traffic, conversion, revenue, satisfaction, inventory,
 budget, awareness, sentiment, competitor pressure, trend}
       │
       ▼
 Choose action {price_adjustment, marketing_spend, restock_amount, product_focus}
       │
       ▼
 Budget validation and action capping
       │
       ▼
 Market simulation
 student clusters -> competitor pressure -> trend -> traffic -> conversion
       │
       ▼
 Business update
 revenue -> inventory -> auto-restock -> satisfaction
       │
       ▼
 Event injection
 inflation / supply shortage / competitor discount
       │
       ▼
 Reward shaping
 profit + satisfaction delta + inventory progress - penalties
       │
       ▼
 Advance phase/day and update rolling 7-day memory
```

---

## The Four Action Controls

| Control | Type | What It Does | Main Tradeoff |
|---------|------|--------------|---------------|
| `price_adjustment` | `float` in `[-1.0, 1.0]` | Raises or lowers effective price | Higher prices can improve revenue per sale but reduce conversion and add overpricing penalty |
| `marketing_spend` | `float >= 0` | Improves awareness and future traffic | Immediate budget burn versus long-term demand lift |
| `restock_amount` | `int >= 0` | Adds inventory units before sales are realized | Prevents stockouts but can create overstock and budget drain |
| `product_focus` | `cafe \| food \| tech \| stationary` | Reorients the shop toward a category | Better alignment with student preferences and seasonal demand, but can mismatch current traffic |

### Observation Space

Each step returns a structured observation with the business signals the agent must react to:

- `day`, `phase`
- `shop_traffic`, `conversion_rate`, `revenue`
- `customer_satisfaction`, `inventory_level`
- `monthly_budget`, `awareness`, `market_sentiment`
- `competitor_pressure`, `trend_factor`

The Gymnasium wrapper projects this into an 11-feature fixed vector for standard RL tooling.

---

## Reward Function

The reward is **profit-first**, but shaped to discourage brittle strategies such as overpricing, overstocking, or letting service quality collapse.

```text
R = profit_term
  + satisfaction_delta_term
  - inventory_balance_penalty
  - overstock_penalty
  + inventory_progress_reward
  - controllable_stockout_penalty
  - overpricing_penalty
```

### Reward Components

| Component | Description |
|-----------|-------------|
| `profit_term` | Normalized gross profit after marketing, manual restocking, and auto-restocking costs |
| `satisfaction_delta_term` | Rewards improvement in customer satisfaction from the previous step |
| `inventory_balance_penalty` | Penalizes inventory drifting too far from the target operating range |
| `overstock_penalty` | Extra penalty when inventory becomes excessively high |
| `inventory_progress_reward` | Rewards movement back toward healthy inventory levels |
| `controllable_stockout_penalty` | Penalizes stockouts caused by poor policy decisions |
| `overpricing_penalty` | Penalizes aggressive positive price adjustments |

### Concrete Shaping Used in `engine.py`

```text
normalized_profit = (revenue - marketing - manual_restock - auto_restock) / 5000
profit_term = normalized_profit * 8
satisfaction_term = (satisfaction - previous_satisfaction) * 6
reward is clamped to [-20, 20]
```

### Inventory Penalties

- Inventory target level: `0.45`
- Inventory tolerance band: `+/- 0.15`
- Overstock threshold: `0.8`
- Stockout penalty applies only when the shortage is caused by the policy, not by an exogenous supply-shock event

---

## Seasonal Trends and Random Events

The campus market is not stationary. Demand shifts over time, and the environment injects occasional shocks that test policy robustness.

### Seasonal Trend Types

| Trend | Multiplier | Typical Effect |
|-------|------------|----------------|
| `normal` | `1.0x` | Baseline traffic and conversion |
| `festival` | `1.3x` | Higher demand and stronger spending |
| `exam` | `0.7x` | Lower traffic and more cautious purchasing |
| `holiday` | `0.5x` | Reduced campus activity and fewer student clusters |

Trend selection is deterministic from **day + quarter + seed**, so experiments are reproducible.

### Random Event Types

| Event | Probability | Effect |
|-------|-------------|--------|
| `inflation` | `3%` | Raises effective base price via multiplier |
| `supply_shortage` | `3%` | Reduces inventory level unexpectedly |
| `competitor_discount` | `3%` | Temporarily increases competitor pressure |

### Day Structure

| Phase | Traffic Multiplier | Purpose |
|-------|--------------------|---------|
| `morning` | `0.78x` | Lighter opening traffic |
| `active` | `1.00x` | Main shopping window |
| `closing` | `0.68x` | End-of-day slowdown |

---

## Evaluation Pipeline

```text
 ┌──────────────┐     ┌───────────────────┐     ┌──────────────────────┐
 │  Policy /    │────►│  CampusMarketEnv   │────►│  Benchmark Tasks      │
 │  Heuristic   │     │  reset + step      │     │  easy / medium / hard │
 └──────────────┘     └───────────────────┘     └──────────┬───────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────────┐
                                                   │  tasks/grader.py      │
                                                   │  weighted scoring      │
                                                   │  grading_report.txt    │
                                                   └──────────────────────┘
```

### Benchmark Tasks

| Task | Horizon | What It Tests | Main Metrics |
|------|---------|---------------|--------------|
| `easy_steady_state` | 30 days | Basic retail control | Revenue, average satisfaction, stockout fraction |
| `medium_adaptive_pricing` | 60 days | Stronger adaptation to trend and pressure | Revenue, satisfaction, stockouts, average reward |
| `hard_full_horizon` | 90 days | Full-episode budget and awareness management | Revenue, satisfaction, stockouts, reward, final budget, final awareness |

### How It Works

1. `task_easy.py`, `task_medium.py`, and `task_hard.py` run reference policies against the same environment.
2. Each task captures cumulative revenue, satisfaction, reward, and stockout behavior.
3. `tasks/grader.py` normalizes each criterion into a score in `[0, 1]`.
4. The final weighted grade uses `easy=20%`, `medium=30%`, and `hard=50%`.
5. A text report is saved to `tasks/grading_report.txt`.

### Submission Validation

`validate-submission.sh` performs the final OpenEnv submission checks:

1. Pings the deployed Hugging Face Space
2. Attempts a local Docker build
3. Runs `openenv validate`

---

## Interfaces

This project does not ship a dedicated frontend application. The repo exposes the environment through lightweight interfaces that are actually present in the folder.

### Included Access Paths

- Interactive API documentation from the OpenEnv/FastAPI app
- Raw REST endpoints for reset/step/state/schema
- OpenEnv WebSocket client support through `CampusMarketEnvClient`
- Gymnasium wrapper support through `CampusMarketGymEnv`
- A static HTML page asset in `static/index.html`

This keeps the repo lightweight while still making the environment easy to inspect and use locally or in deployment.

---

## Supported Models

The environment itself is **model-agnostic**. It does not require an LLM to run. The only model-related file in this repo is `inference.py`, which can drive the environment with an external **OpenAI-compatible chat model** that returns the required JSON action schema.

| Mode / Model | Backend | Notes |
|--------------|---------|-------|
| Any Hugging Face-hosted chat model | Hugging Face Router | Set the model name through `MODEL_NAME` |
| Any OpenAI-compatible endpoint | Custom `API_BASE_URL` | Usable if the model can follow the JSON-only action schema |
| No model available | Built-in heuristic fallback | `inference.py` falls back to a safe default policy if the API call fails or no token is set |

---

## API Reference

### REST Endpoints (port `7860`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `POST` | `/reset` | Start a new episode and return the initial observation |
| `POST` | `/step` | Apply one action and return the next observation + reward |
| `GET` | `/state` | Current environment state snapshot |
| `GET` | `/schema` | OpenEnv action / observation schema metadata |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `WS /ws` | Persistent OpenEnv session channel |

### Example Step Payload

```json
{
  "action": {
    "price_adjustment": 0.08,
    "marketing_spend": 150.0,
    "restock_amount": 20,
    "product_focus": "food"
  }
}
```

### Example OpenEnv Client Usage

```python
import asyncio

from campus_market_env import CampusMarketAction, CampusMarketEnvClient


async def run() -> None:
    env = CampusMarketEnvClient(base_url="http://localhost:7860")
    await env.connect()
    try:
        result = await env.reset(seed=7)
        result = await env.step(
            CampusMarketAction(
                price_adjustment=0.05,
                marketing_spend=120.0,
                restock_amount=12,
                product_focus="cafe",
            )
        )
        print(result.reward, result.done)
    finally:
        await env.close()


asyncio.run(run())
```

---

## Project Structure

```text
campusMarketRL/
│
├── __init__.py                         # Public package exports
├── client.py                           # OpenEnv WebSocket client
├── config.py                           # Core constants and simulation settings
├── enums.py                            # Phase, shop type, and trend enums
├── gym_env.py                          # Gymnasium wrapper + vector projection
├── inference.py                        # Sample LLM/heuristic inference loop
├── main.py                             # Local server entrypoint
├── models.py                           # Pydantic action, observation, and state models
├── openenv.yaml                        # OpenEnv environment descriptor
├── pyproject.toml                      # Package metadata and dependencies
├── requirements.txt                    # Python runtime dependencies
├── test_env.py                         # Local smoke test
├── validate-submission.sh              # HF Space + Docker + openenv validator
│
├── server/
│   ├── app.py                          # FastAPI/OpenEnv app factory and CLI server
│   ├── environment.py                  # CampusMarketEnv reset/step implementation
│   ├── engine.py                       # Core simulation, reward, inventory, demand logic
│   ├── state_manager.py                # Day/phase transitions and rolling memory
│   ├── student_model.py                # Student cluster demand generation
│   ├── competitor_model.py             # Competitor generation and pressure score
│   ├── trend_model.py                  # Seasonal trend selection and multipliers
│   ├── requirements.txt                # Alternate server dependency list
│   └── Dockerfile                      # Server-focused container build
│
├── tasks/
│   ├── task_easy.py                    # 30-day baseline benchmark
│   ├── task_medium.py                  # 60-day adaptive-pricing benchmark
│   ├── task_hard.py                    # 90-day full-horizon benchmark
│   ├── grader.py                       # Weighted grading and report generator
│   └── grading_report.txt              # Latest generated benchmark report
│
├── static/
│   └── index.html                      # Static HTML page asset
│
└── docs/
    ├── GETTING_STARTED.md              # Onboarding notes
    ├── QUICK_REFERENCE.md              # Endpoints and commands
    └── IMPLEMENTATION_STATUS.md        # Architecture summary
```

---

## Configuration

### Environment Variables

Create a `.env` file at the project root if you want to use `inference.py` with a hosted model:

```bash
HF_TOKEN=your_api_key_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=your-model-id
CAMPUS_MARKET_ENV_BASE_URL=http://localhost:7860
TASK_NAME=campus_market_inference
BENCHMARK=campus_market_env
LOCAL_IMAGE_NAME=campus-market:latest
```

### Market Configuration

Key constants in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_DAYS_PER_EPISODE` | `90` | Maximum episode length |
| `PHASES_PER_DAY` | `3` | `morning`, `active`, `closing` |
| `DEFAULT_BUDGET` | `10000.0` | Monthly operating budget |
| `DEFAULT_AWARENESS` | `0.42` | Initial awareness level |
| `DEFAULT_INVENTORY_LEVEL` | `0.72` | Starting normalized inventory |
| `INVENTORY_CAPACITY_UNITS` | `400` | Capacity used for inventory conversion |
| `INVENTORY_THRESHOLD` | `0.2` | Auto-restock trigger threshold |
| `AUTO_RESTOCK_TARGET_LEVEL` | `0.45` | Auto-restock target level |
| `AUTO_RESTOCK_UNIT_COST` | `1.8` | Cost per inventory unit |
| `EVENT_INFLATION_PROBABILITY` | `0.03` | Inflation event chance |
| `EVENT_SUPPLY_SHORTAGE_PROBABILITY` | `0.03` | Supply shortage event chance |
| `EVENT_COMPETITOR_DISCOUNT_PROBABILITY` | `0.03` | Competitor discount event chance |

### Gymnasium Interface

`CampusMarketGymEnv` exposes:

- A dictionary action space for the four control knobs
- An 11-dimensional observation vector
- Extra reset/step info containing the structured observation and market state

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Deterministic seeded simulation** | Reproducible experiments are critical for evaluation and debugging |
| **Structured action interface** | Four interpretable controls keep the task learnable while still rich |
| **Budget-constrained execution** | The environment clips unaffordable actions instead of allowing impossible spend |
| **Student clusters instead of undifferentiated traffic** | Demand responds to preference, budget, and price sensitivity in a more realistic way |
| **Competitor pressure as a first-class signal** | The agent must adapt to nearby shops, not just static demand |
| **Inventory-aware reward shaping** | Policies are pushed away from both stockouts and bloated inventory |
| **OpenEnv + Gym dual support** | Makes the environment usable for hosted evaluation, custom agents, and standard RL pipelines |
| **Lightweight built-in interfaces** | The repo focuses on API access, an OpenEnv client, and a Gym wrapper instead of a separate app UI |

---

## Research Inspirations

- **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** for the hosted environment interface pattern
- **[Gymnasium](https://gymnasium.farama.org/)** for the vectorized RL wrapper interface
- Retail operations heuristics around **pricing, stock control, and demand seasonality**
- Deterministic simulation design for repeatable evaluation across policy variants

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes
4. Run local checks:
   `python test_env.py`
   `python tasks/grader.py`
5. If relevant, run submission validation:
   `openenv validate`
6. Commit and open a pull request

### Areas We'd Love Help With

- New benchmark tasks with tougher business constraints
- Better reward calibration for long-horizon budget preservation
- Richer random events and campus calendar effects
- Additional observation channels or partial-observability variants
- Baseline RL agents and training scripts
- Better environment inspection or debugging utilities around the existing API-based workflow

---

## License

No license file is currently included in this repository. If you plan to share or reuse it publicly, add a `LICENSE` file first so the usage terms are explicit.

---

<div align="center">

**Built as an OpenEnv-ready campus retail simulation**

*A compact but expressive environment where pricing, inventory, marketing, seasonality, and competition all matter at the same time.*

</div>

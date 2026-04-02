# campus_market_env

`campus_market_env` is a single-repo reinforcement learning project that simulates a campus shop operating inside a small competitive market.

The idea is simple:

- you control one shop on a campus
- students move in and out based on demand, awareness, price, satisfaction, and competition
- your actions change pricing, marketing, restocking, and product focus
- the environment returns observations and reward signals that are suitable for RL training or agent evaluation

This project is also designed to be easy to demo:

- one Docker image
- one container
- one exposed port
- backend and frontend served together
- no separate frontend server in production

## Project idea

The environment models a campus market where one main shop competes with nearby alternatives. The agent tries to balance:

- revenue growth
- customer satisfaction
- inventory health
- marketing efficiency
- competition pressure

The simulation includes:

- student demand clusters
- competitor shops
- seasonal trends and events
- inventory and restocking
- dense reward shaping
- deterministic seeded behavior

The frontend turns the environment into a visual 2D scene so you can watch the market evolve step by step instead of only reading JSON.

## What you can do with it

- run the environment manually from the browser
- use the FastAPI endpoints directly
- connect a Python client
- test deterministic rollouts with seeds
- evaluate a baseline LLM policy with the OpenAI SDK
- use it as a starting point for RL experiments

## Repository layout

```text
repo/
├── campus_market_env/
│   ├── client.py
│   ├── config.py
│   ├── models.py
│   ├── openenv.yaml
│   ├── utils/
│   └── server/
│       ├── app.py
│       ├── engine.py
│       ├── environment.py
│       ├── state_manager.py
│       ├── student_model.py
│       ├── competitor_model.py
│       ├── trend_model.py
│       └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   └── app/
├── baseline/
│   └── inference.py
├── scripts/
│   └── test_env.py
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

## Architecture

This is a single-container architecture.

### Backend

The backend is a FastAPI app that:

- exposes the RL environment API under `/api`
- serves the built frontend static files
- keeps the simulation logic in Python

Important API routes:

- `GET /api/health`
- `POST /api/reset`
- `POST /api/step`
- `GET /api/state`

### Frontend

The frontend is a static-exported Next.js app that:

- calls the backend through relative `/api/...` routes
- runs in the same container as the backend
- shows a visual 2D campus market scene
- lets you reset and step the environment from the browser

### Runtime model

In production:

- Next.js is built once into static files
- FastAPI serves those files
- Uvicorn serves everything from port `8080`

That means:

- no extra frontend process
- no CORS setup needed
- no multiple ports to manage

## How the environment works

At a high level, one environment step does the following:

1. determine the current trend for the day
2. generate student demand clusters
3. generate competitor shops
4. compute competitor pressure
5. compute traffic
6. compute conversion
7. compute revenue
8. update inventory
9. update customer satisfaction
10. apply random seeded market events
11. compute reward
12. update environment state and rolling memory

### Action space

The agent controls:

- `price_adjustment`
- `marketing_spend`
- `restock_amount`
- `product_focus`

### Observation space

The environment returns observations such as:

- current day and phase
- traffic
- conversion rate
- revenue
- customer satisfaction
- inventory level
- awareness
- market sentiment
- competitor pressure
- trend factor

### Determinism

The environment is designed to be reproducible.

If you use the same seed and the same action sequence, you should get the same trajectory.

Seeded behavior controls:

- student generation
- competitor generation
- seasonal trend selection
- random event selection

## Visual frontend

The browser UI is not just a form on top of JSON. It includes:

- a 2D campus ground scene
- a main shop that changes style based on product focus
- competing shops
- animated students entering, wandering, and leaving
- queue and crowd density cues
- live metrics for traffic, awareness, satisfaction, revenue, and competition

This makes the project much easier to demo in a hackathon, class, or product pitch.

## Requirements

To run locally without Docker, you should have:

- Python `>= 3.10`
- Node.js `>= 20`
- npm

To run the recommended production-style setup, you only need:

- Docker

## Environment variables

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

The example file contains:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `CAMPUS_MARKET_ENV_BASE_URL`

Typical values:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
CAMPUS_MARKET_ENV_BASE_URL=http://localhost:8080/api
```

## Quick start with Docker

This is the easiest way to run the full project.

### 1. Build everything

```bash
docker build -t campus-market .
```

### 2. Run everything

```bash
docker run -p 8080:8080 campus-market
```

### 3. Open the app

Open this in your browser:

`http://localhost:8080`

### 4. Check health

```bash
curl http://localhost:8080/api/health
```

Expected output:

```json
{"status":"ok"}
```

## Local development without Docker

If you want to run the Python package directly:

```bash
cp .env.example .env
python -m venv .venv
. .venv/bin/activate
pip install -r campus_market_env/server/requirements.txt
pip install .
uvicorn campus_market_env.server.app:app --reload
```

Then open:

`http://localhost:8080`

If you want to build the frontend locally as well:

```bash
cd frontend
npm install
npm run build
cd ..
```

## Python client example

You can control the environment from Python with the included client.

```python
from campus_market_env.client import CampusMarketEnvClient
from campus_market_env.models import CampusMarketAction
from campus_market_env.utils.enums import ShopTypeEnum

env = CampusMarketEnvClient(base_url="http://localhost:8080/api")

result = env.reset(seed=42)
print(result.observation.model_dump())

result = env.step(
    CampusMarketAction(
        price_adjustment=0.1,
        marketing_spend=250.0,
        restock_amount=40,
        product_focus=ShopTypeEnum.CAFE.value,
    )
)

print(result.reward)
print(result.done)
print(result.info)
```

## API usage

### Reset the environment

```bash
curl -X POST http://localhost:8080/api/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'
```

### Step the environment

```bash
curl -X POST http://localhost:8080/api/step \
  -H "Content-Type: application/json" \
  -d '{
    "price_adjustment": 0.05,
    "marketing_spend": 150,
    "restock_amount": 12,
    "product_focus": "food"
  }'
```

### Read current state

```bash
curl http://localhost:8080/api/state
```

## Baseline agent

The project includes a baseline agent in [baseline/inference.py](/home/zoro/Documents/Meta/campusMarketRL/baseline/inference.py).

What it does:

- loads `.env` automatically if present
- connects to the running environment API
- resets 3 evaluation tasks with different seeds
- sends each observation to an OpenAI model
- expects a structured JSON action back
- validates the action
- falls back to a safe default action if parsing fails

### Run the baseline

First make sure the app is already running.

Then run:

```bash
python baseline/inference.py
```

### Baseline requirements

You need:

- a valid `OPENAI_API_KEY`
- the backend running at `CAMPUS_MARKET_ENV_BASE_URL`

## Local smoke test

To test the environment loop quickly without the frontend:

```bash
python scripts/test_env.py
```

This script:

- resets the environment
- samples random actions
- steps through the environment
- prints observation, reward, and done status

## Why this project is useful

This repo is a good fit if you want:

- a visually understandable RL demo
- a reproducible environment for experimentation
- a hackathon-friendly single-container deployment
- a small but structured full-stack ML systems example

It combines:

- RL environment design
- backend API engineering
- static frontend visualization
- Docker packaging
- baseline model evaluation

## Troubleshooting

### The site opens but I do not see the frontend

Make sure the frontend was built into the Docker image by rebuilding:

```bash
docker build -t campus-market .
```

### The API works but the baseline fails

Check:

- `OPENAI_API_KEY` is set
- the backend is running
- `CAMPUS_MARKET_ENV_BASE_URL` points to `http://localhost:8080/api`

### Docker build is slow

That is expected the first time because Docker downloads:

- Node base image
- Python base image
- npm dependencies
- Python dependencies

Later builds are faster due to layer caching.

### The environment should be deterministic, but results differ

Determinism depends on:

- same reset seed
- same action sequence
- same environment version

If those match, trajectories should also match.

## Main technologies

- Python
- FastAPI
- Pydantic v2
- Next.js
- Docker
- OpenAI Python SDK

## Current status

The project currently provides:

- a runnable RL environment
- a packaged single-container deployment
- a visual browser demo
- API routes
- a Python client
- a baseline evaluator

This makes it a solid foundation for future work such as:

- training real RL policies
- richer crowd simulation
- better charts and dashboards
- persistent experiment logging
- leaderboard or multi-agent scenarios

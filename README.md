---
title: Campus Market Environment
sdk: docker
app_port: 7860
---

# Campus Market Environment

`campus_market_env` is an OpenEnv-compatible reinforcement learning environment for a campus shop simulation.

It now follows the real OpenEnv SDK pattern:

- `CampusMarketEnvClient` is an `openenv.core.EnvClient`
- the server is created with `openenv.core.env_server.create_app(...)`
- clients connect over the standard OpenEnv WebSocket session endpoint
- inference can use `from_docker_image(...)` with the local Docker image

## OpenEnv Endpoints

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

## Project Layout

```text
.
├── __init__.py
├── client.py
├── config.py
├── enums.py
├── gym_env.py
├── models.py
├── openenv.yaml
├── server/
│   ├── app.py
│   ├── competitor_model.py
│   ├── engine.py
│   ├── environment.py
│   ├── state_manager.py
│   ├── student_model.py
│   └── trend_model.py
├── docs/
├── static/
├── Dockerfile
├── main.py
├── requirements.txt
├── run_agent.py
└── test_env.py
```

## Local Run

Build and start the container:

```bash
docker build -t campus-market .
docker run -p 7860:7860 campus-market
```

Then open:

- `http://localhost:7860/docs`

Health check:

```bash
curl http://localhost:7860/health
```

## Hugging Face Spaces

This repository is configured for a Docker Space:

- `README.md` includes the required `sdk: docker` metadata
- `Dockerfile` starts the FastAPI service on port `7860`
- `main.py` respects the `PORT` environment variable, defaulting to `7860`

Push this repository to a Hugging Face Docker Space and the container should start without needing a separate frontend build step.

## OpenEnv Client Usage

```python
import asyncio

from campus_market_env import CampusMarketAction, CampusMarketEnvClient


async def run() -> None:
    env = await CampusMarketEnvClient.from_docker_image("campus-market:latest")
    try:
        result = await env.reset(seed=7)
        result = await env.step(
            CampusMarketAction(
                price_adjustment=0.1,
                marketing_spend=100.0,
                restock_amount=10,
                product_focus="food",
            )
        )
        print(result.reward, result.done)
    finally:
        await env.close()


asyncio.run(run())
```

## Raw HTTP Examples

Reset:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 7}'
```

Step:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "price_adjustment": 0.1,
      "marketing_spend": 100.0,
      "restock_amount": 10,
      "product_focus": "food"
    }
  }'
```

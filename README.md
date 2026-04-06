---
title: Campus Market Environment
sdk: docker
app_port: 7860
---

# Campus Market Environment

`campus_market_env` is a single-container reinforcement learning environment service for a campus shop simulation.

It runs as a FastAPI app with:

- a root landing page at `/`
- interactive docs at `/docs`
- environment endpoints under `/api`

## API

- `GET /api/health`
- `POST /api/reset`
- `POST /api/step`
- `GET /api/state`

## Project Layout

```text
.
├── campus_market_env/
│   ├── client.py
│   ├── config.py
│   ├── enums.py
│   ├── gym_env.py
│   ├── models.py
│   ├── openenv.yaml
│   └── server/
│       ├── app.py
│       ├── competitor_model.py
│       ├── engine.py
│       ├── environment.py
│       ├── state_manager.py
│       ├── student_model.py
│       └── trend_model.py
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

- `http://localhost:7860/`
- `http://localhost:7860/docs`

Health check:

```bash
curl http://localhost:7860/api/health
```

## Hugging Face Spaces

This repository is configured for a Docker Space:

- `README.md` includes the required `sdk: docker` metadata
- `Dockerfile` starts the FastAPI service on port `7860`
- `main.py` respects the `PORT` environment variable, defaulting to `7860`

Push this repository to a Hugging Face Docker Space and the container should start without needing a separate frontend build step.

## Example Requests

Reset:

```bash
curl -X POST http://localhost:7860/api/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 7}'
```

Step:

```bash
curl -X POST http://localhost:7860/api/step \
  -H "Content-Type: application/json" \
  -d '{
    "price_adjustment": 0.1,
    "marketing_spend": 100.0,
    "restock_amount": 10,
    "product_focus": "food"
  }'
```

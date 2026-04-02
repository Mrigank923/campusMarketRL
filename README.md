# campus_market_env

Production-grade, OpenEnv-compatible reinforcement learning environment package for a realistic campus market simulation.

## What’s included

- Typed Pydantic v2 action, observation, and state models
- Deterministic seeded environment with `reset()`, `step()`, and `state`
- Pure simulation engine functions separated from state transitions
- FastAPI server exposing `/health`, `/reset`, `/step`, and `/state`
- Thin HTTP client for interacting with the server
- Baseline LLM-driven evaluation script
- Local randomized smoke test script
- OpenEnv manifest at `campus_market_env/openenv.yaml`

## Quick start

```bash
cp .env.example .env
python -m venv .venv
. .venv/bin/activate
pip install -r campus_market_env/server/requirements.txt
pip install .
uvicorn campus_market_env.server.app:app --reload
```

## Example

```python
from campus_market_env.client import CampusMarketEnvClient
from campus_market_env.models import CampusMarketAction
from campus_market_env.utils.enums import ShopTypeEnum

env = CampusMarketEnvClient(base_url="http://localhost:8080/api")
result = env.reset(seed=42)

result = env.step(
    CampusMarketAction(
        price_adjustment=0.1,
        marketing_spend=250.0,
        restock_amount=40,
        product_focus=ShopTypeEnum.CAFE.value,
    )
)

print(result.observation.model_dump())
print(result.reward, result.done, result.info)
```

### Run everything
```bash
docker build -t campus-market .
docker run -p 8080:8080 campus-market
```

Open: `http://localhost:8080`

### Test
```bash
curl http://localhost:8080/api/health
```

### Run baseline
`.env` is loaded automatically by `baseline/inference.py`.

```bash
python baseline/inference.py
```

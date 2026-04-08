# campus_market_env

`campus_market_env` is the Python package for the Campus Market reinforcement-learning environment.

It includes:

- the transport models for actions, observations, and step results
- an OpenEnv `EnvClient` implementation for `/reset`, `/step`, `/state`, and `/ws`
- the in-process simulation environment
- a Gymnasium wrapper for local RL experiments
- `openenv.yaml` metadata for OpenEnv-style tooling

## OpenEnv Metadata

The package-level `openenv.yaml` defines:

- `CampusMarketEnvClient` as the client entrypoint
- `CampusMarketAction` as the action schema
- `CampusMarketObservation` as the observation schema
- `campus-market:latest` as the default Docker image name

## Action Schema

Actions use this shape:

```json
{
  "price_adjustment": 0.0,
  "marketing_spend": 100.0,
  "restock_amount": 10,
  "product_focus": "food"
}
```

Valid `product_focus` values are:

- `cafe`
- `food`
- `tech`
- `stationary`

## Observation Highlights

Observations expose market state such as:

- `day`
- `phase`
- `shop_traffic`
- `revenue`
- `customer_satisfaction`
- `inventory_level`
- `monthly_budget`
- `awareness`
- `market_sentiment`
- `competitor_pressure`
- `trend_factor`

Each step also includes `reward`, `done`, `info`, and `metadata`.

## Inference Script

The submission-style inference entrypoint lives at the repository root in `inference.py`.

That script uses these environment variables:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`
- `CAMPUS_MARKET_ENV_BASE_URL`
- `TASK_NAME`
- `BENCHMARK`

## Local Usage

Start the API server from the repository root:

```bash
python main.py
```

Then point the client to:

```text
http://localhost:7860
```

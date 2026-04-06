# Implementation Status

## Current architecture

- `campus_market_env/` contains the reusable environment package, simulation logic, HTTP client, and Gym wrapper.
- `campus_market_env/server/` contains the FastAPI server and the core simulation runtime.
- `static/` contains the landing page served from the root path.
- root scripts contain lightweight usage and smoke-test examples.
- `docs/` contains onboarding and quick reference material.

## Runtime model

- FastAPI serves the `/api` endpoints and the static landing page.
- The environment is deterministic under seeded control.
- The full service runs from one Python container in production.

## Supported interfaces

- HTTP environment API
- Gymnasium wrapper
- HTTP client for remote interaction

# Quick Reference

## Important endpoints

- `GET /api/health`
- `POST /api/reset`
- `POST /api/step`
- `GET /api/state`

## Important folders

- `campus_market_env/`: reusable RL environment package
- `campus_market_env/server/`: FastAPI server and simulation runtime
- `static/`: landing page served at `/`
- `docs/`: project documentation
- repo root scripts: `run_agent.py`, `test_env.py`, `inference.py`

## Common commands

```bash
docker build -t campus-market .
docker run -p 7860:7860 campus-market
python main.py
python test_env.py
```

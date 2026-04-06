# Getting Started

## Fastest path

1. Build the container:

```bash
docker build -t campus-market .
```

2. Run the application:

```bash
docker run -p 7860:7860 campus-market
```

3. Open the service:

`http://localhost:7860`

4. Verify the API:

```bash
curl http://localhost:7860/api/health
```

## Local Python setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Useful entrypoints

- `python main.py`
- `python test_env.py`
- `python run_agent.py`

"""OpenEnv FastAPI application exposing the campus market environment."""

from __future__ import annotations

import argparse
import os

import uvicorn
from openenv.core.env_server.http_server import create_app

from campus_market_env.models import CampusMarketAction, CampusMarketObservation
from campus_market_env.server.environment import CampusMarketEnv

app = create_app(
    CampusMarketEnv,
    CampusMarketAction,
    CampusMarketObservation,
    env_name="campus_market_env",
    max_concurrent_envs=1,
)


def run_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """CLI/entry-point wrapper expected by OpenEnv validation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

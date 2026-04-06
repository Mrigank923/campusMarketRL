"""Convenience entrypoint for running the demo server locally."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("campus_market_env.server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

"""FastAPI application exposing the campus market environment."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from campus_market_env.models import (
    CampusMarketAction,
    CampusMarketSessionState,
    CampusMarketState,
    CampusMarketStepResult,
)
from campus_market_env.server.environment import CampusMarketEnv

InfoValue = str | int | float | bool


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None


class StateResponse(BaseModel):
    state: CampusMarketSessionState
    market_state: CampusMarketState


class HealthResponse(BaseModel):
    status: str


class SPAStaticFiles(StaticFiles):
    """Static file handler that falls back to index.html for client-side routes."""

    async def get_response(self, path: str, scope: dict[str, object]) -> FileResponse | object:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            return await super().get_response("index.html", scope)


def create_app(environment: CampusMarketEnv | None = None) -> FastAPI:
    app = FastAPI(
        title="Campus Market Environment",
        version="0.1.0",
        description="Campus market reinforcement learning environment service.",
    )
    api = APIRouter(prefix="/api")
    env = environment or CampusMarketEnv()
    lock = Lock()
    static_dir = Path(__file__).resolve().parents[2] / "static"

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.post("/reset", response_model=CampusMarketStepResult)
    def reset_environment(payload: ResetRequest) -> CampusMarketStepResult:
        with lock:
            observation = env.reset(seed=payload.seed)
        return CampusMarketStepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
            info=observation.info,
        )

    @api.post("/step", response_model=CampusMarketStepResult)
    def step_environment(action: CampusMarketAction) -> CampusMarketStepResult:
        try:
            with lock:
                observation = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CampusMarketStepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
            info=observation.info,
        )

    @api.get("/state", response_model=StateResponse)
    def get_state() -> StateResponse:
        try:
            with lock:
                current_state = env.state
                current_market_state = env.market_state
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StateResponse(state=current_state, market_state=current_market_state)

    app.include_router(api)
    app.mount(
        "/",
        SPAStaticFiles(directory=static_dir, html=True, check_dir=False),
        name="static",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Landing page not found.")
        return FileResponse(index_file)

    return app


app = create_app()

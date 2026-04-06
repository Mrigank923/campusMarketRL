"""HTTP client for the campus market environment server."""

from __future__ import annotations

import json
from urllib import error, request

from campus_market_env.models import CampusMarketAction, CampusMarketStepResult


class CampusMarketEnvClient:
    """Thin HTTP client for the FastAPI environment server."""

    def __init__(self, base_url: str = "http://127.0.0.1:7860/api", timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def reset(self, seed: int | None = None) -> CampusMarketStepResult:
        payload = {"seed": seed}
        response = self._request("POST", "/reset", payload)
        return CampusMarketStepResult.model_validate(response)

    def step(self, action: CampusMarketAction) -> CampusMarketStepResult:
        payload = action.model_dump(mode="json")
        response = self._request("POST", "/step", payload)
        return CampusMarketStepResult.model_validate(response)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, int | float | str | None],
    ) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=f"{self._base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with request.urlopen(http_request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8")
            raise RuntimeError(f"Request failed with status {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Unable to connect to environment server: {exc.reason}") from exc

        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError("Server response was not a JSON object.")
        return parsed

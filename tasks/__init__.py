from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_source_package() -> None:
    if "campus_market_env" in sys.modules:
        return

    try:
        if importlib.util.find_spec("campus_market_env") is not None:
            return
    except ValueError:
        pass

    project_root = Path(__file__).resolve().parent.parent
    init_path = project_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "campus_market_env",
        init_path,
        submodule_search_locations=[str(project_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create package spec for {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["campus_market_env"] = module
    spec.loader.exec_module(module)


ensure_source_package()

# Task registry: List all available graders as strings for platform validators
# The platform checks for at least 3 graders during validation
GRADERS = [
    "grade_easy",
    "grade_medium",
    "grade_hard",
]

__all__ = [
    "task_easy",
    "task_medium",
    "task_hard",
    "GRADERS",
    "ensure_source_package",
]

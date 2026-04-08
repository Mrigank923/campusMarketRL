from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tasks import task_easy, task_medium, task_hard


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))



def _score_at_least(actual: float, target: float) -> float:
    if target <= 0:
        return 1.0
    return clamp01(actual / target)


def _score_at_most(actual: float, limit: float) -> float:
    if limit <= 0:
        return 0.0 if actual > 0 else 1.0
    return clamp01(1.0 - (actual / limit))


@dataclass
class CriterionResult:
    name: str
    actual: float
    target: float
    direction: str  
    score: float = 0.0


@dataclass
class TaskGrade:
    task_name: str
    difficulty: str
    criteria: list[CriterionResult] = field(default_factory=list)
    grade: float = 0.0
    elapsed_seconds: float = 0.0


def grade_easy(result: task_easy.TaskResult) -> TaskGrade:
    criteria = [
        CriterionResult(
            "cumulative_revenue",
            result.cumulative_revenue,
            task_easy.REVENUE_TARGET,
            "at_least",
        ),
        CriterionResult(
            "avg_satisfaction",
            result.avg_satisfaction,
            task_easy.MIN_AVG_SATISFACTION,
            "at_least",
        ),
        CriterionResult(
            "stockout_fraction",
            result.stockout_fraction,
            task_easy.MAX_STOCKOUT_FRACTION,
            "at_most",
        ),
    ]
    for c in criteria:
        c.score = (
            _score_at_least(c.actual, c.target)
            if c.direction == "at_least"
            else _score_at_most(c.actual, c.target)
        )
    grade = sum(c.score for c in criteria) / len(criteria)
    return TaskGrade(
        task_name=result.task_name,
        difficulty="easy",
        criteria=criteria,
        grade=round(grade, 4),
    )


def grade_medium(result: task_medium.TaskResult) -> TaskGrade:
    criteria = [
        CriterionResult(
            "cumulative_revenue",
            result.cumulative_revenue,
            task_medium.REVENUE_TARGET,
            "at_least",
        ),
        CriterionResult(
            "avg_satisfaction",
            result.avg_satisfaction,
            task_medium.MIN_AVG_SATISFACTION,
            "at_least",
        ),
        CriterionResult(
            "stockout_fraction",
            result.stockout_fraction,
            task_medium.MAX_STOCKOUT_FRACTION,
            "at_most",
        ),
        CriterionResult(
            "avg_reward",
            result.avg_reward,
            task_medium.MIN_AVG_REWARD,
            "at_least",
        ),
    ]
    for c in criteria:
        c.score = (
            _score_at_least(c.actual, c.target)
            if c.direction == "at_least"
            else _score_at_most(c.actual, c.target)
        )
    grade = sum(c.score for c in criteria) / len(criteria)
    return TaskGrade(
        task_name=result.task_name,
        difficulty="medium",
        criteria=criteria,
        grade=round(grade, 4),
    )


def grade_hard(result: task_hard.TaskResult) -> TaskGrade:
    criteria = [
        CriterionResult(
            "cumulative_revenue",
            result.cumulative_revenue,
            task_hard.REVENUE_TARGET,
            "at_least",
        ),
        CriterionResult(
            "avg_satisfaction",
            result.avg_satisfaction,
            task_hard.MIN_AVG_SATISFACTION,
            "at_least",
        ),
        CriterionResult(
            "stockout_fraction",
            result.stockout_fraction,
            task_hard.MAX_STOCKOUT_FRACTION,
            "at_most",
        ),
        CriterionResult(
            "avg_reward",
            result.avg_reward,
            task_hard.MIN_AVG_REWARD,
            "at_least",
        ),
        CriterionResult(
            "final_budget",
            result.final_budget,
            task_hard.MIN_FINAL_BUDGET,
            "at_least",
        ),
        CriterionResult(
            "final_awareness",
            result.final_awareness,
            task_hard.MIN_FINAL_AWARENESS,
            "at_least",
        ),
    ]
    for c in criteria:
        c.score = (
            _score_at_least(c.actual, c.target)
            if c.direction == "at_least"
            else _score_at_most(c.actual, c.target)
        )
    grade = sum(c.score for c in criteria) / len(criteria)
    return TaskGrade(
        task_name=result.task_name,
        difficulty="hard",
        criteria=criteria,
        grade=round(grade, 4),
    )



DIFFICULTY_WEIGHTS = {
    "easy": 0.20,
    "medium": 0.30,
    "hard": 0.50,
}


def compute_overall_grade(task_grades: list[TaskGrade]) -> float:
    """Weighted average of task grades."""
    if not task_grades:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    for tg in task_grades:
        w = DIFFICULTY_WEIGHTS.get(tg.difficulty, 0.0)
        weighted_sum += tg.grade * w
        weight_sum += w
    return round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.0


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


def format_report(
    task_grades: list[TaskGrade],
    overall_grade: float,
) -> str:
    buf = StringIO()

    buf.write("\n")
    buf.write(_separator("═") + "\n")
    buf.write("  CAMPUS MARKET ENVIRONMENT — GRADING REPORT\n")
    buf.write(_separator("═") + "\n\n")

    for tg in task_grades:
        buf.write(f"  Task: {tg.task_name}  [{tg.difficulty.upper()}]\n")
        buf.write(f"  Time: {tg.elapsed_seconds:.2f}s\n")
        buf.write(_separator() + "\n")

        header = f"  {'Criterion':<25} {'Actual':>12} {'Target':>12} {'Dir':>9} {'Score':>8}\n"
        buf.write(header)
        buf.write(_separator() + "\n")

        for c in tg.criteria:
            dir_label = "≥" if c.direction == "at_least" else "≤"
            buf.write(
                f"  {c.name:<25} {c.actual:>12.4f} {c.target:>12.4f} {dir_label:>9} {c.score:>8.4f}\n"
            )
        buf.write(_separator() + "\n")
        buf.write(f"  Task Grade: {tg.grade:.4f}\n\n")

    buf.write(_separator("═") + "\n")
    buf.write(f"  OVERALL GRADE: {overall_grade:.4f}  (range 0.0 – 1.0)\n")
    buf.write(_separator("═") + "\n\n")

    weights_info = "  Weights: " + " · ".join(
        f"{k} {int(v * 100)}%" for k, v in DIFFICULTY_WEIGHTS.items()
    )
    buf.write(weights_info + "\n\n")

    return buf.getvalue()


def main() -> None:
    print("\nRunning all the tasks\n")

    task_grades: list[TaskGrade] = []


    print("EASY task", end=" ", flush=True)
    t0 = time.perf_counter()
    easy_result = task_easy.run()
    easy_grade = grade_easy(easy_result)
    easy_grade.elapsed_seconds = round(time.perf_counter() - t0, 3)
    task_grades.append(easy_grade)
    print(f"done ({easy_grade.elapsed_seconds:.2f}s)")

    # ── Medium ──
    print("MEDIUM task", end=" ", flush=True)
    t0 = time.perf_counter()
    medium_result = task_medium.run()
    medium_grade = grade_medium(medium_result)
    medium_grade.elapsed_seconds = round(time.perf_counter() - t0, 3)
    task_grades.append(medium_grade)
    print(f"done ({medium_grade.elapsed_seconds:.2f}s)")

    # ── Hard ──
    print("HARD task", end=" ", flush=True)
    t0 = time.perf_counter()
    hard_result = task_hard.run()
    hard_grade = grade_hard(hard_result)
    hard_grade.elapsed_seconds = round(time.perf_counter() - t0, 3)
    task_grades.append(hard_grade)
    print(f"done ({hard_grade.elapsed_seconds:.2f}s)")

    overall = compute_overall_grade(task_grades)
    report = format_report(task_grades, overall)
    print(report)

    # Persist to file
    report_path = Path(__file__).resolve().parent / "grading_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {report_path}\n")


if __name__ == "__main__":
    main()

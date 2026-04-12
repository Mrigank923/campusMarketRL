# tasks/graders.py
# Campus Market task graders with normalized criterion-based scoring.

from dataclasses import dataclass, field
from statistics import mean
from typing import Optional


@dataclass
class CriterionScore:
    """Individual criterion evaluation."""
    name: str
    actual: float
    target: float
    direction: str  # "at_least" or "at_most"
    score: float = 0.0


@dataclass
class TaskGrade:
    """Complete task evaluation result."""
    task_name: str
    difficulty: str
    criteria: list[CriterionScore] = field(default_factory=list)
    grade: float = 0.0


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp value to [lower, upper]."""
    return max(lower, min(upper, value))


def clamp_exclusive(value: float, epsilon: float = 0.001) -> float:
    """Clamp value to strictly (epsilon, 1-epsilon) to avoid 0.0 and 1.0 boundaries."""
    # First clamp to [0, 1], then shift to (epsilon, 1-epsilon)
    clamped = clamp(value, 0.0, 1.0)
    # If exactly 0, shift to epsilon; if exactly 1, shift to 1-epsilon; otherwise squeeze into range
    if clamped == 0.0:
        return epsilon
    elif clamped == 1.0:
        return 1.0 - epsilon
    else:
        # Map value from (0, 1) to (epsilon, 1-epsilon)
        return epsilon + clamped * (1.0 - 2.0 * epsilon)


def score_at_least(actual: float, target: float) -> float:
    """Score for 'at least' criteria: scaled ratio, bounded to (0, 1) exclusive."""
    if target <= 0:
        # If target is invalid, return middle value strictly between 0 and 1
        return 0.5
    score = clamp(actual / target, 0.0, 1.0)
    return clamp_exclusive(score)


def score_at_most(actual: float, limit: float) -> float:
    """Score for 'at most' criteria: penalize exceeding limit, bounded to (0, 1) exclusive."""
    if limit <= 0:
        # If limit is invalid, return middle value strictly between 0 and 1
        return 0.5 if actual <= 0 else 0.5
    else:
        score = clamp(1.0 - (actual / limit), 0.0, 1.0)
    return clamp_exclusive(score)


def grade_easy(
    cumulative_revenue: float,
    avg_satisfaction: float,
    stockout_fraction: float,
) -> TaskGrade:
    """Grade easy task: 30-day steady state."""
    criteria = [
        CriterionScore(
            "cumulative_revenue",
            cumulative_revenue,
            75000.0,
            "at_least",
        ),
        CriterionScore(
            "avg_satisfaction",
            avg_satisfaction,
            0.55,
            "at_least",
        ),
        CriterionScore(
            "stockout_fraction",
            stockout_fraction,
            0.10,
            "at_most",
        ),
    ]
    
    for c in criteria:
        c.score = (
            score_at_least(c.actual, c.target)
            if c.direction == "at_least"
            else score_at_most(c.actual, c.target)
        )
    
    grade = mean(c.score for c in criteria) if criteria else 0.5
    grade = clamp_exclusive(round(grade, 4))
    return TaskGrade(
        task_name="easy_steady_state",
        difficulty="easy",
        criteria=criteria,
        grade=grade,
    )


def grade_medium(
    cumulative_revenue: float,
    avg_satisfaction: float,
    stockout_fraction: float,
    avg_reward: float,
) -> TaskGrade:
    """Grade medium task: 60-day adaptive pricing."""
    criteria = [
        CriterionScore(
            "cumulative_revenue",
            cumulative_revenue,
            180000.0,
            "at_least",
        ),
        CriterionScore(
            "avg_satisfaction",
            avg_satisfaction,
            0.58,
            "at_least",
        ),
        CriterionScore(
            "stockout_fraction",
            stockout_fraction,
            0.08,
            "at_most",
        ),
        CriterionScore(
            "avg_reward",
            avg_reward,
            3.5,
            "at_least",
        ),
    ]
    
    for c in criteria:
        c.score = (
            score_at_least(c.actual, c.target)
            if c.direction == "at_least"
            else score_at_most(c.actual, c.target)
        )
    
    grade = mean(c.score for c in criteria) if criteria else 0.5
    grade = clamp_exclusive(round(grade, 4))
    return TaskGrade(
        task_name="medium_adaptive_pricing",
        difficulty="medium",
        criteria=criteria,
        grade=grade,
    )


def grade_hard(
    cumulative_revenue: float,
    avg_satisfaction: float,
    stockout_fraction: float,
    avg_reward: float,
    final_budget: float,
    final_awareness: float,
) -> TaskGrade:
    """Grade hard task: 90-day full-horizon management."""
    criteria = [
        CriterionScore(
            "cumulative_revenue",
            cumulative_revenue,
            400000.0,
            "at_least",
        ),
        CriterionScore(
            "avg_satisfaction",
            avg_satisfaction,
            0.60,
            "at_least",
        ),
        CriterionScore(
            "stockout_fraction",
            stockout_fraction,
            0.06,
            "at_most",
        ),
        CriterionScore(
            "avg_reward",
            avg_reward,
            4.0,
            "at_least",
        ),
        CriterionScore(
            "final_budget",
            final_budget,
            2000.0,
            "at_least",
        ),
        CriterionScore(
            "final_awareness",
            final_awareness,
            0.65,
            "at_least",
        ),
    ]
    
    for c in criteria:
        c.score = (
            score_at_least(c.actual, c.target)
            if c.direction == "at_least"
            else score_at_most(c.actual, c.target)
        )
    
    grade = mean(c.score for c in criteria) if criteria else 0.5
    grade = clamp_exclusive(round(grade, 4))
    return TaskGrade(
        task_name="hard_full_horizon",
        difficulty="hard",
        criteria=criteria,
        grade=grade,
    )


DIFFICULTY_WEIGHTS = {
    "easy": 0.20,
    "medium": 0.30,
    "hard": 0.50,
}


def compute_overall_grade(task_grades: list[TaskGrade]) -> float:
    """Compute weighted average of task grades, clamped to (0, 1) exclusive."""
    if not task_grades:
        return clamp_exclusive(0.5)  # Default to middle of range if no grades
    
    weighted_sum = 0.0
    weight_sum = 0.0
    for tg in task_grades:
        w = DIFFICULTY_WEIGHTS.get(tg.difficulty, 0.0)
        weighted_sum += tg.grade * w
        weight_sum += w
    
    final_grade = weighted_sum / weight_sum if weight_sum > 0 else 0.5
    return clamp_exclusive(round(final_grade, 4))


def main() -> None:
    """Demo: show campus market grader in action."""
    print("=" * 70)
    print("CAMPUS MARKET GRADER — DEMO RUN")
    print("=" * 70)
    
    # Example results from running tasks
    print("\n📊 Easy Task (30 days):")
    easy = grade_easy(
        cumulative_revenue=72000.0,
        avg_satisfaction=0.52,
        stockout_fraction=0.12,
    )
    print(f"  Grade: {easy.grade:.6f}")
    for c in easy.criteria:
        direction = "≥" if c.direction == "at_least" else "≤"
        print(f"    {c.name}: {c.actual:.6f} {direction} {c.target:.6f} → {c.score:.6f}")
    
    print("\n📊 Medium Task (60 days):")
    medium = grade_medium(
        cumulative_revenue=185000.0,
        avg_satisfaction=0.58,
        stockout_fraction=0.07,
        avg_reward=3.6,
    )
    print(f"  Grade: {medium.grade:.6f}")
    for c in medium.criteria:
        direction = "≥" if c.direction == "at_least" else "≤"
        print(f"    {c.name}: {c.actual:.6f} {direction} {c.target:.6f} → {c.score:.6f}")
    
    print("\n📊 Hard Task (90 days):")
    hard = grade_hard(
        cumulative_revenue=410000.0,
        avg_satisfaction=0.61,
        stockout_fraction=0.05,
        avg_reward=4.2,
        final_budget=2500.0,
        final_awareness=0.67,
    )
    print(f"  Grade: {hard.grade:.6f}")
    for c in hard.criteria:
        direction = "≥" if c.direction == "at_least" else "≤"
        print(f"    {c.name}: {c.actual:.6f} {direction} {c.target:.6f} → {c.score:.6f}")
    
    overall = compute_overall_grade([easy, medium, hard])
    print("\n" + "=" * 70)
    print(f"OVERALL GRADE: {overall:.6f} (range 0.0 – 1.0)")
    print(f"Weights: easy 20% | medium 30% | hard 50%")
    print("=" * 70)


if __name__ == "__main__":
    main()

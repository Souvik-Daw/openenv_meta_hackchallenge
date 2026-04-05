"""
Deterministic graders for all three Healthcare Appointment Scheduling tasks.

Each grader returns a score in [0.0, 1.0] based on:
  - correct_department   (0.25 points)
  - correct_doctor       (0.30 points)
  - booking_successful   (0.35 points)
  - efficiency bonus     (0.10 points when done in ≤ expected_min_steps + 1)

The graders are deterministic: given the same episode state and task config,
they always produce the same score.
"""

from typing import Any, Dict

try:
    from healthcare_scheduling.models import AppointmentState
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models import AppointmentState  # type: ignore


# ---------------------------------------------------------------------------
# Core grader
# ---------------------------------------------------------------------------

def grade_episode(state: AppointmentState, task_config: Dict[str, Any]) -> float:
    """
    Grade a completed episode deterministically.

    Parameters
    ----------
    state       : AppointmentState  — the environment's internal state after episode ends.
    task_config : dict              — produced by easy/medium/hard get_task_config().

    Returns
    -------
    float in [0.0, 1.0]
    """
    correct_dept   = task_config["correct_department"]
    correct_doctor = task_config["correct_doctor"]
    min_steps      = task_config.get("expected_min_steps", 4)
    requires_clr   = task_config.get("requires_clarification", False)

    score = 0.0

    # 0. Get departments
    is_rebook = task_config.get("is_rebook", False)
    used_get_departments = any(
        entry.get("tool") == "get_departments"
        for entry in state.conversation_history
    )
    if not is_rebook and used_get_departments:
        score += 1.0

    if is_rebook:
        score += 2.0

    # 1. Department correctness
    if not is_rebook and state.identified_department == correct_dept:
        score += 1.0

    # 2. Doctor correctness
    is_doctor_correct = state.selected_doctor == correct_doctor
    if is_rebook and state.selected_doctor and hasattr(state, "user_request"):
        req_l = state.user_request.lower()
        doc_l = state.selected_doctor.lower()
        doc_last = state.selected_doctor.split(" ")[-1].lower()
        if doc_l in req_l or doc_last in req_l:
            is_doctor_correct = True

    if is_doctor_correct:
        score += 1.0

    # 3. Booking success
    if state.booking_successful:
        score += 1.0


    # 5. Penalty deductions
    #    Reduce score if agent ignored required clarification (hard task)
    if requires_clr:
        used_clarification = any(
            entry.get("tool") == "ask_user_clarification"
            for entry in state.conversation_history
        )
        if not used_clarification:
            score -= 0.50  # penalise skipping clarification

    # 6. Lower bound score to 0.0
    score = max(0.0, round(score, 4))
    return score


# ---------------------------------------------------------------------------
# Per-task grader wrappers (for clarity and testability)
# ---------------------------------------------------------------------------

def grade_easy(state: AppointmentState) -> float:
    """Grade the EASY (chest pain) task."""
    from tasks.easy import get_task_config
    return grade_episode(state, get_task_config())


def grade_medium(state: AppointmentState) -> float:
    """Grade the MEDIUM (skin rash) task."""
    from tasks.medium import get_task_config
    return grade_episode(state, get_task_config())


def grade_hard(state: AppointmentState) -> float:
    """Grade the HARD (ambiguous pain) task."""
    from tasks.hard import get_task_config
    return grade_episode(state, get_task_config())


# ---------------------------------------------------------------------------
# Diagnostic breakdown helper
# ---------------------------------------------------------------------------

def grade_full_breakdown(
    state: AppointmentState, task_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Return a detailed scoring breakdown for logging and debugging.

    Returns a dict with individual component scores and the final total.
    """
    correct_dept   = state.correct_department or task_config.get("correct_department")
    correct_doctor = state.correct_doctor or task_config.get("correct_doctor")
    min_steps      = task_config.get("expected_min_steps", 4)
    requires_clr   = task_config.get("requires_clarification", False)
    is_rebook      = task_config.get("is_rebook", False)

    used_get_departments = any(
        entry.get("tool") == "get_departments"
        for entry in state.conversation_history
    )
    missing_stage_bonus = 2.0 if is_rebook else 0.0
    get_dept_score = 1.0 if not is_rebook and used_get_departments else 0.0
    dept_score  = 1.0 if not is_rebook and state.identified_department == correct_dept else 0.0
    is_doctor_correct = state.selected_doctor == correct_doctor
    if is_rebook and state.selected_doctor and hasattr(state, "user_request"):
        req_l = state.user_request.lower()
        doc_l = state.selected_doctor.lower()
        doc_last = state.selected_doctor.split(" ")[-1].lower()
        if doc_l in req_l or doc_last in req_l:
            is_doctor_correct = True

    doc_score   = 1.0 if is_doctor_correct else 0.0
    book_score  = 1.0 if state.booking_successful else 0.0


    used_clarification = any(
        entry.get("tool") == "ask_user_clarification"
        for entry in state.conversation_history
    )
    clr_penalty = -0.50 if requires_clr and not used_clarification else 0.0

    total = max(0.0, get_dept_score + dept_score + doc_score + book_score + missing_stage_bonus + clr_penalty)

    return {
        "task_id": task_config["task_id"],
        "difficulty": task_config["difficulty"],
        "used_get_departments": used_get_departments,
        "get_departments_score": get_dept_score,
        "is_rebook": is_rebook,
        "missing_stage_bonus": missing_stage_bonus,
        "department_correct": True if is_rebook else state.identified_department == correct_dept,
        "department_score": dept_score,
        "doctor_correct": is_doctor_correct,
        "doctor_score": doc_score,
        "booking_successful": state.booking_successful,
        "booking_score": book_score,
        "clarification_penalty": clr_penalty,
        "steps_taken": state.step_count,
        "cumulative_env_reward": round(state.cumulative_reward, 4),
        "final_score": round(total, 4),
    }

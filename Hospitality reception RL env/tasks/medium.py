"""
MEDIUM Task — Healthcare Appointment Scheduling

User request: "I have a skin rash for 2 weeks"
Expected path:
  get_departments() → get_doctors("Dermatology") → check_availability("Dr. Priya Patel")
  → book_appointment("Dr. Priya Patel", <any valid slot>)

Ground truth:
  department : Dermatology
  doctor     : Dr. Priya Patel (skin rashes, eczema, chronic skin conditions)

Difficulty note:
  The symptom "skin rash" is moderately specific. The agent must correctly interpret
  the chronic nature ("2 weeks") to prefer the chronic-conditions specialist
  (Dr. Priya Patel) over Dr. Kevin Lee (acne / mole removal).
"""

try:
    from healthcare_scheduling.server.environment import HealthcareAppointmentEnvironment
    from healthcare_scheduling.models import AppointmentAction
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from server.environment import HealthcareAppointmentEnvironment  # type: ignore
    from models import AppointmentAction  # type: ignore


TASK_ID = "medium_skin_rash"
USER_REQUEST = "I have a skin rash for 2 weeks"
CORRECT_DEPARTMENT = "Dermatology"
CORRECT_DOCTOR = "Dr. Priya Patel"


def make_env() -> HealthcareAppointmentEnvironment:
    """Return a fresh environment pre-loaded with the medium task."""
    env = HealthcareAppointmentEnvironment()
    env.reset(
        user_request=USER_REQUEST,
        correct_department=CORRECT_DEPARTMENT,
        correct_doctor=CORRECT_DOCTOR,
    )
    return env


def get_task_config() -> dict:
    """Return the task configuration dictionary used by agents and graders."""
    return {
        "task_id": TASK_ID,
        "difficulty": "medium",
        "user_request": USER_REQUEST,
        "correct_department": CORRECT_DEPARTMENT,
        "correct_doctor": CORRECT_DOCTOR,
        "description": (
            "The user has had a skin rash for two weeks. "
            "The agent must identify Dermatology, select Dr. Priya Patel "
            "(skin rashes, eczema, chronic skin conditions specialist), and book."
        ),
        "max_steps": 10,
        "expected_min_steps": 4,
        "hints": [
            "Skin rash → Dermatology",
            "2-week rash signals a chronic condition → prefer Dr. Priya Patel",
        ],
    }


if __name__ == "__main__":
    from tasks.graders import grade_episode

    env = HealthcareAppointmentEnvironment()
    obs = env.reset(
        user_request=USER_REQUEST,
        correct_department=CORRECT_DEPARTMENT,
        correct_doctor=CORRECT_DOCTOR,
    )
    print(f"[MEDIUM] Episode started: {obs.message}")

    # Simulate an oracle agent
    actions = [
        AppointmentAction(tool="get_departments", parameters={}),
        AppointmentAction(tool="get_doctors", parameters={"department": "Dermatology"}),
        AppointmentAction(
            tool="check_availability", parameters={"doctor": "Dr. Priya Patel"}
        ),
        AppointmentAction(
            tool="book_appointment",
            parameters={"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"},
        ),
    ]

    for action in actions:
        obs = env.step(action)
        print(f"  [{action.tool}] reward={obs.reward:.2f} | {obs.message}")
        if obs.done:
            break

    score = grade_episode(env.state, get_task_config())
    print(f"\n[MEDIUM] Final Score: {score:.3f}")

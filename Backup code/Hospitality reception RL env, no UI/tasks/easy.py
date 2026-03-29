"""
EASY Task — Healthcare Appointment Scheduling

User request: "I have chest pain"
Expected path:
  get_departments() → get_doctors("Cardiology") → check_availability("Dr. Sarah Smith")
  → book_appointment("Dr. Sarah Smith", <any valid slot>)

Ground truth:
  department : Cardiology
  doctor     : Dr. Sarah Smith (general cardiology / chest pain specialist)
"""

try:
    from healthcare_scheduling.server.environment import HealthcareAppointmentEnvironment
    from healthcare_scheduling.models import AppointmentAction
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from server.environment import HealthcareAppointmentEnvironment  # type: ignore
    from models import AppointmentAction  # type: ignore


TASK_ID = "easy_chest_pain"
USER_REQUEST = "I have chest pain"
CORRECT_DEPARTMENT = "Cardiology"
CORRECT_DOCTOR = "Dr. Sarah Smith"


def make_env() -> HealthcareAppointmentEnvironment:
    """Return a fresh environment pre-loaded with the easy task."""
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
        "difficulty": "easy",
        "user_request": USER_REQUEST,
        "correct_department": CORRECT_DEPARTMENT,
        "correct_doctor": CORRECT_DOCTOR,
        "description": (
            "The user reports chest pain. "
            "The agent must identify Cardiology, select Dr. Sarah Smith "
            "(general cardiology / chest pain specialist), and complete a booking."
        ),
        "max_steps": 10,
        "expected_min_steps": 4,
        "hints": [
            "Chest pain → Cardiology",
            "Dr. Sarah Smith specialises in chest pain",
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
    print(f"[EASY] Episode started: {obs.message}")

    # Simulate an oracle agent following the optimal path
    actions = [
        AppointmentAction(tool="get_departments", parameters={}),
        AppointmentAction(tool="get_doctors", parameters={"department": "Cardiology"}),
        AppointmentAction(
            tool="check_availability", parameters={"doctor": "Dr. Sarah Smith"}
        ),
        AppointmentAction(
            tool="book_appointment",
            parameters={"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"},
        ),
    ]

    for action in actions:
        obs = env.step(action)
        print(f"  [{action.tool}] reward={obs.reward:.2f} | {obs.message}")
        if obs.done:
            break

    score = grade_episode(env.state, get_task_config())
    print(f"\n[EASY] Final Score: {score:.3f}")

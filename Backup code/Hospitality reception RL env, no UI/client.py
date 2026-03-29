"""
OpenEnv client wrapper for the Healthcare Appointment Scheduling environment.

Usage (sync)
------------
from healthcare_scheduling.client import HealthcareEnv
from healthcare_scheduling.models import AppointmentAction

with HealthcareEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    obs = env.step(AppointmentAction(tool="get_departments", parameters={}))
    print(obs.observation.tool_result)

Usage (async)
-------------
async with HealthcareEnv(base_url="http://localhost:8000").async_() as env:
    obs = await env.reset()
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

try:
    from healthcare_scheduling.models import (
        AppointmentAction,
        AppointmentObservation,
        AppointmentState,
    )
except ImportError:
    from models import (  # type: ignore
        AppointmentAction,
        AppointmentObservation,
        AppointmentState,
    )


class HealthcareEnv(EnvClient[AppointmentAction, AppointmentObservation, AppointmentState]):
    """Typed HTTP client for the Healthcare Appointment Scheduling environment."""

    def _step_payload(self, action: AppointmentAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        is_done = payload.get("done", False)
        obs_data = payload.get("observation", {})
        if "done" not in obs_data:
            obs_data["done"] = is_done
        return StepResult(
            observation=AppointmentObservation(**obs_data),
            reward=payload.get("reward", 0.0),
            done=is_done,
        )

    def _parse_state(self, payload: dict) -> AppointmentState:
        state_data = payload.get("state", payload)
        return AppointmentState(**state_data)

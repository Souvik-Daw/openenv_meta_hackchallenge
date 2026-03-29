"""
FastAPI application entry-point for the Healthcare Appointment Scheduling environment.

Exposes the standard OpenEnv HTTP endpoints:
  POST /reset   → start a new episode
  POST /step    → take an action
  GET  /state   → read internal episode state
  GET  /health  → liveness probe

The `create_fastapi_app` factory from openenv-core wires these up automatically.
"""

from openenv.core.env_server import create_fastapi_app

try:
    from healthcare_scheduling.server.environment import HealthcareAppointmentEnvironment
    from healthcare_scheduling.models import (
        AppointmentAction,
        AppointmentObservation,
        AppointmentState,
    )
except ImportError:
    # Local dev mode: run from project root with uvicorn server.app:app
    from server.environment import HealthcareAppointmentEnvironment  # type: ignore
    from models import (  # type: ignore
        AppointmentAction,
        AppointmentObservation,
        AppointmentState,
    )

app = create_fastapi_app(
    HealthcareAppointmentEnvironment,
    AppointmentAction,
    AppointmentObservation,
)


def main() -> None:
    """Entry-point used by the `server` script defined in pyproject.toml."""
    import uvicorn
    uvicorn.run(
        "healthcare_scheduling.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()

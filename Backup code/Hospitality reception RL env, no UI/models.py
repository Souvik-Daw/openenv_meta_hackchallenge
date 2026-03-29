"""
Pydantic models for the Healthcare Appointment Scheduling RL Environment.

Defines Action, Observation, and State schemas that are shared between
the environment server and the OpenEnv client.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class AppointmentAction(BaseModel):
    """
    A single tool call issued by the agent.

    The agent must set `tool` to one of:
        get_departments | get_doctors | check_availability |
        book_appointment | ask_user_clarification

    `parameters` carries the keyword arguments for that tool.

    Examples
    --------
    AppointmentAction(tool="get_departments", parameters={})
    AppointmentAction(tool="get_doctors", parameters={"department": "Cardiology"})
    AppointmentAction(tool="check_availability", parameters={"doctor": "Dr. Smith"})
    AppointmentAction(tool="book_appointment",
                      parameters={"doctor": "Dr. Smith", "slot": "2024-01-15 10:00 AM"})
    AppointmentAction(tool="ask_user_clarification",
                      parameters={"question": "Is the pain in your chest or abdomen?"})
    """

    tool: str = Field(
        ...,
        description=(
            "The tool to call. Must be one of: get_departments, get_doctors, "
            "check_availability, book_appointment, ask_user_clarification"
        )
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the tool call."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class AppointmentObservation(BaseModel):
    """
    What the agent sees after each step.

    Contains the tool result, current conversation history, and episode
    status so the agent can plan its next action.
    """

    done: bool = Field(
        False,
        description="True when the episode has ended (booked or max steps reached)."
    )
    reward: Optional[float] = Field(
        0.0,
        description="Reward earned in this step."
    )
    # Tool result returned to the agent
    tool_result: Any = Field(
        None,
        description="The data returned by the tool that was just called."
    )
    tool_called: Optional[str] = Field(
        None,
        description="Name of the tool that was called in this step."
    )
    # Running state for the agent's situational awareness
    user_request: str = Field(
        "",
        description="The original natural-language request from the user."
    )
    identified_department: Optional[str] = Field(
        None,
        description="Department identified so far (None until confirmed)."
    )
    selected_doctor: Optional[str] = Field(
        None,
        description="Doctor chosen so far (None until confirmed)."
    )
    selected_slot: Optional[str] = Field(
        None,
        description="Slot chosen so far (None until booking is attempted)."
    )
    steps_taken: int = Field(0, description="Number of steps taken in this episode.")
    max_steps: int = Field(10, description="Maximum allowed steps per episode.")
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full list of (tool, parameters, result) tuples so far."
    )
    message: str = Field("", description="Human-readable feedback from the environment.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State  (internal episode metadata)
# ---------------------------------------------------------------------------

class AppointmentState(BaseModel):
    """Internal episode state (not sent to agent directly, available via /state)."""

    episode_id: Optional[str] = None
    step_count: int = 0
    user_request: str = ""
    # Ground truth for graders
    correct_department: Optional[str] = None
    correct_doctor: Optional[str] = None
    # What the agent has actually done
    identified_department: Optional[str] = None
    selected_doctor: Optional[str] = None
    selected_slot: Optional[str] = None
    booking_successful: bool = False
    # Tracking
    tools_called: List[str] = Field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    max_steps: int = 10

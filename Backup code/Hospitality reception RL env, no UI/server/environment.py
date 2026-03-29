"""
Core environment implementation for the Healthcare Appointment Scheduling RL environment.

Follows the OpenEnv interface pattern:
  - reset()        → AppointmentObservation
  - step(action)   → AppointmentObservation
  - state property → AppointmentState

State tracks the full episode: department identified, doctor selected,
booking outcome, and conversation history for graders.
"""

import uuid
from typing import Optional

try:
    # Installed package mode
    from healthcare_scheduling.models import AppointmentAction, AppointmentObservation, AppointmentState
    from healthcare_scheduling.server.data import DOCTORS, map_symptoms_to_department
    from healthcare_scheduling.server.tools import VALID_TOOLS, dispatch_tool
except ImportError:
    # Local / sys.path mode (run_baseline.py adds project root to sys.path)
    from models import AppointmentAction, AppointmentObservation, AppointmentState  # type: ignore
    from server.data import DOCTORS, map_symptoms_to_department  # type: ignore
    from server.tools import VALID_TOOLS, dispatch_tool  # type: ignore


class HealthcareAppointmentEnvironment:
    """
    Healthcare Appointment Scheduling RL Environment.

    The agent receives a natural-language user request and must:
      1. Call get_departments() to understand available departments.
      2. Call get_doctors(department) to find the right specialist.
      3. Call check_availability(doctor) to see open slots.
      4. Call book_appointment(doctor, slot) to complete the task.
      5. Optionally call ask_user_clarification(question) for ambiguous cases.

    The episode ends when the agent successfully books an appointment or
    reaches max_steps.
    """

    MAX_STEPS = 10

    def __init__(self) -> None:
        self._state = AppointmentState()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        user_request: str = "I have chest pain",
        correct_department: Optional[str] = None,
        correct_doctor: Optional[str] = None,
    ) -> AppointmentObservation:
        """
        Start a new episode.

        Parameters
        ----------
        user_request : str
            The natural-language symptom description given to the agent.
        correct_department : str | None
            Overrides the auto-detected ground-truth department (used by tasks).
        correct_doctor : str | None
            Overrides the auto-detected ground-truth doctor (used by tasks).
        """
        # Derive ground truth if not provided
        auto_dept = map_symptoms_to_department(user_request)
        resolved_dept = correct_department or auto_dept
        resolved_doctor = correct_doctor or self._default_doctor(resolved_dept)

        self._state = AppointmentState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            user_request=user_request,
            correct_department=resolved_dept,
            correct_doctor=resolved_doctor,
            max_steps=self.MAX_STEPS,
        )

        return AppointmentObservation(
            done=False,
            reward=0.0,
            tool_result=None,
            tool_called=None,
            user_request=user_request,
            identified_department=None,
            selected_doctor=None,
            selected_slot=None,
            steps_taken=0,
            max_steps=self.MAX_STEPS,
            conversation_history=[],
            message=(
                f"New episode started. User says: \"{user_request}\". "
                f"Available tools: {sorted(VALID_TOOLS)}. "
                "Call get_departments() to begin."
            ),
        )

    async def reset_async(
        self,
        user_request: str = "I have chest pain",
        correct_department: Optional[str] = None,
        correct_doctor: Optional[str] = None,
    ) -> AppointmentObservation:
        return self.reset(user_request, correct_department, correct_doctor)

    def step(self, action: AppointmentAction) -> AppointmentObservation:
        """
        Apply one tool call action and return the resulting observation.

        Reward is computed at each step based on:
          - Correctness of department/doctor selection
          - Successful booking
          - Efficiency (fewer wasted steps)
          - Invalid or repeated tool usage
        """
        s = self._state
        s.step_count += 1

        # --- Guard: already done? -------------------------------------------
        if s.booking_successful or s.step_count > self.MAX_STEPS:
            return self._make_observation(
                tool_result={"error": "Episode already ended."},
                reward=-0.1,
                done=True,
                message="Episode has already ended.",
            )

        # --- Dispatch the tool call -----------------------------------------
        tool_name = action.tool
        params = action.parameters

        if tool_name not in VALID_TOOLS:
            reward = -0.2
            result = {
                "error": f"Invalid tool '{tool_name}'.",
                "valid_tools": sorted(VALID_TOOLS),
            }
            s.tools_called.append(tool_name)
            s.conversation_history.append(
                {"tool": tool_name, "parameters": params, "result": result}
            )
            return self._make_observation(
                tool_result=result,
                reward=reward,
                done=False,
                message=f"Invalid tool '{tool_name}'. Use one of: {sorted(VALID_TOOLS)}",
            )

        # Detect unnecessary repetitions (same tool + same params twice in a row)
        if len(s.tools_called) >= 2 and s.tools_called[-1] == tool_name:
            reward = -0.1
        else:
            reward = 0.0

        result = dispatch_tool(tool_name, params)
        s.tools_called.append(tool_name)
        s.conversation_history.append(
            {"tool": tool_name, "parameters": params, "result": result}
        )

        # --- Update state based on which tool was called --------------------
        done = False
        message = ""

        if tool_name == "get_departments":
            message = "Retrieved department list. Now call get_doctors(department)."

        elif tool_name == "get_doctors":
            dept = params.get("department", "")
            resolved = self._resolve_department(dept)
            if resolved is None:
                reward += -0.3
                message = f"Department '{dept}' not found. Try again with a valid department."
            else:
                if resolved == s.correct_department:
                    if s.identified_department != resolved:
                        # First time correctly identifying
                        reward += 0.2
                        message = f"Correct department '{resolved}' identified! (+0.2)"
                    else:
                        message = f"Department '{resolved}' already identified."
                else:
                    reward += -0.3
                    message = (
                        f"Department '{resolved}' may not match the user's symptoms. "
                        "Consider the symptom description more carefully."
                    )
                s.identified_department = resolved

        elif tool_name == "check_availability":
            doctor = params.get("doctor", "")
            doc_record = self._find_doctor(doctor)
            if doc_record is None:
                reward += -0.2
                message = f"Doctor '{doctor}' not found."
            else:
                dept_of_doc = doc_record["department"]
                if (
                    s.correct_department
                    and dept_of_doc == s.correct_department
                    and doc_record["name"] == s.correct_doctor
                ):
                    reward += 0.1
                    message = f"Checking availability for the right doctor ({doctor}). (+0.1)"
                elif dept_of_doc != s.correct_department:
                    reward += -0.1
                    message = (
                        f"Doctor '{doctor}' is in '{dept_of_doc}', "
                        "which may not match the user's symptoms."
                    )
                else:
                    message = f"Checking availability for {doctor}."
                s.selected_doctor = doc_record["name"]

        elif tool_name == "book_appointment":
            doctor = params.get("doctor", "")
            slot = params.get("slot", "")
            s.selected_slot = slot

            if result.get("success"):
                # Evaluate booking correctness
                booked_dept = result.get("department", "")
                booked_doctor = result.get("doctor", "")

                dept_correct = booked_dept == s.correct_department
                doctor_correct = booked_doctor == s.correct_doctor

                booking_reward = 0.0
                if dept_correct and doctor_correct:
                    booking_reward = 0.5
                    # Efficiency bonus: fewer steps = better
                    if s.step_count <= 4:
                        booking_reward += 0.2
                        message = (
                            f"Perfect booking! Correct department and doctor. "
                            f"Efficiency bonus applied! (+0.7 total)"
                        )
                    else:
                        message = (
                            f"Booking successful with correct department and doctor. (+0.5)"
                        )
                elif dept_correct and not doctor_correct:
                    booking_reward = 0.1
                    reward += -0.4
                    message = (
                        f"Booking in the right department but wrong doctor specialization. "
                        f"(-0.4 specialization penalty)"
                    )
                else:
                    booking_reward = 0.0
                    reward += -0.3  # wrong department
                    reward += -0.4  # wrong doctor
                    message = (
                        "Booking with wrong department and doctor! "
                        "This is a major failure. (-0.7 total)"
                    )

                reward += booking_reward
                s.booking_successful = True
                s.selected_doctor = booked_doctor
                done = True
            else:
                reward += -0.2
                message = f"Booking failed: {result.get('error', 'Unknown error.')}"

        elif tool_name == "ask_user_clarification":
            # Clarification is acceptable for ambiguous requests
            if s.step_count <= 2:
                reward += 0.05  # small positive for appropriate use
                message = "Got user clarification. (+0.05)"
            else:
                reward += -0.05  # slight penalty for late clarification
                message = "Clarification obtained (late penalty -0.05)."

        # Check max steps
        if s.step_count >= self.MAX_STEPS and not done:
            done = True
            reward += -0.1
            message = f"Maximum steps ({self.MAX_STEPS}) reached. Episode ended."

        s.cumulative_reward += reward

        return self._make_observation(
            tool_result=result,
            reward=round(reward, 4),
            done=done,
            message=message,
            tool_called=tool_name,
        )

    async def step_async(self, action: AppointmentAction) -> AppointmentObservation:
        return self.step(action)

    @property
    def state(self) -> AppointmentState:
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        tool_result,
        reward: float,
        done: bool,
        message: str,
        tool_called: Optional[str] = None,
    ) -> AppointmentObservation:
        s = self._state
        return AppointmentObservation(
            done=done,
            reward=reward,
            tool_result=tool_result,
            tool_called=tool_called or (s.tools_called[-1] if s.tools_called else None),
            user_request=s.user_request,
            identified_department=s.identified_department,
            selected_doctor=s.selected_doctor,
            selected_slot=s.selected_slot,
            steps_taken=s.step_count,
            max_steps=s.max_steps,
            conversation_history=list(s.conversation_history),
            message=message,
        )

    def _resolve_department(self, dept_name: str) -> Optional[str]:
        """Case-insensitive department lookup. Returns canonical name or None."""
        try:
            from healthcare_scheduling.server.data import DEPARTMENTS
        except ImportError:
            from server.data import DEPARTMENTS  # type: ignore
        for dept in DEPARTMENTS:
            if dept.lower() == dept_name.lower():
                return dept
        return None

    def _find_doctor(self, doctor_name: str) -> Optional[dict]:
        """Find a doctor record by name (case-insensitive), returns augmented dict or None."""
        for dept, doctors in DOCTORS.items():
            for doc in doctors:
                if doc["name"].lower() == doctor_name.lower():
                    return {**doc, "department": dept}
        return None

    def _default_doctor(self, department: Optional[str]) -> Optional[str]:
        """Return the first doctor in a department (default ground truth)."""
        if department and department in DOCTORS:
            return DOCTORS[department][0]["name"]
        return None

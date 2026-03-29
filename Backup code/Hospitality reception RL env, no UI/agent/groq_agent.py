"""
Groq-powered LLM agent for the Healthcare Appointment Scheduling environment.

Uses the `custom` model via the Groq API.

The agent reads the current observation and generates a structured tool-call
action (tool name + parameters) using a JSON-mode prompt.  It loops until the
episode ends or max_steps is reached.

Environment variable required:
    GROQ_API_KEY   — your Groq API key

Usage (standalone):
    python agent/groq_agent.py

Usage (from run_baseline.py):
    from agent.groq_agent import GroqAgent
"""

import json
import os
import re
import sys
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv

try:
    from groq import Groq
except ImportError:
    sys.exit(
        "groq package not found. Install it with:  pip install groq\n"
    )

try:
    from healthcare_scheduling.models import AppointmentObservation, AppointmentAction
    from healthcare_scheduling.server.tools import VALID_TOOLS
except ImportError:
    # Local sys.path mode
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models import AppointmentObservation, AppointmentAction  # type: ignore
    from server.tools import VALID_TOOLS  # type: ignore


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a healthcare appointment scheduling assistant agent.

Your task is to help a patient book an appointment with the correct doctor.

## Available Tools
You may call ONLY ONE of these tools per turn:
1. get_departments()
   → Returns the list of available medical departments.

2. get_doctors(department: str)
   → Returns doctors in a department with their specializations.

3. check_availability(doctor: str)
   → Returns available time slots for a specific doctor.

4. book_appointment(doctor: str, slot: str)
   → Books an appointment. Use EXACT doctor name and EXACT slot string.

5. ask_user_clarification(question: str)
   → Ask the patient a clarifying question when the symptom is ambiguous.

## Strategy
1. If the user request is ambiguous, call ask_user_clarification FIRST.
2. Call get_departments() to see available departments.
3. Match the symptom to the correct department.
4. Call get_doctors(department) to find the appropriate specialist.
5. Call check_availability(doctor) to find an open slot.
6. Call book_appointment(doctor, slot) with exact strings to complete.

## Output Format
You MUST respond with ONLY valid JSON in this exact format:
{
  "tool": "<tool_name>",
  "parameters": {<parameters_dict>},
  "reasoning": "<brief explanation of why you chose this tool>"
}

No extra text, no markdown code blocks — only raw JSON.
"""


# ---------------------------------------------------------------------------
# GroqAgent class
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

class GroqAgent:
    """LLM agent using Groq (llama-3.1-8b-instant) to solve healthcare scheduling tasks."""

    MODEL = os.getenv("GROQ_MODEL")
    MAX_RETRIES = 3

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Export it as an environment variable or pass api_key= directly."
            )
        self.client = Groq(api_key=key)
        self.verbose = verbose
        self._conversation: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_action(self, obs: AppointmentObservation) -> AppointmentAction:
        """
        Given the current observation, query the LLM and return an action.

        Retries up to MAX_RETRIES times on JSON parse failures.
        Falls back to get_departments() if all retries fail.
        """
        user_message = self._format_observation(obs)
        self._conversation.append({"role": "user", "content": user_message})

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *self._conversation,
                    ],
                    temperature=0.0,   # deterministic
                    max_tokens=512,
                )
                raw = response.choices[0].message.content.strip()

                if self.verbose:
                    print(f"    [LLM raw] {raw[:200]}...")

                action = self._parse_action(raw)
                self._conversation.append({"role": "assistant", "content": raw})
                return action

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                if self.verbose:
                    print(f"    [LLM parse error attempt {attempt}] {exc}")
                if attempt == self.MAX_RETRIES:
                    # Final fallback
                    return AppointmentAction(
                        tool="get_departments",
                        parameters={},
                        metadata={"fallback": True},
                    )

        # Unreachable, but satisfies type checker
        return AppointmentAction(tool="get_departments", parameters={})

    def reset_conversation(self) -> None:
        """Clear the in-context conversation history between episodes."""
        self._conversation = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_observation(self, obs: AppointmentObservation) -> str:
        """Convert an observation into a text message for the LLM."""
        lines = [
            f"## Current Observation (Step {obs.steps_taken}/{obs.max_steps})",
            f"**User Request:** {obs.user_request}",
            f"**Last Tool Used:** {obs.tool_called or 'None'}",
            f"**Tool Result:**",
            json.dumps(obs.tool_result, indent=2) if obs.tool_result else "(no result yet)",
            "",
            f"**Identified Department:** {obs.identified_department or 'Not yet determined'}",
            f"**Selected Doctor:** {obs.selected_doctor or 'Not yet selected'}",
            f"**Selected Slot:** {obs.selected_slot or 'Not yet selected'}",
            "",
            f"**Environment Message:** {obs.message}",
            "",
            "Based on the above, what is your next tool call?",
            "Respond ONLY with JSON: {\"tool\": \"...\", \"parameters\": {...}, \"reasoning\": \"...\"}",
        ]
        return "\n".join(lines)

    def _parse_action(self, raw: str) -> AppointmentAction:
        """
        Parse a JSON string from the LLM into an AppointmentAction.

        Handles common LLM quirks:
          - Markdown code fences (```json ... ```)
          - Leading/trailing whitespace
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = cleaned.rstrip("`").strip()

        data: Dict[str, Any] = json.loads(cleaned)

        tool = data.get("tool", "").strip()
        params = data.get("parameters", {})

        if not isinstance(params, dict):
            params = {}

        if tool not in VALID_TOOLS:
            raise ValueError(f"LLM returned invalid tool: '{tool}'")

        return AppointmentAction(
            tool=tool,
            parameters=params,
            metadata={"reasoning": data.get("reasoning", "")},
        )


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from server.environment import HealthcareAppointmentEnvironment
    from tasks.easy import get_task_config as easy_config, USER_REQUEST as EASY_REQ, CORRECT_DEPARTMENT as EASY_DEPT, CORRECT_DOCTOR as EASY_DOC
    from tasks.medium import get_task_config as medium_config, USER_REQUEST as MED_REQ, CORRECT_DEPARTMENT as MED_DEPT, CORRECT_DOCTOR as MED_DOC
    from tasks.hard import get_task_config as hard_config, USER_REQUEST as HARD_REQ, CORRECT_DEPARTMENT as HARD_DEPT, CORRECT_DOCTOR as HARD_DOC
    from tasks.graders import grade_full_breakdown

    TASKS = [
        ("easy",   EASY_REQ,  EASY_DEPT,  EASY_DOC,  easy_config),
        ("medium", MED_REQ,   MED_DEPT,   MED_DOC,   medium_config),
        ("hard",   HARD_REQ,  HARD_DEPT,  HARD_DOC,  hard_config),
    ]

    agent = GroqAgent(verbose=True)

    for difficulty, user_req, dept, doctor, config_fn in TASKS:
        print(f"\n{'='*60}")
        print(f" TASK: {difficulty.upper()}")
        print(f" Request: \"{user_req}\"")
        print(f"{'='*60}")

        env = HealthcareAppointmentEnvironment()
        obs = env.reset(
            user_request=user_req,
            correct_department=dept,
            correct_doctor=doctor,
        )
        agent.reset_conversation()

        step = 0
        while not obs.done and step < 10:
            step += 1
            print(f"\n--- Step {step} ---")
            action = agent.decide_action(obs)
            print(f"  Action: {action.tool}({action.parameters})")
            obs = env.step(action)
            print(f"  Reward: {obs.reward:.4f} | Done: {obs.done}")
            print(f"  Message: {obs.message}")

        breakdown = grade_full_breakdown(env.state, config_fn())
        print(f"\n  [GRADER] Score: {breakdown['final_score']:.3f}")
        print(f"  Breakdown: {json.dumps(breakdown, indent=4)}")

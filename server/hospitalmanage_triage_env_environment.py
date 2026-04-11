# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hospitalmanage Triage Env Environment Implementation.

Uses direct tool dispatch (no MCP/WebSocket) for maximum reliability.
"""

from typing import Any, Optional
from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.interfaces import Action
from openenv.core.env_server.mcp_types import CallToolAction

from .tools import ToolSpec
try:
    from ..models import HospitalmanageTriageObservation, HospitalState
except ImportError:
    from models import HospitalmanageTriageObservation, HospitalState


class HospitalmanageTriageEnvironment(Environment):
    """
    Hospital triage environment that dispatches tool calls directly
    (no MCP / WebSocket transport). This avoids all keepalive-timeout and
    transfer_data_task errors that plagued the MCPEnvironment base class.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Map action tool_name -> ToolSpec method
    TOOL_DISPATCH = {
        "get_department":  "get_department",
        "get_opd_doctor":  "get_opd_doctor",
        "make_appointment": "make_appointment",
        "get_appointment": "get_appointment",
    }

    def __init__(self):
        super().__init__()
        self.tools = ToolSpec()
        self._state: Optional[HospitalState] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        patient_id: Optional[int] = None,
        tool_call_sequence: Optional[list] = None,
        output_sequence: Optional[list] = None,
        **kwargs: Any,
    ) -> HospitalmanageTriageObservation:
        self._state = HospitalState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            patient_id=patient_id or random.randint(1000, 9999),
            tool_call_sequence=tool_call_sequence or [],
            output_sequence=output_sequence or [],
        )
        return HospitalmanageTriageObservation(
            result="Hospitalmanage Triage Env environment ready!",
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step  (sync + async share the same logic via _execute_step)
    # ------------------------------------------------------------------
    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HospitalmanageTriageObservation:
        return self._execute_step(action)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HospitalmanageTriageObservation:
        return self._execute_step(action)

    # ------------------------------------------------------------------
    # core dispatch logic
    # ------------------------------------------------------------------
    def _execute_step(self, action: Action) -> HospitalmanageTriageObservation:
        if self._state is None:
            return HospitalmanageTriageObservation(
                result="Environment not reset. Call reset() first.",
                done=False,
                reward=-0.1,
            )

        if not isinstance(action, CallToolAction):
            return HospitalmanageTriageObservation(
                result=f"Unknown action type: {type(action).__name__}. Use CallToolAction.",
                done=False,
                reward=-0.1,
            )

        tool_name = action.tool_name
        arguments = action.arguments or {}

        if tool_name not in self.TOOL_DISPATCH:
            return HospitalmanageTriageObservation(
                tool_name=tool_name,
                result=f"Unknown tool: {tool_name}. Use one of: {list(self.TOOL_DISPATCH)}",
                tool_executed=tool_name,
                done=False,
                reward=-0.5,
            )

        # --- direct call to ToolSpec ---
        try:
            method = getattr(self.tools, self.TOOL_DISPATCH[tool_name])
            tool_output = method(**arguments)
        except TypeError as e:
            return HospitalmanageTriageObservation(
                tool_name=tool_name,
                result=f"Tool argument error for '{tool_name}': {e}",
                tool_executed=tool_name,
                done=False,
                reward=-0.5,
            )
        except Exception as e:
            return HospitalmanageTriageObservation(
                tool_name=tool_name,
                result=f"Tool '{tool_name}' raised an error: {e}",
                tool_executed=tool_name,
                done=False,
                reward=-0.1,
            )

        tool_called = getattr(tool_output, "tool", tool_name)
        output_message = getattr(tool_output, "message", str(tool_output))
        tool_state = getattr(tool_output, "tool_state", {})

        print(f"[TOOL] {tool_called}: {output_message}")

        # --- update state ---
        self._state.step_count += 1
        for key, value in tool_state.items():
            if key != "episode_id" and value is not None:
                try:
                    setattr(self._state, key, value)
                except Exception:
                    pass

        # --- reward calculation ---
        reward = 0.0
        done_tag = False
        try:
            seq_len = len(self._state.tool_call_sequence)
            if seq_len > 0 and self._state.tool_state_step < seq_len:
                expected_tool = self._state.tool_call_sequence[self._state.tool_state_step]
                expected_out = self._state.output_sequence[self._state.tool_state_step]
                if tool_called == expected_tool and expected_out in output_message:
                    reward = 1.0 / seq_len
                    self._state.tool_state_step += 1
                else:
                    reward = -0.5 / seq_len

            if self._state.tool_state_step >= seq_len and seq_len > 0:
                done_tag = True
        except Exception:
            reward = 0.0

        return HospitalmanageTriageObservation(
            tool_name=tool_called,
            result=f"Patient id {self._state.patient_id}\n{output_message}",
            tool_executed=tool_called,
            done=done_tag,
            reward=reward,
            metadata={
                "original_message": output_message,
                "step": self._state.step_count,
            },
        )

    # ------------------------------------------------------------------
    # state property (required by base class)
    # ------------------------------------------------------------------
    @property
    def state(self) -> HospitalState:
        return self._state

    def close(self) -> None:
        pass

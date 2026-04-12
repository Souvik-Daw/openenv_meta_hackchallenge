# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hospitalmanage Triage Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from typing import Any, Optional
from uuid import uuid4
import random
import inspect

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.interfaces import Action
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import State
from .tools import ToolSpec
try:
    from ..models import  HospitalmanageTriageObservation,HospitalState
except ImportError:
    from models import  HospitalmanageTriageObservation,HospitalState
from fastmcp import FastMCP

class HospitalmanageTriageEnvironment(MCPEnvironment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = HospitalmanageTriageEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Hospitalmanage Triage Env environment ready!"
        >>>
        >>> obs = env.step(HospitalmanageTriageAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def _compute_failure_score(self, tool_name: str, arguments: dict):
        # Compute a failure score for the given tool and arguments.
        # This is a placeholder implementation.
        reward = 1.0
        message = ""
        if tool_name not in dir(self.tools):
            reward -= 0.3
            message += f"Invalid tool name: {tool_name}"
        for arg_name,_ in arguments.items():
            if arg_name not in inspect.getfullargspec(getattr(self.tools, tool_name)).args:
                reward -= 0.3
                message += f"Invalid argument: {arg_name} "
                break
        for _,arg_value in arguments.items():
            if arg_value not in self._state.output_sequence[self._state.tool_state_step]:
                reward -= 0.4
                message += f"Incorrect argument: {arg_value},type({type(arg_value)}). in {self._state.output_sequence[self._state.tool_state_step]}"
                break
        return message, reward

    def __init__(self):
        """Initialize the hospitalmanage_triage_env environment."""

        mcp = FastMCP("hospital_env")
        self._reset_count = 0
        self.tools = ToolSpec()
        mcp.add_tool(self.tools.get_department)
        mcp.add_tool(self.tools.get_opd_doctor)
        mcp.add_tool(self.tools.make_appointment)
        mcp.add_tool(self.tools.get_appointment)
        super().__init__(mcp)


    def reset(self,
              episode_id: Optional[str] = None,
              patient_id: Optional[int]= None,
              tool_call_sequence: Optional[list] = [],
              output_sequence: Optional[list] = [],
              ) -> HospitalmanageTriageObservation:
        """
        Reset the environment.

        Returns:
            HospitalmanageTriageObservation with a ready message
        """
        self._state = HospitalState(episode_id=episode_id or str(uuid4()), 
                            step_count=0,
                            patient_id=patient_id or random.randint(1000, 9999),
                            tool_call_sequence=tool_call_sequence,
                            output_sequence= output_sequence)
        self._reset_count += 1

        return HospitalmanageTriageObservation(
            result="Hospitalmanage Triage Env environment ready!",
            done=False,
            reward=0.0,
        )
    
    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HospitalmanageTriageObservation:
        """
        Handle non-MCP actions.

        This environment only supports MCP actions (ListToolsAction, CallToolAction).
        Any other action type returns an error observation.

        Args:
            action: The action to execute
            timeout_s: Optional timeout (unused)
            **kwargs: Additional arguments

        Returns:
            Observation with error for unknown action types
        """
        return HospitalmanageTriageObservation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(self, action: Action, timeout_s: Optional[float] = 120) -> HospitalmanageTriageObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: HospitalmanageTriageAction containing the message to echo

        Returns:
            HospitalmanageTriageObservation with the echoed message and its length
        """
        self._state.step_count += 1


        if isinstance(action, CallToolAction):
            message, reward = self._compute_failure_score(action.tool_name, action.arguments)
            if reward < 1.0:
                return HospitalmanageTriageObservation(
                    tool_name=action.tool_name,
                    done=False,
                    reward=reward,
                    result=message,
                )


        message = super().step(action, timeout_s=timeout_s)
        # Avoid referencing an undefined CallToolResult symbol; check for a result attribute instead.
        if hasattr(message, "result") and message.result is not None:
            message_dict = message.result.structured_content
            tool_called = message_dict.get("tool")
            output_message = message_dict.get("message")
            tool_state = message_dict.get("tool_state")

        # state update
        for key, value in tool_state.items():
            if key != "episode_id":
                setattr(self._state, key, value)
        # tool reward  + add output reward
        self._state.tool_state_step += 1
        done_tag = False
        if self._state.tool_state_step >= len(self._state.tool_call_sequence):
            done_tag = True

        return HospitalmanageTriageObservation(
            tool_name=tool_called,
            result=f'Patient id {self._state.patient_id} \n ' + (output_message or '<Tool Error>'),
            tool_executed=tool_called,
            done=done_tag,
            reward=max(0.01, min(0.99, reward)),
            metadata={"original_message": output_message, "step": self._state.step_count},
        )
    
    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = 120.0,
    ) -> HospitalmanageTriageObservation:
        """
        Async step used by the WebSocket handler.

        Increments step count then delegates to MCPEnvironment.step_async,
        which routes MCP actions without going through run_async_safely.
        """
        self._state.step_count += 1

        if isinstance(action, CallToolAction):
            message, reward = self._compute_failure_score(action.tool_name, action.arguments)
            if reward < 1.0:
                return HospitalmanageTriageObservation(
                    tool_name=action.tool_name,
                    done=False,
                    reward=reward,
                    result=message,
                )
            
        print(reward)
        message =  await super().step_async(action, timeout_s=timeout_s)

        if hasattr(message, "result") and message.result is not None:
            message_dict = message.result.structured_content
            tool_called = message_dict.get("tool")
            output_message = message_dict.get("message")
            tool_state = message_dict.get("tool_state")

        # update state variables + think of edge cases if (if fails, if tool fails, if wrong tool called) as message changes '3 + add grading
        for key, value in tool_state.items():
            if key != "episode_id":
                setattr(self._state, key, value)
        
        # tool reward  + add output reward
        self._state.tool_state_step += 1

        # condition for done
        done_tag = False
        if self._state.tool_state_step >= len(self._state.tool_call_sequence):
            done_tag = True
        
        return HospitalmanageTriageObservation(
            tool_name=tool_called,
            result=f'Patient id {self._state.patient_id} \n ' + (output_message or '<Tool Error>'),
            tool_executed=tool_called,
            done=done_tag,
            reward=max(0.01, min(0.99, reward)),
            metadata={"original_message": output_message, "step": self._state.step_count},
        )

    @property
    def state(self) -> HospitalState:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        print(self._state)
        return self._state

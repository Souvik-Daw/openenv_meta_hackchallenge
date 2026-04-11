# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Hospitalmanage Triage Env Environment.

Uses a single persistent environment instance with a simple REST API.
No MCP / WebSocket transport — direct HTTP only.

Endpoints:
    POST /reset  - Reset the environment and start a new episode
    POST /step   - Execute a tool action
    GET  /state  - Get current environment state
    GET  /health - Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import uvicorn

try:
    from .hospitalmanage_triage_env_environment import HospitalmanageTriageEnvironment
except ImportError:
    from server.hospitalmanage_triage_env_environment import HospitalmanageTriageEnvironment

# ---------------------------------------------------------------------------
# Singleton environment — state is kept across /reset + /step calls
# ---------------------------------------------------------------------------
_env = HospitalmanageTriageEnvironment()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    patient_id: Optional[int] = None
    tool_call_sequence: Optional[List[str]] = None
    output_sequence: Optional[List[str]] = None
    episode_id: Optional[str] = None


class StepActionRequest(BaseModel):
    type: str = Field(default="call_tool")
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    action: StepActionRequest


class ObservationResponse(BaseModel):
    tool_name: str = ""
    result: str = ""
    tool_executed: str = ""
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EnvResponse(BaseModel):
    observation: ObservationResponse
    reward: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hospital Triage Environment",
    description="Hospital triage RL environment — direct HTTP API",
    version="1.0.0",
)


def _obs_to_dict(obs) -> Dict[str, Any]:
    """Convert a HospitalmanageTriageObservation to a plain dict."""
    return {
        "tool_name":     getattr(obs, "tool_name", ""),
        "result":        getattr(obs, "result", ""),
        "tool_executed": getattr(obs, "tool_executed", ""),
        "done":          getattr(obs, "done", False),
        "reward":        getattr(obs, "reward", 0.0),
        "metadata":      getattr(obs, "metadata", {}),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=EnvResponse)
def reset(request: ResetRequest = None):
    """Reset the environment and begin a new episode."""
    req = request or ResetRequest()
    obs = _env.reset(
        patient_id=req.patient_id,
        tool_call_sequence=req.tool_call_sequence,
        output_sequence=req.output_sequence,
        episode_id=req.episode_id,
    )
    obs_dict = _obs_to_dict(obs)
    return EnvResponse(
        observation=ObservationResponse(**obs_dict),
        reward=obs_dict["reward"],
        done=obs_dict["done"],
    )


@app.post("/step", response_model=EnvResponse)
def step(request: StepRequest):
    """Execute a tool action and return the observation."""
    from openenv.core.env_server.mcp_types import CallToolAction

    action = CallToolAction(
        tool_name=request.action.tool_name,
        arguments=request.action.arguments,
    )
    obs = _env.step(action)
    obs_dict = _obs_to_dict(obs)
    return EnvResponse(
        observation=ObservationResponse(**obs_dict),
        reward=obs_dict["reward"],
        done=obs_dict["done"],
    )


@app.get("/state")
def state():
    """Return current episode state."""
    s = _env.state
    if s is None:
        return {"state": None}
    return {"state": s.model_dump() if hasattr(s, "model_dump") else vars(s)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

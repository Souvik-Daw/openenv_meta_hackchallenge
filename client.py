# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hospitalmanage Triage Env - Pure HTTP client.

Uses the FastAPI /reset and /step endpoints directly (no WebSocket / MCP).
This avoids all WebSocket keepalive and handshake timeout issues.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Dict, Optional

import requests

from openenv.core.client_types import StepResult


# ---------------------------------------------------------------------------
# Lightweight observation wrapper so callers can do result.observation.result
# ---------------------------------------------------------------------------
class _Obs:
    """Wraps the raw observation dict returned by the server."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    # Allow attribute access: obs.result, obs.reward, obs.done, obs.tool_name …
    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'Observation' has no field '{name}'") from None

    def __repr__(self):
        return f"Observation({self._data})"


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------
class HospitalmanageTriageEnv:
    """
    Synchronous HTTP client for the Hospital Triage environment.

    Usage:
        env = HospitalmanageTriageEnv(base_url="http://localhost:8000").sync()
        result = env.reset(patient_id=1001, tool_call_sequence=[...], output_sequence=[...])
        result = env.step(CallToolAction(tool_name="get_department", arguments={...}))
        env.close()

    Also works as a context manager:
        with HospitalmanageTriageEnv(base_url="http://localhost:8000").sync() as env:
            ...
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # .sync() — returns self (already synchronous)
    # ------------------------------------------------------------------
    def sync(self) -> "HospitalmanageTriageEnv":
        return self

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "HospitalmanageTriageEnv":
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        patient_id: Optional[int] = None,
        tool_call_sequence: Optional[list] = None,
        output_sequence: Optional[list] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> StepResult:
        payload: Dict[str, Any] = {}
        if patient_id is not None:
            payload["patient_id"] = patient_id
        if tool_call_sequence is not None:
            payload["tool_call_sequence"] = tool_call_sequence
        if output_sequence is not None:
            payload["output_sequence"] = output_sequence
        if episode_id is not None:
            payload["episode_id"] = episode_id

        resp = self._session.post(
            f"{self._base_url}/reset",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = _Obs(data.get("observation", data))
        return StepResult(
            observation=obs,
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action) -> StepResult:
        # action is a CallToolAction (Pydantic); serialize to dict
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif hasattr(action, "__dict__"):
            action_dict = vars(action)
        else:
            action_dict = dict(action)

        payload = {"action": action_dict}
        resp = self._session.post(
            f"{self._base_url}/step",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = _Obs(data.get("observation", data))
        return StepResult(
            observation=obs,
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
        )

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------
    def close(self):
        self._session.close()

    # ------------------------------------------------------------------
    # Docker / HF stub (keeps the inference.py IMAGE_NAME path compiling)
    # ------------------------------------------------------------------
    @classmethod
    def from_docker_image(cls, image_name: str, **kwargs) -> "HospitalmanageTriageEnv":
        raise NotImplementedError(
            "Docker mode is not supported by this pure-HTTP client. "
            "Start the server manually and pass base_url."
        )

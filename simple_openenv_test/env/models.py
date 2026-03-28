from pydantic import BaseModel, Field
from typing import List, Literal

class Observation(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the request.")
    request_text: str = Field(..., description="The user's input request text.")
    possible_departments: List[str] = Field(..., description="List of valid departments to route to.")
    step_count: int = Field(..., description="Current step in the environment.")

class Action(BaseModel):
    selected_department: Literal["billing", "tech", "general"] = Field(..., description="The chosen department.")

class Reward(BaseModel):
    reward: float = Field(..., description="Numerical reward for the routing decision.")
    correctness: bool = Field(..., description="Whether the routing was correct.")

class State(BaseModel):
    request_id: str = Field(..., description="The processed request ID.")
    request_text: str = Field(..., description="The user's input request text.")
    true_department: str = Field(..., description="The ground-truth department.")
    action_taken: str | None = Field(None, description="The department the agent assigned.")
    score: float = Field(0.0, description="Final score 1.0 or 0.0.")
    done: bool = Field(False, description="Whether the episode is complete.")

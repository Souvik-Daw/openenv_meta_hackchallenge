from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Inheriting from BaseModel fixes the 'model_dump' error

class WordGameAction(BaseModel):
    """Player guesses a single letter."""
    guess: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WordGameObservation(BaseModel):
    """What the player sees after each guess."""
    done: bool
    reward: Optional[float] = 0.0
    masked_word: str             # e.g., "p_th_n"
    guessed_letters: List[str]   # All letters tried
    attempts_remaining: int
    message: str                 # Feedback text
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WordGameState(BaseModel):
    """Episode metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    target_word: str = ""
    max_attempts: int = 6
import random
import uuid
from word_game.models import WordGameAction, WordGameObservation, WordGameState
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

WORDS = [
    "python", "neural", "tensor", "matrix", "vector",
    "kernel", "lambda", "signal", "binary", "cipher",
    "model", "layer", "epoch", "batch", "token",
]

class WordGameEnvironment:
    """A letter-guessing game environment following the OpenEnv pattern."""

    def __init__(self):
        self._state = WordGameState()
        self._target = ""
        self._guessed = set()
        self._remaining = 6

    def reset(self) -> WordGameObservation:
        """Start a new episode with a random word."""
        self._target = random.choice(WORDS)
        self._guessed = set()
        self._remaining = 6
        self._state = WordGameState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            target_word=self._target,
            max_attempts=6,
        )
        return WordGameObservation(
            done=False,
            reward=None,
            masked_word=self._mask(),
            guessed_letters=[],
            attempts_remaining=self._remaining,
            message=f"Guess letters in a {len(self._target)}-letter word!",
        )

    async def reset_async(self) -> WordGameObservation:
        """Start a new episode with a random word."""
        self._target = random.choice(WORDS)
        self._guessed = set()
        self._remaining = 6
        self._state = WordGameState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            target_word=self._target,
            max_attempts=6,
        )
        return WordGameObservation(
            done=False,
            reward=None,
            masked_word=self._mask(),
            guessed_letters=[],
            attempts_remaining=self._remaining,
            message=f"Guess letters in a {len(self._target)}-letter word!",
        )

    def step(self, action: WordGameAction) -> WordGameObservation:
        """Process a letter guess."""
        letter = action.guess.lower().strip()
        self._state.step_count += 1

        # Already guessed?
        if letter in self._guessed:
            return WordGameObservation(
                done=False,
                reward=0.0,
                masked_word=self._mask(),
                guessed_letters=sorted(self._guessed),
                attempts_remaining=self._remaining,
                message=f"Already guessed '{letter}'. Try another.",
            )

        self._guessed.add(letter)

        if letter in self._target:
            message = f"'{letter}' is in the word!"
        else:
            self._remaining -= 1
            message = f"'{letter}' is not in the word."

        # Check win/lose
        masked = self._mask()
        won = "_" not in masked
        lost = self._remaining <= 0
        done = won or lost

        if won:
            reward = 1.0
            message = f"You got it! The word was '{self._target}'."
        elif lost:
            reward = 0.0
            message = f"Out of attempts. The word was '{self._target}'."
        else:
            reward = 0.0

        return WordGameObservation(
            done=done,
            reward=reward,
            masked_word=masked,
            guessed_letters=sorted(self._guessed),
            attempts_remaining=self._remaining,
            message=message,
        )

    async def step_async(self, action: WordGameAction) -> WordGameObservation:
        """Process a letter guess."""
        letter = action.guess.lower().strip()
        self._state.step_count += 1

        # Already guessed?
        if letter in self._guessed:
            return WordGameObservation(
                done=False,
                reward=0.0,
                masked_word=self._mask(),
                guessed_letters=sorted(self._guessed),
                attempts_remaining=self._remaining,
                message=f"Already guessed '{letter}'. Try another.",
            )

        self._guessed.add(letter)

        if letter in self._target:
            message = f"'{letter}' is in the word!"
        else:
            self._remaining -= 1
            message = f"'{letter}' is not in the word."

        # Check win/lose
        masked = self._mask()
        won = "_" not in masked
        lost = self._remaining <= 0
        done = won or lost

        if won:
            reward = 1.0
            message = f"You got it! The word was '{self._target}'."
        elif lost:
            reward = 0.0
            message = f"Out of attempts. The word was '{self._target}'."
        else:
            reward = 0.0

        return WordGameObservation(
            done=done,
            reward=reward,
            masked_word=masked,
            guessed_letters=sorted(self._guessed),
            attempts_remaining=self._remaining,
            message=message,
        )

    @property
    def state(self) -> WordGameState:
        return self._state

    def _mask(self) -> str:
        """Show guessed letters, hide the rest."""
        return "".join(c if c in self._guessed else "_" for c in self._target)

print("WordGameEnvironment defined.")
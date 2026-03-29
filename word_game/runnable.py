
from word_game.server.word_game_environment import WordGameEnvironment
from word_game.models import WordGameAction, WordGameObservation

env = WordGameEnvironment()
obs: WordGameObservation = env.reset()
print(f"Word: {obs.masked_word} ({len(obs.masked_word)} letters)")
print(f"Message: {obs.message}")
print(f"Attempts: {obs.attempts_remaining}")
print()

# Play with common letters
for letter in ["e", "a", "t", "n", "o", "r", "s", "i", "l"]:
    if obs.done:
        break
    obs = env.step(WordGameAction(guess=letter))
    print(f"  Guess '{letter}': {obs.masked_word}  ({obs.message})")

print(f"\nFinal: reward={obs.reward}, done={obs.done}")
print(f"State: episode={env.state.episode_id[:8]}..., steps={env.state.step_count}")
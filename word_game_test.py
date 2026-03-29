from word_game.client import WordGameEnv, WordGameAction

#with WordGameEnv(base_url="https://Fergus2000-word-game.hf.space").sync() as env:
with WordGameEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    result = env.step(WordGameAction(guess="e"))
    print(result.observation.masked_word)
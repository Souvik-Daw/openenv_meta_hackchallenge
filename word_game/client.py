from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from word_game.models import WordGameAction, WordGameObservation, WordGameState

class WordGameEnv(EnvClient[WordGameAction, WordGameObservation,WordGameState]):
    def _step_payload(self, action):
        return {"guess": action.guess}

    def _parse_result(self, payload):
        # 1. Get the 'done' status from the main payload
        is_done = payload.get("done", False)
        
        # 2. Extract the observation data
        obs_data = payload.get("observation", {})
        
        # 3. Manually add 'done' to the obs_data so WordGameObservation is happy
        if "done" not in obs_data:
            obs_data["done"] = is_done
            
        return StepResult(
            observation=WordGameObservation(**obs_data),
            reward=payload.get("reward", 0),
            done=is_done,
        )

    def _parse_state(self, payload):
        # Check if the state is nested inside a 'state' key
        state_data = payload.get("state", payload)
        return WordGameState(**state_data)

#from word_game.client import WordGameEnv, WordGameAction

# #with WordGameEnv(base_url="https://Fergus2000-word-game.hf.space").sync() as env:
# with WordGameEnv(base_url="http://localhost:8000").sync() as env:
#     result = env.reset()
#     result = env.step(WordGameAction(guess="e"))
#     print(result.observation.masked_word)
from openenv.core.env_server import create_fastapi_app
from word_game.server.word_game_environment import WordGameEnvironment
from word_game.models import WordGameAction, WordGameObservation, WordGameState

app = create_fastapi_app(WordGameEnvironment,WordGameAction,WordGameObservation)
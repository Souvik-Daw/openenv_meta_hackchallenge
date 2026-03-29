# openenv_meta_hackchallenge

Meta Pytorch AI Hackathon

. RL env project ideas
Sample of the RL env project ideas (Done)

. simple_agent
Sample of the simple agent project (Done)

. word_game
RL env project (Working locally but not working in HF space)

Running locally:
python -m uvicorn word_game.server.app:app --host 0.0.0.0 --port 8000 --reload
python -m word_game.runnable

Push to HF space:
openenv push --repo-id Fergus2000/word_game

. openspiel_env hf_test
openspiel folder structure modify to work in HF space

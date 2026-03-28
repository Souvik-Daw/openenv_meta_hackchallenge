from env.env import TaskRoutingEnv
from env.models import Action

class GroqRoutingTool:
    def __init__(self, env: TaskRoutingEnv):
        self.env = env

    def reset_env(self):
        obs = self.env.reset()
        return obs.model_dump()

    def step_env(self, action_dict: dict):
        action = Action(**action_dict)
        obs, reward, done, info = self.env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }

    def get_state(self):
        return self.env.state().model_dump()

    @staticmethod
    def get_tool_schema():
        return [
            {
                "type": "function",
                "function": {
                    "name": "step_env",
                    "description": "Routes the user request to a specific department.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_department": {
                                "type": "string",
                                "enum": ["billing", "tech", "general"],
                                "description": "The department to route to."
                            }
                        },
                        "required": ["selected_department"]
                    }
                }
            }
        ]

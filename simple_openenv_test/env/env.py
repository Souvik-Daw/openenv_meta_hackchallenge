from env.models import Observation, Action, Reward, State
from env.tasks import get_task

class TaskRoutingEnv:
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self._state = None
        self.departments = ["billing", "tech", "general"]

    def reset(self) -> Observation:
        task_data = get_task(self.task_id)
        self._state = State(
            request_id=task_data["req_id"],
            request_text=task_data["text"],
            true_department=task_data["true_dept"],
            action_taken=None,
            score=0.0,
            done=False
        )
        return self._get_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if getattr(self, "_state", None) is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            return self._get_observation(), Reward(reward=0.0, correctness=False), True, {}

        # Episode ends immediately after 1 step
        self._state.action_taken = action.selected_department
        self._state.done = True

        correct = (action.selected_department == self._state.true_department)
        reward_val = 1.0 if correct else -1.0
        self._state.score = 1.0 if correct else 0.0

        reward = Reward(reward=reward_val, correctness=correct)
        
        info = {
            "true_department": self._state.true_department
        }

        return self._get_observation(), reward, self._state.done, info

    def state(self) -> State:
        return self._state

    def _get_observation(self) -> Observation:
        return Observation(
            request_id=self._state.request_id,
            request_text=self._state.request_text,
            possible_departments=self.departments,
            step_count=0 if not self._state.done else 1
        )

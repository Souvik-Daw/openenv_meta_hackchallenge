# 📁 PROJECT: AI Technical Decision Maker (Architect Simulator)

## 1. Problem Statement
Designing a scalable, cost-effective cloud architecture requires balancing tradeoffs between performance, reliability, and budget. The AI agent acts as a Cloud Architect, receiving project requirements (e.g., target RPS, data volume, budget) and iteratively selecting components (Compute, Database, Caching, Messaging) to build an architecture that meets the SLA while minimizing costs.

## 2. Observation Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import List, Dict, Optional

class Requirements(BaseModel):
    target_rps: int
    data_volume_gb: float
    requires_acid: bool
    budget_usd_per_month: float

class CurrentArchitecture(BaseModel):
    compute: Optional[str] = None
    database: Optional[str] = None
    cache: Optional[str] = None
    queue: Optional[str] = None
    current_estimated_cost: float = 0.0
    current_estimated_rps: int = 0

class Observation(BaseModel):
    requirements: Requirements
    architecture: CurrentArchitecture
    step_count: int
    feedback: str
```

## 3. Action Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import Literal

class Action(BaseModel):
    action_type: Literal["select_compute", "select_db", "select_cache", "select_queue", "finalize"]
    component_choice: Optional[str] = None  # e.g., "ec2_t3_micro", "rds_postgres", "dynamodb", "redis"
```

## 4. Reward Model (Pydantic) + Code
```python
from pydantic import BaseModel

class Reward(BaseModel):
    step_reward: float
    total_cost_penalty: float
    sla_met_bonus: float
    message: str
```

## 5. Environment Implementation (env.py)
```python
import openenv
from models import Observation, Action, Reward, Requirements, CurrentArchitecture

COMPONENT_SPECS = {
    "rds_postgres": {"cost": 100.0, "rps": 2000, "acid": True},
    "dynamodb": {"cost": 50.0, "rps": 10000, "acid": False},
    "ec2_large": {"cost": 80.0, "rps": 5000},
    "lambda": {"cost": 20.0, "rps": 2000},
    "redis": {"cost": 30.0, "rps": 20000},
    "sqs": {"cost": 10.0, "rps": 5000}
}

class ArchitectEnv(openenv.BaseEnv):
    def __init__(self, requirements: dict):
        self.init_reqs = requirements
        self.reset()

    def reset(self) -> Observation:
        self.reqs = Requirements(**self.init_reqs)
        self.arch = CurrentArchitecture()
        self.step_count = 0
        self.done = False
        return self.state()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            raise ValueError("Environment already done. Call reset().")
        
        self.step_count += 1
        reward_val = 0.0
        feedback = "Action accepted."

        if action.action_type == "finalize":
            self.done = True
            final_reward, feedback = self._evaluate_architecture()
            return self.state(feedback), final_reward, self.done, {}

        # Update architecture
        choice = action.component_choice
        if choice not in COMPONENT_SPECS:
            reward_val = -0.1
            feedback = f"Invalid component: {choice}"
        else:
            specs = COMPONENT_SPECS[choice]
            if action.action_type == "select_compute": self.arch.compute = choice
            elif action.action_type == "select_db": self.arch.database = choice
            elif action.action_type == "select_cache": self.arch.cache = choice
            elif action.action_type == "select_queue": self.arch.queue = choice
            
            # Recalculate metrics
            self._update_metrics()
            reward_val = 0.01

        return self.state(feedback), Reward(step_reward=reward_val, total_cost_penalty=0, sla_met_bonus=0, message=feedback), self.done, {}

    def _update_metrics(self):
        cost = 0.0
        rps = 0
        for comp in [self.arch.compute, self.arch.database, self.arch.cache, self.arch.queue]:
            if comp and comp in COMPONENT_SPECS:
                cost += COMPONENT_SPECS[comp]["cost"]
                rps = min(rps, COMPONENT_SPECS[comp]["rps"]) if rps > 0 else COMPONENT_SPECS[comp]["rps"]
        if self.arch.cache: rps += COMPONENT_SPECS[self.arch.cache]["rps"]
        self.arch.current_estimated_cost = cost
        self.arch.current_estimated_rps = rps

    def _evaluate_architecture(self) -> tuple[Reward, str]:
        if not self.arch.compute or not self.arch.database:
            return Reward(step_reward=-1.0, total_cost_penalty=0, sla_met_bonus=0, message="Incomplete arch"), "Missing core components."
        
        cost_penalty = 0.0
        sla_bonus = 0.0
        if self.arch.current_estimated_cost > self.reqs.budget_usd_per_month:
            cost_penalty = -0.5
        
        if self.arch.current_estimated_rps >= self.reqs.target_rps:
            sla_bonus = 1.0
        else:
            sla_bonus = -0.5
            
        final_score = sla_bonus + cost_penalty
        return Reward(step_reward=final_score, total_cost_penalty=cost_penalty, sla_met_bonus=sla_bonus, message="Arch evaluated"), f"Final score: {final_score}"

    def state(self, feedback: str = "") -> Observation:
        return Observation(requirements=self.reqs, architecture=self.arch, step_count=self.step_count, feedback=feedback)
```

## 6. Reward Function Design (Explanation + Formula)
**Explanation:** Dense rewards are given for valid component selections (+0.01) to encourage action. Invalid actions yield a small penalty (-0.1). Upon finalization, the environment checks if the chosen components meet the target RPS and fit within the budget, adding `-0.5` if strictly over budget, `+1.0` if SLA met, `-0.5` if SLA failed.
**Formula:** `R = sum(valid_steps * 0.01) - sum(invalid_steps * 0.1) + (1.0 if RPS >= Target else -0.5) - (0.5 if Cost > Budget else 0.0)`

## 7. Tasks Definition (3 Tasks: Easy / Medium / Hard)
* **Easy:** Target RPS: 1000, Budget: \$200. No ACID requirement.
* **Medium:** Target RPS: 6000, Budget: \$120. Needs ACID capability.
* **Hard:** Target RPS: 20000, Budget: \$150. Requires integrating caching and queues to offset compute limits while staying under a tight budget.

## 8. Graders (Code + Deterministic Logic)
```python
def grade_architecture(final_obs: Observation) -> float:
    reqs = final_obs.requirements
    arch = final_obs.architecture
    
    if not arch.compute or not arch.database: return 0.0
    
    score = 0.0
    # 50% for SLA
    if arch.current_estimated_rps >= reqs.target_rps:
        score += 0.5
    else:
        score += 0.5 * (arch.current_estimated_rps / reqs.target_rps)
        
    # 50% for Budget
    if arch.current_estimated_cost <= reqs.budget_usd_per_month:
        score += 0.5
    else:
        ratio = reqs.budget_usd_per_month / arch.current_estimated_cost
        score += 0.5 * max(0, ratio - 0.5) # drops quickly if over budget

    # ACID constraint hard fail
    if reqs.requires_acid:
        db_spec = COMPONENT_SPECS.get(arch.database, {})
        if not db_spec.get("acid", False):
            return 0.0

    return min(1.0, max(0.0, score))
```

## 9. openenv.yaml
```yaml
name: "architect-simulator"
version: "1.0.0"
description: "AI Agent acts as a Cloud Architect designing production systems."
entrypoint: "env.py:ArchitectEnv"
tasks:
  - id: "task_easy"
    params: {"target_rps": 1000, "data_volume_gb": 10.0, "requires_acid": False, "budget_usd_per_month": 200.0}
  - id: "task_medium"
    params: {"target_rps": 6000, "data_volume_gb": 100.0, "requires_acid": True, "budget_usd_per_month": 120.0}
  - id: "task_hard"
    params: {"target_rps": 20000, "data_volume_gb": 500.0, "requires_acid": True, "budget_usd_per_month": 150.0}
models:
  observation: "models.py:Observation"
  action: "models.py:Action"
  reward: "models.py:Reward"
```

## 10. Baseline Agent Script (OpenAI API)
```python
import os
from openai import OpenAI
from env import ArchitectEnv
from models import Action

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_agent(env):
    obs = env.reset()
    system_prompt = "You are a cloud architect. Choose compute, db, cache, queue, then finalize."
    messages = [{"role": "system", "content": system_prompt}]
    
    while not env.done:
        messages.append({"role": "user", "content": f"Current State: {obs.model_dump_json()}"})
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[{
                "name": "take_action",
                "parameters": Action.model_json_schema()
            }],
            function_call={"name": "take_action"}
        )
        
        action_args = response.choices[0].message.function_call.arguments
        action = Action.model_validate_json(action_args)
        
        obs, reward, done, info = env.step(action)
        messages.append({"role": "assistant", "content": f"Took action: {action_args}"})
```

## 11. Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "evaluator.py"]
```

## 12. Project Structure
```text
architect_simulator/
├── env.py
├── models.py
├── evaluator.py
├── baseline_agent.py
├── requirements.txt
├── Dockerfile
└── openenv.yaml
```

## 13. Evaluation Metrics
* **Score (0.0-1.0):** From the deterministic grader evaluating SLA and Budget compliance.
* **Cost Efficiency:** Ratio of (SLA achieved) / (Budget Spent).
* **Steps Taken:** Number of steps to reach `finalize`.

## 14. Edge Cases
* Agent selecting multiple DBs (only the last one is recorded).
* Agent finalizing on step 1 without selecting components (handled by reward penalty & 0 grader score).
* Agent choosing invalid strings (handled by small penalty and immediate feedback).

## 15. Hackathon Demo Plan
* Show the standard UI rendering the agent's iterative architecture diagram.
* Run the GPT-4 agent on the Hard Task, showing how it realizes `ec2_large` alone fails RPS, and dynamically adds `redis` to boost RPS without blowing the budget.

## 16. Bonus Enhancements
* Introduce realistic latency penalties between components.
* Support real-time lookup mapping to AWS Pricing API instead of static dicts.

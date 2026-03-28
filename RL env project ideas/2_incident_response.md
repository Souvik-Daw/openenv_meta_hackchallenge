# 📁 PROJECT: Production Incident Response Environment (Issue → Response → Evidence)

## 1. Problem Statement
When a PagerDuty alert fires, Site Reliability Engineers (SREs) must quickly identify the root cause and restore service. This environment simulates a microservices architecture facing an active incident. The AI agent must analyze metrics, grep logs, deduce the issue (e.g., bad deployment, memory leak, DB connection exhaustion), and apply the correct remediation (e.g., rollback, restart, scale).

## 2. Observation Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import List, Dict, Optional

class Alert(BaseModel):
    id: str
    service: str
    description: str
    severity: str

class SystemState(BaseModel):
    cpu_usage: Dict[str, float]
    memory_usage: Dict[str, float]
    error_rates: Dict[str, float]

class Observation(BaseModel):
    active_alerts: List[Alert]
    system_state: SystemState
    last_action_result: str
    time_elapsed_minutes: int
    incident_resolved: bool
```

## 3. Action Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import Literal, Optional

class Action(BaseModel):
    action_type: Literal[
        "view_metrics", 
        "query_logs", 
        "restart_service", 
        "rollback_deployment", 
        "scale_up", 
        "mark_resolved"
    ]
    target_service: Optional[str] = None
    log_query: Optional[str] = None
```

## 4. Reward Model (Pydantic) + Code
```python
from pydantic import BaseModel

class Reward(BaseModel):
    step_cost: float
    resolution_bonus: float
    wrong_action_penalty: float
    total_reward: float
```

## 5. Environment Implementation (env.py)
```python
import openenv
from models import Observation, Action, Reward, Alert, SystemState

class IncidentEnv(openenv.BaseEnv):
    def __init__(self, incident_type: str, failing_service: str):
        self.incident_type = incident_type
        self.failing_service = failing_service
        self.reset()
        
    def reset(self) -> Observation:
        self.time = 0
        self.resolved = False
        self.done = False
        self.last_res = "PagerDuty Alert Triggered."
        
        self.state_data = SystemState(
            cpu_usage={"payment": 45.0, "auth": 20.0, "db": 80.0},
            memory_usage={"payment": 99.0 if self.incident_type == "oom" else 50.0, "auth": 30.0, "db": 60.0},
            error_rates={"payment": 0.0, "auth": 0.0, "db": 0.0}
        )
        if self.incident_type == "bad_deploy":
            self.state_data.error_rates[self.failing_service] = 0.8
            
        self.alerts = [Alert(id="INC-101", service=self.failing_service, description="High Error Rate/Latency", severity="SEV-1")]
        return self.state()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            raise ValueError("Environment done.")
        
        self.time += 5  # Each action takes 5 virtual minutes
        step_cost = -0.05
        penalty = 0.0
        bonus = 0.0

        if action.action_type == "mark_resolved":
            self.done = True
            if self.resolved:
                self.last_res = "Incident successfully resolved."
                bonus = 1.0
            else:
                self.last_res = "Incident marked resolved but issues persist!"
                penalty = -1.0
        
        elif action.action_type == "view_metrics":
            self.last_res = f"Metrics for {action.target_service}: CPU={self.state_data.cpu_usage.get(action.target_service, 0)}%, Mem={self.state_data.memory_usage.get(action.target_service, 0)}%"
            
        elif action.action_type == "query_logs":
            if self.incident_type == "bad_deploy" and action.target_service == self.failing_service:
                self.last_res = 'ERROR: NullPointerException in Handler v2.1.0'
            elif self.incident_type == "oom" and action.target_service == self.failing_service:
                self.last_res = 'FATAL: OutOfMemoryError: Java heap space'
            else:
                self.last_res = "INFO: Service starting... INFO: Ready."
                
        elif action.action_type == "rollback_deployment":
            if self.incident_type == "bad_deploy" and action.target_service == self.failing_service:
                self.resolved = True
                self.last_res = f"Service {action.target_service} rolled back to stable version. Errors dropped."
                self.state_data.error_rates[self.failing_service] = 0.0
            else:
                self.last_res = "Rollback applied, but no effect."
                penalty = -0.2
        
        elif action.action_type == "restart_service":
            if self.incident_type == "oom" and action.target_service == self.failing_service:
                self.resolved = True
                self.last_res = f"Service {action.target_service} restarted. Memory cleared."
                self.state_data.memory_usage[self.failing_service] = 20.0
            else:
                self.last_res = "Service restarted. No change in underlying issue."
                penalty = -0.1

        tot_rew = step_cost + bonus + penalty
        reward = Reward(
            step_cost=step_cost, resolution_bonus=bonus, wrong_action_penalty=penalty, total_reward=tot_rew
        )
        return self.state(), reward, self.done, {}

    def state(self) -> Observation:
        return Observation(
            active_alerts=self.alerts if not self.resolved else [],
            system_state=self.state_data,
            last_action_result=self.last_res,
            time_elapsed_minutes=self.time,
            incident_resolved=self.resolved
        )
```

## 6. Reward Function Design (Explanation + Formula)
**Explanation:** 
Every step costs a minor amount of time (`-0.05`), simulating SLA burn. Applying incorrect fixes (e.g., rolling back a DB experiencing a normal traffic surge) incurs a penalty to penalize chaotic "try everything" behavior. Fixing the actual root cause and calling `mark_resolved` gives a massive dense reward (+1.0). Marking resolved falsely gives a `-1.0` penalty.
**Formula:** `R = -0.05*steps + (-0.2 for blind rollback) + (1.0 if fixed AND marked_resolved else -1.0 if not fixed AND marked_resolved)`.

## 7. Tasks Definition (3 Tasks: Easy / Medium / Hard)
* **Easy:** Auth service has an Out Of Memory (OOM) error. Fix: Read metrics, restart auth service.
* **Medium:** Payment service deployed a bad version (NullPointerException). Fix: Read logs to see the exception, rollback payment service.
* **Hard:** Database is overwhelmed (not an error, just CPU at 99%). The agent must notice CPU spikes across metrics, check logs (no errors), and scale up the DB rather than restarting or rolling back.

## 8. Graders (Code + Deterministic Logic)
```python
def grade_incident_response(env_trajectory: list, final_obs: Observation) -> float:
    # Scale 0.0 to 1.0 based on correctness and speed
    if not final_obs.incident_resolved:
        return 0.0
    
    # Base score for fixing it
    score = 0.5
    
    # 50% based on efficiency (time taken). Max efficiency is 3 steps (15 mins)
    time_taken = final_obs.time_elapsed_minutes
    if time_taken <= 15:
        score += 0.5
    elif time_taken <= 30:
        score += 0.3
    else:
        score += 0.1
        
    return score
```

## 9. openenv.yaml
```yaml
name: "production-incident-response"
version: "1.0.0"
description: "SRE Agent resolving production incidents rapidly."
entrypoint: "env.py:IncidentEnv"
tasks:
  - id: "task_easy"
    params: {"incident_type": "oom", "failing_service": "auth"}
  - id: "task_medium"
    params: {"incident_type": "bad_deploy", "failing_service": "payment"}
  - id: "task_hard"
    params: {"incident_type": "high_load", "failing_service": "db"}
models:
  observation: "models.py:Observation"
  action: "models.py:Action"
  reward: "models.py:Reward"
```

## 10. Baseline Agent Script (OpenAI API)
```python
import os
from openai import OpenAI
from env import IncidentEnv
from models import Action

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_agent(env):
    obs = env.reset()
    system_prompt = "You are an SRE on call. Investigate the alert, use logs/metrics, fix the service, and mark resolved."
    messages = [{"role": "system", "content": system_prompt}]
    
    while not env.done:
        messages.append({"role": "user", "content": f"State: {obs.model_dump_json()}"})
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[{"name": "take_action", "parameters": Action.model_json_schema()}],
            function_call={"name": "take_action"}
        )
        action_args = response.choices[0].message.function_call.arguments
        action = Action.model_validate_json(action_args)
        
        obs, reward, done, info = env.step(action)
        messages.append({"role": "assistant", "content": f"Performed: {action.action_type} on {action.target_service}"})
```

## 11. Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-m", "pytest", "tests/"]
```

## 12. Project Structure
```text
incident_response/
├── env.py
├── models.py
├── evaluator.py
├── baseline_agent.py
├── requirements.txt
├── Dockerfile
└── openenv.yaml
```

## 13. Evaluation Metrics
* **Resolution Success Rate:** % of incidents actually resolved.
* **Mean Time to Recovery (MTTR):** Average virtual minutes taken to resolve.
* **Blind Action Rate:** How often the agent restarts/rolls back without viewing logs/metrics first.

## 14. Edge Cases
* Restarting a database during high load (which might make things worse).
* Querying logs for services that don't exist.
* Marking resolved immediately to try and cheat the environment (handled by negative reward).

## 15. Hackathon Demo Plan
* Connect the environment to an actual local Prometheus/Grafana or mock UI. 
* Show the GPT-4 agent parsing logs, correctly deducing `bad_deploy` from the NullPointerException, and triggering the rollback.

## 16. Bonus Enhancements
* Let the agent write actual Kubernetes `kubectl` commands instead of categorical actions.
* Add multi-service cascading failures (DB failure causes Payment to throw 500s).

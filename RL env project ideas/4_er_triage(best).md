# 📁 PROJECT: Emergency Room Triage & Decision System

## 1. Problem Statement

Emergency rooms are chaotic and resource-constrained. Patients arrive with various symptoms and vitals. The AI agent acts as the Triage Manager. It must correctly prioritize patients, order required lab tests, admit critical patients to limited ICU beds, and discharge stable ones. Mismanagement leads to deteriorating patient health or bed locks.

## 2. Observation Model (Pydantic) + Code

```python
from pydantic import BaseModel
from typing import List, Optional

class Patient(BaseModel):
    id: str
    age: int
    symptoms: str
    heart_rate: int
    blood_pressure: str
    condition_score: float # 0.0 (dead) to 1.0 (perfect)
    wait_time_mins: int
    state: str = "waiting" # waiting, in_bed, tested, treated, discharged

class Observation(BaseModel):
    waiting_room: List[Patient]
    beds_occupied: int
    total_beds: int
    current_time_step: int
```

## 3. Action Model (Pydantic) + Code

```python
from pydantic import BaseModel
from typing import Literal

class Action(BaseModel):
    action_type: Literal["admit_to_bed", "order_test", "treat", "discharge", "wait"]
    target_patient_id: str
```

## 4. Reward Model (Pydantic) + Code

```python
from pydantic import BaseModel

class Reward(BaseModel):
    reward_value: float
    message: str
```

## 5. Environment Implementation (env.py)

```python
import openenv
from models import Observation, Action, Reward, Patient

class ERTriageEnv(openenv.BaseEnv):
    def __init__(self, initial_patients: list, total_beds: int):
        self.init_patients = initial_patients
        self.total_beds = total_beds
        self.reset()

    def reset(self) -> Observation:
        self.patients = [Patient(**p) for p in self.init_patients]
        self.beds_occupied = 0
        self.time_step = 0
        self.done = False
        return self.state()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done: raise ValueError("Done.")
        self.time_step += 1
        reward_val = 0.0
        msg = "Action applied."

        patient = next((p for p in self.patients if p.id == action.target_patient_id), None)

        if action.action_type != "wait" and not patient:
            return self.state(), Reward(reward_value=-1.0, message="Invalid patient ID"), self.done, {}

        # Apply action logic
        if action.action_type == "admit_to_bed":
            if self.beds_occupied < self.total_beds and patient.state == "waiting":
                patient.state = "in_bed"
                self.beds_occupied += 1
                reward_val = 0.5
            else:
                reward_val = -1.0
                msg = "No beds or already admitted."

        elif action.action_type == "order_test":
            if patient.state == "in_bed":
                patient.state = "tested"
                reward_val = 0.2
            else:
                reward_val = -0.5
                msg = "Must be in bed to test."

        elif action.action_type == "treat":
            if patient.state == "tested" or patient.state == "in_bed":
                patient.condition_score = min(1.0, patient.condition_score + 0.3)
                patient.state = "treated"
                reward_val = 1.0
            else:
                reward_val = -0.5

        elif action.action_type == "discharge":
            if patient.state == "treated" and patient.condition_score >= 0.8:
                patient.state = "discharged"
                self.beds_occupied -= 1
                reward_val = 2.0
            else:
                reward_val = -2.0 # Malpractice!
                msg = "Discharged unstable patient!"

        # Environment Global Tick - Unstable patients waiting deteriorate
        for p in self.patients:
            if p.state == "waiting":
                p.wait_time_mins += 15
                if p.condition_score < 0.5:
                    p.condition_score -= 0.1 # Critically ill drop fast
                else:
                    p.condition_score -= 0.02

            if p.condition_score <= 0.0:
                reward_val -= 10.0 # Huge penalty for death
                self.done = True

        # Check termination (All discharged)
        if all(p.state == "discharged" for p in self.patients):
            self.done = True
            reward_val += 5.0

        if self.time_step >= 50:
            self.done = True # Timeout

        return self.state(), Reward(reward_value=reward_val, message=msg), self.done, {}

    def state(self) -> Observation:
        return Observation(
            waiting_room=[p for p in self.patients if p.state != "discharged"],
            beds_occupied=self.beds_occupied,
            total_beds=self.total_beds,
            current_time_step=self.time_step
        )
```

## 6. Reward Function Design (Explanation + Formula)

**Explanation:** Agent gets positive micro-rewards for moving patients through the pipeline (`waiting -> bed -> test -> treat -> discharge`). Severe negative rewards (`-2.0`) for discharging an untreated patient (malpractice). Lethal penalty (`-10.0`) if a patient's condition hits 0.0 while in the waiting room.
**Formula:** `R = (Pipeline Advance: +0.2 to +2.0) - (Invalid State Trigger: -1.0) - (Unstable Discharge: -2.0) - (Patient Death: -10.0)`

## 7. Tasks Definition (3 Tasks: Easy / Medium / Hard)

- **Easy:** 3 patients, 3 beds. No one is critical. Simple FIFO pipeline.
- **Medium:** 5 patients, 2 beds. 1 patient is critical (0.3 score) and must jump the queue, else they will die in the waiting room.
- **Hard:** 10 patients, 3 beds. Multiple critical trauma patients mixed with healthy flu patients. Agent must rapidly treat and discharge healthy patients to free beds, while managing deteriorating critical cases.

## 8. Graders (Code + Deterministic Logic)

```python
def grade_er_triage(env_trajectory: list, final_obs: Observation) -> float:
    # Look at the final state of all patients
    # We must access the env's internal patient list since discharged are removed from obs
    env = env_trajectory[-1]['env_instance']

    dead = sum(1 for p in env.patients if p.condition_score <= 0)
    if dead > 0: return 0.0

    discharged = sum(1 for p in env.patients if p.state == "discharged")
    total = len(env.patients)

    score = discharged / total

    # Time penalty: fast discharge goes from 0.8 to 1.0
    if score == 1.0:
        if final_obs.current_time_step <= (total * 4): # Optimal time
            return 1.0
        return 0.8

    return score
```

## 9. openenv.yaml

```yaml
name: "er-triage-system"
version: "1.0.0"
description: "AI Triage Nurse managing ER beds and patient care pipelines."
entrypoint: "env.py:ERTriageEnv"
tasks:
  - id: "task_easy"
    params:
      total_beds: 3
      initial_patients:
        - {
            "id": "p1",
            "age": 20,
            "symptoms": "Cough",
            "heart_rate": 80,
            "blood_pressure": "120/80",
            "condition_score": 0.9,
            "wait_time_mins": 0,
          }
  - id: "task_hard"
    params:
      total_beds: 2
      initial_patients:
        - {
            "id": "p1",
            "age": 45,
            "symptoms": "Chest Pain",
            "heart_rate": 140,
            "blood_pressure": "190/100",
            "condition_score": 0.3,
            "wait_time_mins": 0,
          }
        - {
            "id": "p2",
            "age": 12,
            "symptoms": "Sprained Ankle",
            "heart_rate": 90,
            "blood_pressure": "110/70",
            "condition_score": 0.9,
            "wait_time_mins": 0,
          }
        - {
            "id": "p3",
            "age": 70,
            "symptoms": "Stroke",
            "heart_rate": 40,
            "blood_pressure": "90/50",
            "condition_score": 0.2,
            "wait_time_mins": 0,
          }
models:
  observation: "models.py:Observation"
  action: "models.py:Action"
  reward: "models.py:Reward"
```

## 10. Baseline Agent Script (OpenAI API)

```python
import os
from openai import OpenAI
from env import ERTriageEnv
from models import Action

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_agent(env):
    obs = env.reset()
    sys = "You are an ER Triage Manager. Prioritize low condition_score patients. Admit, test, treat, discharge."
    messages = [{"role": "system", "content": sys}]

    while not env.done:
        messages.append({"role": "user", "content": obs.model_dump_json()})
        res = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[{"name": "triage_action", "parameters": Action.model_json_schema()}],
            function_call={"name": "triage_action"}
        )
        action_args = res.choices[0].message.function_call.arguments
        action = Action.model_validate_json(action_args)
        obs, r, d, i = env.step(action)
        messages.append({"role": "assistant", "content": f"Action: {action_args}"})
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
er_triage/
├── env.py
├── models.py
├── evaluator.py
├── baseline_agent.py
├── requirements.txt
├── Dockerfile
└── openenv.yaml
```

## 13. Evaluation Metrics

- **Patient Survival Rate:** Target is 100%.
- **Average Wait Time for Critical Patients:** Should be near zero.
- **Bed Utilization Rate:** Percentage of time beds are actively resolving patients vs empty.

## 14. Edge Cases

- Discharging a healthy patient directly from the waiting room without treating (Malpractice penalty).
- Treating a patient multiple times to farm rewards (Capped condition score, reward only applies to state change).

## 15. Hackathon Demo Plan

- Show the JSON blob of the Hard task. The baseline agent correctly identifies the Chest Pain and Stroke patients and pushes them to beds immediately, leaving the Sprained Ankle waiting.

## 16. Bonus Enhancements

- Incorporate LLM context logic where the agent must literally read the `symptoms` field to deduce severity if `condition_score` is hidden.

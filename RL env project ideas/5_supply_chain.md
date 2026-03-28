# 📁 PROJECT: Supply Chain & Logistics + Vehicle Management

## 1. Problem Statement
Logistics companies manage fleets of vehicles to fulfill customer orders across different depots. This environment tasks an AI agent with serving as the Central Dispatch. It must process pending freight orders, load vehicles at depots, dispatch them to destination zones, and manage fuel and vehicle maintenance—all before the Order SLA (Service Level Agreement) expires.

## 2. Observation Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import List

class Order(BaseModel):
    id: str
    origin: str
    destination: str
    weight: int
    sla_time_remaining: int

class Vehicle(BaseModel):
    id: str
    current_location: str
    fuel_level: int
    health_status: int # 0 to 100
    is_loaded: bool
    loaded_order_id: str = None

class Observation(BaseModel):
    pending_orders: List[Order]
    fleet: List[Vehicle]
    current_time: int
```

## 3. Action Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import Literal, Optional

class Action(BaseModel):
    action_type: Literal["load_vehicle", "dispatch", "refuel", "maintain", "wait"]
    vehicle_id: str
    target_order_id: Optional[str] = None
    destination_node: Optional[str] = None
```

## 4. Reward Model (Pydantic) + Code
```python
from pydantic import BaseModel

class Reward(BaseModel):
    step_reward: float
    delivered_count: int
    sla_breaches: int
```

## 5. Environment Implementation (env.py)
```python
import openenv
from models import Observation, Action, Reward, Order, Vehicle

class LogisticsEnv(openenv.BaseEnv):
    def __init__(self, orders: list, vehicles: list):
        self.init_orders = orders
        self.init_vehicles = vehicles
        self.reset()
        
    def reset(self) -> Observation:
        self.orders = [Order(**o) for o in self.init_orders]
        self.vehicles = [Vehicle(**v) for v in self.init_vehicles]
        self.time = 0
        self.delivered_count = 0
        self.sla_breaches = 0
        self.done = False
        return self.state()
        
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done: raise ValueError("Done.")
        self.time += 1
        reward_val = 0.0

        vehicle = next((v for v in self.vehicles if v.id == action.vehicle_id), None)
        
        if action.action_type != "wait" and not vehicle:
            return self.state(), Reward(step_reward=-0.1, delivered_count=self.delivered_count, sla_breaches=self.sla_breaches), self.done, {}

        # Handle Action
        if action.action_type == "load_vehicle":
            order = next((o for o in self.orders if o.id == action.target_order_id), None)
            if order and not vehicle.is_loaded and vehicle.current_location == order.origin:
                vehicle.is_loaded = True
                vehicle.loaded_order_id = order.id
                reward_val += 0.1
            else:
                reward_val -= 0.5
                
        elif action.action_type == "dispatch":
            if vehicle.fuel_level >= 20 and vehicle.health_status >= 10:
                # Simulate movement
                vehicle.current_location = action.destination_node
                vehicle.fuel_level -= 20
                vehicle.health_status -= 10
                
                # Check dropoff
                if vehicle.is_loaded:
                    order = next((o for o in self.orders if o.id == vehicle.loaded_order_id), None)
                    if order and vehicle.current_location == order.destination:
                        # Delivered!
                        self.orders.remove(order)
                        vehicle.is_loaded = False
                        vehicle.loaded_order_id = None
                        self.delivered_count += 1
                        reward_val += 5.0
            else:
                reward_val -= 1.0 # Dispatch failed, vehicle broke down or out of gas
                
        elif action.action_type == "refuel":
            vehicle.fuel_level = 100
            reward_val -= 0.1 # Cost of fuel
            
        elif action.action_type == "maintain":
            vehicle.health_status = 100
            reward_val -= 0.5 # Cost of maintenance
            
        # Global clock tick
        for o in self.orders:
            o.sla_time_remaining -= 1
            if o.sla_time_remaining <= 0:
                self.sla_breaches += 1
                reward_val -= 2.0
                self.orders.remove(o) # Order cancelled
                # Unload any vehicle carrying it
                for v in self.vehicles:
                    if v.loaded_order_id == o.id:
                        v.is_loaded = False
                        v.loaded_order_id = None

        if len(self.orders) == 0:
            self.done = True

        return self.state(), Reward(step_reward=reward_val, delivered_count=self.delivered_count, sla_breaches=self.sla_breaches), self.done, {}

    def state(self) -> Observation:
        return Observation(
            pending_orders=self.orders,
            fleet=self.vehicles,
            current_time=self.time
        )
```

## 6. Reward Function Design (Explanation + Formula)
**Explanation:** 
Successfully delivering an order grants a heavy dense reward (+5.0). Letting an order expire the SLA triggers a harsh penalty (-2.0). Minor penalties (-0.1, -0.5) simulate the real-world operational costs of refueling or fixing a truck. Dispatching an empty or broken truck wastes time and incurs penalties.
**Formula:** `R = (Deliveries * 5) - (SLA_Breaches * 2) - (Fuel/Maint_Cost) + (Valid_Load * 0.1)`

## 7. Tasks Definition (3 Tasks: Easy / Medium / Hard)
* **Easy:** 1 Order, 1 Truck. High fuel, high health. SLA 10 ticks. Just Load -> Dispatch.
* **Medium:** 3 Orders, 2 Trucks. Trucks have low fuel. Agent must Refuel -> Load -> Dispatch.
* **Hard:** 5 Orders, 2 Trucks. Tight SLAs. Trucks need refueling and maintenance mid-journey. Cross-dispatching required (Truck drops off at Zone B, picks up new order from Zone B to go to Zone C).

## 8. Graders (Code + Deterministic Logic)
```python
def grade_logistics(env_trajectory: list, final_obs: Observation) -> float:
    final_reward_state = env_trajectory[-1]['reward']
    delivered = final_reward_state.delivered_count
    breaches = final_reward_state.sla_breaches
    total_orders = delivered + breaches
    
    if total_orders == 0: return 0.0
    
    delivery_ratio = delivered / total_orders
    
    # Penalize inefficient operations (too much refueling)
    score = delivery_ratio
    
    if delivery_ratio == 1.0:
        # Check action count for efficiency 
        if final_obs.current_time <= (total_orders * 4):
            return 1.0 # Perfect Execution
        return 0.9 # Delivered everything but took long
        
    return score
```

## 9. openenv.yaml
```yaml
name: "supply-chain-dispatch"
version: "1.0.0"
description: "AI Logistics Coordinator dispatching fleets and managing SLAs."
entrypoint: "env.py:LogisticsEnv"
tasks:
  - id: "task_easy"
    params:
      orders: [{"id": "o1", "origin": "DepotA", "destination": "DepotB", "weight": 10, "sla_time_remaining": 10}]
      vehicles: [{"id": "v1", "current_location": "DepotA", "fuel_level": 100, "health_status": 100, "is_loaded": false}]
  - id: "task_medium"
    params:
      orders: [{"id": "o1", "origin": "A", "destination": "B", "weight": 5, "sla_time_remaining": 5}]
      vehicles: [{"id": "v1", "current_location": "A", "fuel_level": 10, "health_status": 100, "is_loaded": false}]
models:
  observation: "models.py:Observation"
  action: "models.py:Action"
  reward: "models.py:Reward"
```

## 10. Baseline Agent Script (OpenAI API)
```python
import os
from openai import OpenAI
from env import LogisticsEnv
from models import Action

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_agent(env):
    obs = env.reset()
    sys = "You are a dispatcher. Check fuel, load orders, and dispatch trucks before SLA hits 0."
    messages = [{"role": "system", "content": sys}]
    
    while not env.done:
        messages.append({"role": "user", "content": obs.model_dump_json()})
        res = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[{"name": "dispatch_action", "parameters": Action.model_json_schema()}],
            function_call={"name": "dispatch_action"}
        )
        action_args = res.choices[0].message.function_call.arguments
        action = Action.model_validate_json(action_args)
        obs, r, d, i = env.step(action)
        messages.append({"role": "assistant", "content": f"Action: {action_args}"})
```

## 11. Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "evaluator.py"]
```

## 12. Project Structure
```text
supply_chain/
├── env.py
├── models.py
├── evaluator.py
├── baseline_agent.py
├── requirements.txt
├── Dockerfile
└── openenv.yaml
```

## 13. Evaluation Metrics
* **SLA Compliance Rate:** Goal 100%.
* **Fleet Utilization:** Time spent loaded vs empty.
* **Operational Expense:** Money wasted on unnecessary refueling or bad dispatches.

## 14. Edge Cases
* Truck breaking down mid-journey because the agent forgot to maintain it (-1.0 penalization and lost turn).
* Trying to load a truck that isn't at the right origin depot (Rejected action).

## 15. Hackathon Demo Plan
* Connect a PyGame or Web UI showing nodes (Depot A, Depot B) and little trucks moving. Watch the Agent realize a truck has 10 Fuel and autonomously order a `refuel` before triggering `dispatch`.

## 16. Bonus Enhancements
* Introduce traffic/weather probabilities that randomly drain more fuel or cause delay.

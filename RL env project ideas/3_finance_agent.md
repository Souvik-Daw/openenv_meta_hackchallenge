# 📁 PROJECT: Personal Finance & Budgeting Agent

## 1. Problem Statement
Many people struggle to allocate their monthly income efficiently across bills, savings, and investments while avoiding debt spirals. This environment asks an AI agent to act as a Financial Planner. Given a starting bank balance, a monthly salary, upcoming bills, and high-interest debts, the agent must step through 12 simulated months, deciding exactly how much money to move where.

## 2. Observation Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import Dict

class FinancialState(BaseModel):
    checking_balance: float
    savings_balance: float
    credit_card_debt: float
    investment_balance: float

class Observation(BaseModel):
    month_number: int
    current_state: FinancialState
    monthly_salary: float
    monthly_rent: float
    minimum_cc_payment: float
    cc_interest_rate_monthly: float
    savings_interest_rate_monthly: float
    msg: str
```

## 3. Action Model (Pydantic) + Code
```python
from pydantic import BaseModel

class Action(BaseModel):
    pay_rent: bool
    cc_payment_amount: float
    transfer_to_savings: float
    transfer_to_investments: float
```

## 4. Reward Model (Pydantic) + Code
```python
from pydantic import BaseModel

class Reward(BaseModel):
    net_worth_delta: float
    penalty_missed_rent: float
    penalty_missed_cc_min: float
    total_reward: float
```

## 5. Environment Implementation (env.py)
```python
import openenv
from models import Observation, Action, Reward, FinancialState

class FinanceEnv(openenv.BaseEnv):
    def __init__(self, starting_cc_debt: float, salary: float, rent: float):
        self.init_cc_debt = starting_cc_debt
        self.salary = salary
        self.rent = rent
        self.reset()

    def reset(self) -> Observation:
        self.month = 1
        self.state_data = FinancialState(
            checking_balance=self.salary,
            savings_balance=0.0,
            credit_card_debt=self.init_cc_debt,
            investment_balance=0.0
        )
        self.done = False
        return self.state("Started.")

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            raise ValueError("Environment finished. Call reset().")

        penalty_rent = 0.0
        penalty_cc = 0.0
        msg = ""

        # Calculate pre-action net worth
        nw_before = self._net_worth()

        total_spend = action.cc_payment_amount + action.transfer_to_savings + action.transfer_to_investments
        if action.pay_rent: total_spend += self.rent

        # Enforce checkings limit
        if total_spend > self.state_data.checking_balance:
            msg = "Overdraft attempted! Resetting action bounds."
            # Force minimums if possible, else fail
            action.transfer_to_savings = 0.0
            action.transfer_to_investments = 0.0
            action.cc_payment_amount = min(self.state_data.checking_balance - (self.rent if action.pay_rent else 0), action.cc_payment_amount)
            action.cc_payment_amount = max(0, action.cc_payment_amount)

        # Apply rent
        if action.pay_rent and self.state_data.checking_balance >= self.rent:
            self.state_data.checking_balance -= self.rent
        else:
            penalty_rent = -500.0  # Eviction danger!

        # Apply CC
        if action.cc_payment_amount >= (self.state_data.credit_card_debt * 0.05): # 5% minimum
            self.state_data.checking_balance -= action.cc_payment_amount
            self.state_data.credit_card_debt -= action.cc_payment_amount
        else:
            penalty_cc = -100.0 # Late fee / missed minimum
            if action.cc_payment_amount > 0:
                self.state_data.checking_balance -= action.cc_payment_amount
                self.state_data.credit_card_debt -= action.cc_payment_amount

        # Transfers
        self.state_data.checking_balance -= action.transfer_to_savings
        self.state_data.savings_balance += action.transfer_to_savings
        
        self.state_data.checking_balance -= action.transfer_to_investments
        self.state_data.investment_balance += action.transfer_to_investments

        # End of month accruals
        self.state_data.credit_card_debt *= 1.02  # 2% monthly CC interest
        self.state_data.savings_balance *= 1.003 # 0.3% monthly savings
        self.state_data.investment_balance *= 1.008 # 0.8% monthly investment return
        
        # Next month salary
        self.month += 1
        if self.month > 12:
            self.done = True
        else:
            self.state_data.checking_balance += self.salary

        nw_after = self._net_worth()
        delta = nw_after - nw_before
        
        tot_rew = delta * 0.01 + penalty_rent + penalty_cc

        return self.state(msg), Reward(net_worth_delta=delta, penalty_missed_rent=penalty_rent, penalty_missed_cc_min=penalty_cc, total_reward=tot_rew), self.done, {}

    def _net_worth(self):
        return self.state_data.checking_balance + self.state_data.savings_balance + self.state_data.investment_balance - self.state_data.credit_card_debt

    def state(self, msg="") -> Observation:
        return Observation(
            month_number=self.month,
            current_state=self.state_data,
            monthly_salary=self.salary,
            monthly_rent=self.rent,
            minimum_cc_payment=self.state_data.credit_card_debt * 0.05,
            cc_interest_rate_monthly=0.02,
            savings_interest_rate_monthly=0.003,
            msg=msg
        )
```

## 6. Reward Function Design (Explanation + Formula)
**Explanation:** 
The environment rewards the agent for purely increasing the simulated Net Worth. Paying off high-interest CC debt is mathematically superior to putting money in savings; the dense reward naturally teaches this by comparing pre-step and post-step net worth. Massive sparse penalties (`-500`, `-100`) are applied if basic living expenses (rent, min payments) are missed.
**Formula:** `R = (NetWorth(t+1) - NetWorth(t)) * 0.01 - (500 if NO_RENT) - (100 if NO_CC_MIN)`

## 7. Tasks Definition (3 Tasks: Easy / Medium / Hard)
* **Easy:** Salary \$5k, Rent \$2k, CC Debt \$1k. Generous surplus.
* **Medium:** Salary \$4k, Rent \$2k, CC Debt \$8k. High debt requires aggressive CC paydown before saving.
* **Hard:** Salary \$3k, Rent \$2k, CC Debt \$15k. The agent is trapped. Must pay exactly minimums to survive and avoid penalties, allocating every dollar perfectly.

## 8. Graders (Code + Deterministic Logic)
```python
def grade_financial_performance(env_trajectory: list, final_obs: Observation) -> float:
    # Check for any missed rent in the trajectory
    for step in env_trajectory:
        if step['reward'].penalty_missed_rent < 0 or step['reward'].penalty_missed_cc_min < 0:
            return 0.0  # Automatic failure if you go homeless or face collections
            
    # Calculate pure financial efficiency
    # A perfect agent pays CC completely, then dumps into investments.
    final_nw = final_obs.current_state.checking_balance + final_obs.current_state.savings_balance + final_obs.current_state.investment_balance - final_obs.current_state.credit_card_debt
    
    # Simple baseline normalization: if you end with >0 net worth in hard tasks, you get a good score.
    # Assuming baseline net worth growth expected is 5000 over 12 months.
    score = min(1.0, max(0.0, (final_nw + 10000) / 20000.0))
    return score
```

## 9. openenv.yaml
```yaml
name: "personal-finance-agent"
version: "1.0.0"
description: "AI agent balancing a monthly household budget."
entrypoint: "env.py:FinanceEnv"
tasks:
  - id: "task_easy"
    params: {"starting_cc_debt": 1000.0, "salary": 5000.0, "rent": 2000.0}
  - id: "task_medium"
    params: {"starting_cc_debt": 8000.0, "salary": 4000.0, "rent": 2000.0}
  - id: "task_hard"
    params: {"starting_cc_debt": 15000.0, "salary": 3000.0, "rent": 2000.0}
models:
  observation: "models.py:Observation"
  action: "models.py:Action"
  reward: "models.py:Reward"
```

## 10. Baseline Agent Script (OpenAI API)
```python
import os
from openai import OpenAI
from env import FinanceEnv
from models import Action

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_agent(env):
    obs = env.reset()
    sys_msg = "You are a financial advisor. For 12 months, allocate checking balance to pay rent, CC debt, and investments."
    messages = [{"role": "system", "content": sys_msg}]
    
    while not env.done:
        messages.append({"role": "user", "content": f"Month {obs.month_number}: {obs.model_dump_json()}"})
        res = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[{"name": "allocate_funds", "parameters": Action.model_json_schema()}],
            function_call={"name": "allocate_funds"}
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
CMD ["python", "evaluator.py"]
```

## 12. Project Structure
```text
finance_agent/
├── env.py
├── models.py
├── evaluator.py
├── baseline_agent.py
├── requirements.txt
├── Dockerfile
└── openenv.yaml
```

## 13. Evaluation Metrics
* **Final Net Worth:** Absolute dollar value at Month 12.
* **Survival Rate:** Did the agent miss any rent payments?
* **Interest Paid:** Total dollars lost to CC interest.

## 14. Edge Cases
* Agent transferring out more money than exists in checkings (Blocked & defaults to CC minimums).
* Agent ignoring rent to invest in the stock market (Huge penalty).

## 15. Hackathon Demo Plan
* Print a beautiful terminal table showing Bank Balance, CC Debt, Investments month by month as the agent zeroes out the credit card and then pivots strictly to investments.

## 16. Bonus Enhancements
* Introduce stochastic elements: unexpected car repair bill (-$1500) in month 6.
* Introduce tax implications.

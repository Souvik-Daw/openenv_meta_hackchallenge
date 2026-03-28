# 📁 PROJECT: Mutual Fund Portfolio Manager (Stock Market Management)

## 1. Problem Statement
Managing a Mutual Fund portfolio requires balancing risk and reward over time. The AI agent acts as a Portfolio Manager with an initial capital allocation. Over a simulated period of months, the agent observes market trends, fund NAVs (Net Asset Values), and macroeconomic sentiment. The goal is to aggressively grow the portfolio in bull markets and defensively rotate into fixed-income bonds in bear markets, beating a standard benchmark index while minimizing trading fees and severe drawdowns.

## 2. Observation Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import Literal

class MarketData(BaseModel):
    month_number: int
    eq_growth_nav: float      # High risk, high reward Equity Fund
    fixed_income_nav: float   # Low risk Bond Fund
    benchmark_index: float    # A proxy S&P 500 index score

class Portfolio(BaseModel):
    cash_balance: float
    eq_growth_units: float
    fixed_income_units: float
    total_portfolio_value: float

class Observation(BaseModel):
    market: MarketData
    portfolio: Portfolio
    recent_news_sentiment: Literal["bullish", "bearish", "neutral"]
    months_remaining: int
```

## 3. Action Model (Pydantic) + Code
```python
from pydantic import BaseModel
from typing import Literal

class Action(BaseModel):
    action_type: Literal["buy", "sell", "hold"]
    target_fund: Literal["eq_growth", "fixed_income", "none"]
    amount_usd: float # Amount of USD to spend or liquidate. 0 for hold.
```

## 4. Reward Model (Pydantic) + Code
```python
from pydantic import BaseModel

class Reward(BaseModel):
    alpha_reward: float       # Portfolio growth vs Benchmark growth
    trading_fee_penalty: float
    drawdown_penalty: float   # Penalty for losing >3% in one month
    total_reward: float
```

## 5. Environment Implementation (env.py)
```python
import openenv
from models import Observation, Action, Reward, MarketData, Portfolio

class MFPortfolioEnv(openenv.BaseEnv):
    def __init__(self, market_sequence: list, initial_cash: float):
        self.market_sequence = market_sequence # List of dicts with navs and sentiment
        self.init_cash = initial_cash
        self.reset()
        
    def reset(self) -> Observation:
        self.month = 0
        self.done = False
        
        self.portfolio = Portfolio(
            cash_balance=self.init_cash,
            eq_growth_units=0.0,
            fixed_income_units=0.0,
            total_portfolio_value=self.init_cash
        )
        return self.state()
        
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done: raise ValueError("Done.")
        
        current_market = self.market_sequence[self.month]
        fee = 0.0
        
        # 1. Process Trading Action
        if action.action_type == "buy" and action.target_fund != "none" and action.amount_usd > 0:
            spend = min(action.amount_usd, self.portfolio.cash_balance)
            if spend > 0:
                fee = spend * 0.005 # 0.5% trading fee
                actual_invest = spend - fee
                
                self.portfolio.cash_balance -= spend
                if action.target_fund == "eq_growth":
                    self.portfolio.eq_growth_units += (actual_invest / current_market["eq_growth_nav"])
                elif action.target_fund == "fixed_income":
                    self.portfolio.fixed_income_units += (actual_invest / current_market["fixed_income_nav"])

        elif action.action_type == "sell" and action.target_fund != "none" and action.amount_usd > 0:
            fund_nav = current_market[action.target_fund + "_nav"]
            units_to_sell = action.amount_usd / fund_nav
            
            # Clamp to what we actually own
            if action.target_fund == "eq_growth":
                units_to_sell = min(units_to_sell, self.portfolio.eq_growth_units)
                self.portfolio.eq_growth_units -= units_to_sell
            else:
                units_to_sell = min(units_to_sell, self.portfolio.fixed_income_units)
                self.portfolio.fixed_income_units -= units_to_sell
                
            liquidated = units_to_sell * fund_nav
            fee = liquidated * 0.005 # 0.5% exit load
            self.portfolio.cash_balance += (liquidated - fee)

        # 2. Track Before-Market-Update Value
        val_before = self._calculate_nav()
        benchmark_before = current_market["benchmark_index"]

        # 3. Time steps forward (Market Updates)
        self.month += 1
        if self.month >= len(self.market_sequence) - 1:
            self.done = True
        
        # 4. Track After-Market-Update Value
        new_market = self.market_sequence[self.month]
        val_after = self._calculate_nav()
        benchmark_after = new_market["benchmark_index"]
        
        # 5. Calculate Rewards
        pct_return = (val_after - val_before) / max(val_before, 1.0)
        pct_bench = (benchmark_after - benchmark_before) / max(benchmark_before, 1.0)
        
        alpha = (pct_return - pct_bench) * 100.0 # Points of metric beat
        
        drawdown_pen = 0.0
        if pct_return < -0.03:
            drawdown_pen = -2.0 # Huge penalty for losing more than 3% in a month
            
        total_rew = alpha - fee/1000.0 + drawdown_pen

        # Update absolute total safely
        self.portfolio.total_portfolio_value = val_after

        return self.state(), Reward(alpha_reward=alpha, trading_fee_penalty=-fee/1000.0, drawdown_penalty=drawdown_pen, total_reward=total_rew), self.done, {}

    def _calculate_nav(self) -> float:
        market = self.market_sequence[self.month]
        return self.portfolio.cash_balance + \
               (self.portfolio.eq_growth_units * market["eq_growth_nav"]) + \
               (self.portfolio.fixed_income_units * market["fixed_income_nav"])

    def state(self) -> Observation:
        market_now = self.market_sequence[self.month]
        
        m_data = MarketData(
            month_number=self.month,
            eq_growth_nav=market_now["eq_growth_nav"],
            fixed_income_nav=market_now["fixed_income_nav"],
            benchmark_index=market_now["benchmark_index"]
        )
        return Observation(
            market=m_data,
            portfolio=self.portfolio,
            recent_news_sentiment=market_now["sentiment"],
            months_remaining=len(self.market_sequence) - self.month - 1
        )
```

## 6. Reward Function Design (Explanation + Formula)
**Explanation:** 
Mutual Fund Managers are judged heavily on 'Alpha' (beating the benchmark) and managing systemic risk. If the overall market drops 5%, but the agent only drops 2% (by rotating to bonds), that yields a *positive* Alpha reward, teaching defensive play. Trading fees penalize erratic buy/sell looping. Hard drops (>3%) yield a fixed Drawdown Penalty.
**Formula:** `R = (Portfolio_Pct_Change - Benchmark_Pct_Change) * 100 - (Fee_Paid / 1000) - (2.0 if Portfolio_Pct_Change < -0.03)`

## 7. Tasks Definition (3 Tasks: Easy / Medium / Hard)
* **Easy (Bull Run):** 24 months of "bullish" sentiment. Equities guarantee +2% monthly. Agent just needs to dump all cash into `eq_growth` early and hold.
* **Medium (Correction):** 24 months. Months 1-12 are bullish, 13-18 are heavily bearish (Equities crash, Bonds rise), 18-24 is flat. Agent must read sentiment shift and sell Equities for Fixed Income before month 13.
* **Hard (Stagnant Choppy Market):** 36 months of chaotic sideway drift. High volatility. Buying and holding loses to the benchmark due to drawdown penalties. Active, tightly timed rebalancing based on sentiment is strictly required.

## 8. Graders (Code + Deterministic Logic)
```python
def grade_mf_portfolio(env_trajectory: list, final_obs: Observation) -> float:
    start_value = 10000.0 # Assuming init_cash from task params
    final_value = final_obs.portfolio.total_portfolio_value
    
    # We must look at the env internals or trajectory to get initial benchmark
    env = env_trajectory[0]['env_instance']
    start_bench = env.market_sequence[0]["benchmark_index"]
    final_bench = env.market_sequence[-1]["benchmark_index"]
    
    port_return = (final_value - start_value) / start_value
    bench_return = (final_bench - start_bench) / start_bench
    
    # Score out of 1.0
    # Baseline: If you match benchmark exactly, you get 0.5.
    # If you beat benchmark by 10%+, you get 1.0
    # If you underperform benchmark by 10%+, you get 0.0
    
    alpha = port_return - bench_return
    
    if alpha <= -0.10: return 0.0
    if alpha >= 0.10: return 1.0
    
    # Linearly scale between -0.10 and +0.10
    score = 0.5 + (alpha * 5.0) 
    
    return min(1.0, max(0.0, score))
```

## 9. openenv.yaml
```yaml
name: "mf-portfolio-manager"
version: "1.0.0"
description: "AI Portfolio Manager directing capital to beat market benchmarks."
entrypoint: "env.py:MFPortfolioEnv"
tasks:
  - id: "task_easy_bull_run"
    params: 
      initial_cash: 10000.0
      market_sequence: 
        # (Auto-generated in actual code; listed short for brevity)
        - {"eq_growth_nav": 100, "fixed_income_nav": 50, "benchmark_index": 1000, "sentiment": "bullish"}
        - {"eq_growth_nav": 105, "fixed_income_nav": 50.5, "benchmark_index": 1050, "sentiment": "bullish"}
        - {"eq_growth_nav": 110, "fixed_income_nav": 51, "benchmark_index": 1100, "sentiment": "bullish"}
  - id: "task_medium_crash"
    # Follows a pattern of bullish -> bearish crash -> flat
    params: { "initial_cash": 10000.0, "market_sequence": [...] }
models:
  observation: "models.py:Observation"
  action: "models.py:Action"
  reward: "models.py:Reward"
```

## 10. Baseline Agent Script (OpenAI API)
```python
import os
from openai import OpenAI
from env import MFPortfolioEnv
from models import Action

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_agent(env):
    obs = env.reset()
    sys = "You manage a mutual fund. Given sentiment and NAV, decide whether to buy eq_growth, fixed_income, or hold."
    messages = [{"role": "system", "content": sys}]
    
    while not env.done:
        messages.append({"role": "user", "content": obs.model_dump_json()})
        res = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[{"name": "trade", "parameters": Action.model_json_schema()}],
            function_call={"name": "trade"}
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
ENTRYPOINT ["python", "evaluator.py"]
```

## 12. Project Structure
```text
mf_portfolio_manager/
├── env.py              # Environment Logic
├── models.py           # Pydantic Schemas
├── data_gen.py         # Helper to generate custom market_sequences
├── evaluator.py        # Runs and grades tasks
├── baseline_agent.py   # OpenAI test script
├── requirements.txt
├── Dockerfile
└── openenv.yaml
```

## 13. Evaluation Metrics
* **Total Alpha:** Portfolio % return minus Benchmark % return.
* **Sharpe Ratio (Optional):** Risk-adjusted return calculation.
* **Max Drawdown:** The largest single peak-to-trough drop over the timeframe.

## 14. Edge Cases
* Agent attempting to invest more cash than it possesses (the code safely clamps `spend = min(action_usd, cash)`).
* Agent panic-selling back and forth multiple times a month (penalized heavily by the 0.5% trading fees compounding).

## 15. Hackathon Demo Plan
* Pre-generate a `.csv` representing the 2008 Financial Crisis (Bearish sentiment hitting strongly around month 8).
* Demonstrate standard "Buy & Hold" baseline getting wrecked by the crash.
* Run the GPT-4 Agent, showing it detect the "bearish" sentiment flip in Observation, sell `eq_growth` early, park funds in `fixed_income`, and completely dodge the drawdown penalty before re-entering equities at the bottom.

## 16. Bonus Enhancements
* Abstract the `market_sequence` to scrape real historical ticker data from Yahoo Finance via the `yfinance` python library.
* Add more Asset Classes (e.g., Commodities/Gold, Crypto, Real Estate REITs) for deeper diversification strategies.

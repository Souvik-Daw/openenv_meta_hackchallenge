# 🏥 Healthcare Appointment Scheduling RL Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://python.org)
[![LLM](https://img.shields.io/badge/LLM-Groq%20llama3--70b-orange)](https://groq.com)

A production-quality **OpenEnv-compatible Reinforcement Learning environment** that simulates a real-world healthcare appointment scheduling system.

An AI agent receives a natural-language patient request, navigates a hospital information system via structured tool calls, identifies the correct medical department and specialist, and books an appointment — all within a dense-reward, multi-step interaction loop.

---

## 📁 Project Structure

```
healthcare_scheduling/         ← OpenEnv package root
├── __init__.py                ← Package exports
├── models.py                  ← Pydantic models (Action, Observation, State)
├── client.py                  ← OpenEnv HTTP client
├── openenv.yaml               ← OpenEnv spec (validate with: openenv validate)
├── pyproject.toml             ← Python package / dependency config
├── Dockerfile                 ← Multi-stage container (uv-based)
├── run_baseline.py            ← Evaluation script (LLM + oracle modes)
│
├── server/                    ← FastAPI server components
│   ├── __init__.py
│   ├── app.py                 ← FastAPI app (create_fastapi_app factory)
│   ├── environment.py         ← Core RL environment (reset / step / state)
│   ├── data.py                ← Hospital data + symptom→department mapping
│   ├── tools.py               ← 5 tool implementations + dispatcher
│   └── requirements.txt
│
├── tasks/                     ← Task definitions + deterministic graders
│   ├── __init__.py
│   ├── easy.py                ← "I have chest pain"
│   ├── medium.py              ← "I have a skin rash for 2 weeks"
│   ├── hard.py                ← "I feel pain but not sure where"
│   └── graders.py             ← Deterministic 0.0–1.0 scoring
│
└── agent/                     ← Baseline LLM agent
    ├── __init__.py
    └── groq_agent.py          ← Groq (llama3-70b) structured tool-calling agent
```

---

## 🌍 Environment Design

### Interaction Model

The agent interacts exclusively through **tool calls** — one call per step. The environment routes each tool call and returns a rich observation containing:

- The tool's result data
- Current episode state (department identified, doctor selected, slot chosen)
- Running conversation history
- Dense per-step reward
- Episode completion status

### Available Tools (Action Space)

| Tool                               | Parameters               | Purpose                                           |
| ---------------------------------- | ------------------------ | ------------------------------------------------- |
| `get_departments()`                | —                        | Returns all available medical departments         |
| `get_doctors(department)`          | `department: str`        | Returns doctors + specializations in a department |
| `check_availability(doctor)`       | `doctor: str`            | Returns available time slots for a doctor         |
| `book_appointment(doctor, slot)`   | `doctor: str, slot: str` | Books the appointment and ends the episode        |
| `ask_user_clarification(question)` | `question: str`          | Returns deterministic simulated user answer       |

### Hospital Data

**Departments:** Cardiology, Dermatology, Neurology, Orthopedics

| Department  | Doctor           | Specialization                                |
| ----------- | ---------------- | --------------------------------------------- |
| Cardiology  | Dr. Sarah Smith  | General cardiology, chest pain, heart failure |
| Cardiology  | Dr. James Adams  | Arrhythmia, palpitations, ECG interpretation  |
| Dermatology | Dr. Priya Patel  | Skin rashes, eczema, psoriasis, chronic skin  |
| Dermatology | Dr. Kevin Lee    | Acne, skin cancer screening, mole removal     |
| Neurology   | Dr. Elena Rossi  | Migraines, headaches, epilepsy                |
| Neurology   | Dr. Michael Chen | Stroke, memory disorders, Parkinson's         |
| Orthopedics | Dr. Thomas Grant | Sports injuries, knee pain, fractures         |
| Orthopedics | Dr. Aisha Okafor | Spine disorders, back pain, disc herniation   |

---

## 🎯 Reward Function (Dense)

Rewards are issued at every step to provide continuous learning signal:

| Event                                            | Reward    |
| ------------------------------------------------ | --------- |
| Correct department identified                    | **+0.20** |
| Correct-department doctor's availability checked | **+0.10** |
| Successful booking, correct dept + doctor        | **+0.50** |
| Efficiency bonus (≤ expected_min_steps + 1)      | **+0.20** |
| Timely use of clarification (step ≤ 2)           | **+0.05** |
| Wrong department selected                        | **−0.30** |
| Wrong doctor specialization booked               | **−0.40** |
| Invalid tool name                                | **−0.20** |
| Doctor not found                                 | **−0.20** |
| Repeated tool call (efficiency penalty)          | **−0.10** |
| Episode exceeds max steps                        | **−0.10** |
| Late clarification (step > 2)                    | **−0.05** |

---

## 🧪 Tasks

### EASY — Chest Pain

```
User: "I have chest pain"
Expected: Cardiology → Dr. Sarah Smith → book slot
Minimum steps: 4
```

### MEDIUM — Skin Rash

```
User: "I have a skin rash for 2 weeks"
Expected: Dermatology → Dr. Priya Patel → book slot
Challenge: Must prefer chronic-condition specialist over general dermatologist
Minimum steps: 4
```

### HARD — Ambiguous Pain

```
User: "I feel pain but not sure where"
Expected: ask_user_clarification → Cardiology → Dr. Sarah Smith → book slot
Challenge: Must ask clarification before routing; penalised if skipped
Minimum steps: 5
```

---

## 🧮 Grader Scoring (0.0–1.0)

Each task uses a deterministic grader with four components:

| Component                              | Points |
| -------------------------------------- | ------ |
| Correct department identified          | 0.25   |
| Correct doctor selected                | 0.30   |
| Booking successful                     | 0.35   |
| Efficiency bonus (≤ min_steps + 1)     | 0.10   |
| Clarification skip penalty (hard only) | −0.10  |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd "Hospitality reception RL env"

# With uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### 2. Run the baseline evaluation

```bash
# Oracle (no API key needed — deterministic, scores 1.0 on all tasks)
python run_baseline.py --oracle

# With Groq LLM agent
export GROQ_API_KEY=gsk_your_key_here
python run_baseline.py
#set APi key int he env file

# Run only the hard task
python run_baseline.py --task hard
```

### 3. Start the environment server

```bash
# With uv
uv run server

# Or directly
uvicorn healthcare_scheduling.server.app:app --host 0.0.0.0 --port 8000
```

### 4. Use the HTTP client

```python
from healthcare_scheduling.client import HealthcareEnv
from healthcare_scheduling.models import AppointmentAction

with HealthcareEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset()
    obs = env.step(AppointmentAction(tool="get_departments", parameters={}))
    print(obs.observation.tool_result)
```

---

## 🐳 Docker

```bash
# Build
docker build -t healthcare-scheduling:latest .

# Run server
docker run --rm -p 8000:8000 healthcare-scheduling:latest

# Run baseline (oracle mode, no API key)
docker run --rm healthcare-scheduling:latest python run_baseline.py --oracle

# Run baseline with Groq
docker run --rm -e GROQ_API_KEY=gsk_... healthcare-scheduling:latest python run_baseline.py
```

---

## ✅ OpenEnv Validation

```bash
openenv validate
# ✔ spec_version: 1
# ✔ name: healthcare_scheduling
# ✔ runtime: fastapi
# ✔ action_schema: valid
# ✔ observation_schema: valid
```

---

## 📊 Sample Oracle Output

```
==============================================================
  Healthcare Appointment Scheduling - Baseline Evaluation
==============================================================
  Mode: ORACLE (deterministic)
==============================================================

  TASK [EASY]
  User request: "I have chest pain"
  Expected: Cardiology -> Dr. Sarah Smith
  ----------------------------------------------------------

  [OK] Score: 1.000
      Department correct:  True  (+0.25)
      Doctor correct:      True  (+0.30)
      Booking successful:  True  (+0.35)
      Efficiency bonus:    +0.10
      Steps taken:         4
      Env cumulative reward: 1.0000

  TASK [MEDIUM]
  User request: "I have a skin rash for 2 weeks"
  Expected: Dermatology -> Dr. Priya Patel
  ----------------------------------------------------------

  [OK] Score: 1.000
      Department correct:  True  (+0.25)
      Doctor correct:      True  (+0.30)
      Booking successful:  True  (+0.35)
      Efficiency bonus:    +0.10
      Steps taken:         4
      Env cumulative reward: 1.0000

  TASK [HARD]
  User request: "I feel pain but not sure where"
  Expected: Cardiology -> Dr. Sarah Smith
  ----------------------------------------------------------

  [OK] Score: 1.000
      Department correct:  True  (+0.25)
      Doctor correct:      True  (+0.30)
      Booking successful:  True  (+0.35)
      Efficiency bonus:    +0.10
      Steps taken:         5
      Env cumulative reward: 0.8500

==============================================================
  SUMMARY
==============================================================
  EASY    1.000  |####################|
  MEDIUM  1.000  |####################|
  HARD    1.000  |####################|

  Average Score: 1.000
==============================================================
```

---

## 🔬 Design Decisions

1. **Tool-call only interaction** — mirrors real LLM agent architectures (function calling). No direct state access for the agent.
2. **Dense reward** — every step yields meaningful signal, avoiding sparse-reward dead zones.
3. **Deterministic clarification** — `ask_user_clarification` returns keyword-matched responses, ensuring graders are reproducible without randomness.
4. **Ground-truth injection** — `reset()` accepts `correct_department` and `correct_doctor` overrides, enabling task-specific grading without modifying core environment logic.
5. **Efficiency bonus** — incentivises agents to solve tasks in the minimum number of steps, not just to solve them.

---

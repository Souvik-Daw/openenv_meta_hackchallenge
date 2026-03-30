#!/usr/bin/env python3
"""
test_cases.py — Comprehensive Benchmarking & Testing System
for the Healthcare Appointment Scheduling RL Environment.

Evaluates multiple Groq model variants over an extensive dataset of test cases,
measuring accuracy, reward, efficiency, and error rates.

Outputs:
  - Readable console table
  - results.json (Detailed breakdown format)
  - results.csv (Flattened log)
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

from tqdm import tqdm

# Ensure local imports resolve
sys.path.insert(0, os.path.dirname(__file__))

from server.environment import HealthcareAppointmentEnvironment
import server.data
import server.tools
from tasks.graders import grade_full_breakdown
from agent.groq_agent import GroqAgent, SYSTEM_PROMPT

original_get_clarification = server.data.get_clarification_response


# ---------------------------------------------------------------------------
# Benchmark Agent Subclass (Extending GroqAgent for dynamic config)
# ---------------------------------------------------------------------------

class BenchmarkGroqAgent(GroqAgent):
    """
    Subclass of GroqAgent to allow dynamic temperature and model selection
    without modifying the original agent source code.
    """
    def __init__(self, model_name: str, temperature: float, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.MODEL = model_name
        self.temperature = temperature
        self.client.timeout = 30.0  # Optional safety timeout

    def decide_action(self, obs):
        """Override decide_action to inject specific model and temperature."""
        from models import AppointmentAction
        user_message = self._format_observation(obs)
        self._conversation.append({"role": "user", "content": user_message})

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *self._conversation,
                    ],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                raw = response.choices[0].message.content.strip()
                if self.verbose:
                    print(f"    [LLM raw] {raw[:200]}...")

                action = self._parse_action(raw)
                self._conversation.append({"role": "assistant", "content": raw})
                return action

            except Exception as exc:
                if self.verbose:
                    print(f"    [LLM error attempt {attempt}] {exc}")
                if attempt == self.MAX_RETRIES:
                    return AppointmentAction(
                        tool="get_departments",
                        parameters={},
                        metadata={"fallback": True},
                    )
        return AppointmentAction(tool="get_departments", parameters={})


# ---------------------------------------------------------------------------
# Test Case Generation (Combining static + custom synthesis)
# ---------------------------------------------------------------------------
# Mapping some known doctors to departments to act as Ground Truth
# Cardiology: Dr. Sarah Smith (chest pain), Dr. James Adams (arrhythmia)
# Dermatology: Dr. Priya Patel (rash, eczema), Dr. Kevin Lee (acne, mole)
# Neurology: Dr. Elena Rossi (migraine), Dr. Michael Chen (stroke, memory)
# Orthopedics: Dr. Thomas Grant (knee, fracture), Dr. Aisha Okafor (spine, back pain)

def generate_test_cases() -> List[Dict[str, Any]]:
    cases = []
    
    # helper
    def add(req, diff, dept, doc, req_clr=False, min_steps=4, clr_ans=None):
        cases.append({
            "difficulty": diff,
            "user_request": req,
            "correct_department": dept,
            "correct_doctor": doc,
            "expected_min_steps": min_steps,
            "requires_clarification": req_clr,
            "clarification_answer": clr_ans,
            "task_id": f"{diff}_{len(cases)}"
        })

    # A. Clear Cases (Easy)
    add("I have severe chest pain", "easy", "Cardiology", "Dr. Sarah Smith")
    add("I have a weird skin rash for 3 days", "easy", "Dermatology", "Dr. Priya Patel")
    add("I am experiencing a severe headache", "easy", "Neurology", "Dr. Elena Rossi")
    add("My knee really hurts after running", "easy", "Orthopedics", "Dr. Thomas Grant")
    add("I need an ECG interpretation", "easy", "Cardiology", "Dr. James Adams")
    add("I need to get a mole removed", "easy", "Dermatology", "Dr. Kevin Lee")
    add("My migraines are getting out of control", "easy", "Neurology", "Dr. Elena Rossi")
    add("I have severe lower back pain", "easy", "Orthopedics", "Dr. Aisha Okafor")
    add("I am having memory loss issues", "easy", "Neurology", "Dr. Michael Chen")
    add("I feel rapid heartbeat when I wake up", "easy", "Cardiology", "Dr. James Adams")
    add("My skin is flaking and scaly", "easy", "Dermatology", "Dr. Priya Patel")
    add("I broke my wrist during sports", "easy", "Orthopedics", "Dr. Thomas Grant")
    add("Please help with my chronic psoriasis", "easy", "Dermatology", "Dr. Priya Patel")
    add("I am having sudden palpitations", "easy", "Cardiology", "Dr. James Adams")
    add("I suspect I have a minor fracture", "easy", "Orthopedics", "Dr. Thomas Grant")

    # B. Medium Cases
    add("Feeling heart discomfort sometimes", "medium", "Cardiology", "Dr. Sarah Smith")
    add("Strange irritation on my forearm", "medium", "Dermatology", "Dr. Priya Patel")
    add("I've been feeling dizzy and numb", "medium", "Neurology", "Dr. Elena Rossi")
    add("I fell and my wrist is swelling up rapidly", "medium", "Orthopedics", "Dr. Thomas Grant")
    add("My chest feels tight and breathless", "medium", "Cardiology", "Dr. Sarah Smith")
    add("I noticed awful acne on my shoulders", "medium", "Dermatology", "Dr. Kevin Lee")
    add("Experiencing tremors in my left hand", "medium", "Neurology", "Dr. Elena Rossi")
    add("Need to get my scoliosis checked again", "medium", "Orthopedics", "Dr. Aisha Okafor")
    add("I need a screening for skin cancer", "medium", "Dermatology", "Dr. Kevin Lee")
    add("Recovering from a stroke, need a checkup", "medium", "Neurology", "Dr. Michael Chen")
    add("Ligaments in my ankle are torn", "medium", "Orthopedics", "Dr. Thomas Grant")
    add("Consistent pressure in my chest area", "medium", "Cardiology", "Dr. Sarah Smith")
    add("My back hurts if I sit for long hours", "medium", "Orthopedics", "Dr. Aisha Okafor")
    add("A blistering hive is expanding on my leg", "medium", "Dermatology", "Dr. Priya Patel")
    add("I have awful tingling in my toes", "medium", "Neurology", "Dr. Elena Rossi")

    # C. Hard / Ambiguous Cases (req_clr=True mostly)
    add("I feel pain but not sure where", "hard", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="It's a tight squeezing sensation in my chest.") 
    add("Something feels off in my body", "hard", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="I feel breathless and there is pressure in my chest.")
    add("I am not feeling well", "hard", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="My heart is beating irregularly and fast.")
    add("It's a tight squeezing sensation in the morning", "hard", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="I feel it mainly in my chest area.")
    add("I cannot walk properly", "hard", "Orthopedics", "Dr. Thomas Grant", req_clr=True, min_steps=5, clr_ans="There is sharp pain in my left knee when I step.")
    add("My skin is acting up", "hard", "Dermatology", "Dr. Priya Patel", req_clr=False, clr_ans="It is red, flaky, and itchy on my arms.") 
    add("I've been sick for three days and it's getting worse", "hard", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="I'm having heavy palpitations at night.")
    add("The pain is roughly 6 out of 10", "hard", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="It's a persistent ache in my chest.")
    add("Left side of my body feels awful", "hard", "Neurology", "Dr. Elena Rossi", req_clr=True, min_steps=5, clr_ans="I am getting terrible migraines and throbbing headaches.") 
    add("Need help with an ongoing issue", "hard", "Orthopedics", "Dr. Aisha Okafor", req_clr=True, min_steps=5, clr_ans="My lower back pain is radiating down my spine.")

    # D. Edge Cases
    add("Ahhhh", "edge", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="Sorry, I meant my chest has sudden sharp pain.")
    add("I want pizza", "edge", "Cardiology", "Dr. Sarah Smith", req_clr=True, min_steps=5, clr_ans="Actually, my chest pressure is extremely high right now.") 
    add("Chest pain and skin rash", "edge", "Cardiology", "Dr. Sarah Smith") 
    add("Hello? Is anyone there?", "edge", "Orthopedics", "Dr. Thomas Grant", req_clr=True, min_steps=5, clr_ans="Yes, my knee is very swollen and hurts.")
    add("I am dying", "edge", "Neurology", "Dr. Elena Rossi", req_clr=True, min_steps=5, clr_ans="I just had a mild seizure and feel dizzy.")
    
    # E. Noisy Inputs
    add("i hv chet pan very bad", "noisy", "Cardiology", "Dr. Sarah Smith")
    add("skiinn is reeed and itiichi", "noisy", "Dermatology", "Dr. Priya Patel")
    add("hadake is killing me man!!! :(", "noisy", "Neurology", "Dr. Elena Rossi")
    add("BAack pan. very much hurts!!! plz help fast.", "noisy", "Orthopedics", "Dr. Aisha Okafor")
    add("memory is bad cant remember lol", "noisy", "Neurology", "Dr. Michael Chen")
    add("hrt bit too fasst", "noisy", "Cardiology", "Dr. James Adams")
    add("pimples everywhere what to do", "noisy", "Dermatology", "Dr. Kevin Lee")
    add("Kneeee is broken I think", "noisy", "Orthopedics", "Dr. Thomas Grant")
    add("seizre yesterday night!!", "noisy", "Neurology", "Dr. Elena Rossi")
    add("moles look weird on my bck", "noisy", "Dermatology", "Dr. Kevin Lee")

    return cases


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

def run_tests():
    parser = argparse.ArgumentParser(description="Healthcare Benchmark Suite")
    parser.add_argument("--test-run", action="store_true", help="Run a small 3-case test")
    args = parser.parse_args()

    # Model variants to evaluate
    MODELS_TO_TEST = [
        ("llama-3.1-8b-instant", 0.0) 
    ]

    all_cases = generate_test_cases()
    if args.test_run:
        all_cases = all_cases[:3]

    print("\n" + "=" * 70)
    print(f"  Healthcare Benchmarking Suite initialized")
    print(f"  Test Cases: {len(all_cases)} | Models: {len(MODELS_TO_TEST)}")
    print("=" * 70 + "\n")

    results_data = {}
    csv_rows = []
    
    # Process sequentially 
    for model_name, temp in MODELS_TO_TEST:
        variant_tag = f"{model_name} (temp={temp})"
        print(f"Evaluating model: {variant_tag} ...")
        
        agent = BenchmarkGroqAgent(model_name=model_name, temperature=temp, verbose=False)
        
        variant_results = []
        
        for case in tqdm(all_cases, desc=variant_tag, unit="case"):
            
            # Monkeypatch the data module to return the custom test case clarification constraint
            def dynamic_clarification(question: str) -> str:
                if case.get("clarification_answer"):
                    return case["clarification_answer"]
                return original_get_clarification(question)
            
            server.data.get_clarification_response = dynamic_clarification
            # Initialize env
            env = HealthcareAppointmentEnvironment()
            obs = env.reset(
                user_request=case["user_request"],
                correct_department=case["correct_department"],
                correct_doctor=case["correct_doctor"]
            )
            agent.reset_conversation()
            
            error_details = {"invalid_calls": 0, "wrong_department": False, "wrong_doctor": False}

            # Agent Loop
            step = 0
            while not obs.done and step < 10:
                step += 1
                try:
                    action = agent.decide_action(obs)
                    # Detect fallback/invalid calls
                    if action.metadata and action.metadata.get("fallback"):
                        error_details["invalid_calls"] += 1
                        
                    obs = env.step(action)
                    time.sleep(0.3)  # Anti-rate-limit sleep
                except Exception as e:
                    print(f"Environment/Agent Error on case {case['task_id']}: {e}")
                    break
            
            # Grade
            breakdown = grade_full_breakdown(env.state, case)
            
            # Collect Error Metrics
            if not breakdown["department_correct"]:
                error_details["wrong_department"] = True
            if not breakdown["doctor_correct"]:
                error_details["wrong_doctor"] = True
                
            case_result = {
                "task_id": case["task_id"],
                "input": case["user_request"],
                "difficulty": case["difficulty"],
                "expected_department": case["correct_department"],
                "expected_doctor": case["correct_doctor"],
                "success": breakdown["booking_successful"],
                "reward": breakdown["cumulative_env_reward"],
                "steps": breakdown["steps_taken"],
                "accuracy": breakdown["final_score"],
                "errors": error_details,
                "predicted_department": getattr(env.state, "identified_department", "None"),
                "predicted_doctor": getattr(env.state, "selected_doctor", "None")
            }
            variant_results.append(case_result)
            
            csv_rows.append({
                "model": variant_tag,
                "task_id": case["task_id"],
                "input": case["user_request"],
                "difficulty": case["difficulty"],
                "success": case_result["success"],
                "accuracy": case_result["accuracy"],
                "reward": case_result["reward"],
                "steps": case_result["steps"],
                "invalid_calls": error_details["invalid_calls"]
            })

        # Calculate Summaries
        successes = sum(1 for r in variant_results if r["success"])
        avg_reward = sum(r["reward"] for r in variant_results) / len(variant_results)
        avg_steps = sum(r["steps"] for r in variant_results) / len(variant_results)
        avg_acc = sum(r["accuracy"] for r in variant_results) / len(variant_results)
        
        wrong_depts = sum(1 for r in variant_results if r["errors"]["wrong_department"])
        wrong_docs = sum(1 for r in variant_results if r["errors"]["wrong_doctor"])
        total_invalid_calls = sum(r["errors"]["invalid_calls"] for r in variant_results)

        results_data[variant_tag] = {
            "summary": {
                "Accuracy": avg_acc,
                "Avg Reward": avg_reward,
                "Avg Steps": avg_steps,
                "Completion Rate": successes / len(variant_results),
                "Total Wrong Depts": wrong_depts,
                "Total Wrong Docs": wrong_docs,
                "Total Invalid Calls": total_invalid_calls
            },
            "test_cases": variant_results
        }
            

    # -----------------------------------------------------------------------
    # Exporters and Final Output
    # -----------------------------------------------------------------------
    
    # 1. Detailed JSON Output
    with open("results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # 2. CSV Export
    with open("results.csv", "w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # 3. Console Output (Readable Table Output & Ranking)
    print("\n" + "=" * 70)
    print("  MODEL RUN PERFORMANCE SUMMARY")
    print("=" * 70)
    for model_name, output in results_data.items():
        summary = output["summary"]
        print(f"Model: {model_name}")
        print("-" * 32)
        print(f"Accuracy:         {summary['Accuracy'] * 100:.1f}%")
        print(f"Avg Reward:       {summary['Avg Reward']:.3f}")
        print(f"Avg Steps:        {summary['Avg Steps']:.1f}")
        print(f"Completion Rate:  {summary['Completion Rate'] * 100:.1f}%")
        print(f"Wrong Depts:      {summary['Total Wrong Depts']}")
        print(f"Invalid Calls:    {summary['Total Invalid Calls']}\n")

    # Final Ranking
    print("=" * 70)
    print("  COMPARISON TABLE & RANKINGS")
    print("=" * 70)
    print(f"| {'Model':<30} | {'Accuracy':<8} | {'Reward':<8} | {'Steps':<6} | {'Rank':<4} |")
    print("-" * 70)
    
    # Sort models by Accuracy, then Reward, then Efficiency (Steps ASC)
    ranked_models = sorted(
        results_data.keys(),
        key=lambda k: (
            results_data[k]["summary"]["Accuracy"],
            results_data[k]["summary"]["Avg Reward"],
            -results_data[k]["summary"]["Avg Steps"]
        ),
        reverse=True
    )
    
    for rank, model in enumerate(ranked_models, 1):
        summary = results_data[model]["summary"]
        acc = f"{summary['Accuracy']*100:.1f}%"
        rew = f"{summary['Avg Reward']:.3f}"
        steps = f"{summary['Avg Steps']:.1f}"
        print(f"| {model:<30} | {acc:<8} | {rew:<8} | {steps:<6} | {rank:<4} |")
        
    print("=" * 70)
    print("Saved exported artifacts: results.json and results.csv")


if __name__ == "__main__":
    run_tests()

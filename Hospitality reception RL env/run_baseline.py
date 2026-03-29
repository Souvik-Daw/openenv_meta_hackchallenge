#!/usr/bin/env python3
"""
run_baseline.py — Baseline evaluation script for the Healthcare Appointment
                  Scheduling RL environment.

Runs the Groq agent over all three tasks (easy, medium, hard)
and prints a final score report.

Usage:
    export GROQ_API_KEY=gsk_...
    python run_baseline.py

    # To run without the LLM (oracle mode for testing):
    python run_baseline.py --oracle
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Ensure local packages resolve correctly whether run from project root
# ---------------------------------------------------------------------------
import os
sys.path.insert(0, os.path.dirname(__file__))

from server.environment import HealthcareAppointmentEnvironment
from models import AppointmentAction
from tasks.easy   import get_task_config as easy_config,   USER_REQUEST as EASY_REQ,   CORRECT_DEPARTMENT as EASY_DEPT,   CORRECT_DOCTOR as EASY_DOC
from tasks.medium import get_task_config as medium_config, USER_REQUEST as MED_REQ,    CORRECT_DEPARTMENT as MED_DEPT,    CORRECT_DOCTOR as MED_DOC
from tasks.hard   import get_task_config as hard_config,   USER_REQUEST as HARD_REQ,   CORRECT_DEPARTMENT as HARD_DEPT,   CORRECT_DOCTOR as HARD_DOC
from tasks.graders import grade_full_breakdown


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: List[Tuple[str, str, str, str, Any]] = [
    ("easy",   EASY_REQ,  EASY_DEPT,  EASY_DOC,  easy_config),
    ("medium", MED_REQ,   MED_DEPT,   MED_DOC,   medium_config),
    ("hard",   HARD_REQ,  HARD_DEPT,  HARD_DOC,  hard_config),
]


# ---------------------------------------------------------------------------
# Oracle agent (deterministic, no LLM — for testing without API key)
# ---------------------------------------------------------------------------

ORACLE_ACTIONS: Dict[str, List[AppointmentAction]] = {
    "easy": [
        AppointmentAction(tool="get_departments", parameters={}),
        AppointmentAction(tool="get_doctors", parameters={"department": "Cardiology"}),
        AppointmentAction(tool="check_availability", parameters={"doctor": "Dr. Sarah Smith"}),
        AppointmentAction(tool="book_appointment", parameters={"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"}),
    ],
    "medium": [
        AppointmentAction(tool="get_departments", parameters={}),
        AppointmentAction(tool="get_doctors", parameters={"department": "Dermatology"}),
        AppointmentAction(tool="check_availability", parameters={"doctor": "Dr. Priya Patel"}),
        AppointmentAction(tool="book_appointment", parameters={"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"}),
    ],
    "hard": [
        AppointmentAction(tool="ask_user_clarification", parameters={"question": "Where exactly is the pain located?"}),
        AppointmentAction(tool="get_departments", parameters={}),
        AppointmentAction(tool="get_doctors", parameters={"department": "Cardiology"}),
        AppointmentAction(tool="check_availability", parameters={"doctor": "Dr. Sarah Smith"}),
        AppointmentAction(tool="book_appointment", parameters={"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"}),
    ],
}


def run_oracle_task(
    difficulty: str,
    user_req: str,
    dept: str,
    doctor: str,
    config_fn,
) -> Dict[str, Any]:
    """Run a single task with the deterministic oracle agent."""
    env = HealthcareAppointmentEnvironment()
    obs = env.reset(user_request=user_req, correct_department=dept, correct_doctor=doctor)

    for action in ORACLE_ACTIONS[difficulty]:
        obs = env.step(action)
        if obs.done:
            break

    return grade_full_breakdown(env.state, config_fn())


# ---------------------------------------------------------------------------
# LLM agent runner
# ---------------------------------------------------------------------------

def run_llm_task(
    difficulty: str,
    user_req: str,
    dept: str,
    doctor: str,
    config_fn,
    agent,
) -> Dict[str, Any]:
    """Run a single task with the Groq LLM agent."""
    env = HealthcareAppointmentEnvironment()
    obs = env.reset(user_request=user_req, correct_department=dept, correct_doctor=doctor)
    agent.reset_conversation()

    step = 0
    while not obs.done and step < 10:
        step += 1
        action = agent.decide_action(obs)
        print(f"    Step {step}: {action.tool}({json.dumps(action.parameters)})")
        obs = env.step(action)
        print(f"           → reward={obs.reward:.4f} | {obs.message[:80]}")
        # Small sleep to avoid hitting Groq rate limits
        time.sleep(0.5)

    return grade_full_breakdown(env.state, config_fn())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run healthcare scheduling baseline evaluation."
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Run deterministic oracle agent instead of LLM (no API key needed).",
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task(s) to run (default: all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print LLM raw output (default: True).",
    )
    args = parser.parse_args()

    use_oracle = args.oracle
    task_filter = args.task

    if not use_oracle:
        from agent.groq_agent import GroqAgent
        try:
            agent = GroqAgent(verbose=args.verbose)
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Tip: run with --oracle to use the deterministic agent instead.")
            sys.exit(1)
    else:
        agent = None  # type: ignore

    # Filter tasks
    tasks_to_run = [
        t for t in TASK_REGISTRY
        if task_filter == "all" or t[0] == task_filter
    ]

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 62)
    print("  Healthcare Appointment Scheduling — Baseline Evaluation")
    print("=" * 62)
    mode = "ORACLE (deterministic)" if use_oracle else f"GROQ ({GroqAgent.MODEL if not use_oracle else 'n/a'})"
    print(f"  Mode: {mode}")
    print("=" * 62)

    for difficulty, user_req, dept, doctor, config_fn in tasks_to_run:
        print(f"\n  TASK [{difficulty.upper()}]")
        print(f"  User request: \"{user_req}\"")
        print(f"  Expected: {dept} -> {doctor}")
        print("  " + "-" * 58)

        if use_oracle:
            breakdown = run_oracle_task(difficulty, user_req, dept, doctor, config_fn)
        else:
            breakdown = run_llm_task(difficulty, user_req, dept, doctor, config_fn, agent)

        results.append(breakdown)

        print(f"\n  [OK] Score: {breakdown['final_score']:.3f}")
        print(f"      Department correct:  {breakdown['department_correct']}  (+{breakdown['department_score']:.2f})")
        print(f"      Doctor correct:      {breakdown['doctor_correct']}  (+{breakdown['doctor_score']:.2f})")
        print(f"      Booking successful:  {breakdown['booking_successful']}  (+{breakdown['booking_score']:.2f})")
        print(f"      Efficiency bonus:    +{breakdown['efficiency_bonus']:.2f}")
        print(f"      Clarification penalty: {breakdown['clarification_penalty']:.2f}")
        print(f"      Steps taken:         {breakdown['steps_taken']}")
        print(f"      Env cumulative reward:{breakdown['cumulative_env_reward']:.4f}")

    # Summary table
    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    avg_score = sum(r["final_score"] for r in results) / max(len(results), 1)
    for r in results:
        bar = "#" * int(r["final_score"] * 20)
        print(f"  {r['difficulty'].upper():6s}  {r['final_score']:.3f}  |{bar:<20}|")
    print(f"\n  Average Score: {avg_score:.3f}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()

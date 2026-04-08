import asyncio
import os
import json
import textwrap
import re
import sys
from typing import List, Optional, Dict, Any

from openai import OpenAI

# Path setup so local imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

try:
    from client import HealthcareEnv
    from models import AppointmentAction, AppointmentObservation
except ImportError:
    # Handle cases where it might be installed as a package
    from healthcare_scheduling.client import HealthcareEnv
    from healthcare_scheduling.models import AppointmentAction, AppointmentObservation

# Environment variables for LLM
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Environment variables for Task / Benchmark
TASK_NAME = os.getenv("TASK_NAME", "easy_chest_pain")
BENCHMARK = "healthcare_scheduling"
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 512

SYSTEM_PROMPT = textwrap.dedent("""
    You are a healthcare appointment scheduling assistant agent.
    Your task is to help a patient book an appointment with the correct doctor.

    ## Available Tools
    You may call ONLY ONE of these tools per turn:
    1. get_departments()
        Returns the list of available medical departments.

    2. get_doctors(department: str)
        Returns doctors in a department with their specializations.

    3. check_availability(doctor: str)
        Returns available time slots for a specific doctor.

    4. book_appointment(doctor: str, slot: str)
        Books an appointment. Use EXACT doctor name and EXACT slot string.

    5. ask_user_clarification(question: str)
        Ask the patient a clarifying question when the symptom is ambiguous.

    ## Strategy
    1. If the user request is ambiguous, call ask_user_clarification FIRST.
    2. Call get_departments() to see available departments.
    3. Match the symptom to the correct department.
    4. Call get_doctors(department) to find the appropriate specialist.
    5. Call check_availability(doctor) to find an open slot.
    6. Call book_appointment(doctor, slot) with exact strings to complete.

    ## Output Format
    You MUST respond with ONLY valid JSON in this exact format:
    {
      "tool": "<tool_name>",
      "parameters": {<parameters_dict>},
      "reasoning": "<brief explanation of why you chose this tool>"
    }

    No extra text, no markdown code blocks  only raw JSON.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def format_observation(obs: AppointmentObservation) -> str:
    lines = [
        f"## Current Observation (Step {obs.steps_taken}/{obs.max_steps})",
        f"**User Request:** {obs.user_request}",
        f"**Last Tool Used:** {obs.tool_called or 'None'}",
        f"**Tool Result:**",
        json.dumps(obs.tool_result, indent=2) if obs.tool_result else "(no result yet)",
        "",
        f"**Identified Department:** {obs.identified_department or 'Not yet determined'}",
        f"**Selected Doctor:** {obs.selected_doctor or 'Not yet selected'}",
        f"**Selected Slot:** {obs.selected_slot or 'Not yet selected'}",
        "",
        f"**Environment Message:** {obs.message}",
        "",
        "Based on the above, what is your next tool call?",
        "Respond ONLY with JSON: {\"tool\": \"...\", \"parameters\": {...}, \"reasoning\": \"...\"}",
    ]
    return "\n".join(lines)

def parse_action(raw: str) -> AppointmentAction:
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = cleaned.rstrip("`").strip()
    
    data = json.loads(cleaned)
    tool = data.get("tool", "").strip()
    params = data.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
        
    return AppointmentAction(
        tool=tool,
        parameters=params,
        metadata={"reasoning": data.get("reasoning", "")}
    )

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize environment
    if IMAGE_NAME:
        env = await HealthcareEnv.from_docker_image(IMAGE_NAME)
    else:
        # Fallback for local testing if no image name provided
        env_url = os.getenv("API_BASE_URL_ENV", "http://localhost:8000")
        env = HealthcareEnv(base_url=env_url)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    conversation = []

    try:
        async with env:
            # Start episode
            step_result = await env.reset()
            obs = step_result.observation
            
            for step in range(1, MAX_STEPS + 1):
                if step_result.done:
                    break
                    
                user_msg = format_observation(obs)
                conversation.append({"role": "user", "content": user_msg})
                
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *conversation
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    raw_response = completion.choices[0].message.content or ""
                    action = parse_action(raw_response)
                    conversation.append({"role": "assistant", "content": raw_response})
                except Exception as e:
                    error_msg = str(e)
                    log_step(step=step, action="error", reward=0.0, done=True, error=error_msg)
                    break

                action_str = f"{action.tool}({json.dumps(action.parameters)})"
                
                # Execute action
                step_result = await env.step(action)
                obs = step_result.observation
                reward = step_result.reward or 0.0
                done = step_result.done
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=None)
                
                if done:
                    # Heuristic for success: correct booking gives 0.5 reward
                    if reward >= 0.5:
                        success = True
                    break
        
    except Exception as e:
        # Silent fail or debug log
        pass
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())

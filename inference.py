
import asyncio
import os
import textwrap
from typing import List, Optional
import logging
from openai import OpenAI

from client import HospitalmanageTriageEnv
from openenv.core.env_server.mcp_types import CallToolAction
import time

from dotenv import load_dotenv

# Load variables from .env file into the environment
# load_dotenv(".env.example")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1") #OK
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:groq") #OK
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")  # Support both API_KEY and HF_API_KEY for flexibility
TASK_NAME = os.getenv("TASK_NAME")
BENCHMARK = os.getenv("BENCHMARK") or "hospital_manage_triage" #OK

MAX_STEPS = 10
TEMPERATURE = 0.7

def main(task_name) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    # https://stavust28-hospitalmanage-triage-env.hf.space
    env = HospitalmanageTriageEnv(base_url="https://stavust28-hospitalmanage-triage-env.hf.space").sync()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    task = parse_task(task_name)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(
            patient_id=task["patient_id"] , 
            tool_call_sequence=task['expected_output']['tool_seq'],
            output_sequence=task['expected_output']['tool_output']
        )

        last_message = result.observation.result
        last_reward = result.observation.reward

        if last_message == 'Hospitalmanage Triage Env environment ready!':

            # read task json task
            last_message = task['query']
            # logger.info(f"Task query: {last_message}")
            for step in range(1, MAX_STEPS + 1):
                if result.observation.done:
                    break

                message = get_model_message(client, step, last_message, last_reward, history)

                # logger.info("LLM message: %s", message)
                func_name, tool_args = parse_tool_args(message)
                # logger.info("Parsed function: %s with arguments: %s", func_name, tool_args)
                assert isinstance(tool_args, dict), f"Parsed tool arguments should be a dictionary, got: {tool_args}"

                result = env.step(CallToolAction(tool_name=func_name, arguments=tool_args))
                obs = result.observation

                reward = obs.reward or 0.0
                done = obs.done
                error = None

                rewards.append(reward)
                steps_taken = step
                last_message = obs.result
                last_reward = reward
                # logger.info("Step result: %s", last_message) #
                log_step(task=task_name, step=step, action=message, reward=reward, done=done, error=error)

                history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

                if done:
                    break

            score = sum(rewards) / len(rewards)
            score = max(0.01, min(0.99, score))  # clamp to [0, 1]
            success = score >= 0.5 and done
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)




SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a Hospital environment.
    Each turn you must send a tool call. The environment return a reward and response
    Reward is variable and depends on the tool used and the context of the request.
    Your goal is to maximize total reward by performing correct task in small steps.
    
    ** Remember **:
        1. DO NOT greet the user.
        2. DO NOT provide explanations or justifications for your actions.
        3. DO NOT make assumptions about the user's intent.
        4. DO NOT divert from the `list of tools` and `Output Response`. if you do so, you will be penalized with -inf.

    ** Output Response **
        <selected_tool_with_parameters>
        <example>
        get_department(conditions="chest pain, shortness of breath")
        </example>
    
    ** The list of tools **:
        - Tool 1: get_department(conditions: str) :
            - When to call : when the user is searching for a department given conditions
            - `conditions` : a string describing the conditions to match ex 'chest pain, shortness of breath'; in comma separated
        - Tool 2: get_opd_doctor(department: str) :
            - When to call : when the user is searching for a doctor in a specific department
            - `department` : the name of the department to search for doctors
        - Tool 3: make_appointment(self, doctor_id: str, patient_id: str, doctor_name: str, patient_name: str) :
            - When to call : when the user wants to schedule an appointment with a specific doctor
            - `doctor_id` : the ID of the doctor to schedule the appointment with
            - `patient_id` : the ID of the patient to schedule the appointment for
            - `doctor_name` : the name of the doctor to schedule the appointment with
            - `patient_name` : the name of the patient to schedule the appointment for

    from the given tools select ONLY ONE tool in response
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(task: str, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] task={task} step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} score={score:.3f} success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    # print("User prompt:", user_prompt)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # print("Text from model:", completion.choices[0].message.content)
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"

def parse_tool_args(message: str):
    import re
    import ast

    # 1. Clean up the message: remove markdown backticks and language tags
    message = message.replace("```python", "").replace("```", "").strip()
    
    # 2. Extract ONLY the function call part (e.g., get_doctor(arg='val'))
    # This regex is better at ignoring surrounding conversational text
    match = re.search(r"(\w+)\s*\((.*)\)", message, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract function pattern from: {message}")
    
    func_name = match.group(1)
    args_str = match.group(2)
    
    # 3. Use a safer way to parse the arguments
    # We wrap it in a dummy function call to make it valid AST
    full_call_str = f"{func_name}({args_str})"
    try:
        tree = ast.parse(full_call_str)
        call = tree.body[0].value
        
        arguments = {}
        for kw in call.keywords:
            # kw.value can be ast.Constant (3.8+) or ast.Str/ast.Num (older)
            if hasattr(kw.value, 'value'):
                arguments[kw.arg] = kw.value.value
            elif hasattr(kw.value, 's'):
                arguments[kw.arg] = kw.value.s
            elif hasattr(kw.value, 'n'):
                arguments[kw.arg] = kw.value.n
        
        return func_name, arguments
    except SyntaxError as e:
        raise ValueError(f"Syntax error in tool call: {e}")

def parse_task(task_name: str):
    import json
    with open('task.json', 'r') as f:
        task = json.load(f)
    return next((t for t in task if t["task_id"] == task_name), None)

if __name__ == "__main__":
    import json
    with open('task.json', 'r') as f:
        tasks = json.load(f)
        
    for task in tasks:
        task_id = task["task_id"]
        # print(f"Starting inference for: {task_id}")
        main(task_id)
        # time.sleep(30)
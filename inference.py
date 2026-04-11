
import asyncio
import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()
import textwrap
from typing import List, Optional

from openai import AsyncOpenAI

from client import HospitalmanageTriageEnv
from openenv.core.env_server.mcp_types import CallToolAction


IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1" or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("GROQ_MODEL") or os.getenv("MODEL_NAME") or "llama-3.3-70b-versatile" or "openai/gpt-oss-120b:groq"
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a Hospital environment.
    Each turn you must send a tool call. The environment return a reward and response
    Reward is variable and depends on the tool used and the context of the request.
    Your goal is to maximize total reward by performing correct task in small steps

    The list of tools:
        - Tool 1: get_department(conditions: str) :
            - When to call : when the user is searching for a department given conditions
            - `conditions` : a string describing the conditions to match ex 'chest pain, shortness of breath'; in comma separated
        - Tool 2: get_opd_doctor(department: str) :
            - When to call : when the user is searching for a doctor in a specific department
            - `department` : the name of the department to search for doctors
        - Tool 3: make_appointment(self, doctor_id: int, patient_id: int, doctor_name: str, patient_name: str) :
            - When to call : when the user wants to schedule an appointment with a specific doctor
            - `doctor_id` : the ID of the doctor to schedule the appointment with
            - `patient_id` : the ID of the patient to schedule the appointment for
            - `doctor_name` : the name of the doctor to schedule the appointment with
            - `patient_name` : the name of the patient to schedule the appointment for

    from the given tools select ONLY ONE tool in response

    ***Output Response***
        <selected_tool_with_parameters>
        <example>
        get_department(conditions="chest pain, shortness of breath")
        </example>
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
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


async def get_model_message(client: AsyncOpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"

def parse_task():
    import json
    with open('task.json', 'r') as f:
        task = json.load(f)
    return task[0]

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

async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        raise NotImplementedError("Use base_url with the running server instead of Docker.")
    else:
        env = HospitalmanageTriageEnv(base_url="http://localhost:8000").sync()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    task = parse_task()

    # log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(
            patient_id=task.get("patient_id"),
            tool_call_sequence=task.get("expected_output", {}).get("tool_seq", []),
            output_sequence=task.get("expected_output", {}).get("tool_output", [])
        ) # OpenENV.reset()
        last_message = result.observation.result
        last_reward = result.observation.reward

        for step in range(1, MAX_STEPS + 1):
            if result.observation.done:
                break

            message = await get_model_message(client, step, last_message, last_reward, history)
            print(f"Model raw message: {message}")
            
            try:
                tool_name, arguments = parse_tool_args(message)
                print(f"-> Calling Tool: {tool_name} with args: {arguments}")
                result = env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
            except Exception as e:
                print(f"-> Parse error: {e}")
                # Tell the model it made a syntax format error next step
                last_message = f"Error: Tool parsing failed. Ensure you output exactly ONE function call, like: get_department(conditions='...'). Exception: {e}"
                last_reward = -0.5
                history.append(f"Step {step}: failed to parse '{message!r}'")
                continue
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_message = obs.result
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        # score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        # score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        # success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
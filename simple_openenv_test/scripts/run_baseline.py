import sys
import os
import json
from dotenv import load_dotenv, find_dotenv

# Add the project root to sys.path so it can find the 'env' and 'tools' modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Auto-search and load the .env file (even if it's in a parent directory)
load_dotenv(find_dotenv())

from groq import Groq
from env.env import TaskRoutingEnv
from tools.groq_tool import GroqRoutingTool
from env.graders import grader_task_1, grader_task_2, grader_task_3

def run_task(client, task_id, grader_fn):
    print(f"\n{'='*40}")
    print(f"--- Running Task {task_id} ---")
    
    env = TaskRoutingEnv(task_id=task_id)
    tool = GroqRoutingTool(env)
    
    obs = tool.reset_env()
    print(f"User Request: '{obs['request_text']}'")
    
    system_prompt = (
        "You are an intelligent routing AI. "
        "Your job is to route customer requests to the correct department using the step_env tool. "
        "The possible departments are billing, tech, and general."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"New Request:\n'{obs['request_text']}'\n\nRoute this request to one of: billing, tech, general."}
    ]
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            messages=messages,
            tools=tool.get_tool_schema(),
            tool_choice={"type": "function", "function": {"name": "step_env"}},
            temperature=0.0
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        action_args = json.loads(tool_call.function.arguments)
        print(f"Agent Predicted Route: {action_args.get('selected_department')}")
        
        step_res = tool.step_env(action_args)
        
        reward = step_res["reward"]
        print(f"Reward: {reward['reward']}")
        
        final_score = grader_fn(env.state())
        print(f"Final Grader Score: {final_score}")
        print("----------------------------------")
        
    except Exception as e:
        print(f"Error calling LLM: {e}")

def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is missing. Please set it using export GROQ_API_KEY=\"...\"")
        return
        
    print("Initializing Groq client...")
    client = Groq(api_key=api_key)
    
    run_task(client, 1, grader_task_1)
    run_task(client, 2, grader_task_2)
    run_task(client, 3, grader_task_3)

if __name__ == "__main__":
    main()

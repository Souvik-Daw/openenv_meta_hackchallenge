from client import HealthcareEnv 
from models import AppointmentAction

HF_SPACE_URL = "https://fergus2000-hospital-env.hf.space"

with HealthcareEnv(base_url=HF_SPACE_URL).sync() as env:
    print("--- Connecting to Hospital Env ---")
    
    initial_obs = env.reset()
    print("Environment Reset Successful.")

    action = AppointmentAction(tool="get_departments", parameters={})
    result = env.step(action)
    
    print("\nAvailable Departments:")
    print(result.observation.tool_result)

    action_docs = AppointmentAction(tool="get_doctors", parameters={"department": "Cardiology"})
    result_docs = env.step(action_docs)
    print("\nDoctors List:")
    print(result_docs.observation.tool_result)
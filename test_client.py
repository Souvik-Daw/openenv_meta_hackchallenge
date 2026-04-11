from client import HospitalmanageTriageEnv
from openenv.core.env_server.mcp_types import CallToolAction

with HospitalmanageTriageEnv(base_url="http://localhost:8000").sync() as env:

    ### test reset + get_department
    op = env.reset(patient_id=8118,
                   tool_call_sequence=['get_department'],
                   output_sequence=['The proper department for the patient is:'])
    print("RESET:", op.observation)

    result = env.step(CallToolAction(tool_name="get_department", arguments={"conditions": "headache"}))
    print("get_department:", result.observation)
    print("  reward:", result.reward, "done:", result.done)

    ### test full workflow: get_department -> get_opd_doctor -> make_appointment
    # env.reset(patient_id=8144,
    #           tool_call_sequence=['get_department','get_opd_doctor','make_appointment'],
    #           output_sequence=['The proper department for the patient is: Neurology',
    #                            'The proper doctor for the patient is:',
    #                            'Appointment made successfully.'])
    # result = env.step(CallToolAction(tool_name="get_department", arguments={"conditions": "headache"}))
    # print("get_department:", result.observation)
    # result = env.step(CallToolAction(tool_name="get_opd_doctor", arguments={"department": "Neurology"}))
    # print("get_opd_doctor:", result.observation)
    # result = env.step(CallToolAction(tool_name="make_appointment",
    #                                  arguments={"doctor_id": 3, "patient_id": 8144,
    #                                             "doctor_name": "Dr. Maria Lopez", "patient_name": "Alice Doe"}))
    # print("make_appointment:", result.observation)
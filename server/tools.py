from typing import Dict,List, Any, Optional
from .utils import search, search_like,insert_db, find_department_tfidf
from datetime import datetime
try:
    from ..models import HospitalToolsOutput
except ImportError:
    from models import HospitalToolsOutput

class ToolSpec:
    def __init__(self): pass

    def get_department(self, conditions: str) -> HospitalToolsOutput:

        def __get_department_wrapper(output):
            try:
                if isinstance(output, str):
                    return f'The proper department for the patient is: {output}'
            except Exception as e:
                return 'Error occurred while wrapping department.'
        
        result = find_department_tfidf(conditions)

        return HospitalToolsOutput(tool="get_department", 
                                   message=__get_department_wrapper(result),
                                   tool_state={"department": result})

    def get_opd_doctor(self, department: str) -> HospitalToolsOutput:

        def __get_doctor_wrapper(output):
            try:
                if isinstance(output, dict):
                    return f"The proper doctor for the patient is: {output.get('doctor_name', '<ERROR>')} with doctor ID: {output.get('doctor_id', '<ERROR>')}"
            except Exception as e:
                return 'Error occurred while wrapping doctor.'


        result = search(conn_or_path="server/hospital.db",
                        table='doctors',
                        columns=["doctor_id", "doctor_name"],
                        where=f"department_name = ? AND is_available_opd = ? AND day_of_week = ?",
                        params=(department, True, datetime.now().strftime("%A")),
                        limit=1
                        )[0]
        
        return HospitalToolsOutput(tool="get_opd_doctor",
                                   message=__get_doctor_wrapper(result),
                                   tool_state={"doctor_id": result.get("doctor_id")})

    def make_appointment(self, doctor_id: int, patient_id: int, doctor_name: str, patient_name: str) -> HospitalToolsOutput:
        data={
                      "doctor_id": doctor_id,
                      "patient_id": patient_id,
                      "doctor_name": doctor_name,
                      "patient_name": patient_name,
                      "day": datetime.now().strftime('%Y-%m-%d')}
        result = insert_db(conn_or_path="server/hospital.db",
                  table='appointments',
                  data=data)
        if result:
            return HospitalToolsOutput(tool="make_appointment", message="Appointment made successfully.", tool_state={})

    def get_appointment(self, doctor_id: Optional[int],  patient_id: Optional[int]) -> HospitalToolsOutput:

        def __get_appointment_wrapper(output):
            try:
                if isinstance(output, dict):
                    if patient_id:
                        return f'The upcoming appointment is with: {output.get("doctor_name", "<ERROR>")} is on comming {output.get("day", "<ERROR>")}'
                    return f'The next appointment is with: {output.get("patient_name", "<ERROR>")} is on {output.get("day", "<ERROR>")}'
            except Exception as e:
                return 'Error occurred while wrapping doctor.'

        if patient_id:
            columns,where,params = ["doctor_id", "doctor_name", "day"], f"patient_id = ?", (patient_id,)
        elif doctor_id:
            columns,where,params = ["patient_id", "patient_name", "day"], f"doctor_id = ?", (doctor_id,)

        result = search(conn_or_path="server/hospital.db",
                        table='appointments',
                        columns=columns,
                        where=where,
                        params=params,
                        limit=1
                        )[0]
        
        return HospitalToolsOutput(tool="get_appointment", message=__get_appointment_wrapper(result), tool_state={})
    
if __name__ == "__main__":
    import inspect

    tools = ToolSpec()
    # method to call
    method = "get_department"
    print(getattr(tools, method))
    print(inspect.getfullargspec(getattr(tools, method)).args)

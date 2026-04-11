from typing import Dict,List, Any, Optional
from .utils import search, search_like,insert_db, find_department_tfidf
from datetime import datetime
try:
    from ..models import HospitalToolsOutput
except ImportError:
    from models import HospitalToolsOutput

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hospital.db")
print(f"DATABASE DETECTED AT: {DB_PATH}")

class ToolSpec:
    def __init__(self): pass

    def get_department(self, conditions: str) -> HospitalToolsOutput:

        def __get_department_wrapper(output):
            try:
                if isinstance(output, str):
                    return f'The proper department for the patient is: {output}'
            except Exception as e:
                return 'Error occurred while wrapping department.'
        
        print("calling tfidf models")
        result = find_department_tfidf(conditions)
        print("tfidf models called")

        return HospitalToolsOutput(tool="get_department", 
                                   message=__get_department_wrapper(result),
                                   tool_state={"department": result})

    def make_appointment(self, doctor_id: int, patient_id: int, doctor_name: str, patient_name: str) -> HospitalToolsOutput:
        data={
                      "doctor_id": int(doctor_id),
                      "patient_id": int(patient_id),
                      "doctor_name": doctor_name,
                      "patient_name": patient_name,
                      "day": datetime.now().strftime('%Y-%m-%d')}
        result = insert_db(conn_or_path=DB_PATH,
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
        
        results = search(conn_or_path=DB_PATH,
                        table='appointments',
                        columns=columns,
                        where=where,
                        params=params,
                        limit=1
                        )
        result = results[0] if results else {}
        
        return HospitalToolsOutput(tool="get_appointment", message=__get_appointment_wrapper(result), tool_state={})

    def get_opd_doctor(self, department: str) -> HospitalToolsOutput:
        # 1. Normalize the input (Helps with 'Cardiology' vs 'cardiology')
        department_clean = department.strip().title() 
        current_day = datetime.now().strftime("%A")

        print("Start of function get_opd_doctor")

        def __get_doctor_wrapper(output):
            # Ensure output is a dictionary and has the keys
            if isinstance(output, dict) and output:
                d_name = output.get('doctor_name', 'Unknown')
                d_id = output.get('doctor_id', 'Unknown')
                return f"The proper doctor for the patient is: {d_name} with doctor ID: {d_id}"
            return f"No available doctor found in {department_clean} for today ({current_day})."

        try:
            results = search(
                conn_or_path=DB_PATH,
                table='doctors',
                columns=["doctor_id", "doctor_name"],
                # Use LIKE for more flexible matching if needed
                where="department_name = ? AND is_available_opd = ? AND day_of_week = ?",
                params=(department_clean, True, current_day),
                limit=1
            )
            
            # Safely handle empty list or None
            result = results[0] if (results and len(results) > 0) else {}
            
        except Exception as e:
            # This prevents the whole server from crashing
            return HospitalToolsOutput(
                tool="get_opd_doctor",
                message=f"Database error: {str(e)}",
                tool_state={}
            )

        print("End of function get_opd_doctor")

        return HospitalToolsOutput(
            tool="get_opd_doctor",
            message=__get_doctor_wrapper(result),
            tool_state={"doctor_id": result.get("doctor_id") if result else None}
        )
"""
Tool implementations for the Healthcare Appointment Scheduling environment.

All tools are pure functions that operate on the hospital data.  The
environment calls these in response to agent actions, returning structured
results.
"""

from typing import Any, Dict, List, Optional

try:
    from healthcare_scheduling.server.data import (
        DEPARTMENTS,
        DOCTORS,
        get_clarification_response,
        map_symptoms_to_department,
    )
except ImportError:
    from server.data import (  # type: ignore
        DEPARTMENTS,
        DOCTORS,
        get_clarification_response,
        map_symptoms_to_department,
    )


# ---------------------------------------------------------------------------
# Tool 1 – get_departments
# ---------------------------------------------------------------------------

def get_departments() -> Dict[str, Any]:
    """
    Return the full list of available medical departments.

    Returns
    -------
    Dict with keys:
        departments (List[str]): available department names
        count (int):             total number of departments
    """
    return {
        "departments": DEPARTMENTS,
        "count": len(DEPARTMENTS),
        "hint": (
            "Use get_doctors(department) to see doctors in a specific department."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 2 – get_doctors
# ---------------------------------------------------------------------------

def get_doctors(department: str) -> Dict[str, Any]:
    """
    Return all doctors in a given department with their specializations.

    Parameters
    ----------
    department : str  (must match one of the keys in DOCTORS)

    Returns
    -------
    Dict with keys:
        department (str):          the queried department
        doctors (List[Dict]):      name, specialization, slot count
        error (str | None):        set if department is invalid
    """
    # Normalise input (case-insensitive match)
    matched_dept = None
    for dept in DOCTORS:
        if dept.lower() == department.lower():
            matched_dept = dept
            break

    if matched_dept is None:
        return {
            "error": f"Department '{department}' not found.",
            "valid_departments": DEPARTMENTS,
            "doctors": [],
        }

    doctors_summary = [
        {
            "name": doc["name"],
            "specialization": doc["specialization"],
            "available_slots_count": len(doc["available_slots"]),
        }
        for doc in DOCTORS[matched_dept]
    ]

    return {
        "department": matched_dept,
        "doctors": doctors_summary,
        "hint": (
            "Use check_availability(doctor) with an exact doctor name "
            "to see open time slots."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 3 – check_availability
# ---------------------------------------------------------------------------

def check_availability(doctor: str) -> Dict[str, Any]:
    """
    Return available appointment slots for a specific doctor.

    Parameters
    ----------
    doctor : str  (must match the 'name' field of a doctor record)

    Returns
    -------
    Dict with keys:
        doctor (str):               the queried doctor name
        department (str):           their department
        specialization (str):       their specialization
        available_slots (List[str]): slots the agent can book
        error (str | None):         set if doctor not found
    """
    for dept, doctors in DOCTORS.items():
        for doc in doctors:
            if doc["name"].lower() == doctor.lower():
                return {
                    "doctor": doc["name"],
                    "department": dept,
                    "specialization": doc["specialization"],
                    "available_slots": doc["available_slots"],
                    "hint": (
                        "Use book_appointment(doctor, slot) with an exact slot string."
                    ),
                }

    return {
        "error": f"Doctor '{doctor}' not found.",
        "hint": "Use get_doctors(department) to see valid doctor names.",
        "available_slots": [],
    }


# ---------------------------------------------------------------------------
# Tool 4 – book_appointment
# ---------------------------------------------------------------------------

def book_appointment(doctor: str, slot: str) -> Dict[str, Any]:
    """
    Attempt to book an appointment for the given doctor and slot.

    Parameters
    ----------
    doctor : str   exact doctor name
    slot   : str   exact slot string from check_availability

    Returns
    -------
    Dict with keys:
        success (bool):            True if booking confirmed
        confirmation_id (str):     booking reference (when success)
        doctor (str):              confirmed doctor name
        slot (str):                confirmed slot
        department (str):          the doctor's department
        error (str | None):        set if doctor/slot invalid
    """
    import uuid

    for dept, doctors in DOCTORS.items():
        for doc in doctors:
            if doc["name"].lower() == doctor.lower():
                if slot in doc["available_slots"]:
                    confirmation_id = f"APPT-{uuid.uuid4().hex[:8].upper()}"
                    return {
                        "success": True,
                        "confirmation_id": confirmation_id,
                        "doctor": doc["name"],
                        "slot": slot,
                        "department": dept,
                        "message": (
                            f"Appointment confirmed with {doc['name']} "
                            f"({dept}) on {slot}. "
                            f"Reference: {confirmation_id}"
                        ),
                    }
                else:
                    return {
                        "success": False,
                        "error": (
                            f"Slot '{slot}' is not available for {doc['name']}."
                        ),
                        "available_slots": doc["available_slots"],
                    }

    return {
        "success": False,
        "error": f"Doctor '{doctor}' not found.",
        "hint": "Use get_doctors(department) to see valid doctor names.",
    }


# ---------------------------------------------------------------------------
# Tool 5 – ask_user_clarification
# ---------------------------------------------------------------------------

def ask_user_clarification(question: str) -> Dict[str, Any]:
    """
    Ask the simulated user a clarifying question.

    The response is deterministic based on keywords in the question so that
    graders remain reproducible.

    Parameters
    ----------
    question : str   The clarifying question posed by the agent.

    Returns
    -------
    Dict with keys:
        question (str):   the original question
        user_response (str): the simulated user's answer
    """
    user_response = get_clarification_response(question)
    return {
        "question": question,
        "user_response": user_response,
        "hint": (
            "Based on the user's response, decide on the appropriate department."
        ),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

VALID_TOOLS = {
    "get_departments",
    "get_doctors",
    "check_availability",
    "book_appointment",
    "ask_user_clarification",
}


def dispatch_tool(tool: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route a tool name + parameters dict to the correct tool function.

    Returns a result dict. On invalid tool name, returns an error dict.
    """
    if tool == "get_departments":
        return get_departments()

    if tool == "get_doctors":
        department = parameters.get("department", "")
        if not department:
            return {"error": "Missing required parameter: department"}
        return get_doctors(department)

    if tool == "check_availability":
        doctor = parameters.get("doctor", "")
        if not doctor:
            return {"error": "Missing required parameter: doctor"}
        return check_availability(doctor)

    if tool == "book_appointment":
        doctor = parameters.get("doctor", "")
        slot = parameters.get("slot", "")
        if not doctor or not slot:
            return {"error": "Missing required parameters: doctor and/or slot"}
        return book_appointment(doctor, slot)

    if tool == "ask_user_clarification":
        question = parameters.get("question", "")
        if not question:
            return {"error": "Missing required parameter: question"}
        return ask_user_clarification(question)

    return {
        "error": f"Unknown tool: '{tool}'",
        "valid_tools": sorted(VALID_TOOLS),
    }

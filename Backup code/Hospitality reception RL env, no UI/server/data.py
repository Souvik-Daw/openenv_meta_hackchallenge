"""
Hardcoded healthcare data: departments, doctors, slots, and symptom mappings.

All data is accessed exclusively through tool functions — the agent is never
given this data directly, only through tool calls.
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Hospital data
# ---------------------------------------------------------------------------

DEPARTMENTS: List[str] = [
    "Cardiology",
    "Dermatology",
    "Neurology",
    "Orthopedics",
]

DOCTORS: Dict[str, List[Dict]] = {
    "Cardiology": [
        {
            "name": "Dr. Sarah Smith",
            "department": "Cardiology",
            "specialization": "General cardiology, chest pain, heart failure",
            "available_slots": [
                "2024-01-15 09:00 AM",
                "2024-01-15 11:00 AM",
                "2024-01-16 02:00 PM",
            ],
        },
        {
            "name": "Dr. James Adams",
            "department": "Cardiology",
            "specialization": "Arrhythmia, palpitations, ECG interpretation",
            "available_slots": [
                "2024-01-15 10:00 AM",
                "2024-01-16 09:00 AM",
                "2024-01-17 03:00 PM",
            ],
        },
    ],
    "Dermatology": [
        {
            "name": "Dr. Priya Patel",
            "department": "Dermatology",
            "specialization": "Skin rashes, eczema, psoriasis, chronic skin conditions",
            "available_slots": [
                "2024-01-15 08:00 AM",
                "2024-01-15 01:00 PM",
                "2024-01-16 10:00 AM",
            ],
        },
        {
            "name": "Dr. Kevin Lee",
            "department": "Dermatology",
            "specialization": "Acne, skin cancer screening, mole removal",
            "available_slots": [
                "2024-01-15 03:00 PM",
                "2024-01-16 11:00 AM",
                "2024-01-17 09:00 AM",
            ],
        },
    ],
    "Neurology": [
        {
            "name": "Dr. Elena Rossi",
            "department": "Neurology",
            "specialization": "Migraines, headaches, epilepsy, neurological disorders",
            "available_slots": [
                "2024-01-15 10:00 AM",
                "2024-01-16 02:00 PM",
                "2024-01-17 11:00 AM",
            ],
        },
        {
            "name": "Dr. Michael Chen",
            "department": "Neurology",
            "specialization": "Stroke, memory disorders, Parkinson's, neuropathy",
            "available_slots": [
                "2024-01-15 02:00 PM",
                "2024-01-16 09:00 AM",
                "2024-01-17 04:00 PM",
            ],
        },
    ],
    "Orthopedics": [
        {
            "name": "Dr. Thomas Grant",
            "department": "Orthopedics",
            "specialization": "Sports injuries, knee pain, fractures, joint replacement",
            "available_slots": [
                "2024-01-15 09:00 AM",
                "2024-01-15 03:00 PM",
                "2024-01-16 01:00 PM",
            ],
        },
        {
            "name": "Dr. Aisha Okafor",
            "department": "Orthopedics",
            "specialization": "Spine disorders, back pain, scoliosis, disc herniation",
            "available_slots": [
                "2024-01-15 11:00 AM",
                "2024-01-16 03:00 PM",
                "2024-01-17 10:00 AM",
            ],
        },
    ],
}

# ---------------------------------------------------------------------------
# Symptom → Department mapping (rule-based)
# ---------------------------------------------------------------------------

SYMPTOM_KEYWORDS: Dict[str, List[str]] = {
    "Cardiology": [
        "chest", "heart", "palpitation", "arrhythmia", "cardiac",
        "shortness of breath", "breathless", "angina", "pressure in chest",
        "chest pain", "heart attack", "rapid heartbeat", "irregular heartbeat",
    ],
    "Dermatology": [
        "skin", "rash", "itch", "eczema", "psoriasis", "acne",
        "mole", "lesion", "blister", "hive", "dermatitis",
        "dry skin", "flaky", "scaling", "redness on skin",
    ],
    "Neurology": [
        "headache", "migraine", "seizure", "epilepsy", "memory",
        "dizziness", "vertigo", "numbness", "tingling", "tremor",
        "stroke", "paralysis", "confusion", "brain", "neuro",
    ],
    "Orthopedics": [
        "knee", "bone", "joint", "fracture", "back pain", "spine",
        "muscle pain", "sports injury", "ankle", "shoulder", "hip",
        "arthritis", "ligament", "tendon", "wrist", "elbow",
    ],
}

# ---------------------------------------------------------------------------
# Simulated user clarifications (deterministic for task reproducibility)
# ---------------------------------------------------------------------------

CLARIFICATION_RESPONSES: Dict[str, str] = {
    "default": (
        "The pain is mainly in my chest area and gets worse when I walk fast. "
        "I also feel short of breath sometimes."
    ),
    "location": (
        "It's my chest — a tight, squeezing sensation, especially in the morning."
    ),
    "duration": "It started about three days ago and has been getting worse.",
    "severity": "I'd rate the pain around 6 out of 10. It's uncomfortable but I can still walk.",
    "skin": (
        "The rash is on my forearm and spreads across my lower back. "
        "It's red, itchy, and has been there for about two weeks."
    ),
    "pain": (
        "The pain is sharp in my left knee. It started after I twisted it "
        "during a football game last weekend."
    ),
}


def get_clarification_response(question: str) -> str:
    """Return a deterministic simulated user clarification based on keywords."""
    question_lower = question.lower()
    if any(w in question_lower for w in ["where", "location", "which part", "area"]):
        return CLARIFICATION_RESPONSES["location"]
    if any(w in question_lower for w in ["long", "duration", "when", "started"]):
        return CLARIFICATION_RESPONSES["duration"]
    if any(w in question_lower for w in ["severe", "severity", "scale", "rate", "how bad"]):
        return CLARIFICATION_RESPONSES["severity"]
    if any(w in question_lower for w in ["skin", "rash", "itch"]):
        return CLARIFICATION_RESPONSES["skin"]
    if any(w in question_lower for w in ["knee", "joint", "bone", "muscle"]):
        return CLARIFICATION_RESPONSES["pain"]
    return CLARIFICATION_RESPONSES["default"]


def map_symptoms_to_department(user_request: str) -> Optional[str]:
    """
    Rule-based symptom → department mapper.

    Returns the best-matching department name or None if ambiguous.
    Scores each department by keyword hit count.
    """
    request_lower = user_request.lower()
    scores: Dict[str, int] = {dept: 0 for dept in DEPARTMENTS}

    for dept, keywords in SYMPTOM_KEYWORDS.items():
        for kw in keywords:
            if kw in request_lower:
                scores[dept] += 1

    best_dept = max(scores, key=lambda d: scores[d])
    if scores[best_dept] == 0:
        return None  # Truly ambiguous; agent should ask clarification
    return best_dept

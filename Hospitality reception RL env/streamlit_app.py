"""
streamlit_app.py — Interactive UI for the Healthcare Appointment Scheduling RL Agent.

Run with:
    streamlit run streamlit_app.py

Requirements (add to your env):
    pip install streamlit
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------------
# Path setup — identical to run_baseline.py so local imports resolve
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from server.environment import HealthcareAppointmentEnvironment
from models import AppointmentObservation, AppointmentAction
from tasks.easy   import get_task_config as easy_config,   USER_REQUEST as EASY_REQ,   CORRECT_DEPARTMENT as EASY_DEPT,   CORRECT_DOCTOR as EASY_DOC
from tasks.medium import get_task_config as medium_config, USER_REQUEST as MED_REQ,    CORRECT_DEPARTMENT as MED_DEPT,    CORRECT_DOCTOR as MED_DOC
from tasks.hard   import get_task_config as hard_config,   USER_REQUEST as HARD_REQ,   CORRECT_DEPARTMENT as HARD_DEPT,   CORRECT_DOCTOR as HARD_DOC
from tasks.rebook import get_task_config as rebook_config, USER_REQUEST as REB_REQ,    CORRECT_DEPARTMENT as REB_DEPT,    CORRECT_DOCTOR as REB_DOC
from tasks.graders import grade_full_breakdown
from server.data import map_symptoms_to_department

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Healthcare Scheduling Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — high-contrast dark theme (fully readable)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ═══ BACKGROUND — deep charcoal ═══ */
    .stApp {
        background: #0e1117;
        min-height: 100vh;
    }

    /* ═══ GLOBAL TEXT — bright white everywhere ═══ */
    .stApp p, .stApp span, .stApp div,
    .stApp li, .stApp small {
        color: #e2e8f0;
    }
    .stMarkdown p, .stMarkdown li {
        color: #d1d5db !important;
        font-size: 0.95rem;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    h4, h5, h6 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    strong, b { color: #f8fafc !important; }
    em, i { color: #94a3b8 !important; }
    code {
        color: #c4b5fd !important;
        background: rgba(167,139,250,0.2) !important;
        border-radius: 4px;
        padding: 0 5px;
    }
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 2px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #a5b4fc !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] code {
        background: #1e293b !important;
        color: #7dd3fc !important;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.82rem;
        display: inline-block;
    }

    /* ═══ HERO ═══ */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.25rem;
        line-height: 1.25;
    }
    .hero-subtitle {
        text-align: center;
        color: #6b7280 !important;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* ═══ MAIN CARD ═══ */
    .main-card {
        background: #1a2236;
        border: 1.5px solid #2d3748;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* ═══ TEXT INPUTS ═══ */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #1a2236 !important;
        border: 1.5px solid #3b4f6b !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        caret-color: #38bdf8 !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #4b5563 !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.25) !important;
    }

    /* ═══ BUTTONS ═══ */
    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.03em !important;
        transition: filter 0.2s, transform 0.15s !important;
    }
    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        filter: brightness(1.2) !important;
        transform: translateY(-2px) !important;
    }

    /* ═══ STEP CARDS ═══ */
    .step-card {
        background: #1a2236;
        border-left: 4px solid #818cf8;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.65rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .step-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 20px rgba(56,189,248,0.1);
    }
    .step-card.reward-positive { border-left-color: #22c55e; }
    .step-card.reward-negative { border-left-color: #ef4444; }
    .step-card.reward-neutral  { border-left-color: #818cf8; }

    .step-number {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7dd3fc;
        margin-bottom: 0.25rem;
    }
    .step-tool {
        font-size: 1rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .step-meta {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.2rem;
    }

    /* ═══ METRIC TILES ═══ */
    .metric-tile {
        background: #1a2236;
        border: 1.5px solid #2d3e55;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ═══ BADGES ═══ */
    .badge {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .badge-green  { background: #052e16; color: #4ade80; border: 1.5px solid #22c55e; }
    .badge-red    { background: #450a0a; color: #fca5a5; border: 1.5px solid #ef4444; }
    .badge-purple { background: #1e1b4b; color: #c4b5fd; border: 1.5px solid #818cf8; }
    .badge-blue   { background: #0c1a2e; color: #7dd3fc; border: 1.5px solid #38bdf8; }

    /* ═══ EXPANDER ═══ */
    .streamlit-expanderHeader {
        background: #1a2236 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* ═══ LOG BOX ═══ */
    .log-box {
        background: #080d14;
        border: 1px solid #1b2f46;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.76rem;
        color: #4ade80;
        max-height: 280px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-all;
        line-height: 1.6;
    }

    /* ═══ RESULT BANNERS ═══ */
    .result-banner {
        background: linear-gradient(135deg, #052e16, #0c2740);
        border: 1.5px solid #22c55e;
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1rem;
    }
    .result-banner h3 { color: #f1f5f9 !important; }
    .result-banner-fail {
        background: linear-gradient(135deg, #2d0000, #1a0808);
        border: 1.5px solid #ef4444;
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1rem;
    }
    .result-banner-fail h3 { color: #f1f5f9 !important; }

    /* ═══ SCORE BARS ═══ */
    .score-bar-bg {
        background: #1e293b;
        border-radius: 999px;
        height: 9px;
        margin-top: 6px;
    }
    .score-bar-fill {
        height: 9px;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #38bdf8, #34d399);
        transition: width 0.6s ease;
    }

    /* ═══ MISC ═══ */
    hr { border-color: #1e293b !important; }
    .stSlider > div > div > div > div { background: #38bdf8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "steps": [],          # list of dicts: {step, tool, params, result, reward, message, reasoning}
        "logs": [],           # list of plain strings
        "final_result": None, # grade_full_breakdown dict after episode
        "env_state": None,    # AppointmentState snapshot after episode
        "running": False,
        "episode_done": False,
        "total_reward": 0.0,
        "user_request_used": "",
        "agent_mode": "groq",
        "groq_key_ok": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_ICONS: Dict[str, str] = {
    "get_departments":      "🏛️",
    "get_doctors":          "👨‍⚕️",
    "check_availability":   "📅",
    "book_appointment":     "✅",
    "ask_user_clarification": "❓",
}

TASK_REGISTRY = [
    ("easy",   EASY_REQ,  EASY_DEPT,  EASY_DOC,  easy_config),
    ("medium", MED_REQ,   MED_DEPT,   MED_DOC,   medium_config),
    ("hard",   HARD_REQ,  HARD_DEPT,  HARD_DOC,  hard_config),
    ("rebook", REB_REQ,   REB_DEPT,   REB_DOC,   rebook_config),
]


def _find_best_task_config(user_request: str):
    """Return (correct_dept, correct_doctor, config_fn) for the given request.

    Routing priority:
    1. Exact sentence match → use predefined task config (grader-aligned).
    2. Symptom keyword engine (map_symptoms_to_department) → always correct
       for any free-text input (e.g. "back pain" → Orthopedics, not Cardiology).
    3. Ambiguous (no keyword match) → correct_department = None, agent should
       call ask_user_clarification.
    """
    from server.data import DOCTORS

    req_lower = user_request.strip().lower()

    # ── 1. Exact sentence match against predefined tasks ──────────────────────
    for _, req, dept, doc, cfg_fn in TASK_REGISTRY:
        if req_lower == req.lower():
            return dept, doc, cfg_fn

    # ── 2. Primary router: symptom keyword engine ─────────────────────────────
    #    map_symptoms_to_department has the proper SYMPTOM_KEYWORDS table.
    auto_dept = map_symptoms_to_department(user_request)

    from server.data import DOCTORS
    explicit_doctor = None
    explicit_doc_dept = None
    req_lower = user_request.lower()
    for dept, doc_list in DOCTORS.items():
        for doc_obj in doc_list:
            d_name = doc_obj["name"].lower()
            d_last = doc_obj["name"].split()[-1].lower()
            if d_name in req_lower or d_last in req_lower:
                explicit_doctor = doc_obj["name"]
                explicit_doc_dept = dept
                break
        if explicit_doctor:
            break

    if explicit_doctor:
        auto_dept = explicit_doc_dept
        auto_doc = explicit_doctor
    elif auto_dept and auto_dept in DOCTORS:
        from server.data import department_to_doctor
        auto_doc = department_to_doctor(user_request, auto_dept)
    else:
        # Truly ambiguous — no keyword hit; agent should clarify
        auto_dept = None
        auto_doc = None

    # ── 3. Build a dynamic task config ───────────────────────────────────────
    _dept = auto_dept   # capture for closure
    _doc  = auto_doc

    is_rebook_intent = "rebook" in req_lower or explicit_doctor is not None

    def dynamic_config():
        return {
            "task_id": "custom",
            "difficulty": "custom",
            "user_request": user_request,
            "correct_department": _dept,
            "correct_doctor": _doc,
            "expected_min_steps": 2 if is_rebook_intent else 4,
            "requires_clarification": _dept is None and not is_rebook_intent,
            "is_rebook": is_rebook_intent,
        }

    return _dept, _doc, dynamic_config


def _log(msg: str) -> None:
    st.session_state.logs.append(msg)


def _reward_class(reward: float) -> str:
    if reward > 0:
        return "reward-positive"
    if reward < 0:
        return "reward-negative"
    return "reward-neutral"


def _render_step_card(step_data: Dict[str, Any], idx: int) -> None:
    tool = step_data["tool"]
    params = step_data["params"]
    result = step_data["result"]
    reward = step_data["reward"]
    message = step_data["message"]
    reasoning = step_data.get("reasoning", "")

    rc = _reward_class(reward)
    icon = TOOL_ICONS.get(tool, "🔧")
    reward_color = "#34d399" if reward > 0 else ("#f87171" if reward < 0 else "#a78bfa")

    card_html = f"""
    <div class="step-card {rc}">
        <div class="step-number">Step {idx}</div>
        <div class="step-tool">{icon} {tool}</div>
        <div class="step-meta">
            Params: <code style="color:#c4b5fd">{json.dumps(params)}</code>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander(f"📋 Step {idx} details — reward: {reward:+.4f}", expanded=False):
        if reasoning:
            st.markdown(f"**🧠 Agent Reasoning:** _{reasoning}_")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Tool Output (Observation)**")
            st.json(result)
        with col_b:
            st.markdown("**Environment Feedback**")
            st.info(message or "—")
            rew_str = f"+{reward:.4f}" if reward > 0 else f"{reward:.4f}"
            badge_cls = "badge-green" if reward > 0 else ("badge-red" if reward < 0 else "badge-purple")
            st.markdown(
                f'Reward: <span class="badge {badge_cls}">{rew_str}</span>',
                unsafe_allow_html=True,
            )


def _render_final_result() -> None:
    result = st.session_state.final_result
    if result is None:
        return

    success = result.get("booking_successful", False)
    score = result.get("final_score", 0.0)
    steps = result.get("steps_taken", 0)

    banner_cls = "result-banner" if success else "result-banner-fail"
    st.markdown(
        f'<div class="{banner_cls}">'
        f'<h3 style="margin:0;color:#e2e8f0">{"✅ Booking Successful!" if success else "❌ Episode Ended — No Booking"}</h3>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Booking summary
    st.markdown("### 📋 Booking Summary")
    env_state = st.session_state.env_state

    col1, col2, col3 = st.columns(3)
    with col1:
        dept = env_state.identified_department or "—"
        dept_correct = result.get("department_correct", False)
        badge = '<span class="badge badge-green">✓ Correct</span>' if dept_correct else '<span class="badge badge-red">✗ Wrong</span>'
        st.markdown(
            f'<div class="metric-tile"><div class="metric-label">Department</div><div style="font-size:1.2rem;color:#e2e8f0;font-weight:600;">{dept}</div>{badge}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        doc = env_state.selected_doctor or "—"
        doc_correct = result.get("doctor_correct", False)
        badge = '<span class="badge badge-green">✓ Correct</span>' if doc_correct else '<span class="badge badge-red">✗ Wrong</span>'
        st.markdown(
            f'<div class="metric-tile"><div class="metric-label">Doctor</div><div style="font-size:1.1rem;color:#e2e8f0;font-weight:600;">{doc}</div>{badge}</div>',
            unsafe_allow_html=True,
        )
    with col3:
        slot = env_state.selected_slot or "—"
        st.markdown(
            f'<div class="metric-tile"><div class="metric-label">Time Slot</div><div style="font-size:0.95rem;color:#e2e8f0;font-weight:600;">{slot}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-tile"><div class="metric-value">{score:g}/4</div><div class="metric-label">Final Score</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-tile"><div class="metric-value">{steps}</div><div class="metric-label">Steps Taken</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        env_reward = result.get("cumulative_env_reward", 0.0)
        st.markdown(
            f'<div class="metric-tile"><div class="metric-value">{env_reward:.3f}</div><div class="metric-label">Total Reward</div></div>',
            unsafe_allow_html=True,
        )

    is_rebook = result.get("is_rebook", False)
    # Score breakdown bar
    st.markdown("#### Score Breakdown")
    components = []
    if not is_rebook:
        components.extend([
            ("Get Depts",  result.get("get_departments_score", 0.0), 1.0, "#93c5fd"),
            ("Department", result.get("department_score", 0.0), 1.0, "#a78bfa"),
        ])
    components.extend([
        ("Doctor",     result.get("doctor_score",     0.0), 1.0, "#60a5fa"),
        ("Booking",    result.get("booking_score",    0.0), 1.0, "#34d399"),
    ])
    for label, earned, max_val, color in components:
        pct = earned / max_val if max_val > 0 else 0
        fill_w = int(pct * 100)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
            f"  <div style='width:90px;font-size:0.8rem;color:rgba(255,255,255,0.6)'>{label}</div>"
            f"  <div style='flex:1;background:rgba(255,255,255,0.08);border-radius:999px;height:8px'>"
            f"    <div style='width:{fill_w}%;background:{color};height:8px;border-radius:999px'></div>"
            f"  </div>"
            f"  <div style='width:60px;text-align:right;font-size:0.78rem;color:#e2e8f0'>{earned:.2f}/{max_val:.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    clr_penalty = result.get("clarification_penalty", 0.0)
    if clr_penalty < 0:
        st.markdown(
            f'<span class="badge badge-red">Clarification Penalty: {clr_penalty:.2f}</span>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def init_agent_episode(
    user_request: str, mode: str, groq_api_key: Optional[str],
    correct_dept: Optional[str], correct_doctor: Optional[str], config_fn
) -> None:
    """Initialize state tracking for an interactive episode."""
    st.session_state.steps = []
    st.session_state.logs = []
    st.session_state.final_result = None
    st.session_state.env_state = None
    st.session_state.total_reward = 0.0
    st.session_state.episode_done = False
    st.session_state.user_request_used = user_request

    _log(f"[INIT] Episode started for: \"{user_request}\"")
    
    env = HealthcareAppointmentEnvironment()
    obs = env.reset(user_request=user_request, correct_department=correct_dept, correct_doctor=correct_doctor)
    _log(f"[ENV ] {obs.message}")

    if mode == "groq":
        from agent.groq_agent import GroqAgent
        agent = GroqAgent(api_key=groq_api_key, verbose=False)
        agent.reset_conversation()
    else:
        agent = None
        from run_baseline import ORACLE_ACTIONS
        difficulty = config_fn().get("difficulty", "easy")
        st.session_state.oracle_actions = ORACLE_ACTIONS.get(difficulty, ORACLE_ACTIONS["easy"])
        st.session_state.oracle_idx = 0

    st.session_state.env = env
    st.session_state.obs = obs
    st.session_state.agent = agent
    st.session_state.step_num = 0
    st.session_state.config_fn = config_fn
    st.session_state.agent_running = True
    st.session_state.waiting_for_clarification = False
    st.session_state.mode = mode


def step_agent_machine(status_placeholder, steps_placeholder, log_placeholder, step_delay: float) -> None:
    """Runs a single step of the environment loop, persisting via st.rerun(). 
       If clarification is needed, it halts and shows an interactive form."""
    env = st.session_state.env
    obs = st.session_state.obs
    agent = st.session_state.agent
    mode = st.session_state.mode
    step_num = st.session_state.step_num

    # Re-render UI elements every frame
    with steps_placeholder.container():
        _render_steps_section()
    _refresh_log(log_placeholder)

    # 1. Check for termination
    if obs.done or step_num >= 10:
        st.session_state.agent_running = False
        try:
            breakdown = grade_full_breakdown(env.state, st.session_state.config_fn())
        except Exception as exc:
            _log(f"[ERROR] Grader failed: {exc}")
            breakdown = {
                "task_id": "unknown", "difficulty": "custom",
                "final_score": 0.0, "cumulative_env_reward": round(env.state.cumulative_reward, 4)
            }
        st.session_state.final_result = breakdown
        st.session_state.env_state = env.state
        st.session_state.episode_done = True
        status_placeholder.empty()
        st.rerun()

    # 2. Check if waiting for interactive user response
    if st.session_state.waiting_for_clarification:
        action = st.session_state.pending_action
        question = action.parameters.get("question", "Could you provide more details?")
        
        status_placeholder.warning("⚠️ The agent has paused to ask you a question.", icon="⏳")
        st.markdown(
            f"<div class='result-banner' style='border-color:#38bdf8; background:rgba(56,189,248,0.1)'>"
            f"<h3 style='margin-top:0'>🤖 Agent asks:</h3><p style='font-size:1.1rem'>{question}</p>"
            f"</div>", 
            unsafe_allow_html=True
        )
        
        with st.form("clarification_form", clear_on_submit=True):
            user_ans = st.text_input("Your Response (type here...)", key="user_ans_input")
            submitted = st.form_submit_button("Reply to Agent")
            if submitted:
                if not user_ans.strip():
                    st.error("Please type a response.")
                else:
                    # Monkeypatch the answer directly into the tool
                    from unittest.mock import patch
                    with patch('server.tools.get_clarification_response', return_value=user_ans.strip()):
                        obs = env.step(action)
                    
                    st.session_state.step_num += 1
                    reasoning = action.metadata.get("reasoning", "")
                    st.session_state.steps.append({
                        "step": st.session_state.step_num,
                        "tool": action.tool,
                        "params": action.parameters,
                        "result": {"question": question, "user_response": user_ans.strip()},
                        "reward": obs.reward,
                        "message": obs.message,
                        "reasoning": reasoning,
                    })
                    st.session_state.total_reward += obs.reward
                    st.session_state.obs = obs
                    st.session_state.waiting_for_clarification = False
                    st.rerun()
        st.stop()  # Halt Streamlit execution until the form is submitted

    # 3. Normal step logic
    with status_placeholder.container():
        st.markdown(
            f'<div style="color:#a78bfa;font-weight:600;font-size:0.95rem;">'
            f'⚡ Agent is thinking... (Step {step_num + 1}/10)'
            f'</div>',
            unsafe_allow_html=True,
        )

    reasoning = ""
    try:
        if mode == "groq":
            action = agent.decide_action(obs)
            reasoning = action.metadata.get("reasoning", "")
        else:
            if st.session_state.oracle_idx < len(st.session_state.oracle_actions):
                action = st.session_state.oracle_actions[st.session_state.oracle_idx]
                st.session_state.oracle_idx += 1
            else:
                _log(f"[WARN] Oracle ran out of actions")
                st.session_state.agent_running = False
                st.rerun()
    except Exception as exc:
        _log(f"[ERROR] Agent failed at step {step_num + 1}: {exc}")
        status_placeholder.error(f"Agent error: {exc}")
        st.session_state.agent_running = False
        st.stop()

    _log(f"[STEP {step_num + 1}] Tool: {action.tool} | Params: {json.dumps(action.parameters)}")

    # Intercept clarification tools before stepping
    if action.tool == "ask_user_clarification":
        st.session_state.waiting_for_clarification = True
        st.session_state.pending_action = action
        st.rerun()

    # Step environment
    try:
        obs = env.step(action)
    except Exception as exc:
        _log(f"[ERROR] env.step failed: {exc}")
        status_placeholder.error(f"Environment error: {exc}")
        st.session_state.agent_running = False
        st.stop()

    st.session_state.step_num += 1
    st.session_state.steps.append({
        "step": st.session_state.step_num,
        "tool": action.tool,
        "params": action.parameters,
        "result": obs.tool_result,
        "reward": obs.reward,
        "message": obs.message,
        "reasoning": reasoning,
    })
    st.session_state.total_reward += obs.reward
    st.session_state.obs = obs

    if step_delay > 0:
        time.sleep(step_delay)
    
    st.rerun()


def _render_steps_section() -> None:
    """Called inside a placeholder — renders all step cards so far."""
    steps = st.session_state.steps
    if not steps:
        return
    for s in steps:
        _render_step_card(s, s["step"])


def _refresh_log(log_placeholder) -> None:
    logs = st.session_state.logs
    text = "\n".join(logs[-60:])  # last 60 lines
    with log_placeholder.container():
        st.markdown(
            f'<div class="log-box">{text}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='color:#a78bfa;margin-bottom:0.5rem'>⚙️ Configuration</h2>",
        unsafe_allow_html=True,
    )

    st.markdown("**Agent Mode**")
    agent_mode = st.radio(
        "agent_mode_radio",
        options=["groq", "oracle"],
        format_func=lambda x: "🤖 Groq LLM Agent" if x == "groq" else "🔬 Oracle (Deterministic)",
        label_visibility="collapsed",
    )
    st.session_state.agent_mode = agent_mode

    groq_api_key = None
    if agent_mode == "groq":
        st.markdown("---")
        st.markdown("**Groq API Key**")

        # Try loading from .env first
        load_dotenv(find_dotenv())
        env_key = os.getenv("GROQ_API_KEY", "")

        groq_key_input = st.text_input(
            "groq_key",
            value=env_key,
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed",
        )
        groq_api_key = groq_key_input or env_key or None

        if groq_api_key:
            st.success("Key loaded ✓", icon="🔑")
        else:
            st.warning("No key — switch to Oracle mode", icon="⚠️")

        st.markdown("**Groq Model**")
        groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        st.code(groq_model, language=None)

    st.markdown("---")
    st.markdown("**Step Delay (seconds)**")
    step_delay = st.slider("step_delay", min_value=0.0, max_value=2.0, value=0.7, step=0.1, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:rgba(255,255,255,0.3);text-align:center'>"
        "Healthcare Scheduling RL · OpenEnv"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

# Hero
st.markdown(
    "<div class='hero-title'>🏥 AI Healthcare Scheduling Agent</div>"
    "<div class='hero-subtitle'>RL-powered doctor booking system · OpenEnv Hackathon Demo</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("### 💬 Describe your health issue")

# Prefill from sidebar button
prefill = st.session_state.pop("prefill_request", None)
default_text = prefill if prefill else ""

user_input = st.text_input(
    "user_request_input",
    value=default_text,
    placeholder='e.g. "I have chest pain since morning" or "I have a skin rash for 2 weeks"',
    label_visibility="collapsed",
)

col_btn1, col_btn2, col_spacer = st.columns([1, 1, 4])
with col_btn1:
    run_clicked = st.button("🚀 Run Agent", use_container_width=True)
with col_btn2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if clear_clicked:
    st.session_state.steps = []
    st.session_state.logs = []
    st.session_state.final_result = None
    st.session_state.env_state = None
    st.session_state.total_reward = 0.0
    st.session_state.episode_done = False
    st.session_state.user_request_used = ""
    st.rerun()

# ---------------------------------------------------------------------------
# Validate + run
# ---------------------------------------------------------------------------
if run_clicked:
    if not user_input.strip():
        st.error("⚠️ Please describe your health issue before running the agent.")
        st.stop()

    if agent_mode == "groq" and not groq_api_key:
        st.error("⚠️ No Groq API key found. Please enter your key in the sidebar or set GROQ_API_KEY in your .env file.")
        st.stop()

    # Resolve task config
    correct_dept, correct_doctor, config_fn = _find_best_task_config(user_input.strip())

    st.markdown(
        f"<div style='font-size:0.82rem;color:rgba(255,255,255,0.5);margin-bottom:0.5rem'>"
        f"🎯 Routing to: <b style='color:#a78bfa'>{correct_dept}</b> → "
        f"<b style='color:#60a5fa'>{correct_doctor}</b>"
        f" &nbsp;|&nbsp; Config: <b style='color:#34d399'>{config_fn().get('difficulty','custom').upper()}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    init_agent_episode(
        user_request=user_input.strip(),
        mode=agent_mode,
        groq_api_key=groq_api_key,
        correct_dept=correct_dept,
        correct_doctor=correct_doctor,
        config_fn=config_fn,
    )
    st.rerun()

# ---------------------------------------------------------------------------
# Continuous Event Loop for Agent Execution
# ---------------------------------------------------------------------------
if st.session_state.get("agent_running"):
    st.markdown("---")
    
    # Render placeholders
    status_ph   = st.empty()
    st.markdown("### 🔍 Agent Trajectory")
    
    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.markdown("#### Step-by-Step Actions")
        steps_ph = st.empty()
    with right_col:
        st.markdown("#### 📟 Live Log")
        log_ph = st.empty()

    step_agent_machine(status_ph, steps_ph, log_ph, step_delay)

# ---------------------------------------------------------------------------
# Persistent display (after run)
# ---------------------------------------------------------------------------
if st.session_state.steps or st.session_state.episode_done:

    if st.session_state.user_request_used:
        st.markdown(
            f"<div style='font-size:0.9rem;color:rgba(255,255,255,0.55);margin-bottom:0.8rem'>"
            f'User request: <em>"{st.session_state.user_request_used}"</em>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ----- Step-by-step trace -----
    if st.session_state.steps:
        st.markdown("---")
        st.markdown("### 🔍 Agent Trajectory")

        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown("#### Step-by-Step Actions")
            for s in st.session_state.steps:
                _render_step_card(s, s["step"])

        with right_col:
            st.markdown("#### 📟 Live Log")
            logs = st.session_state.logs
            log_text = "\n".join(logs)
            st.markdown(
                f'<div class="log-box">{log_text}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Running Reward")
            running_reward = 0.0
            for i, s in enumerate(st.session_state.steps):
                running_reward += s["reward"]
                reward_color = "#34d399" if running_reward >= 0 else "#f87171"
                st.markdown(
                    f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.4)'>After step {i+1}: "
                    f"<span style='color:{reward_color};font-weight:600'>{running_reward:+.4f}</span></div>",
                    unsafe_allow_html=True,
                )

    # ----- Final result -----
    if st.session_state.episode_done and st.session_state.final_result is not None:
        st.markdown("---")
        st.markdown("### 🏁 Final Result")
        _render_final_result()

        st.markdown("---")

        # ---- Download / JSON export ----
        dl_col1, dl_col2 = st.columns([1, 1])

        trajectory_export = {
            "user_request": st.session_state.user_request_used,
            "agent_mode": st.session_state.agent_mode,
            "steps": [
                {
                    "step": s["step"],
                    "tool": s["tool"],
                    "parameters": s["params"],
                    "tool_result": s["result"],
                    "reward": s["reward"],
                    "message": s["message"],
                    "reasoning": s.get("reasoning", ""),
                }
                for s in st.session_state.steps
            ],
            "grader_result": st.session_state.final_result,
            "total_steps": len(st.session_state.steps),
            "total_reward": round(st.session_state.total_reward, 4),
        }
        trajectory_json = json.dumps(trajectory_export, indent=2)

        with dl_col1:
            with st.expander("🗂️ Show Full Trajectory JSON"):
                st.code(trajectory_json, language="json")

        with dl_col2:
            st.download_button(
                label="⬇️ Download Logs as JSON",
                data=trajectory_json,
                file_name="agent_trajectory.json",
                mime="application/json",
                use_container_width=True,
            )

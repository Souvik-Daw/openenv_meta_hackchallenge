# openenv_meta_hackchallenge

Meta Pytorch AI Hackathon

## RL env project ideas

Sample of the RL env project ideas (Done)

## simple_agent

Sample of the simple agent project (Done)

## word_game

RL env project (Working locally but not working in HF space)

Running locally:
python -m uvicorn word_game.server.app:app --host 0.0.0.0 --port 8000 --reload
python -m word_game.runnable

Push to HF space:
openenv push --repo-id Fergus2000/word_game

## openspiel_env hf_test

openspiel folder structure modify to work in HF space

## Hospitality reception RL env

Hospitality reception env (Done)
Enhancement possible :

1. MCP server
2. HF space, fast api and docker
3. Database to replace static data

https://fergus2000-hospital-env.hf.space/health

python run_baseline.py
python run_baseline.py --task hard

docker build -t healthcare-scheduling:latest .
docker run --rm -p 8000:8000 healthcare-scheduling:latest

Output below

TASK [EASY]
User request: "I have chest pain"
Expected: Cardiology -> Dr. Sarah Smith

---

    [LLM raw] {

"tool": "get_departments",
"parameters": {},
"reasoning": "Begin by getting a list of available medical departments to narrow down the options."
}...
Step 1: get_departments({})
→ reward=0.0000 | Retrieved department list. Now call get_doctors(department).
[LLM raw] {
"tool": "get_doctors",
"parameters": {"department": "Cardiology"},
"reasoning": "Match the symptom 'chest pain' to the Cardiology department, which deals with heart-related issues."
}...
Step 2: get_doctors({"department": "Cardiology"})
→ reward=0.2000 | Correct department 'Cardiology' identified! (+0.2)
[LLM raw] {
"tool": "check_availability",
"parameters": {"doctor": "Dr. Sarah Smith"},
"reasoning": "Choose Dr. Sarah Smith as the specialist for chest pain and check her available time slots."
}...
Step 3: check_availability({"doctor": "Dr. Sarah Smith"})
→ reward=0.1000 | Checking availability for the right doctor (Dr. Sarah Smith). (+0.1)
[LLM raw] {
"tool": "book_appointment",
"parameters": {"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"},
"reasoning": "Book an appointment with Dr. Sarah Smith on January 15th at 9:00 AM, as it...
Step 4: book_appointment({"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"})
→ reward=0.7000 | Perfect booking! Correct department and doctor. Efficiency bonus applied! (+0.7

[OK] Score: 1.000
Department correct: True (+0.25)
Doctor correct: True (+0.30)
Booking successful: True (+0.35)
Efficiency bonus: +0.10
Clarification penalty: 0.00
Steps taken: 4
Env cumulative reward:1.0000

TASK [MEDIUM]
User request: "I have a skin rash for 2 weeks"
Expected: Dermatology -> Dr. Priya Patel

---

    [LLM raw] {

"tool": "get_departments",
"parameters": {},
"reasoning": "Begin by getting a list of available medical departments to narrow down the options"
}...
Step 1: get_departments({})
→ reward=0.0000 | Retrieved department list. Now call get_doctors(department).
[LLM raw] {
"tool": "get_doctors",
"parameters": {"department": "Dermatology"},
"reasoning": "Dermatology is the most relevant department for a skin rash, so let's see the doctors available in that depart...
Step 2: get_doctors({"department": "Dermatology"})
→ reward=0.2000 | Correct department 'Dermatology' identified! (+0.2)
[LLM raw] {
"tool": "check_availability",
"parameters": {"doctor": "Dr. Priya Patel"},
"reasoning": "Dr. Priya Patel's specialization matches the user's symptom, so let's check her availability first"
}...
Step 3: check_availability({"doctor": "Dr. Priya Patel"})
→ reward=0.1000 | Checking availability for the right doctor (Dr. Priya Patel). (+0.1)
[LLM raw] {
"tool": "book_appointment",
"parameters": {"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"},
"reasoning": "Dr. Priya Patel has an available slot on 2024-01-15 08:00 AM, so let's boo...
Step 4: book_appointment({"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"})
→ reward=0.7000 | Perfect booking! Correct department and doctor. Efficiency bonus applied! (+0.7

[OK] Score: 1.000
Department correct: True (+0.25)
Doctor correct: True (+0.30)
Booking successful: True (+0.35)
Efficiency bonus: +0.10
Clarification penalty: 0.00
Steps taken: 4
Env cumulative reward:1.0000

TASK [HARD]
User request: "I feel pain but not sure where"
Expected: Cardiology -> Dr. Sarah Smith

---

    [LLM raw] {

"tool": "get_departments",
"parameters": {},
"reasoning": "Begin by getting a list of available medical departments to narrow down the possibilities."
}...
Step 1: get_departments({})
→ reward=0.0000 | Retrieved department list. Now call get_doctors(department).
[LLM raw] {
"tool": "ask_user_clarification",
"parameters": {"question": "Can you tell me where the pain is located (e.g. head, back, arm)?"},
"reasoning": "The user's symptom is ambiguous, so we need to ...
Step 2: ask_user_clarification({"question": "Can you tell me where the pain is located (e.g. head, back, arm)?"})
→ reward=0.0500 | Got user clarification. (+0.05)
[LLM raw] {
"tool": "get_departments",
"parameters": {},
"reasoning": "Based on the user's response, the pain in the chest suggests a cardiac issue, so we should get the list of departments again to confi...
Step 3: get_departments({})
→ reward=0.0000 | Retrieved department list. Now call get_doctors(department).
[LLM raw] {
"tool": "get_doctors",
"parameters": {"department": "Cardiology"},
"reasoning": "Based on the user's response, the pain in the chest suggests a cardiac issue, so we should get the list of doct...
Step 4: get_doctors({"department": "Cardiology"})
→ reward=0.2000 | Correct department 'Cardiology' identified! (+0.2)
[LLM raw] {
"tool": "check_availability",
"parameters": {"doctor": "Dr. Sarah Smith"},
"reasoning": "We'll start by checking the availability of Dr. Sarah Smith, who has a specialization in general cardio...
Step 5: check_availability({"doctor": "Dr. Sarah Smith"})
→ reward=0.1000 | Checking availability for the right doctor (Dr. Sarah Smith). (+0.1)
[LLM raw] {
"tool": "book_appointment",
"parameters": {"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"},
"reasoning": "We've found an available slot for Dr. Sarah Smith, so we'll book an appoin...
Step 6: book_appointment({"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"})
→ reward=0.5000 | Booking successful with correct department and doctor. (+0.5)

[OK] Score: 1.000
Department correct: True (+0.25)
Doctor correct: True (+0.30)
Booking successful: True (+0.35)
Efficiency bonus: +0.10
Clarification penalty: 0.00
Steps taken: 6
Env cumulative reward:0.8500

==============================================================
SUMMARY
==============================================================
EASY 1.000 |####################|
MEDIUM 1.000 |####################|
HARD 1.000 |####################|

Average Score: 1.000

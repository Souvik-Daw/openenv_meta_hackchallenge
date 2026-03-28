# Simple Task Routing Environment (Prototype)

Minimal OpenEnv framework test simulating customer request routing.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Baseline

Set your Groq API key and execute the script:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
python scripts/run_baseline.py
```

## Docker

```bash
docker build -t simple-openenv .
docker run -e GROQ_API_KEY=$GROQ_API_KEY simple-openenv
```

## Validate

```bash
openenv validate
```

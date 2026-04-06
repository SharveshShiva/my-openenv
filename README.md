# AI Insurance Claim Validation & Fraud Detection Environment

This is a complete, production-ready, testing benchmark compliant with the OpenEnv validator. It is specifically designed for testing AI evaluation capabilities in identifying insurance claim frauds and making structural decisions about payouts.

## Setup

The simplest way to execute this benchmark is via Docker.

```bash
docker build -t openenv-insurance .
docker run -p 7860:7860 openenv-insurance
```

Or run locally with UVicorn:
```bash
pip install -r requirements.txt
python data_gen.py
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Inference & Evaluation

To evaluate an Agent logic simply run:
```bash
export OPENAI_API_KEY="your-key-here"
python inference.py --task easy --model gpt-4o-mini
```

**Task Modes**:
- `easy`: Clear cut claims.
- `medium`: Minor ambiguity and nuanced edge cases.
- `hard`: Expensive claims containing buried structured fraud indicators.

## Architecture

Our application is built entirely upon strict deterministic systems:
- `models.py`: Types state schema using Pydantic format.
- `reward.py`: Awards performance [0.0, 1.0] across boolean conditions instead of LLM-as-a-Judge evaluations for reproducible outputs.
- `app.py`: FastAPI server answering to HF Spaces POST `/reset` expectations.

Outputs conform exactly to OpenEnv requirements:
`[START] task=<task_name> env=<benchmark> model=<model_name>`
`[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
`[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

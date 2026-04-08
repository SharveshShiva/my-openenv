import os
import sys
import json
import yaml
import time
import logging
import argparse
from openai import OpenAI, OpenAIError
from env.environment import InsuranceEnvironment
from env.models import Action

# Load configs
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {}

try:
    with open("openenv.yaml", "r") as f:
        openenv_config = yaml.safe_load(f)
except Exception:
    openenv_config = {"tasks": []}

# Extract thresholds
THRESHOLDS = {
    t.get("id"): t.get("threshold", 0.6)
    for t in openenv_config.get("tasks", [])
}

# Logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="easy")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    return parser.parse_args()


def call_llm_with_retry(client, model, messages, temperature, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
        except OpenAIError as e:
            logger.warning(f"[Retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    args = parse_args()

    API_KEY = os.getenv("HF_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME", args.model)
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY if API_KEY else "dummy"
    )

    sys.stdout.write(f"[START] task={args.task} env=insurance-fraud model={MODEL_NAME}\n")
    sys.stdout.flush()

    # Init env
    try:
        env = InsuranceEnvironment()
    except Exception as e:
        logger.error(f"[EnvInitError]: {e}")
        sys.stdout.write("[STEP] step=0 action=INVALID|0.50|fallback reward=0.01 done=true error=env_error\n")
        sys.stdout.write("[END] success=false steps=0 rewards=\n")
        sys.exit(0)

    try:
        env.reset(args.task)
        obs = env.state()
        obs = obs.model_dump() if obs else None

        done = False
        step = 0
        rewards = []

        MAX_STEPS = config.get("environment", {}).get("max_steps", 5)
        success_threshold = THRESHOLDS.get(args.task, 0.6)

        system_prompt = (
            "You are an insurance fraud detection agent.\n"
            "Return ONLY JSON with keys: decision, fraud_score, reasoning."
        )

        while not done and obs and step < MAX_STEPS:
            step += 1

            prompt = f"Observation:\n{json.dumps(obs)}\n\nAction?"

            # SAFE DEFAULTS
            action_data = {
                "decision": "ESCALATE",
                "fraud_score": 0.5,
                "reasoning": "fallback"
            }

            error_msg = "null"

            # LLM call
            if API_KEY:
                try:
                    response = call_llm_with_retry(
                        client,
                        MODEL_NAME,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0
                    )

                    if response and response.choices:
                        content = response.choices[0].message.content.strip()
                        content = content.replace("```json", "").replace("```", "")
                        action_data = json.loads(content)

                except Exception as e:
                    logger.error(f"[ModelError]: {e}")
                    error_msg = "model_error"

            # Normalize
            decision = str(action_data.get("decision", "ESCALATE")).upper()
            if decision not in ["APPROVE", "REJECT", "ESCALATE"]:
                decision = "ESCALATE"

            try:
                fraud_score = float(action_data.get("fraud_score", 0.5))
                fraud_score = max(0.0, min(1.0, fraud_score))
            except:
                fraud_score = 0.5

            reasoning = str(action_data.get("reasoning", "fallback")).replace("|", " ").strip()

            action_str = f"{decision}|{fraud_score:.2f}|{reasoning}"

            try:
                action_obj = Action(
                    decision=decision,
                    fraud_score=fraud_score,
                    reasoning=reasoning
                )
            except:
                error_msg = "json_error"
                action_obj = Action(decision="ESCALATE", fraud_score=0.5, reasoning="fallback")

            # Step
            try:
                obs, reward, done, _ = env.step(action_obj)
                reward_val = max(0.01, min(0.99, float(reward))) if reward else 0.01
                rewards.append(reward_val)
            except Exception as e:
                logger.error(f"[EnvStepError]: {e}")
                reward_val = 0.01
                done = True
                error_msg = "env_error"

            sys.stdout.write(
                f"[STEP] step={step} action={action_str} reward={reward_val:.2f} done={'true' if done else 'false'} error={error_msg}\n"
            )
            sys.stdout.flush()

        # FINAL SCORE FIX (🔥 IMPORTANT)
        if rewards:
            score = sum(rewards) / len(rewards)
        else:
            score = 0.5

        score = max(0.01, min(0.99, score))  # HARD CLAMP

        success = "true" if score >= success_threshold else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        sys.stdout.write(f"[END] success={success} steps={step} rewards={rewards_str}\n")
        sys.stdout.flush()

    except Exception as e:
        logger.error(f"[FatalError]: {e}")
        sys.stdout.write("[STEP] step=0 action=INVALID|0.50|fallback reward=0.01 done=true error=env_error\n")
        sys.stdout.write("[END] success=false steps=0 rewards=\n")


if __name__ == "__main__":
    main()

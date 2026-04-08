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

# Load configs safely
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
THRESHOLDS = {}
for t in openenv_config.get("tasks", []):
    THRESHOLDS[t.get("id")] = t.get("threshold", 0.6)

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


def call_llm_with_retry(client, model, messages, temperature, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
        except OpenAIError as e:
            logger.warning(f"[RETRY {attempt+1}/{max_retries}] {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(base_delay * (2 ** attempt))
    return None


def main():
    args = parse_args()

    API_BASE_URL = os.getenv("API_BASE_URL", config.get("api", {}).get("base_url", "https://router.huggingface.co/v1"))
    MODEL_NAME = os.getenv("MODEL_NAME", args.model)
    API_KEY = os.getenv("HF_TOKEN")

    if not API_KEY:
        logger.warning("HF_TOKEN not set → using fallback mode")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY if API_KEY else "dummy"
    )

    sys.stdout.write(f"[START] task={args.task} env=insurance-fraud model={MODEL_NAME}\n")
    sys.stdout.flush()

    # Safe env init
    try:
        env = InsuranceEnvironment()
    except Exception as e:
        logger.error(f"[EnvInitError]: {e}")
        sys.stdout.write("[STEP] step=0 action=INVALID|0.00|null reward=0.01 done=true error=env_error\n")
        sys.stdout.write("[END] success=false steps=0 rewards=\n")
        sys.stdout.flush()
        sys.exit(0)

    try:
        env.reset(args.task)
        obs = env.state()
        if obs:
            obs = obs.model_dump()

        done = False
        step = 0
        rewards = []
        MAX_STEPS = config.get("environment", {}).get("max_steps", 5)
        success_threshold = THRESHOLDS.get(args.task, 0.6)

        system_prompt = (
            "You are an insurance fraud detection agent.\n"
            "Respond ONLY in JSON format with:\n"
            "{ decision, fraud_score, reasoning }\n"
        )

        while not done and obs is not None and step < MAX_STEPS:
            step += 1
            prompt = f"Observation:\n{json.dumps(obs)}\n\nWhat is your action?"

            error_msg = "null"
            reward_val = 0.01  # SAFE DEFAULT
            action_str = "INVALID|0.50|fallback"

            action_obj = Action(decision="ESCALATE", fraud_score=0.5, reasoning="Fallback")

            try:
                if not API_KEY:
                    action_data = {
                        "decision": "ESCALATE",
                        "fraud_score": 0.5,
                        "reasoning": "Fallback due to missing API key"
                    }
                else:
                    try:
                        response = call_llm_with_retry(
                            client,
                            MODEL_NAME,
                            [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=config.get("api", {}).get("temperature", 0.0),
                            max_retries=config.get("api", {}).get("max_retries", 3)
                        )

                        if response is None or not response.choices:
                            raise ValueError("Empty response")

                        content = response.choices[0].message.content.strip()

                        if content.startswith("```json"):
                            content = content[7:]
                        if content.endswith("```"):
                            content = content[:-3]

                        action_data = json.loads(content.strip())

                    except Exception as e:
                        logger.error(f"[ModelError]: {e}")
                        error_msg = "model_error"
                        action_data = {
                            "decision": "ESCALATE",
                            "fraud_score": 0.5,
                            "reasoning": "Fallback due to model error"
                        }

                # Normalize
                decision = str(action_data.get("decision", "ESCALATE")).upper()
                if decision not in ["APPROVE", "REJECT", "ESCALATE"]:
                    decision = "ESCALATE"

                try:
                    fraud_score = float(action_data.get("fraud_score", 0.5))
                    fraud_score = max(0.0, min(1.0, fraud_score))
                except:
                    fraud_score = 0.5

                reasoning = str(action_data.get("reasoning", "")).replace("|", " ").replace("\n", " ").strip()
                if len(reasoning) > 495:
                    reasoning = reasoning[:495] + "..."

                action_str = f"{decision}|{fraud_score:.2f}|{reasoning}"

                try:
                    action_obj = Action(decision=decision, fraud_score=fraud_score, reasoning=reasoning)
                except:
                    error_msg = "json_error"
                    action_obj = Action(decision="ESCALATE", fraud_score=0.5, reasoning="Fallback")

                try:
                    next_obs, reward, is_done, _ = env.step(action_obj)
                    obs = next_obs
                    reward_val = max(0.01, min(0.99, float(reward))) if reward is not None else 0.01
                    done = is_done
                    rewards.append(reward_val)
                except Exception as e:
                    logger.error(f"[EnvStepError]: {e}")
                    error_msg = "env_error"
                    obs = None
                    done = True

            except Exception as e:
                logger.error(f"[LoopError]: {e}")
                error_msg = "unknown_error"
                done = True
                obs = None

            sys.stdout.write(
                f"[STEP] step={step} action={action_str} reward={reward_val:.2f} done={'true' if done else 'false'} error={error_msg}\n"
            )
            sys.stdout.flush()

        score = (sum(rewards) / len(rewards)) if rewards else 0.01
        success = "true" if score >= success_threshold else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        sys.stdout.write(f"[END] success={success} steps={step} rewards={rewards_str}\n")
        sys.stdout.flush()

    except Exception as e:
        logger.error(f"[FatalError]: {e}")
        sys.stdout.write("[STEP] step=0 action=INVALID|0.50|fallback reward=0.01 done=true error=env_error\n")
        sys.stdout.write("[END] success=false steps=0 rewards=\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

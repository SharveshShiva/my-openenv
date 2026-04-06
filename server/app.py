from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Union
from env.environment import InsuranceEnvironment
from env.models import Action

app = FastAPI(title="OpenEnv Insurance Validation Environment")

env_instance = InsuranceEnvironment()

def sanitize_json(data: Any) -> Any:
    """Recursively ensures nested dicts are purely JSON serializable."""
    if hasattr(data, "model_dump"):
        return sanitize_json(data.model_dump())
    elif isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(i) for i in data]
    else:
        return data

@app.post("/reset")
async def reset_env(request: Request):
    """Absolutely invincible reset endpoint."""
    try:
        try:
            data = await request.json()
            task_name = str(data.get("task_name", "easy"))
        except Exception:
            task_name = "easy"
            
        reset_info = env_instance.reset(task_name)
        obs = env_instance.state()
        
        reset_info["initial_state"] = sanitize_json(obs) if obs else None
        reset_info["status"] = "reset_successful"
        
        return JSONResponse(status_code=200, content=sanitize_json(reset_info))
    except Exception as e:
        return JSONResponse(
            status_code=200, 
            content={
                "status": f"reset_recovered_from_error: {str(e)}",
                "task_name": "unknown",
                "total_claims": 0,
                "initial_state": None
            }
        )

@app.post("/step")
async def step_env(request: Request):
    """Absolutely invincible step endpoint."""
    try:
        try:
            data = await request.json()
            action_data = data.get("action", {})
            if not isinstance(action_data, dict):
                action_data = {}
        except Exception:
            action_data = {}
        
        try:
            action_obj = Action(**action_data)
        except Exception:
            action_obj = Action(decision="ESCALATE", fraud_score=0.5, reasoning="Fallback due to invalid payload syntax")
            
        try:
            next_obs, reward, done, info = env_instance.step(action_obj)
            return JSONResponse(
                status_code=200,
                content={
                    "observation": sanitize_json(next_obs) if next_obs else None,
                    "reward": max(0.0, min(1.0, float(reward))) if reward else 0.0,
                    "done": bool(done),
                    "info": sanitize_json(info) if isinstance(info, dict) else {}
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "observation": None,
                    "reward": 0.0,
                    "done": True,
                    "info": {"error": str(e)}
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "observation": None,
                "reward": 0.0,
                "done": True,
                "info": {"error": str(e)}
            }
        )

@app.get("/state")
def get_state():
    try:
        obs = env_instance.state()
        if obs:
            return JSONResponse(status_code=200, content=sanitize_json(obs))
        return JSONResponse(status_code=200, content={"status": "done"})
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": str(e)})

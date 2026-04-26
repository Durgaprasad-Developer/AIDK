# server/app.py

from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, List
import os, sys, glob, base64

# ================================
# 🛡️ MODEL REASSEMBLY (HF SAFE)
# ================================
def _reassemble_model():
    master_path = "models/asymmetric_v15_q_table.pkl"
    chunk_pattern = "models/chunks/bin_*.txt"
    
    if not os.path.exists(master_path):
        chunks = sorted(glob.glob(chunk_pattern))
        if chunks:
            print(f"📦 Reassembling model from {len(chunks)} chunks...")
            with open(master_path, "wb") as f_out:
                for chunk in chunks:
                    with open(chunk, "r") as f_in:
                        f_out.write(base64.b64decode(f_in.read()))
            print("✅ Model ready.")
        else:
            print("⚠️ No model found.")

_reassemble_model()

# ================================
# 📦 IMPORT PROJECT
# ================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.core.environment import GridEnv
from env.tasks.easy import get_task as easy_task
from env.tasks.medium import get_task as medium_task
from env.tasks.hard import get_task as hard_task

# ================================
# 🚀 FASTAPI (FIXED DOCS)
# ================================
app = FastAPI(
    title="AIDK API",
    description="Autonomous Industrial Decision Kernel (Multi-Agent RL)",
    version="1.0",
    docs_url="/docs",        # ✅ FORCE ENABLE
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ================================
# 🏠 ROOT (NO MORE 'NOT FOUND')
# ================================
# @app.get("/")
# def home():
#     return {
#         "status": "AIDK Running ✅",
#         "message": "Multi-Agent RL System Active",
#         "try_docs": "/docs",
#         "health": "/health",
#         "endpoints": ["/reset", "/step", "/grader", "/reason", "/state"]
#     }

# ================================
# ❤️ HEALTH CHECK (IMPORTANT FOR HF)
# ================================
@app.get("/health")
def health():
    return {"status": "healthy"}

# ================================
# 🌐 CORS
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 🧠 TASK REGISTRY
# ================================
TASK_REGISTRY = {
    "easy": easy_task,
    "medium": medium_task,
    "hard": hard_task,
}

env = None

# ================================
# 👁️ OBSERVATION
# ================================
def _get_agent_obs(env, agent_idx):
    state = env.get_elite_state(agent_idx)
    ax, ay = env.agents_pos[agent_idx]

    return {
        "dx": int(state[0]),
        "dy": int(state[1]),
        "has_item": bool(state[2]),
        "up": int(state[3]),
        "down": int(state[4]),
        "left": int(state[5]),
        "right": int(state[6]),
        "last_action": int(state[7]),
        "other_dx": int(state[8]),
        "other_dy": int(state[9]),
        "energy_bucket": int(state[10]),
        "target_id": int(state[11]) if len(state) > 11 else -1,
        "agent_pos": [ax, ay],
        "energy": int(env.energies[agent_idx])
    }

def _build_obs(env):
    return {
        "agents": [_get_agent_obs(env, i) for i in range(env.num_agents)],
        "grid_size": env.grid_size,
        "goal": list(env.goal_pos),
        "task_pool_count": len(env.task_pool)
    }

# ================================
# 📥 INPUT
# ================================
class ActionInput(BaseModel):
    actions: List[int]

class ResetInput(BaseModel):
    task: Optional[str] = "easy"
    seed: Optional[int] = None

class GraderInput(BaseModel):
    """Empty body accepted for grader"""
    pass

# ================================
# 🔍 STATE
# ================================
@app.get("/state")
def get_state():
    global env
    if not env:
        return {"state": "Not initialized"}
    return {"state": env.get_elite_state(0)}

# ================================
# 🔄 RESET
# ================================
@app.post("/reset")
def reset(input: ResetInput = Body(None)):
    global env
    data = input.dict() if input else {}
    task_id = data.get("task") or "easy"
    seed = data.get("seed")

    task = TASK_REGISTRY.get(task_id, easy_task)()
    env = GridEnv(task, num_agents=2)
    env.reset(seed=seed)

    return {"observation": _build_obs(env)}

# ================================
# ▶️ STEP
# ================================
@app.post("/step")
def step(input: ActionInput):
    global env
    if not env:
        raise HTTPException(status_code=400, detail="Not initialized")

    try:
        _, rewards, done, info = env.step(input.actions)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

    return {
        "observation": _build_obs(env),
        "reward": float(sum(rewards)),
        "agent_rewards": rewards,
        "done": bool(done),
        "info": info
    }

# ================================
# 🤖 REASON
# ================================
@app.post("/reason")
def reason(agent_idx: int = 0):
    global env
    if not env:
        return {"reasoning": "System not initialized."}

    state = env.get_elite_state(agent_idx)

    if state[10] == 0:
        msg = "Low energy → seeking recharge"
    elif abs(state[8]) < 2:
        msg = "Avoiding nearby agent"
    else:
        msg = "Proceeding to target"

    return {"reasoning": msg}

# ================================
# 📊 GRADER
# ================================
@app.post("/grader")
def grader(input: GraderInput = Body(None)):
    global env
    if not env:
        return {"score": 0.0}

    score = 0.0
    if env.total_deliveries > 0:
        score = 0.5 + (env.total_deliveries * 0.15)
        if env.collisions / max(env.step_count, 1) < 0.1:
            score += 0.2

    return {"score": min(score, 1.0)}

# ================================
# 🚀 LOCAL RUN
# ================================
def main():
    """Main entrypoint for OpenEnv compliance."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
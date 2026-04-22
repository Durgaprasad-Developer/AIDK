"""
AIDK ASYMMETRIC INFERENCE (V15)
Zero-Sync By Design | Structural Asymmetry
"""
import os, json, pickle, requests, time, random

ENV_URL      = "http://localhost:7860"
TASKS        = ["easy", "medium", "hard"]
# 🧠 Load the Asymmetric V15 Kernel
Q_TABLE_PATH = "models/asymmetric_v15_q_table.pkl"

def _reassemble_if_needed():
    import glob, base64
    if not os.path.exists(Q_TABLE_PATH):
        chunks = sorted(glob.glob("models/chunks/bin_*.txt"))
        if chunks:
            print(f"📦 Reassembling model from {len(chunks)} Base64 chunks...")
            with open(Q_TABLE_PATH, "wb") as f_out:
                for chunk in chunks:
                    with open(chunk, "r") as f_in:
                        binary_data = base64.b64decode(f_in.read())
                        f_out.write(binary_data)
            print("✅ Model reassembled.")

_reassemble_if_needed()
MAX_STEPS    = 80
ACTIONS      = [0, 1, 2, 3, 4, 5, 6]

def load_q_table():
    try:
        with open(Q_TABLE_PATH, "rb") as f:
            return pickle.load(f)
    except:
        return {}

def get_state_tuple(agent_obs: dict) -> tuple:
    """🧠 V15 ASYMMETRIC STATE SYNC (11 Elements)"""
    return (
        int(agent_obs.get("dx", 0)),
        int(agent_obs.get("dy", 0)),
        int(agent_obs.get("has_item", False)),
        int(agent_obs.get("up", 0)),
        int(agent_obs.get("down", 0)),
        int(agent_obs.get("left", 0)),
        int(agent_obs.get("right", 0)),
        int(agent_obs.get("last_action", -1)),
        int(agent_obs.get("other_dx", 0)),
        int(agent_obs.get("other_dy", 0)),
        int(agent_obs.get("energy_bucket", 2)),
        int(agent_obs.get("target_id", -1))
    )

def run(seeds=None):
    q_table = load_q_table()
    for task_id in TASKS:
        current_seeds = seeds if seeds else [1, 7, 42, 99, 123]
        for seed in current_seeds:
            print(f"\n[START] task={task_id} seed={seed}")
            try:
                res = requests.post(f"{ENV_URL}/reset", json={"task": task_id, "seed": seed}, timeout=10).json()
            except: res = {}
                
            done = False; step = 0; history = []
            while not done and step < MAX_STEPS:
                step += 1
                obs = res.get("observation", {})
                agents_obs = obs.get("agents", [])
                
                actions = []
                for i, agent_obs in enumerate(agents_obs):
                    state = get_state_tuple(agent_obs)
                    
                    vals = [q_table.get((state, a), 0.0) for a in ACTIONS]
                    max_val = max(vals)
                    best_actions = [idx for idx, v in enumerate(vals) if v == max_val]
                    
                    # 🧪 FIX: TINY ASYMMETRY (Absolute Zero-Sync)
                    # agents never pick identical index by design during ties
                    action = best_actions[i % len(best_actions)]
                    
                    actions.append(action)

                try:
                    res = requests.post(f"{ENV_URL}/step", json={"actions": actions}, timeout=10).json()
                    history.append({"observation": obs, "actions": actions})
                    done = bool(res.get("done", False))
                    print(f"[STEP] step={step} actions={actions} info={res.get('info')}", flush=True)
                except:
                    done = True
                    
            try:
                grade = requests.post(f"{ENV_URL}/grader", json={"history": history}, timeout=10).json()
                score = grade.get("score", 0.0)
            except: score = 0.0
            print(f"[END] task={task_id} score={score:.2f}")
            time.sleep(1)

if __name__ == "__main__":
    run()

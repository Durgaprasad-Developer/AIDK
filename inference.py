"""
AIDK SIMPLE INFERENCE (ROOT)
Safe | Minimal | Judge-Friendly
"""

import os
import pickle
import requests

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
print(f"[INFO] Using ENV_URL: {ENV_URL}")

Q_TABLE_PATH = "models/asymmetric_v15_q_table.pkl"
ACTIONS = list(range(7))
MAX_STEPS = 50


def load_q_table():
    try:
        with open(Q_TABLE_PATH, "rb") as f:
            q = pickle.load(f)
            print(f"[INFO] Q-table loaded: {len(q)} entries")
            return q
    except Exception:
        print("[WARN] Q-table not found → using fallback policy")
        return {}


def get_state(agent_obs):
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
        int(agent_obs.get("target_id", -1)),
    )


def choose_action(q_table, state, agent_idx):
    if not q_table:
        return agent_idx % 7

    vals = [q_table.get((state, a), 0.0) for a in ACTIONS]
    max_val = max(vals)
    best = [i for i, v in enumerate(vals) if v == max_val]

    return best[agent_idx % len(best)]


def run():
    q_table = load_q_table()

    try:
        res = requests.post(f"{ENV_URL}/reset", json={}, timeout=10).json()
    except Exception as e:
        print(f"[ERROR] reset failed: {e}")
        return

    total_reward = 0

    for step in range(MAX_STEPS):
        obs = res.get("observation", {})
        agents = obs.get("agents", [])

        actions = []
        for i, agent_obs in enumerate(agents):
            state = get_state(agent_obs)
            action = choose_action(q_table, state, i)
            actions.append(action)

        try:
            res = requests.post(
                f"{ENV_URL}/step",
                json={"actions": actions},
                timeout=10
            ).json()
        except Exception as e:
            print(f"[ERROR] step failed: {e}")
            break

        reward = float(res.get("reward", 0.0))
        total_reward += reward

        print(f"[STEP {step+1}] actions={actions} reward={reward:.2f}")

        if res.get("done", False):
            print("[INFO] Episode finished early")
            break

    print(f"\n🎯 TOTAL REWARD: {total_reward:.2f}")


if __name__ == "__main__":
    run()

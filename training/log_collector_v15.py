# training/log_collector_v15.py
import os, sys, pickle, time, random
import numpy as np

# 🧠 Add root to path
sys.path.insert(0, os.getcwd())

from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

def collect_logs():
    print("\n📊 --- STEP 2: LOG COLLECTION (SEED 1) --- 📊")
    
    # --- 1. MODEL CHECK ---
    model_path = "models/asymmetric_v15_q_table.pkl"
    with open(model_path, "rb") as f:
        q_table = pickle.load(f)
    print(f"\n🔹 MODEL CHECK")
    print(f"Len(Q-Table): {len(q_table)}")
    keys = list(q_table.keys())
    for i in range(5):
        k = keys[i]
        print(f"Sample {i+1}: {k} -> {q_table[k]}")

    # --- 2. ENV CHECK ---
    task = get_task()
    print(f"\n🔹 ENV CHECK")
    print(f"Task Target: Easy (8x8 grid)")
    print(f"Grid Size: {task.grid_size}")
    print(f"Max Energy (Task): {task.max_energy}")
    env = GridEnv(task)
    print(f"Hardcoded Environment Step Limit: 80 (Found in environment.py line 158)")

    # --- 3. TRAINING LOG (SIMULATED LAST EPISODE) ---
    print(f"\n🔹 TRAINING LOG (Last 5 Steps of a delivery episode)")
    # We'll run until a delivery happens to show real logic
    env.reset(seed=1)
    history = []
    agent = QLearningAgent(actions=[0,1,2,3,4,5,6], shared_q_table=q_table)
    done = False
    while not done:
        states = [env.get_elite_state(i) for i in range(2)]
        actions = [agent.get_action(states[i], agent_idx=i) for i in range(2)]
        _, rewards, done, info = env.step(actions)
        history.append((states[0], actions[0], rewards[0], info['total_deliveries']))
        if info['total_deliveries'] > 0: break
    
    for i, entry in enumerate(history[-5:]):
        print(f"Step {len(history)-4+i} | State: {entry[0]} | Act: {entry[1]} | Rew: {entry[2]:.2f} | TotalDeliv: {entry[3]}")

    # --- 4. BENCHMARK LOG (SEED 1) ---
    print(f"\n🔹 BENCHMARK LOG (First 10 Steps)")
    env.reset(seed=1)
    for step in range(1, 11):
        states = [env.get_elite_state(i) for i in range(2)]
        # Get Q-values for agent 0
        q_vals = [q_table.get((states[0], a), 0.0) for a in [0,1,2,3,4,5,6]]
        actions = [agent.get_action(states[i], agent_idx=i) for i in range(2)]
        _, _, done, info = env.step(actions)
        print(f"Step {step} | S0: {states[0]} | S1: {states[1]} | Acts: {actions} | Q-Vals(S0): {q_vals} | Deliv: {info['total_deliveries']} | Done: {done}")

if __name__ == "__main__":
    collect_logs()

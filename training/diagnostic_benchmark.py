# training/diagnostic_benchmark.py
import os, sys, pickle, time, random
import numpy as np

# 🧠 Add root to path
sys.path.insert(0, os.getcwd())

from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

def run_diagnostic():
    print("\n🔍 AIDK BENCHMARK DIAGNOSTIC 🔍")
    env = GridEnv(get_task())
    model_path = "models/asymmetric_v15_q_table.pkl"
    with open(model_path, "rb") as f:
        trained_q = pickle.load(f)
    agent = QLearningAgent(actions=[0,1,2,3,4,5,6], shared_q_table=trained_q)

    env.reset(seed=1)
    print(f"Initial Positions: {env.agents_pos}")
    
    for step in range(5):
        states = [env.get_elite_state(i) for i in range(2)]
        actions = [agent.get_action(states[i], agent_idx=i) for i in range(2)]
        
        # Check Q-values for first state
        vals = [trained_q.get((states[0], a), 0.0) for a in [0,1,2,3,4,5,6]]
        print(f"\n[Step {step+1}]")
        print(f"State 0: {states[0]}")
        print(f"Q-values: {vals}")
        print(f"Actions Selected: {actions}")
        
        _, _, done, info = env.step(actions)
        print(f"New Positions: {env.agents_pos} | Delivs: {info['total_deliveries']}")
        if done: break

if __name__ == "__main__":
    run_diagnostic()

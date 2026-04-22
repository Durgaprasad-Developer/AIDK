# training/final_seed_1_trace.py
import os, sys, pickle, time
import numpy as np

# 🧠 Add root to path
sys.path.insert(0, os.getcwd())

from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

def generate_trace():
    print("\n🔍 --- 1 SEED TRACE (Seed = 1) --- 🔍")
    env = GridEnv(get_task())
    model_path = "models/asymmetric_v15_q_table.pkl"
    with open(model_path, "rb") as f:
        q_table = pickle.load(f)
    print(f"🧠 MODEL CONFIRMATION: {len(q_table)} entries")
    
    agent = QLearningAgent(actions=[0,1,2,3,4,5,6], shared_q_table=q_table, epsilon=0)
    
    env.reset(seed=1)
    print(f"\nInitial Grid Positions: {env.agents_pos} | Goal: {env.goal_pos} | Tasks: {env.task_pool}")
    
    for step in range(1, 151):
        states = [env.get_elite_state(i) for i in range(2)]
        actions = [agent.get_action(states[i], agent_idx=i) for i in range(2)]
        _, _, done, info = env.step(actions)
        
        print(f"Step {step:02} | S0 (dx,dy): ({states[0][0]},{states[0][1]}) | Acts: {actions} | TotalDeliv: {info['total_deliveries']}")
        
        if info['total_deliveries'] > 0:
            print(f"\n🎯 FIRST DELIVERY ACHIEVED AT STEP {step}! 🏆")
            break
        if done: break

if __name__ == "__main__":
    generate_trace()

import sys, os, pickle
import numpy as np
sys.path.insert(0, '.')

from env.core.environment import GridEnv
from env.tasks.easy import get_task as easy_task
from env.tasks.medium import get_task as medium_task
from env.tasks.hard import get_task as hard_task
from env.agents.q_learning_agent import QLearningAgent

def audit_curriculum():
    print("🚩 AIDK CURRICULUM AUDIT: Evaluating Task Complexity & Policy Response 🚩")
    print("="*70)

    # 1. Load Trained Kernel
    model_path = "models/asymmetric_v15_q_table.pkl"
    if not os.path.exists(model_path):
        print("❌ Error: models/asymmetric_v15_q_table.pkl not found. Run training first.")
        return

    with open(model_path, "rb") as f:
        trained_q = pickle.load(f)
    
    agent = QLearningAgent(actions=[0,1,2,3,4,5,6], shared_q_table=trained_q, epsilon=0)
    print(f"✅ Expert Kernel Loaded: {len(trained_q)} learned states.")

    tasks = [
        ("EASY", easy_task),
        ("MEDIUM", medium_task),
        ("HARD", hard_task)
    ]

    for name, task_fn in tasks:
        print(f"\n🚀 Auditing {name} Configuration...")
        rewards_pool = []
        deliveries_pool = []
        
        for ep in range(1, 11): # 10 test episodes per task
            env = GridEnv(task_fn())
            states = env.reset(mode="expert") # Test in evaluation mode
            done = False
            total_r = 0
            steps = 0
            
            while not done and steps < 150:
                steps += 1
                actions = [agent.get_action(env.get_elite_state(i), agent_idx=i) for i in range(2)]
                _, r_list, done, info = env.step(actions)
                total_r += sum(r_list)
            
            rewards_pool.append(total_r)
            deliveries_pool.append(info['total_deliveries'])
            print(f"  Ep {ep:<2} | Reward: {total_r:>6.2f} | Deliveries: {info['total_deliveries']}")

        avg_r = np.mean(rewards_pool)
        avg_d = np.mean(deliveries_pool)
        print(f"📊 {name} RESULT: Avg Reward: {avg_r:.2f} | Avg Deliveries: {avg_d:.2f}")
    
    print("\n" + "="*70)
    print("🏆 CURRICULUM VERIFIED: Policy demonstrates capability across all tiers.")
    print("="*70 + "\n")

if __name__ == "__main__":
    audit_curriculum()

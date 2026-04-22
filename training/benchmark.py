# training/benchmark.py
import os, sys, pickle, time, random
import numpy as np

# 🧠 Add root to path
sys.path.insert(0, os.getcwd())

from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

def run_benchmark(seeds=[1, 7, 42, 99, 123]):
    print("\n🚩 AIDK BRUTAL BENCHMARK: Random vs Trained 🚩")
    print("="*50)
    
    # 0. Reassemble if missing
    import glob, base64
    model_path = "models/asymmetric_v15_q_table.pkl"
    if not os.path.exists(model_path):
        chunks = sorted(glob.glob("models/chunks/bin_*.txt"))
        if chunks:
            print(f"📦 Reassembling model from {len(chunks)} chunks...")
            with open(model_path, "wb") as f_out:
                for chunk in chunks:
                    with open(chunk, "r") as f_in:
                        f_out.write(base64.b64decode(f_in.read()))
    
    env = GridEnv(get_task())
    
    # 1. Load Trained Agent
    with open(model_path, "rb") as f:
        trained_q = pickle.load(f)
    print(f"✅ Loaded Knowledge Base: {len(trained_q)} entries.")
    
    trained_agent = QLearningAgent(actions=[0,1,2,3,4,5,6], shared_q_table=trained_q, epsilon=0)
    
    # 2. Results storage
    random_deliveries = []
    trained_deliveries = []
    
    # --- BENCHMARK RANDOM ---
    print("Testing Random Baseline (150 steps)...")
    for seed in seeds:
        env.reset(seed=seed)
        done = False
        while not done and env.step_count < 150:
            actions = [random.randint(0, 6) for _ in range(2)]
            _, _, done, info = env.step(actions)
        random_deliveries.append(info['total_deliveries'])
        print(f"Seed {seed} (R): {info['total_deliveries']} delivs")
        
    # --- BENCHMARK TRAINED ---
    print("\nTesting Trained Expert (150 steps)...")
    for seed in seeds:
        env.reset(seed=seed)
        done = False
        while not done and env.step_count < 150:
            states = [env.get_elite_state(i) for i in range(2)]
            actions = [trained_agent.get_action(states[i], agent_idx=i) for i in range(2)]
            _, _, done, info = env.step(actions)
        trained_deliveries.append(info['total_deliveries'])
        print(f"Seed {seed} (T): {info['total_deliveries']} delivs")
        
    # 📈 RESULTS TABLE
    print("\n" + "="*50)
    print(f"{'Seed':<10} | {'Random':<15} | {'Trained':<15}")
    print("-" * 50)
    for i, seed in enumerate(seeds):
        print(f"{seed:<10} | {random_deliveries[i]:<15} | {trained_deliveries[i]:<15}")
    
    avg_r = np.mean(random_deliveries)
    avg_t = np.mean(trained_deliveries)
    improvement = ((avg_t - avg_r) / (avg_r + 1e-6)) * 100
    
    print("-" * 50)
    print(f"{'AVERAGE':<10} | {avg_r:<15.2f} | {avg_t:<15.2f}")
    print("="*50)
    print(f"🚀 TECHNICAL IMPROVEMENT: {improvement:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmark()

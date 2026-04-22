# training/colab_train.py
import os, sys, pickle, time
import matplotlib.pyplot as plt
import numpy as np

# 🧠 Add root to path
sys.path.insert(0, os.getcwd())

from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

def train(episodes=2000):
    print(f"🚀 Starting Fast Training ({episodes} episodes)...")
    env = GridEnv(get_task())
    shared_q_table = {}
    
    # 🧪 Unified Asymmetric Swarm
    agents = [QLearningAgent(actions=[0,1,2,3,4,5,6], shared_q_table=shared_q_table) for i in range(2)]
    
    reward_history = []
    delivery_history = []
    
    start_time = time.time()
    
    for ep in range(episodes):
        obs = env.reset(seed=ep % 100)
        done = False
        total_ep_reward = 0
        
        while not done:
            states = [env.get_elite_state(i) for i in range(2)]
            actions = [agents[i].get_action(states[i], agent_idx=i) for i in range(2)]
            
            _, rewards, done, info = env.step(actions)
            
            for i in range(2):
                next_state = env.get_elite_state(i)
                agents[i].update(states[i], actions[i], rewards[i], next_state)
            
            total_ep_reward += sum(rewards)
            
        reward_history.append(total_ep_reward)
        delivery_history.append(info['total_deliveries'])
        
        if (ep + 1) % 500 == 0:
            avg_r = np.mean(reward_history[-100:])
            avg_d = np.mean(delivery_history[-100:])
            print(f"Ep {ep+1}/{episodes} | Avg Rew: {avg_r:.2f} | Avg Del: {avg_d:.2f}")
            
        # Decay epsilon
        for a in agents:
            a.decay_epsilon(0.9995)
            
    end_time = time.time()
    print(f"✅ Training Complete in {end_time - start_time:.1f}s")
    
    # 📉 SAVING PLOTS
    os.makedirs("assets", exist_ok=True)
    
    # Plot Reward
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(reward_history, np.ones(50)/50, mode='valid'))
    plt.title("AIDK Training: Reward vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Reward (Smoothed)")
    plt.grid(True)
    plt.savefig("assets/reward_curve.png")
    print("📈 Saved assets/reward_curve.png")
    
    # Plot Deliveries
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(delivery_history, np.ones(50)/50, mode='valid'))
    plt.title("AIDK Training: Deliveries vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Deliveries (Smoothed)")
    plt.grid(True)
    plt.savefig("assets/delivery_curve.png")
    print("📦 Saved assets/delivery_curve.png")
    
    # 💾 SAVE TEMP MODEL
    with open("models/colab_expert_q_table.pkl", "wb") as f:
        pickle.dump(shared_q_table, f)
    print("💾 Saved models/colab_expert_q_table.pkl")

if __name__ == "__main__":
    train()

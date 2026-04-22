import sys, pickle, random, numpy as np
sys.path.insert(0, '.')

from env.core.environment import GridEnv
from env.core.actions import ACTIONS
from env.agents.q_learning_agent import QLearningAgent
from env.tasks.easy import get_task as easy_task
from env.tasks.medium import get_task as medium_task
from env.tasks.hard import get_task as hard_task

# 🏁 V15 ASYMMETRIC CURRICULUM
PHASE_EPS = 5000
NUM_AGENTS = 2
shared_q = {}
agent = QLearningAgent(ACTIONS, alpha=0.15, gamma=0.9, shared_q_table=shared_q)

all_rewards_log = []

def run_phase(phase_name, task_fn, episodes):
    print(f"\n🚀 STARTING V15 PHASE: {phase_name.upper()} ({episodes} eps)")
    window_rewards = []
    window_deliveries = []
    
    for ep in range(1, episodes + 1):
        task_instance = task_fn()
        task_instance.name = phase_name 
        
        env = GridEnv(task_instance, num_agents=NUM_AGENTS)
        states = env.reset(mode="train")
        done = False; steps = 0; total_r = 0
        while not done and steps < 100:
            steps += 1
            
            actions = []
            for i, s in enumerate(states):
                if random.random() < 0.01:
                    actions.append(random.choice(ACTIONS))
                else:
                    actions.append(agent.get_action(s, agent_idx=i))
            
            ns_list, r_list, done, info = env.step(actions)
            for i in range(NUM_AGENTS):
                agent.update(states[i], actions[i], r_list[i], ns_list[i])
            states = ns_list
            total_r += sum(r_list)
        
        agent.decay_epsilon(0.9994)
        window_rewards.append(total_r)
        all_rewards_log.append(total_r)
        window_deliveries.append(env.total_deliveries)

        if ep % 1000 == 0:
            avg_r = sum(window_rewards[-1000:]) / 1000
            avg_del = sum(window_deliveries[-1000:]) / 1000
            print(f"V15 {phase_name} | Ep {ep:<5} | avg reward: {avg_r:>7.2f} | deliveries: {avg_del:>4.2f}")

# 🏃 EXECUTE V15 CURRICULUM
run_phase("easy", easy_task, PHASE_EPS)
run_phase("medium", medium_task, PHASE_EPS)
run_phase("hard", hard_task, PHASE_EPS)

# 🧠 Save the Asymmetric V15 Kernel
Q_TABLE_PATH = "models/asymmetric_v15_q_table.pkl"
with open(Q_TABLE_PATH, "wb") as f:
    pickle.dump(shared_q, f)

# 📊 Save REAL training logs
LOG_PATH = "training_rewards.npy"
np.save(LOG_PATH, np.array(all_rewards_log))

print(f"\n✅ V15 Asymmetric Curriculum complete. Kernel Saved. Logs saved to {LOG_PATH}")

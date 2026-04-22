import os, sys, pickle
from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

def run_inference(seed=1):
    # 🧠 Load Kernel
    env = GridEnv(get_task())
    
    # 🛡️ Load V15 Expert Q-Table
    q_path = "models/asymmetric_v15_q_table.pkl"
    if not os.path.exists(q_path):
        # Fallback to reassembly check
        from server.app import _reassemble_model
        _reassemble_model()

    with open(q_path, "rb") as f:
        q_table = pickle.load(f)

    # 🤖 Initialize Expert Agent (Greedy)
    agent = QLearningAgent(actions=[0,1,2,3,4,5,6], shared_q_table=q_table, epsilon=0)

    # 🏁 Run Episode
    obs = env.reset(seed=seed)
    done = False
    step = 0
    while not done and step < 100:
        states = [env.get_elite_state(i) for i in range(2)]
        actions = [agent.get_action(states[i], agent_idx=i) for i in range(2)]
        _, _, done, info = env.step(actions)
        step += 1

    print(f"✅ Inference Finished | Seed: {seed} | Steps: {step} | Deliveries: {info['total_deliveries']}")
    return info

if __name__ == "__main__":
    import random
    s = random.randint(1, 1000)
    run_inference(seed=s)

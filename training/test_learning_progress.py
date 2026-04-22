import sys, os, pickle, numpy as np
sys.path.insert(0, '.')

from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

EPISODES = 50
MAX_STEPS = 150

def evaluate_agent(agent, episodes=EPISODES):
    rewards = []
    deliveries = []

    for ep in range(episodes):
        env = GridEnv(get_task())
        env.reset(mode="expert")

        total_r = 0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            actions = [
                agent.get_action(env.get_elite_state(i), agent_idx=i)
                for i in range(2)
            ]
            _, r_list, done, info = env.step(actions)
            total_r += sum(r_list)
            steps += 1

        rewards.append(total_r)
        deliveries.append(info["total_deliveries"])

    return np.mean(rewards), np.mean(deliveries)


def test_learning_progress():
    print("\n📈 LEARNING PROGRESS TEST")
    print("="*50)

    model_path = "models/asymmetric_v15_q_table.pkl"
    with open(model_path, "rb") as f:
        q_table = pickle.load(f)

    # ❌ RANDOM AGENT
    random_agent = QLearningAgent(
        actions=[0,1,2,3,4,5,6],
        shared_q_table={},  # empty
        epsilon=1.0
    )

    # ✅ TRAINED AGENT
    trained_agent = QLearningAgent(
        actions=[0,1,2,3,4,5,6],
        shared_q_table=q_table,
        epsilon=0
    )

    r_reward, r_del = evaluate_agent(random_agent)
    t_reward, t_del = evaluate_agent(trained_agent)

    print(f"RANDOM   → Reward: {r_reward:.2f} | Deliveries: {r_del:.2f}")
    print(f"TRAINED  → Reward: {t_reward:.2f} | Deliveries: {t_del:.2f}")

    # Validation
    if t_del <= r_del:
        print("❌ Agent did NOT learn (deliveries not improved)")
        sys.exit(1)
    if t_reward <= r_reward:
        print("❌ Reward did NOT improve")
        sys.exit(1)

    print("✅ LEARNING VERIFIED (Reward + Deliveries Improved)")


if __name__ == "__main__":
    test_learning_progress()

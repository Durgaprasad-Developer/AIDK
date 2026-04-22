import sys, os, pickle, random
import numpy as np
sys.path.insert(0, '.')

from env.core.environment import GridEnv
from env.tasks.easy import get_task
from env.agents.q_learning_agent import QLearningAgent

EPISODES = 30
MAX_STEPS = 150

def run_episode(env, policy_fn):
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        actions = policy_fn(env)
        _, rewards, done, info = env.step(actions)
        total_reward += sum(rewards)
        steps += 1

    return total_reward, info["total_deliveries"]


# ------------------ POLICIES ------------------

def random_policy(env):
    return [random.randint(0,6) for _ in range(2)]

def idle_policy(env):
    return [0, 0]

def oscillation_policy(env):
    # left-right spam using step_count
    return [env.step_count % 2, env.step_count % 2]


def expert_policy(agent):
    def policy(env):
        return [
            agent.get_action(env.get_elite_state(i), agent_idx=i)
            for i in range(2)
        ]
    return policy


# ------------------ TEST ------------------

def test_reward_hacking():
    print("\n🛡️ REWARD HACKING TEST")
    print("="*50)

    model_path = "models/asymmetric_v15_q_table.pkl"
    if not os.path.exists(model_path):
        print("❌ Missing trained model")
        return

    with open(model_path, "rb") as f:
        q_table = pickle.load(f)

    agent = QLearningAgent(
        actions=[0,1,2,3,4,5,6],
        shared_q_table=q_table,
        epsilon=0
    )

    policies = {
        "RANDOM": random_policy,
        "IDLE": idle_policy,
        "OSCILLATION": oscillation_policy,
        "EXPERT": expert_policy(agent)
    }

    results = {}

    for name, policy in policies.items():
        rewards = []
        deliveries = []

        for _ in range(EPISODES):
            env = GridEnv(get_task())
            env.reset(mode="expert")

            r, d = run_episode(env, policy)
            rewards.append(r)
            deliveries.append(d)

        avg_r = np.mean(rewards)
        avg_d = np.mean(deliveries)

        results[name] = (avg_r, avg_d)

        print(f"{name:<12} | Reward: {avg_r:>7.2f} | Deliveries: {avg_d:>4.2f}")

    # ---------------- VALIDATION ----------------

    exp_r, exp_d = results["EXPERT"]
    rand_r, rand_d = results["RANDOM"]
    idle_r, idle_d = results["IDLE"]
    osc_r, osc_d = results["OSCILLATION"]

    # ✅ 1. Expert must be best at task
    assert exp_d > rand_d, "Expert not better than random"
    assert exp_d > osc_d, "Oscillation achieving deliveries (BAD)"
    assert exp_d > idle_d, "Idle achieving deliveries (BAD)"

    # ✅ 2. Bad policies must NOT exploit reward
    assert idle_r <= 0, "Idle policy getting positive reward (BAD)"
    assert osc_r <= 0, "Oscillation exploiting reward (BAD)"

    print("\n✅ REWARD SYSTEM IS ROBUST (NO EXPLOITATION)")


if __name__ == "__main__":
    test_reward_hacking()

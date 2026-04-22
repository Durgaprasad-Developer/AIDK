import sys, os
sys.path.insert(0, '.')

from env.core.environment import GridEnv
from env.tasks.easy import get_task

def test_env_core():
    print("🔍 Testing CORE ENV (no wrapper)...")

    env = GridEnv(get_task())
    obs = env.reset(seed=1)

    assert isinstance(obs, list), "Reset must return list"

    for _ in range(10):
        actions = [0, 1]
        obs, rewards, done, info = env.step(actions)

        # In modernized environment, rewards are a list of floats
        assert isinstance(rewards, list), "Rewards must be list"
        assert isinstance(info, dict), "Info must be dict"
        assert "total_deliveries" in info

    print("✅ CORE ENV OK")

def test_expert_vs_train_mode():
    print("🔍 Testing BOTH MODES...")

    env = GridEnv(get_task())

    # mode="expert" ensures deterministic static map
    obs1 = env.reset(mode="expert", seed=1)
    
    # mode="train" (or default) allows dynamic modernization randomization
    obs2 = env.reset(mode="train", seed=1)

    # obs are the starting positions; with same seed 1, expert and train should yield different world-states due to randomization inclusion
    assert obs1 != obs2, "Expert & Train mode should differ in dynamic environment"

    print("✅ MODE SEPARATION OK")

if __name__ == "__main__":
    test_env_core()
    test_expert_vs_train_mode()
    print("🏁 REGRESSION PASSED")

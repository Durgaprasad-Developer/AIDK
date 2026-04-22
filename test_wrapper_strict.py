import sys
sys.path.insert(0, '.')

from env.openenv_wrapper import AIDKEnv

def test_wrapper_format():
    print("🔍 Testing WRAPPER FORMAT...")

    env = AIDKEnv()
    res = env.reset(seed=1)

    assert "observation" in res

    step = env.step([0,1])

    assert "observation" in step
    assert "reward" in step
    assert "done" in step
    assert "info" in step

    # OpenEnv requires total reward as float
    assert isinstance(step["reward"], float), f"Expected float reward, got {type(step['reward'])}"

    print("✅ WRAPPER FORMAT OK")

if __name__ == "__main__":
    test_wrapper_format()

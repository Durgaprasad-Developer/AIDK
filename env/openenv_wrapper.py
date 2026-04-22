# env/openenv_wrapper.py
import os, sys
from typing import Any, Optional

# 🧠 Add root to path
sys.path.insert(0, os.getcwd())

from openenv.core.env_server.interfaces import Environment
from env.core.environment import GridEnv
from env.tasks.easy import get_task

class AIDKEnv(Environment):
    """
    AIDK Judge-Compliant OpenEnv Wrapper.
    Wraps the V15 High-Performance GridEnv Kernel.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_env = GridEnv(get_task())

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any):
        """Reset the core V15 kernel."""
        obs = self.grid_env.reset(seed=seed)
        return {
            "observation": obs
        }

    def step(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any):
        """Step the core V15 kernel with 2-agent action list."""
        obs, rewards, done, info = self.grid_env.step(action)
        
        # 🛡️ Reward Verification (Meta Compliance)
        total_reward = float(sum(rewards))
        assert isinstance(total_reward, float), "Total reward must be a float"
        assert -100 <= total_reward <= 100, f"Reward out of bounds: {total_reward}"

        return {
            "observation": obs,
            "reward": float(sum(rewards)),
            "done": bool(done),
            "info": info
        }

    @property
    def state(self) -> Any:
        """Expose the Elite 12-Element State Vector for Agent 0."""
        return {
            "agent_0_state": self.grid_env.get_elite_state(0)
        }

if __name__ == "__main__":
    # Internal Validation
    wrapper = AIDKEnv()
    res_reset = wrapper.reset(seed=1)
    print(f"AIDKEnv Reset Successful. Initial Obs: {res_reset}")
    
    print(f"Current State Check: {wrapper.state}")
    
    res_step = wrapper.step([1, 1])
    print(f"Step Result: Done={res_step['done']}, Delivs={res_step['info']['total_deliveries']}")

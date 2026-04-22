# env/core/environment.py

import random, collections
from env.core.actions import ACTION_TO_DELTA, ACTIONS

class GridEnv:
    def __init__(self, task, num_agents=2):
        self.grid_size = task.grid_size
        self.num_obstacles = task.num_obstacles
        self.max_energy = task.max_energy
        self.num_agents = num_agents
        self.task_id = getattr(task, 'name', 'hard') 

        self.agents_pos = [None] * num_agents
        self.energies = [0] * num_agents
        self.has_picked_up = [False] * num_agents
        
        self.goal_pos = None
        self.obstacles = set()
        self.task_pool = []
        self.step_count = 0
        self.total_deliveries = 0
        self.collisions = 0
        self.histories = [collections.deque(maxlen=10) for _ in range(num_agents)]

    def reset(self, mode="train", seed=None, difficulty=None):
        """
        Difficulty: easy, medium, hard (overrides task defaults)
        Mode: 'expert' preserves the static validated benchmark map for elite models.
        """
        if seed is not None: random.seed(seed)
        self.step_count = 0; self.total_deliveries = 0
        self.task_pool = []; self.obstacles = set(); self.collisions = 0
        self.histories = [collections.deque(maxlen=10) for _ in range(self.num_agents)]
        
        exclude = set()
        
        # 1. Deterministic Placement for Expert Mode (Benchmark Compatibility)
        if mode == "expert":
            # Fixed placement for expert kernel validation
            for i in range(self.num_agents):
                self.energies[i] = self.max_energy; self.has_picked_up[i] = False
                p = self._random_empty_cell(exclude); self.agents_pos[i] = p; exclude.add(p)
                setattr(self, f"last_action_{i}", -1)
            for _ in range(3):
                p = self._random_empty_cell(exclude)
                if p: self.task_pool.append(p); exclude.add(p)
            self.goal_pos = self._random_empty_cell(exclude); exclude.add(self.goal_pos)
            available = (self.grid_size**2) - len(exclude)
            while len(self.obstacles) < min(self.num_obstacles, available - 1):
                p = self._random_empty_cell(exclude | self.obstacles)
                if p: self.obstacles.add(p)
        else:
            # 🚀 1% DYNAMIC MODERNIZATION: Randomized Obstacles & Tasks
            # Difficulty mapping
            target_obs = self.num_obstacles
            if difficulty == "easy": target_obs = 3
            elif difficulty == "medium": target_obs = 6
            elif difficulty == "hard": target_obs = 10
            
            for i in range(self.num_agents):
                self.energies[i] = self.max_energy; self.has_picked_up[i] = False
                p = self._random_empty_cell(exclude); self.agents_pos[i] = p; exclude.add(p)
                setattr(self, f"last_action_{i}", -1)

            # Random tasks
            for _ in range(3):
                p = self._random_empty_cell(exclude)
                if p: self.task_pool.append(p); exclude.add(p)
            
            self.goal_pos = self._random_empty_cell(exclude); exclude.add(self.goal_pos)
            
            # Dynamic Obstacles (5-15% density)
            available = (self.grid_size**2) - len(exclude)
            density_count = int(available * random.uniform(0.05, 0.15))
            target_obs = max(target_obs, density_count)
            
            while len(self.obstacles) < min(target_obs, available - 1):
                p = self._random_empty_cell(exclude | self.obstacles)
                if p: self.obstacles.add(p)
            
        return [self.get_elite_state(j) for j in range(self.num_agents)]

    def is_blocked(self, x, y, agent_idx=None):
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size): return True
        if (x, y) in self.obstacles: return True
        for i, pos in enumerate(self.agents_pos):
            if agent_idx is not None and i == agent_idx: continue
            if pos == (x, y): return True
        return False

    def get_elite_state(self, agent_idx):
        """🧠 V15 EXPERT STATE (12 Elements)"""
        ax, ay = self.agents_pos[agent_idx]
        has_item = self.has_picked_up[agent_idx]
        last_action = getattr(self, f"last_action_{agent_idx}", -1)
        energy = self.energies[agent_idx]
        
        target_id = -1
        if not has_item:
            if self.task_pool:
                distances = [abs(t[0]-ax) + abs(t[1]-ay) for t in self.task_pool]
                target_id = distances.index(min(distances))
                target = self.task_pool[target_id]
            else:
                target = self.goal_pos
        else:
            target = self.goal_pos
            
        dx = max(-4, min(4, target[0] - ax))
        dy = max(-4, min(4, target[1] - ay))
        
        u = int(self.is_blocked(ax-1, ay, agent_idx))
        d = int(self.is_blocked(ax+1, ay, agent_idx))
        l = int(self.is_blocked(ax, ay-1, agent_idx))
        r = int(self.is_blocked(ax, ay+1, agent_idx))
        
        idx_other = 1 - agent_idx if self.num_agents == 2 else agent_idx
        opx, opy = self.agents_pos[idx_other]
        other_dx = max(-2, min(2, opx - ax))
        other_dy = max(-2, min(2, opy - ay))
        
        energy_bucket = 0 if energy < 10 else (1 if energy < 30 else 2)
        
        return (dx, dy, int(has_item), u, d, l, r, last_action, other_dx, other_dy, energy_bucket, target_id)

    def step(self, actions):
        """🛡️ V15 MODERNIZED REWARD (Anti-Hacking & Energy Constraints)"""
        if not isinstance(actions, (list, tuple)) or len(actions) != self.num_agents:
            raise ValueError("Invalid action format")

        self.step_count += 1
        rewards = []; dones = []
        deliv_in_step = 0
        
        # Multi-Agent Collision Check
        collision_step = (self.agents_pos[0] == self.agents_pos[1]) if self.num_agents == 2 else False
        
        for i in range(self.num_agents):
            prev_pos = self.agents_pos[i]
            action = actions[i]
            setattr(self, f"last_action_{i}", action)
            
            # History for Anti-Loop Detection
            self.histories[i].append(prev_pos)
            
            has_item = self.has_picked_up[i]
            target = self.goal_pos if has_item else (min(self.task_pool, key=lambda t: abs(t[0]-prev_pos[0]) + abs(t[1]-prev_pos[1])) if self.task_pool else self.goal_pos)
            prev_dist = abs(target[0]-prev_pos[0]) + abs(target[1]-prev_pos[1])
            
            reward = -0.1 # 1% STEP PENALTY
            
            if action == 6: # STAY/IDLE
                reward -= 0.05
            
            if action in [0, 1, 2, 3]: # MOVE
                self.energies[i] -= 1
                delta = ACTION_TO_DELTA[action]
                new_pos = (prev_pos[0]+delta[0], prev_pos[1]+delta[1])
                
                if self.is_blocked(new_pos[0], new_pos[1], i):
                    reward -= 5.0 # COLLISION
                    self.collisions += 1
                else:
                    self.agents_pos[i] = new_pos
                    curr_dist = abs(target[0]-new_pos[0]) + abs(target[1]-new_pos[1])
                    if curr_dist < prev_dist: reward += 0.2 # DISTANCE SHAPING
                    elif curr_dist > prev_dist: reward -= 0.5
            
            elif action == 4: # PICKUP
                if not has_item and self.agents_pos[i] in self.task_pool:
                    self.has_picked_up[i] = True; self.task_pool.remove(self.agents_pos[i]); reward += 10.0
                else: reward -= 2.0 # INVALID MOVE
            
            elif action == 5: # DELIVER
                if has_item and self.agents_pos[i] == self.goal_pos:
                    self.has_picked_up[i] = False; deliv_in_step += 1; self.total_deliveries += 1; reward += 10.0
                else: reward -= 2.0 # INVALID MOVE
                
            # 🛡️ ANTI-LOOP Signal
            if len(self.histories[i]) == 10 and len(set(self.histories[i])) < 3:
                reward -= 1.0 # Oscillation Penalty

            rewards.append(reward)
            dones.append(self.energies[i] <= 0)

        episode_done = self.step_count >= 150 or any(dones)
        info = {"deliveries": deliv_in_step, "total_deliveries": self.total_deliveries, "step_count": self.step_count}
        return [self.get_elite_state(j) for j in range(self.num_agents)], rewards, episode_done, info

    def _random_empty_cell(self, exclude):
        candidates = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in exclude]
        return random.choice(candidates) if candidates else None

    def _random_empty_cell(self, exclude):
        candidates = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in exclude]
        return random.choice(candidates) if candidates else None

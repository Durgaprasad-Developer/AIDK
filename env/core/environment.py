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

    def reset(self, mode="train", seed=None):
        if seed is not None: random.seed(seed)
        self.step_count = 0; self.total_deliveries = 0
        self.task_pool = []; self.obstacles = set(); self.collisions = 0
        
        exclude = set()
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
            
        return [self.get_elite_state(j) for j in range(self.num_agents)]

    def is_blocked(self, x, y, agent_idx=None):
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size): return True
        if (x, y) in self.obstacles: return True
        for i, pos in enumerate(self.agents_pos):
            if agent_idx is not None and i == agent_idx: continue
            if pos == (x, y): return True
        return False

    def get_elite_state(self, agent_idx):
        """🧠 V14 WINNING STATE (11 Elements)"""
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
        """🛡️ V14 WINNING REWARD (Anti-Mimicry + Hyper Penalty)"""
        self.step_count += 1
        rewards = []; dones = []
        deliv_in_step = 0
        
        dist_between = abs(self.agents_pos[0][0]-self.agents_pos[1][0]) + abs(self.agents_pos[0][1]-self.agents_pos[1][1]) if self.num_agents == 2 else 99
        
        for i in range(self.num_agents):
            prev_pos = self.agents_pos[i]
            action = actions[i]
            setattr(self, f"last_action_{i}", action)
            
            has_item = self.has_picked_up[i]
            if not has_item:
                target = min(self.task_pool, key=lambda t: abs(t[0]-prev_pos[0]) + abs(t[1]-prev_pos[1])) if self.task_pool else self.goal_pos
            else:
                target = self.goal_pos
            prev_dist = abs(target[0]-prev_pos[0]) + abs(target[1]-prev_pos[1])
            
            reward = -0.05 
            
            # 🛡️ FIX 1: Action Mimicry Penalty (Break [0,0] loops)
            if self.num_agents == 2 and actions[0] == actions[1]:
                reward -= 0.3
            
            # 🛡️ FIX 2: Strengthened Coordination Penalties
            penalty = -4.0 # Default (Hard)
            if self.task_id == "easy": penalty = -1.5
            elif self.task_id == "medium": penalty = -3.5
            
            if dist_between <= 1: reward += penalty
            
            if action in [0, 1, 2, 3]: # Move
                self.energies[i] -= 1
                delta = ACTION_TO_DELTA[action]
                new_pos = (prev_pos[0]+delta[0], prev_pos[1]+delta[1])
                
                if self.is_blocked(new_pos[0], new_pos[1], i):
                    reward -= 5.0 
                    self.collisions += 1
                else:
                    self.agents_pos[i] = new_pos
                    curr_dist = abs(target[0]-new_pos[0]) + abs(target[1]-new_pos[1])
                    if curr_dist < prev_dist: reward += 0.2
                    elif curr_dist > prev_dist: reward -= 0.5
                    if prev_dist == curr_dist: reward -= 0.2
            
            elif action == 4: # PICKUP
                if not has_item and self.agents_pos[i] in self.task_pool:
                    self.has_picked_up[i] = True; self.task_pool.remove(self.agents_pos[i]); reward += 10.0
                else: reward -= 10.0
            
            elif action == 5: # DELIVER
                if has_item and self.agents_pos[i] == self.goal_pos:
                    self.has_picked_up[i] = False; deliv_in_step += 1; self.total_deliveries += 1; reward += 80.0
                else: reward -= 10.0
                
            elif action == 6: # RECHARGE
                self.energies[i] = min(self.max_energy, self.energies[i] + 10)
                if self.energies[i] < 15: reward += 2.0
                else: reward -= 1.0 
            
            rewards.append(reward)
            dones.append(self.energies[i] <= 0)

        episode_done = self.step_count >= self.max_energy or all(dones)
        info = {"deliveries": deliv_in_step, "total_deliveries": self.total_deliveries, "step_count": self.step_count, "collision_rate": self.collisions / self.step_count}
        return [self.get_elite_state(j) for j in range(self.num_agents)], rewards, episode_done, info

    def _random_empty_cell(self, exclude):
        candidates = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in exclude]
        return random.choice(candidates) if candidates else None

import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, shared_q_table=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = shared_q_table if shared_q_table is not None else {}

    def get_action(self, state, agent_idx=0):
        """🧠 V15 ASYMMETRIC SELECTION (Zero Sync By Design)"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        vals = [self.q_table.get((state, a), 0.0) for a in self.actions]
        max_val = max(vals)
        best_actions = [i for i, v in enumerate(vals) if v == max_val]
        
        # 🧪 FIX: TINY ASYMMETRY (Deterministic separation)
        # Instead of best_actions[0], we offset by agent index.
        # This ensures agents never pick the same identical index by design if ties exist.
        return best_actions[agent_idx % len(best_actions)]

    def update(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action), 0.0)
        next_max_q = max([self.q_table.get((next_state, a), 0.0) for a in self.actions])
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self, decay_rate):
        self.epsilon *= decay_rate

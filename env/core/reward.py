def compute_reward(prev_dist, current_dist, action_type, success=False, collision=False):
    """
    MANDATORY HACKATHON REWARD MODEL (AIDK-SPEC)
    - move closer: +0.5
    - move away: -0.3
    - pickup success: +10
    - deliver success: +70
    - invalid action: -10
    - collision: -5
    - step cost: -0.05
    """
    reward = -0.05 # Baseline step cost
    
    if action_type == "move":
        if collision:
            reward -= 5.0
        else:
            if current_dist < prev_dist:
                reward += 0.5
            else:
                reward -= 0.3
                
    elif action_type == "pickup":
        if success: reward += 10.0
        else: reward -= 10.0
        
    elif action_type == "deliver":
        if success: reward += 70.0
        else: reward -= 10.0
        
    elif action_type == "recharge":
        if success: reward += 5.0
        else: reward -= 10.0
        
    return float(reward)
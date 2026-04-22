def grade_episode(fulfilled, crashed, steps, optimal_steps, energy_left, max_energy):
    """
    SCIENTIFIC GRADER (Round 2 compliant).
    Ensures 0.05 - 0.95 range.
    Only successful if FULFILLED (Pickup + Delivery + Verification).
    """
    if crashed or energy_left <= 0:
        return 0.05 # Baseline failure score
    
    if not fulfilled:
        return 0.10 # Partial score for survival but mission failure
    
    # Success Scaling
    # 70% Speed Efficiency, 30% Energy Conservation
    speed_eff = min(1.0, optimal_steps / max(steps, 1))
    energy_eff = energy_left / max_energy
    
    raw_score = (0.7 * speed_eff) + (0.3 * energy_eff)
    
    # Map to [0.2, 0.95] to distinguish from failure
    return 0.2 + (raw_score * 0.75)
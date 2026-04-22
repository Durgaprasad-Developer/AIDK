# env/grading/metrics.py

def compute_efficiency(optimal_steps, actual_steps):
    if actual_steps == 0:
        return 0.0
    return optimal_steps / actual_steps
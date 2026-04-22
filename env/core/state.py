def create_state(agent_pos, goal_pos, energy, ascii_grid=""):
    return {
        "agent": agent_pos,
        "goal": goal_pos,
        "energy": energy,
        "ascii_grid": ascii_grid,
    }
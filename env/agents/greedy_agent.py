def greedy_action(state):
    ax, ay = state["agent"]
    gx, gy = state["goal"]

    if ax < gx:
        return 1
    elif ax > gx:
        return 0
    elif ay < gy:
        return 3
    elif ay > gy:
        return 2

    return 0
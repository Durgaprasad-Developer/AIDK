from env.core.environment import GridEnv
from env.agents.greedy_agent import greedy_action
from env.grading.grader import grade_episode

from env.tasks import TASK_REGISTRY

def run_task(task_fn, name):
    print(f"\nRunning {name}...")

    task = task_fn()
    env = GridEnv(task)

    state = env.reset()

    initial_agent = state["agent"]
    goal = state["goal"]

    done = False
    steps = 0

    reached = False
    crashed = False

    while not done:
        action = greedy_action(state)
        state, reward, done, info = env.step(action)

        steps += 1
        reached = info["reached"]
        crashed = info["crashed"]

    optimal_steps = abs(initial_agent[0] - goal[0]) + abs(initial_agent[1] - goal[1])

    score = grade_episode(
        reached,
        crashed,
        steps,
        optimal_steps,
        state["agent"],
        goal
    )

    print(f"{name} → Steps: {steps}, Score: {round(score, 2)}")


if __name__ == "__main__":
    for name, task_fn in TASK_REGISTRY.items():
        run_task(task_fn, name.capitalize())
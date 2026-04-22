from env.tasks.easy import get_task as easy_task
from env.tasks.medium import get_task as medium_task
from env.tasks.hard import get_task as hard_task
from env.tasks.extreme import get_task as extreme_task

TASK_REGISTRY = {
    "easy": easy_task,
    "medium": medium_task,
    "hard": hard_task,
    "extreme": extreme_task,
}

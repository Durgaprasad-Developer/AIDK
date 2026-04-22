class BaseTask:
    def __init__(self, grid_size, num_obstacles, max_energy, grader_id="default"):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.max_energy = max_energy
        self.grader_id = grader_id
---
title: AIDK Navigation Env
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# AIDK — Autonomous Industrial Decision Kernel

## Summary
AIDK is a multi-agent reinforcement learning environment designed to study long-horizon planning and coordination under constraints. It simulates a warehouse logistics scenario where agents must cooperate to fulfill delivery tasks while managing resource constraints and avoiding collisions.

## System Architecture

- **Environment**: Grid-based warehouse simulation
- **Agents**: Tabular Q-learning (asymmetric multi-agent coordination)
- **Knowledge Base**: ~968K learned state-action pairs
- **Action Space**: 7 discrete actions per agent
- **Interface**: OpenEnv compliant wrapper
- **Deployment**: FastAPI + Docker

### Separation of Concerns
- `env/`: Core simulation kernel and task definitions
- `agents/`: Learning logic and policy implementations
- `server/`: API interface and deployment configuration

## Environment Specifications

AIDK abstracts real-world industrial logistics into a learnable framework:
- **Agents**: 2 autonomous actors with partial observability
- **Task**: Stochastic pickup and delivery sequences
- **Constraints**: 150-step horizon, collision penalties, and energy management
- **State Representation**: 12-dimensional optimized state vector

## Real-World Applications

AIDK abstracts real-world coordination problems into a reproducible RL framework:
- **Warehouse Robotics**: Multi-agent routing and automated delivery optimization.
- **Autonomous Fleet Coordination**: Path planning for drones and AGVs in shared spaces.
- **Supply Chain Optimization**: Managing throughput under physical resource constraints.
- **Task Scheduling**: Decentralized decision-making in multi-agent industrial systems.

This environment generalizes to any multi-agent system requiring coordination under resource constraints.

## Reward Design

The reward kernel is designed to discourage shortcut behavior and ensure policy stability:
- **+Delivery Completion**: Positive reward for successful task fulfillment.
- **-Step Penalty**: Encourages temporal efficiency.
- **-Collision Penalty**: Discourages unsafe agent interactions.
- **-Energy Penalty**: Enforces resource awareness.

### Constraints & Safeguards
- **State Isolation**: Agents only access local observations (no global state bypass).
- **Hard Horizon**: Episode termination strictly enforced at 150 steps.
- **Deterministic Transitions**: Ensures learning reflects actual environment outcomes.

This structure prevents reward hacking and ensures improvement reflects genuine task proficiency.

## Performance Benchmarks

### Evaluation Results (Expert V15)

| Policy | Average Deliveries (5-Seed Avg) |
| :--- | :--- |
| Random Baseline | 0.00 |
| **AIDK Expert Swarm** | **2.60** |

### Learning Curve

![Training Curve](assets/training_curve.png)

This curve represents evaluation performance across deterministic seeds during knowledge acquisition checkpoints. Original variance is expected due to task distribution differences between seeds.

The trend indicates consistent improvement in policy effectiveness as the q-table size increases.

## LLM Integration

AIDK provides a verified interface for LLM-based decision systems:
- **Interface**: LLM → Deterministic Action Mapping → Environment → Reward
- **Compatibility**: Integrated with TRL for alignment experiments
- **Signal**: Weighted ordinal sum mapping ensures robust signal capture from reasoning strings

## Deployment & Reproducibility

### Docker Configuration
This system is containerized to ensure identical behavior across different infrastructures.

```bash
# Build the production image
docker build -t aidk-swarm .

# Run the API server
docker run -p 7860:7860 aidk-swarm

# Standardized Validation
BASE_URL=http://localhost:7860 python validate.py
```

### API Endpoints
- `POST /reset`: Resets the environment to a specific or random seed.
- `POST /step`: Executes actions for both agents and returns the next state, rewards, and info.
- `POST /grader`: Standardized endpoint for automated evaluation.

## Limitations

- **Static Geometry**: The current kernel uses a static grid (no dynamic shelf movement during episodes).
- **Discrete Control**: Agents are restricted to discrete step-based actions.
- **Partial Observability**: Real-world sensors may have higher noise profiles than the simulated local view.

### Future Work
- Integration of dynamic/moving obstacles.
- Continuous action space support via PPO/DDPG layers.
- High-noise environmental sensor modeling.

## Status

- **Compliance**: OpenEnv compliant wrapper
- **Validation**: Benchmark validated (2.60 avg deliveries)
- **Portability**: Fully reproducible via Docker
- **Alignment**: LLM interaction loop verified

System is stable and ready for evaluation.
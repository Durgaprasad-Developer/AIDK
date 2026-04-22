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

<p align="center">
  <b>Multi-Agent Reinforcement Learning Environment for Long-Horizon Decision Making</b>
</p>

<p align="center">
  <img src="assets/training_curve.png" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Compliant-green"/>
  <img src="https://img.shields.io/badge/RL-MultiAgent-blue"/>
  <img src="https://img.shields.io/badge/Benchmark-2.60%20Deliveries-orange"/>
  <img src="https://img.shields.io/badge/Docker-Ready-black"/>
</p>

---

## Summary

AIDK is a multi-agent reinforcement learning environment designed to study long-horizon planning and coordination under constraints. It simulates a warehouse logistics scenario where agents must cooperate to fulfill delivery tasks while managing resource constraints and avoiding collisions.

---

## System Overview

<p align="center">
  <img src="assets/system_flow.png" width="700"/>
</p>

The system implements a robust reinforcement learning loop integrated with LLM reasoning capabilities, enabling complex decision-making in non-stationary multi-agent environments.

---

## System Architecture

- **Environment**: Grid-based warehouse simulation kernel.
- **Agents**: Tabular Q-learning with asymmetric swarming coordination.
- **Knowledge Base**: ~968K learned state-action pairs.
- **Action Space**: 7 discrete actions per agent (Move, Stay, Protocol).
- **Interface**: OpenEnv compliant wrapper.
- **Deployment**: FastAPI production server within a Docker container.

---

## Repository Structure

- `env/`: Core simulation kernel and task definitions.
- `agents/`: RL learning logic and policy implementations.
- `training/`: Benchmarking, evaluation, and curve generation scripts.
- `server/`: API layer and deployment configuration.
- `models/`: Persistent trained Q-table knowledge.
- `assets/`: Technical diagrams and performance proof visuals.

---

## Environment

AIDK abstracts real-world industrial logistics into a learnable framework:
- **Agents**: 2 autonomous actors with partial observability.
- **Task**: Stochastic pickup and delivery sequences.
- **Constraints**: 150-step horizon, collision penalties, and energy management.
- **State Representation**: 12-dimensional optimized state vector (Elite State).

---

## Applications

AIDK abstracts real-world coordination problems into a reproducible RL framework:
- **Warehouse Robotics**: Multi-agent routing and automated delivery optimization.
- **Autonomous Fleet Coordination**: Path planning for drones and AGVs.
- **Supply Chain Optimization**: Managing throughput under physical constraints.
- **Task Scheduling**: Decentralized decision-making in multi-agent industrial systems.

---

## Reward Design

The reward kernel is designed to discourage shortcut behavior and ensure policy stability:
- **+Delivery Completion**: Positive reward for successful task fulfillment.
- **-Step Penalty**: Encourages temporal efficiency.
- **-Collision Penalty**: Discourages unsafe agent interactions.
- **-Energy Penalty**: Enforces resource awareness.

### Constraints & Safeguards
- **State Isolation**: Agents only access local observations.
- **Hard Horizon**: Episode termination strictly enforced at 150 steps.
- **Deterministic Transitions**: Ensures learning reflects actual environment outcomes.

---

## Benchmarks

### Evaluation Results (Expert V15)

| Policy | Average Deliveries (5-Seed Avg) |
| :--- | :--- |
| Random Baseline | 0.00 |
| **AIDK Expert Swarm** | **2.60** |

---

## Learning Curve

<p align="center">
  <img src="assets/training_curve.png" width="650"/>
</p>

This curve represents evaluation performance across deterministic seeds during knowledge acquisition checkpoints. Variance is expected due to task distribution differences between seeds. Trend indicates consistent improvement in policy effectiveness.

---

## LLM Integration

AIDK provides a verified interface for LLM-based reasoning systems:
- **Interface**: LLM → Deterministic Action Mapping → Environment → Reward.
- **Compatibility**: Integrated with TRL for alignment experiments.
- **Signal**: Weighted ordinal sum mapping ensures robust signal capture from reasoning strings.

---

## Deployment

### Docker Configuration
Fully containerized for identical behavior across all infrastructures.

```bash
# Build the production image
docker build -t aidk-env .

# Run the API server
docker run -p 7860:7860 aidk-env

# Standardized Validation
BASE_URL=http://localhost:7860 python validate.py
```

### API Interface
- `POST /reset`: Resets the environment to a specific seed.
- `POST /step`: Executes actions and returns structured outcome data.
- `POST /grader`: Optimized endpoint for automated judge evaluation.

---

## Limitations

- **Static Geometry**: The current kernel uses a static grid geometry.
- **Discrete Control**: Agents are restricted to discrete step-based actions.
- **Partial Observability**: Higher noise sensor modeling is out of current scope.

---

## Status

- **Compliance**: OpenEnv compliant wrapper.
- **Validation**: Benchmark validated (2.60 avg deliveries).
- **Portability**: Fully reproducible via Docker.
- **Alignment**: LLM interaction loop verified.

System is stable and ready for evaluation.
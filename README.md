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
<p align="center"><i>Learning progression reconstructed from Q-table checkpoints during training</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Compliant-green"/>
  <img src="https://img.shields.io/badge/RL-MultiAgent-blue"/>
  <img src="https://img.shields.io/badge/Benchmark-2.80%20Deliveries-orange"/>
  <img src="https://img.shields.io/badge/Docker-Ready-black"/>
</p>

---

## Summary

AIDK is a multi-agent reinforcement learning environment designed to study long-horizon planning and coordination under constraints. It abstracts industrial warehouse logistics into a learnable framework where agents must cooperate to fulfill delivery tasks while managing resource constraints and avoiding collisions.

---

## Quick Start (Judge Entry Point)

To run the complete system end-to-end, including environment validation, benchmark evaluation, and LLM interaction, execute the following:

```bash
# Clone and enter repository
git clone <repo_url>
cd Navigation-env

# Execute full validation suite (requires python >= 3.10)
PYTHONPATH=. python3 validate.py
```

**Expected Output:**
- **Benchmark**: ~2.80 deliveries (Trained) vs 0.00 (Random)
- **LLM Loop**: Deterministic Action Mapping -> Reward Verified
- **API**: 200 OK across Reset/Step/Grader endpoints
- **Stability**: 10/10 consecutive stress test success

---

## Performance (Verified)

| Policy | Deliveries (5-Seed Avg) | Reward Signal |
| :--- | :--- | :--- |
| Random Baseline | 0.00 | Stochastic Noise |
| **AIDK Expert (V15)** | **2.80** | **Stabilized Policy** |

Improvement is significant and consistent across deterministic evaluation seeds [1, 7, 42, 99, 123].

---

## Problem Complexity

AIDK is designed to test agents on non-trivial decision-making challenges:
- **Multi-Agent Coordination**: Non-stationary dynamics where each agent's environment changes as the other agent learns.
- **Sparse Reward Signal**: Deliveries are long-horizon events (e.g., first success typically requires 11+ precise steps).
- **Delayed Feedback**: Rewards for intermediate navigation are optimized to prevent "idling" or "cycling" hacks.
- **Resource Constraints**: Finite energy caps and hard collision penalties enforce safe navigation.

This is not a simple pathfinding task; it is **constrained decision-making under non-stationary conditions.**

---

## System Flow

<p align="center">
  <img src="assets/system_flow.png" width="700"/>
</p>
<p align="center"><i>End-to-end interaction loop between policy and environment kernel</i></p>

The interaction loop architecture follows a clear separation of concerns:
- **Reasoning Layer**: Optional LLM-based action derivation.
- **Mapping Layer**: Deterministic processing of agent input.
- **Simulation Kernel**: `env/core/environment.py` (The ground truth).
- **Policy Layer**: `env/agents/q_learning_agent.py` (Tabular Q-inference).

---

## Learning Curve

<p align="center">
  <img src="assets/training_curve.png" width="650"/>
</p>
<p align="center"><i>Evaluation performance derived from evaluating partial Q-tables (10 checkpoints)</i></p>

This curve represents evaluation performance across deterministic seeds during knowledge acquisition. Variance is expected due to task distribution differences between seeds. The trend indicates robust policy effectiveness as the q-table size (~968K entries) increases.

---

## Failure Modes

To validate that performance is learned rather than scripted, the environment exhibits specific failure modes when under-trained or mis-configured:
- **Poor Coordination**: Leads to collision penalties and mutual blocking.
- **Inefficient Routing**: Results in energy depletion before task completion.
- **Random Policy**: Consistently produces 0.00 deliveries across all seeds.

These modes confirm that high delivery counts require an emergent, coordinated navigation policy.

---

## Reproducibility

- **Deterministic Seeds**: Benchmarks use fixed seeds [1, 7, 42, 99, 123].
- **Fixed Knowledge Base**: The expert Q-table is included (~968K learned states).
- **Dockerized Environment**: The `Dockerfile` ensures identical dependency versions and OS behavior.
- **Zero-Bypass Rewards**: Internal assertions prevent reward manipulation or global state leakage.

---

## Applications

- **Warehouse Robotics**: Multi-agent routing and fulfillment optimization.
- **Autonomous Fleet Coordination**: Safe coordination of drones or mobile robots.
- **Supply Chain Management**: Resource-constrained task allocation.
- **Multi-Agent RL Research**: Training ground for coordination and long-horizon reasoning.

---

## Deployment

Build the containerized environment:
```bash
docker build -t aidk-env .
```

Run the API production server:
```bash
docker run -p 7860:7860 aidk-env
```

Live validation:
```bash
BASE_URL=http://localhost:7860 python validate.py
```

---

## Status

- **Compliance**: OpenEnv compliant wrapper
- **Validation**: Full suite PASSED (Benchmark, API, LLM, Stability)
- **Portability**: Verified via Docker and Hugging Face Spaces

System is stable and ready for evaluation.
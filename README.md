---
title: AIDK Navigation Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AIDK — Autonomous Industrial Decision Kernel (V15)

AIDK is a Multi-Agent Reinforcement Learning system designed to simulate warehouse navigation under real-world constraints such as obstacles, energy limits, and coordination challenges.

The goal is to study how multiple agents learn navigation, avoid collisions, and complete pickup-delivery tasks efficiently.

---

## 🚀 Core Idea

AIDK focuses on **learning-based decision making**, not rule-based systems.

* Agents learn from interaction (Q-learning)
* No hardcoded paths or heuristics
* Behavior emerges from rewards and state representation

---

## 🧠 Key Design

### 1. State Representation

Each agent observes:

* Relative target position (dx, dy)
* Obstacle proximity (4 directions)
* Whether carrying item
* Last action (loop awareness)
* Nearby agent information

---

### 2. Learning Method

* Tabular Q-learning
* Shared Q-table across agents
* Curriculum training (easy → medium → hard)

---

### 3. Multi-Agent Behavior

* Agents act independently
* Shared learning enables knowledge transfer
* Asymmetric action selection avoids identical behavior

---

### 4. Reward Design

Encourages:

* Moving toward goals
* Successful pickup & delivery
* Avoiding collisions
* Efficient movement

---

## ⚙️ How to Run

### Start Server

```bash
PYTHONPATH=. python3 server/app.py
```

---

### Run Inference

```bash
PYTHONPATH=. python3 inference/inference_v15.py
```

---

### Train Model

```bash
PYTHONPATH=. python3 training/train_v15.py
```

---

## 📊 What to Expect

* Agents learn navigation over time
* Reduced collisions after training
* Increasing delivery success

---

## 🧪 OpenEnv Compatibility

* reset / step API implemented
* Multi-agent support
* Compatible with validator

---

## 🧩 What Makes It Interesting

* Multi-agent RL in constrained grid world
* Emergent coordination (not hardcoded)
* Handles obstacles + energy + interaction

---

## ⚠️ Notes

This is a learning system, not a deterministic planner.

Performance depends on:

* training duration
* reward design
* state representation

---

## 🏁 Summary

AIDK demonstrates how simple RL agents can learn structured behavior in a shared environment without explicit coordination rules.
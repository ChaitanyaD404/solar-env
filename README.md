# 🌞 Solar Energy Optimizer – OpenEnv RL Environment

## 📌 Overview

Solar Energy Optimizer is a **real-world reinforcement learning (RL) environment** built using the OpenEnv framework. It simulates a small solar farm where an AI agent must make decisions to maximize energy output under changing environmental conditions.

This project models real-world challenges like:

* Dust accumulation on panels
* Weather variability (sunlight changes)
* Battery storage management

The goal is to train and evaluate agents that can **optimize clean energy production efficiently**.

---

## 🎯 Motivation

In real solar farms (especially in regions like India), energy output is affected by:

* Dust buildup
* Suboptimal panel positioning
* Poor battery usage

This environment provides a simplified but realistic simulation to:

* Train AI agents
* Evaluate decision-making strategies
* Optimize renewable energy systems

---

## ⚙️ Environment Design

### 🧠 Observation Space

Each state includes:

```python
dirt: List[float]   # Dirt level on panels (0–1)
sun: float          # Sun intensity (0–1)
battery: float      # Battery level (0–100)
hour: int           # Time step (0–23)
```

---

### 🎮 Action Space

Agent can perform:

* `clean_1`, `clean_2`, `clean_3` → Clean panels
* `charge` → Store energy in battery
* `discharge` → Use battery energy
* `noop` → Do nothing

---

### 🔁 Environment API

* `reset()` → Initializes environment
* `step(action)` → Executes action and returns:

  * next state
  * reward
  * done
* `state()` → Returns current state

---

## 🏆 Tasks & Difficulty Levels

### 🟢 Easy

* Basic energy optimization
* Focus on cleaning and simple battery usage

### 🟡 Medium

* Introduces fluctuating sunlight
* Requires smarter battery management

### 🔴 Hard

* Complex trade-offs between:

  * cleaning cost
  * battery usage
  * energy output

---

## 📊 Reward Function

The reward is designed to reflect real-world trade-offs:

* ✅ Positive:

  * Energy generated
* ❌ Negative:

  * Cleaning cost
  * Battery overuse

👉 Provides **continuous feedback (not just final reward)**

---

## 🧪 Grading System

Each task is evaluated using a deterministic grader:

```python
score ∈ [0.0, 1.0]
```

* Based on total energy efficiency
* Compared to maximum achievable output

---

## 🤖 Baseline Inference

The project includes `inference.py` which:

* Uses OpenAI client for decision-making
* Runs the agent through the environment
* Outputs structured logs:

```
[START]
[STEP]
[END]
```

* Produces reproducible scores

---

## 🚀 Deployment

This environment is deployed on Hugging Face Spaces using Docker.

🔗 **Live URL**:
https://chaitanya2840-solar-env.hf.space

---

## 🐳 Docker Support

The project includes a Dockerfile for containerized execution.

To build locally:

```bash
docker build -t solar-env .
docker run -p 8000:8000 solar-env
```

---

## 🧪 API Endpoints

* `/` → Health check
* `/reset` → Reset environment (POST/GET)
* `/state` → Get current state

---

## 🛠️ Setup Instructions

### 1. Clone repo

```bash
git clone https://github.com/<your-username>/solar-env.git
cd solar-env
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run locally

```bash
uvicorn app:app --reload
```

---

### 4. Run inference

```bash
python inference.py
```

---

## 🔐 Environment Variables

```bash
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=your_key_here
HF_TOKEN=your_hf_token
```

---

## 📌 Key Features

* Real-world RL simulation
* OpenEnv compliant
* Structured evaluation system
* Dockerized deployment
* Hugging Face hosting

---

## 🧠 Future Improvements

* Add real weather API integration
* Visual dashboard for panel monitoring
* Advanced RL training support

---

## 👨‍💻 Author

Chaitanya

---

## 📄 License

MIT License

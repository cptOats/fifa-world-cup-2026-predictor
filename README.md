<div align="center">

[![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![Framework: Marimo](https://img.shields.io/badge/UI-Marimo-green.svg)](https://marimo.io/)
[![Deployment: WASM](https://img.shields.io/badge/WebAssembly-Supported-orange.svg)](https://webassembly.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# ⚽ FIFA World Cup 2026 Prediction Engine 🏆

**End-to-end MLOps pipeline and reactive simulation dashboard designed to forecast the 2026 FIFA World Cup.**

<!-- Hero Image -->
[<img width="98.5%" alt="tournament-progression-5x4" src="https://github.com/user-attachments/assets/a8c1bb1f-8bb6-4263-b929-5d8f00ae036b" />](https://cptoats.github.io/fifa-world-cup-2026-predictor/)
<br>

<!-- Additional Images -->
<p align="center">
  <img width="49%" alt="tournament-matches-1x1" src="https://github.com/user-attachments/assets/409bef02-d046-4f8f-8ee5-66d5f8dcc650" />
  <img width="49%" alt="sandbox-1x1" src="https://github.com/user-attachments/assets/5823e653-bc26-4784-a67d-2348f5052776" />
</p>

</div>

---

### ℹ️ Overview

This project is a complete predictive architecture built to model, simulate, and visualize the 2026 FIFA World Cup. 

The predictor features a **Meta-Learner** that blends multiple machine learning models into a cohesive expected goals (xG) framework. Trained on decades of historical international match data with rigorous safeguards against target leakage, this engine predicts individual match outcomes to project teams through the Group & Knockout stages via both deterministic and stochastic pathways. The simulation results are displayed on an interactive front-end dashboard built with Marimo.

### 🌟 Highlights

* **🧠 Multi-Model Ensemble:** Blends Poisson, dynamic Elo ratings, and XGBoost Count Trees using Ridge regression or a Bounded SLSQP solver.
* **🛡️ Leak-Proof Feature Engineering:** Strict Point-in-Time (PiT) data shifting ensures models only ever see historical form vectors *prior* to a match taking place.
* **🎲 Stochastic Multiverse:** Executes 10,000+ Monte Carlo parallel simulations using Bivariate Copula draw inflation to accurately capture low-scoring tournament upsets.
* **⚡ Reactive UI:** A beautiful dashboard built in Marimo; toggle between deterministic and probabilistic logic on the fly to view tournament progression and sandbox matchups.
* **🌐 Edge WASM Deployment:** The front end compiles natively to WebAssembly. Dive into the data now ➡️ [Launch Live Dashboard](https://cptOats.github.io/fifa-world-cup-2026-predictor/).

---

### 🚀 Usage
* **1. Configure the Engine**<br>
Adjust model hyperparameters directly within the _MLOPS FLIGHT CONTROLS_ block of main.py

* **2. Execute the Pipeline**<br>
Retrain models with fresh data, execute the Monte Carlo simulations, and validate results:
```bash
uv run main.py
```
* **3. Visualize the Results**<br>
Unpack your run data in an interactive and immersive python native dashboard.
```bash
uv run marimo run dashboard.py
```

---

### ⬇️ Installation
* [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
* [Git](https://git-scm.com/install/) installed

#### Clone the repository
```bash
git clone https://github.com/cptOats/fifa-world-cup-2026-predictor.git
cd fifa-world-cup-2026-predictor
```

####  Create the environment and install dependencies
```bash
uv sync
```
---

**Daniel John Barlow | Data Scientist & AI Engineer**<br>
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/cptOats) [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/danieljohnbarlow/) 

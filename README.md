# George-Daniel Gherasim

**AI Infrastructure & Systems** — ENSTA Paris, 2nd year

---

### What I'm building

**[ChaosAI](https://github.com/ElMonstroDelBrest/ChaosAI)** — Time-series world model trained from scratch.

- 38M-parameter Mamba-2 JEPA encoder, trained on 838M tokens across 8,969 financial assets
- Full JAX/Flax pipeline: FSQ tokenizer → SSM encoder → OT-CFM stochastic predictor → TD-MPC2 RL agent
- Auto-sharding on TPU v6e clusters (GSPMD, 2D mesh, XLA production flags)
- Data lake: raw parquet → ArrayRecord on GCS, zero idle cost

The core insight: JEPA (Joint Embedding Predictive Architecture) learns *structured latent representations* of relationships and context — not next-token prediction. Same philosophy as knowledge graphs for agents.

---

### Stack

**AI/Compute:** JAX, Flax, Optax, PyTorch, XLA, TPU Pod topology  
**Infra:** GCP, GCS, Grain/ArrayRecord, Docker, FinOps  
**Systems:** Python, C, Bash  

---

📫 [george-daniel.gherasim@ensta.fr](mailto:george-daniel.gherasim@ensta.fr)

# 🧩 Project One-Pager

## **Predictive Maintenance & ML Infrastructure for Distributed SMR Powerplants**


**By:** \[Your Name\]**Role Proposed:** Engineering Lead – Data & Machine Learning, Applied Atomics


---

## 🧭 Why This Project?

Applied Atomics is tackling one of the hardest and most important problems in energy: delivering co-located, carbon-free power to high-intensity industrial applications—**hyperscalers, e-fuels, desalination, hydrogen production**—using a full-stack SMR powerplant architecture.

While much of the industry chases theoretical tech, Applied Atomics is building for *deployment reality*—leveraging existing supply chains, known fuels, and proven Gen III+ light water reactor architecture. The company is moving fast, and building not just the reactor, but the whole plant, with vertical integration across design, testing, and operations.

This project exists to match that ambition on the **data and machine learning side**—anticipating the challenges that come after the physical plant is certified, commissioned, and running across customer sites.


---

## 👋 Why I’m Building This

I’m an ML engineer and mathematician by background, with published research in dynamical systems, an MBA, PMP certification, and years of experience building machine learning pipelines in production settings—particularly in regulated industries (e.g., healthcare). I’ve also worked in IoT startups and cross-functional engineering orgs.



Most ML work today is drifting toward hype cycles: generative AI, LLMs, etc. I’m not interested in that.**I want to build things that need to be right.**Things that operate in the physical world, in critical infrastructure, under regulatory oversight.

This project is not a case study or a toy. It’s a **strategic prototype**: a practical, production-aware system that anticipates what Applied Atomics will need when your reactors are live and producing power—across **multiple sites**, with **distributed telemetry**, **scheduled maintenance**, and eventually, **auditable ML pipelines** that must stand up to inspection.


---

## ⚙️ What the System Is

This is a **full-stack predictive maintenance and ML infrastructure system** built for SMR deployments that are:

* **Modular** (1–10 reactors per site)
* **Co-located** (at customer sites)
* **Vertically integrated** (entire powerplant stack)
* **Data-rich** (via digital twin + real sensors)

It’s designed to support your reactors *once they’re running*—and to do it in a way that reflects your values: **speed, pragmatism, reliability, and ownership.**

### Key Features

* 🏭 **Component-aware synthetic simulator**
  * Simulates realistic telemetry for multiple reactor modules
  * Site-aware: `site_id`, `reactor_id`, `component` identifiers
  * Fault injection and degradation modeling for testing ML pipelines
* ⚙️ **Kafka-based streaming ingestion layer**
  * Designed for real-time ingestion from sensors or digital twin
  * Routes data to storage, inference engine, and monitoring dashboards
* 🤖 **ML models for Remaining Useful Life (RUL)**
  * Built using PyTorch + MLflow
  * gRPC-based inference for speed and traceability
  * All predictions are logged with full context (model version, input, timestamp)
* 📊 **Monitoring dashboards for operators and engineers**
  * Grafana for sensor health, component wear, model predictions
  * Alerting when components show signs of failure or drift
* 🔐 **Compliance-aware audit trail**
  * Every prediction logged and versioned
  * Traceability aligned with NRC regulatory posture (10 CFR Part 50/52)
  * Supports future readiness for system safety classification and digital twin integration
* ☁️ **Cloud-native, reproducible infrastructure**
  * Dockerized, runs locally or on GCP
  * Cloud Run for ML inference, InfluxDB for TS data, Cloud SQL for audit logs

This is **not an academic exercise**. It’s a real system built as if it were being prepared for use in a regulated production environment—*because one day, it will be.*


---

## 🎯 Why I’m Building This

This project is not a portfolio piece. It's a **strategic demonstration of readiness**.

I built it to show how I think, how I design systems, and how I would approach leading data and machine learning efforts in a company like Applied Atomics—where uptime, integration, regulation, and real-world reliability all matter more than academic novelty.

It’s meant to highlight that:

* I bring not just ML expertise, but systems and infrastructure thinking
* I understand the engineering realities of high-integrity hardware environments
* I can integrate with simulation, physics, ops, and compliance teams—not slow them down
* I’ve worked in regulated domains, shipped pipelines to production, and managed technical projects from concept to deployment

And ultimately, it’s meant to open the door for a conversation—about where ML and data fit in the Applied Atomics roadmap, and how I can help build that capability from the ground up.



This is what I want to be working on.This is where I want to do it.And this project is how I’ve chosen to show it.

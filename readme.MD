# 🧠 Pi-HPC-Lab

> **High-Performance Computing on Raspberry Pi Cluster**  
> A lightweight distributed computing lab for mathematics, physics, and AI simulation — built entirely on Raspberry Pi devices.

---

## 🌍 Overview

**Pi-HPC-Lab** is a collection of distributed computation experiments designed for small ARM clusters  
(such as Raspberry Pi 5 or Jetson Orin Nano).  
It demonstrates how fundamental mathematical, physical, and optimization problems can be solved in **parallel** across multiple nodes — using pure Python and lightweight frameworks (e.g., sockets, Dask, NumPy).

This project aims to:
- Build a **mini supercomputer** environment with low-cost hardware.  
- Study **parallel algorithms** for mathematical & physical simulations.  
- Provide reusable code templates for distributed computing education and experimentation.

---

## ⚙️ Cluster Setup

### Hardware Example
| Role | Device | Notes |
|------|---------|-------|
| Master | Raspberry Pi 5 (8GB) | Runs Dask Scheduler or custom socket server |
| Worker 1–3 | Raspberry Pi 5 / Jetson Nano | Execute compute tasks |
| Network | Gigabit Ethernet Switch | Optional PoE power |
| Storage | NVMe SSD / USB3 External | For caching & logs |

### Software Stack
- **Python 3.10+**
- **NumPy**, **Dask**, **Matplotlib**
- **SSH** access between nodes  
- Optional: **Ray**, **MPI4py**, or **FastAPI** for extended experiments

### Install Dependencies
```bash
sudo apt update && sudo apt install -y python3-pip
pip install numpy dask distributed matplotlib
```

## 🧩 Project Structure

pi-hpc-lab/
├── matmul/              # Distributed matrix multiplication
├── linear_regression/   # Distributed gradient descent
├── heat_equation/       # PDE simulation (2D heat diffusion)
├── montecarlo_pi/       # Monte Carlo π estimation
├── fft_image/           # Distributed FFT and image filtering
├── benchmark/           # Performance comparison utilities
└── README.md

# 🌡️ Distributed Heat Equation (2D)

Solve 2D heat diffusion using parallel computation on Raspberry Pi cluster.

---

## 🔬 Mathematical Model
∂T/∂t = α∇²T

## ⚙️ Implementation
- Dask Array `map_overlap` for ghost cell exchange  
- Each worker computes one subgrid block  
- Reflective boundaries (Neumann)

---

## 📊 Parameters
| Variable | Description | Default |
|-----------|-------------|----------|
| nx, ny | Grid size | 400×400 |
| steps | Time steps | 300 |
| alpha | Diffusion coefficient | 0.1 |

---

## 🖼️ Output
- `heat_final.png` → final temperature map  
- `heat_sim.gif` (optional) → time evolution animation

---

## 🧠 Concepts
- PDE numerical solution (finite difference)
- Ghost cell exchange in parallel computing
- Dask `map_overlap` for boundary handling
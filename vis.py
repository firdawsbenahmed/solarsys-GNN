import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from Nbody import make_sys, simulate     
except ImportError:
    from Nbody import make_sys, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--n_planets", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_steps", type=int, default=8000)
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--save_every", type=int, default=20)
ap.add_argument("--trail", type=int, default=60, help="how many past frames to draw as a trail")
ap.add_argument("--gif", default="solar_system.gif")
ap.add_argument("--png", default="orbits.png")
ap.add_argument("--show", action="store_true")
a = ap.parse_args()

# 1) simulate one system
rng = np.random.default_rng(a.seed)
pos, vel, mass = make_sys(rng, a.n_planets)
P, V, _ = simulate(pos, vel, mass, a.n_steps, a.dt, a.eps, a.save_every)   # P: (T, N, 3)
T, N, _ = P.shape ### t= nb of screenshot n = nb of bodies _ for the coordonates 
print(f"simulated {T} snapshots of {N} bodies")


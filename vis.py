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

lim = 1.05 * np.abs(P).max() ### this finds the largest coordinate value anywhere in the data
### and ignore the sign then it adds a margen for the axis ranges so all the orbits fit in the pic 
colors = [to_rgba("gold")] + [plt.cm.tab10(k) for k in range(N - 1)] ### this makes one color per body the palatte is tab10 and the to_rgba turns the name into 4 numbers 
sizes = [250] + [25 + 60 * (mass[k] / mass[1:].max()) for k in range(1, N)] ## this makes one dot size per body 250 for the star and each planet gets 25 plus up to 60 more depending on its mass compared with the heaviest planet

def make_axes(figsize=(7, 7)):
    fig = plt.figure(figsize=figsize) ### a new blank figure 7 by 7 inches
    ax = fig.add_subplot(111, projection="3d") ### this adds one 3d plotting area to it the 111 means 1 row and 1 column and plot number 1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ####Each axis runs from −lim to +lim with the same range on all three so the picture isnt stretched
    ax.set_box_aspect((1, 1, 1)) ### this makes the plot box a cube
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=25, azim=40) ### 25 above the horizontal plane turned 40 around
    return fig, ax

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
 ### full orbits pic
fig, ax = make_axes()
for b in range(N):
    ##### fresh plot then goes through the bodies one by one
    ax.plot(P[:, b, 0], P[:, b, 1], P[:, b, 2], color=colors[b], lw=0.8, alpha=0.8) ### here we drow the path of the body b where the x coord of the body at every time is drawn written as a loop
    ### the three lists are the whole route through space the line with is 0.8 making it lil bit transp
    ax.scatter(*P[-1, b], color=colors[b], s=sizes[b], depthshade=False)
    ### this draws a dot at the body’s final position its lastest coords * works as a separator s is the dot size 
    #### depthshade=False stops matplotlib from fading the color with distance
ax.set_title("Orbits (full trajectories)")
fig.savefig(a.png, dpi=150, bbox_inches="tight")
print("saved", a.png) ## saving the pic 

###animation
fig, ax = make_axes()
## new figure with one dot per body at its time 0 position
scat = ax.scatter(P[0, :, 0], P[0, :, 1], P[0, :, 2], c=colors, s=sizes, depthshade=False)
####### we keep the name scat so we can move the dots later
trails = [ax.plot([], [], [], color=colors[b], lw=1.0, alpha=0.7)[0] for b in range(N)]
###creates one empty line per body to be filled in frame by frame ax.plot returns list of lines and 0 takes the first one
label = ax.text2D(0.03, 0.95, "", transform=ax.transAxes) ###text label for the time

def update(t): ### is called once per frame
    scat._offsets3d = (P[t, :, 0], P[t, :, 1], P[t, :, 2]) ### move all the dots to their positions at time t
    lo = max(0, t - a.trail) ### starting of the tail and the last a.trail frames 60 by default
    for b, line in enumerate(trails):
        ### for each body the trail is drawn through its positions from frame lo up to and including frame t
        line.set_data(P[lo:t + 1, b, 0], P[lo:t + 1, b, 1]) ### adding x and y
        line.set_3d_properties(P[lo:t + 1, b, 2]) ### adding the z 
    label.set_text(f"t = {t * a.dt * a.save_every:.1f}") ### updates the clock with the real simulated time and t is the frame nb * the time btwn the snapshots
    return [scat, label, *trails] ### returns all the things that has changed 

anim = FuncAnimation(fig, update, frames=T, interval=33, blit=False)
#### the set up of the animation and we call the update for every frame fro, 0 to t-1 33 is the delay btwn the time frames in ms the false means the frame is redrawn cmptly 
anim.save(a.gif, writer=PillowWriter(fps=30), dpi=70)
#### pillow lib is the one tht helps with of gif 30 per sec
print("saved", a.gif)

if a.show:
    plt.show()

    ### note use --show to have interractive view 
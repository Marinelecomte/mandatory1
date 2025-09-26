import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from Wave2D import Wave2D_Neumann

mx = 2
my = 3
c  = 1.0

periods_to_show = 2
target_frames = 60

fps      = 10
figsize  = (4, 4)
dpi      = 70
cmap     = "viridis"

def choose_params(mx, my, c, periods_to_show, target_frames):
    M = max(mx, my)
    R = math.sqrt(mx*mx + my*my)
    N = max(32, 8 * M)  
    cfl = 0.5  
    steps_per_period = max(1, int(round(4 * N / R)))
    Nt = max(steps_per_period, int(round(periods_to_show * steps_per_period)))
    store_every = max(1, int(round(Nt / target_frames)))
    return N, Nt, cfl, store_every

def main():
    N, Nt, cfl, store_every = choose_params(mx, my, c, periods_to_show, target_frames)
    solver = Wave2D_Neumann()
    data = solver(N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=store_every)
    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError("Wave2D_Neumann did not return snapshots. "
                           "Call it with store_data > 0 and return {tstep: U}.")

    steps  = sorted(data.keys())
    frames = [data[k] for k in steps]
    vmax = max(abs(frames[0]).max(), abs(frames[-1]).max())
    vmin = -vmax

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(frames[0], vmin=vmin, vmax=vmax, cmap=cmap,
                   origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
    ax.set_title(f"Neumann wave: mx={mx}, my={my}, N={N}, CFL={cfl}")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def update(k):
        im.set_data(frames[k])
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000.0/max(fps,1), blit=True)

    os.makedirs("report", exist_ok=True)
    out_path = "report/neumannwave.gif"
    ani.save(out_path, writer="pillow", fps=fps,
             savefig_kwargs={"dpi": dpi, "bbox_inches": "tight"})
    print(f"Saved {out_path}")
    print(f"Frames in GIF: {len(frames)} | Grid N={N} | Nt={Nt} | store_every={store_every}")

if __name__ == "__main__":
    main()

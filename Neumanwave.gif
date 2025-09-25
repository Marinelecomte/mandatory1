import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from Wave2D import Wave2D.py

N  = 80       
Nt = 120      
cfl = 0.5
c = 1.0
mx = 2
my = 3
store_every = 2 

def main():
    sol = Wave2D_Neumann()
    data = sol(N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=store_every)

    steps  = sorted(data.keys())
    frames = [data[k] for k in steps]

    vmax = max(abs(frames[0]).max(), abs(frames[-1]).max())
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=70)
    im = ax.imshow(frames[0], vmin=vmin, vmax=vmax, cmap="viridis", origin="lower",
                   extent=[0, 1, 0, 1], interpolation="nearest")
    ax.set_title(f"Neumann wave: mx={mx}, my={my}")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    def update(k):
        im.set_data(frames[k])
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)

    os.makedirs("report", exist_ok=True)
    out_path = "report/neumannwave.gif"
    ani.save(out_path, writer="pillow", fps=10,
             savefig_kwargs={"dpi": 70, "bbox_inches": "tight"})
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for GitHub runners
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Wave2D import Wave2D_Neumann

N, Nt, cfl, c, mx, my, store_every = 60, 120, 0.5, 1.0, 2, 3, 2
fps, figsize, dpi, cmap = 10, (4, 4), 70, "viridis"
OUT_PATH = "neumannwave.gif"  # save to repo root

def main(output_path: str = OUT_PATH):
    solver = Wave2D_Neumann()
    data = solver(N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=store_every)
    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError("Expected snapshots {tstep: U}. Call with store_data > 0.")

    steps  = sorted(data.keys())
    frames = [np.asarray(data[k]) for k in steps]

    vmax = max(abs(frames[0]).max(), abs(frames[-1]).max())
    vmin = -vmax

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(frames[0], vmin=vmin, vmax=vmax, cmap=cmap,
                   origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
    ax.set_title(f"Neumann wave: mx={mx}, my={my}, N={N}, CFL={cfl}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(k):
        im.set_data(frames[k])
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000.0/max(fps, 1), blit=True)

    writer = animation.PillowWriter(fps=fps, metadata={"artist": "neumann_movie"})
    ani.save(output_path, writer=writer, dpi=dpi, savefig_kwargs={"bbox_inches": "tight"})
    plt.close(fig)
    size = os.path.getsize(output_path)
    print(f"[ok] Saved {output_path} ({size} bytes)")

if __name__ == "__main__":
    main()

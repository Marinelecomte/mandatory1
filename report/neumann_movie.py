import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Wave2D import Wave2D_Neumann  

N, Nt = 60, 60
cfl, c = 1.0 / np.sqrt(2.0), 1.0
mx, my = 3, 3
store_every = 5
fps = 5
figsize, dpi = (6, 6), 100

OUT_PATH = os.path.join(os.path.dirname(__file__), "neumannwave.gif")


def main(output_path: str = OUT_PATH):
    sol = Wave2D_Neumann()
    results = sol(N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=store_every)

    if not isinstance(results, dict) or len(results) == 0:
        raise RuntimeError("Expected {tstep: U} snapshots; call with store_data > 0.")

    xij, yij = sol.xij, sol.yij

    tsteps = sorted(results.keys())
    frames = [results[k] for k in tsteps]

    amp = max(float(np.abs(F).max()) for F in frames)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Neumann wave: mx={mx}, my={my}, N={N}, CFL={cfl:.4f}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u(x,y,t)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-amp, amp)          
    ax.set_box_aspect((1, 1, 0.5)) 
    ax.view_init(elev=25, azim=35)

    artists = []
    for F in frames:
        wf = ax.plot_wireframe(xij, yij, F, rstride=2, cstride=2)
        artists.append([wf])

    ani = animation.ArtistAnimation(fig, artists, interval=400, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=fps, metadata={"artist": "neumann_movie"})
    ani.save(output_path, writer=writer, dpi=dpi, savefig_kwargs={"bbox_inches": "tight"})
    plt.close(fig)

    size = os.path.getsize(output_path)
    print(f"[ok] Saved {output_path} ({size} bytes)")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Wave2D import Wave2D_Neumann


N  = 60        
Nt = 120       
cfl = 0.5     
c   = 1.0     
mx  = 2       
my  = 3     
store_every = 2  

fps = 10      
figsize = (4, 4)
dpi = 70
cmap = "viridis"


def main():
    solver = Wave2D_Neumann()
    data = solver(N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=store_every)
    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError("Expected snapshots {tstep: U}. Call with store_data > 0.")

    steps  = sorted(data.keys())
    frames = [data[k] for k in steps]

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

    # Save directly as neumann.gif in the current directory
    out_path = "neumann.gif"   
    ani.save(out_path, writer="pillow", fps=fps,
             savefig_kwargs={"dpi": dpi, "bbox_inches": "tight"})
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

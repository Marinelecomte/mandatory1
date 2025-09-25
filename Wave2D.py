import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        x = np.linspace(0.0, 1.0, N+1)
        y = np.linspace(0.0, 1.0, N+1)
        self.dx = 1.0 / N
        self.dy = 1.0 / N
        self.xij, self.yij = np.meshgrid(x, y, indexing="ij")

    def D2(self, N):
        """Return second order differentiation matrix"""
        Dx2 = sparse.diags([np.ones(N+1), -2*np.ones(N+1), np.ones(N+1)],
                           offsets=[-1, 0, 1], shape=(N+1, N+1), format="csr")
        I = sparse.eye(N+1, format="csr")
        return sparse.kron(I, Dx2, format="csr") + sparse.kron(Dx2, I, format="csr")

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.create_mesh(N)
        self.mx, self.my = mx, my
        ue_fun = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")
        self.unm1 = ue_fun(self.xij, self.yij, -self.dt) 
        self.un   = ue_fun(self.xij, self.yij,  0.0)      
        return self.un, self.unm1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * min(self.dx, self.dy) / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue_fun = sp.lambdify((x, y, t), self.ue(self.mx, self.my), "numpy")
        ue_vals = ue_fun(self.xij, self.yij, t0)
        return np.sqrt(np.mean((u - ue_vals)**2))
        
    def apply_bcs(self):
        self.unp1[0,  :] = 0
        self.unp1[-1, :] = 0
        self.unp1[:,  0] = 0
        self.unp1[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.c   = c
        self.initialize(N, mx, my)
        D = self.D2(N)
        C = self.cfl  # C = c*dt/dx = cfl on uniform grid
        data = {} if (store_data and store_data > 0) else None
        errs = [] if store_data == -1 else None
        if data is not None:
            data[0] = self.unm1.copy()
        for n in range(1, Nt+1):
            lap_un = (D @ self.un.reshape(-1)).reshape(self.un.shape)
            self.unp1 = 2.0*self.un - self.unm1 + (C**2)*lap_un
            self.apply_bcs()
            self.unm1, self.un = self.un, self.unp1
            if data is not None and (n % store_data == 0):
                data[n] = self.un.copy()
            if errs is not None:
                errs.append(self.l2_error(self.un, n*self.dt))
        h = self.dx
        if data is not None:
            return data
        else:
            return h, np.array(errs)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        Dx2 = sparse.diags([np.ones(N+1), -2*np.ones(N+1), np.ones(N+1)],
                           offsets=[-1, 0, 1], shape=(N+1, N+1), format="lil")
        Dx2[0, 0]  = -2; Dx2[0, 1]  =  2
        Dx2[-1,-1] = -2; Dx2[-1,-2] =  2
        Dx2 = Dx2.tocsr()
        I = sparse.eye(N+1, format="csr")
        return sparse.kron(I, Dx2, format="csr") + sparse.kron(Dx2, I, format="csr")
   
    def ue(self, mx, my):
         return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        return
        
def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    h, err = sol(N=16, Nt=10, cfl=0.1, mx=2, my=3, store_data=-1)
    assert err[-1] < 5e-3

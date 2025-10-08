import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y, t = sp.symbols('x, y, t')


class Wave2D:
    """
    2D wave equation solver on [0, L] x [0, L] with uniform grid.
    Dirichlet BCs (u=0) in the base class; Neumann handled in subclass.
    """

    def __init__(self, L: float = 1.0):
        self.L = L
        self.N = None
        self.h = None
        self.x = self.y = None
        self.xij = self.yij = None
        self.Un = self.Um1 = self.Unp1 = None
        self.c = 1.0
        self.cfl = 0.5
        self.mx = self.my = 1
        self._L2 = None  

    def create_mesh(self, N: int):
        self.N = N
        self.h = self.L / N
        self.x = np.linspace(0.0, self.L, N + 1)
        self.y = np.linspace(0.0, self.L, N + 1)
        self.xij, self.yij = np.meshgrid(self.x, self.y, indexing='ij')
        return self.xij, self.yij

    def _d2_dirichlet_1d(self, N: int, h: float):
        """1D second-difference with Dirichlet structure (interior stencil)."""
        main = (-2.0 / h**2) * np.ones(N + 1)
        off  = (1.0 / h**2) * np.ones(N)
        D = sparse.diags([off, main, off], offsets=[-1, 0, 1], shape=(N + 1, N + 1), format='csr')
        return D

    def _d2_neumann_1d(self, N: int, h: float):
        """1D second-difference with Neumann BCs using 1-sided boundary rows."""
        D = sparse.lil_matrix((N + 1, N + 1))
        invh2 = 1.0 / h**2
        for i in range(1, N):
            D[i, i - 1] = 1.0 * invh2
            D[i, i]     = -2.0 * invh2
            D[i, i + 1] = 1.0 * invh2
        D[0, 0] = -2.0 * invh2
        D[0, 1] =  2.0 * invh2
        D[N, N]     = -2.0 * invh2
        D[N, N - 1] =  2.0 * invh2
        return D.tocsr()

    def D2(self, N: int):
        return self._d2_dirichlet_1d(N, self.h)

    def _build_L2(self):
        D = self.D2(self.N)  # 1D operator already scaled by 1/h^2 (and BC flavor)
        I = sparse.eye(self.N + 1, format='csr')
        self._L2 = sparse.kron(I, D, format='csr') + sparse.kron(D, I, format='csr')
        return self._L2


    @property
    def w(self):
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx, my):
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    @property
    def dt(self):
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        ue_fun = sp.lambdify((x, y, t), self.ue(self.mx, self.my), 'numpy')
        ue_vals = ue_fun(self.xij, self.yij, t0)
        return np.sqrt((self.h * self.h) * np.sum((u - ue_vals) ** 2))


    def initialize(self, N: int, mx: int, my: int):
        self.create_mesh(N)
        self.mx, self.my = mx, my
        ue_fun = sp.lambdify((x, y, t), self.ue(mx, my), 'numpy')
        self.Um1 = ue_fun(self.xij, self.yij, -self.dt)
        self.Un  = ue_fun(self.xij, self.yij,  0.0)
        self._build_L2()
        return self.Un, self.Um1

    def apply_bcs(self):
        self.Unp1[0,  :] = 0.0
        self.Unp1[-1, :] = 0.0
        self.Unp1[:,  0] = 0.0
        self.Unp1[:, -1] = 0.0
        return self.Unp1

    def __call__(self, N: int, Nt: int, cfl: float = 0.5, c: float = 1.0,
                 mx: int = 3, my: int = 3, store_data: int = -1):
        self.cfl, self.c = cfl, c
        self.initialize(N, mx, my)

        dt = self.dt
        cdt2 = (self.c * dt) ** 2
        shape = self.Un.shape
        L2 = self._L2

        results = {} if store_data and store_data > 0 else None
        errors  = [] if store_data == -1 else None
        if results is not None:
            results[0] = self.Un.copy()

        for n in range(1, Nt + 1):
            lap_Un = (L2 @ self.Un.reshape(-1)).reshape(shape)
            self.Unp1 = 2.0 * self.Un - self.Um1 + cdt2 * lap_Un

            self.apply_bcs()

            if results is not None and (n % store_data == 0):
                results[n] = self.Unp1.copy()
            if errors is not None:
                errors.append(self.l2_error(self.Unp1, n * dt))

            self.Um1, self.Un = self.Un, self.Unp1

        if results is not None:
            return results
        else:
            return self.h, np.array(errors)

class Wave2D_Neumann(Wave2D):
    def D2(self, N: int):
        return self._d2_neumann_1d(N, self.h)

    def ue(self, mx, my):
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self):
        return self.Unp1
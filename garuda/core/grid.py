"""Grid module - Mesh generation and management for reservoir simulation.

Supports structured and unstructured grids with full 3D capability.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Grid:
    """Base class for reservoir simulation grids."""

    dim: int = 3
    num_cells: int = 0
    num_faces: int = 0
    num_vertices: int = 0

    # Cell properties
    cell_volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    cell_centroids: np.ndarray = field(default_factory=lambda: np.array([]))

    # Face properties
    face_areas: np.ndarray = field(default_factory=lambda: np.array([]))
    face_centroids: np.ndarray = field(default_factory=lambda: np.array([]))
    face_normals: np.ndarray = field(default_factory=lambda: np.array([]))

    # Connectivity
    cell_faces: np.ndarray = field(default_factory=lambda: np.array([]))
    face_cells: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Validate grid properties."""
        if self.dim not in [1, 2, 3]:
            raise ValueError(f"Grid dimension must be 1, 2, or 3, got {self.dim}")

    @property
    def num_boundaries(self) -> int:
        """Number of boundary faces."""
        if self.face_cells.size == 0:
            return 0
        return int(np.sum(np.any(self.face_cells < 0, axis=1)))


@dataclass
class StructuredGrid(Grid):
    """Structured Cartesian grid for reservoir simulation.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells in each direction
    dx, dy, dz : float or array-like
        Cell sizes in each direction (can be heterogeneous)

    Examples
    --------
    >>> grid = StructuredGrid(nx=10, ny=10, nz=5, dx=100, dy=100, dz=10)
    >>> print(f"Grid has {grid.num_cells} cells")
    Grid has 500 cells

    """

    nx: int = 10
    ny: int = 10
    nz: int = 1
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0

    @staticmethod
    def _ensure_spacing(val, n):
        """Convert scalar or array input to 1D float array of length n."""
        arr = np.asarray(val)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if arr.shape == (1,):
            return np.full(n, float(arr[0]))
        if arr.shape != (n,):
            raise ValueError(f"Spacing array must have shape ({n},) or scalar, got {arr.shape}")
        return arr.astype(float)

    def __post_init__(self):
        """Initialize grid geometry."""
        super().__post_init__()
        self.dim = 3 if self.nz > 1 else (2 if self.ny > 1 else 1)
        self.num_cells = self.nx * self.ny * self.nz

        # Normalise spacing inputs to 1D arrays
        self.dx = self._ensure_spacing(self.dx, self.nx)
        self.dy = self._ensure_spacing(self.dy, self.ny)
        self.dz = self._ensure_spacing(self.dz, self.nz)

        self._generate_grid()

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------
    def _generate_grid(self):
        """Generate grid geometry (vertices, centroids, volumes, faces)."""
        # Cumulative coordinates (edges)
        xc = np.cumsum(np.r_[0.0, self.dx])
        yc = np.cumsum(np.r_[0.0, self.dy])
        zc = np.cumsum(np.r_[0.0, self.dz])

        # Cell center coordinates
        xc_centers = (xc[:-1] + xc[1:]) / 2
        yc_centers = (yc[:-1] + yc[1:]) / 2
        zc_centers = (zc[:-1] + zc[1:]) / 2

        # Build centroids & volumes in *natural* i-major order:
        #   for k: for j: for i:
        # This matches get_cell_index / get_ijk.
        if self.dim == 3:
            # meshgrid with indexing='ij' returns shape (nx, ny, nz). To match
            # get_cell_index = i + j*nx + k*nx*ny we need i-fastest ordering,
            # which is Fortran-order ravel.
            xx, yy, zz = np.meshgrid(xc_centers, yc_centers, zc_centers, indexing="ij")
            self.cell_centroids = np.column_stack([xx.ravel("F"), yy.ravel("F"), zz.ravel("F")])
            dx3, dy3, dz3 = np.meshgrid(self.dx, self.dy, self.dz, indexing="ij")
            self.cell_volumes = (dx3 * dy3 * dz3).ravel("F")

        elif self.dim == 2:
            xx, yy = np.meshgrid(xc_centers, yc_centers, indexing="ij")
            zz = np.zeros_like(xx)
            self.cell_centroids = np.column_stack([xx.ravel("F"), yy.ravel("F"), zz.ravel("F")])
            dx2, dy2 = np.meshgrid(self.dx, self.dy, indexing="ij")
            self.cell_volumes = (dx2 * dy2).ravel("F") * self.dz[0]

        else:  # 1D
            self.cell_centroids = np.column_stack([xc_centers, np.zeros(self.nx), np.zeros(self.nx)])
            self.cell_volumes = self.dx.copy() * self.dy[0] * self.dz[0]

        # Generate faces and connectivity
        self._generate_faces(xc, yc, zc, xc_centers, yc_centers, zc_centers)
        self._generate_cell_face_connectivity()

    # ------------------------------------------------------------------
    # Face generation
    # ------------------------------------------------------------------
    def _generate_faces(self, xc, yc, zc, xc_c, yc_c, zc_c):
        """Generate face geometry, areas, normals, and face_cells connectivity.
        Uses pre-computed edge / centre coordinates.
        """
        if self.dim == 1:
            self.num_faces = self.nx + 1
            # Face area perpendicular to flow = dy * dz (the cross-section).
            self.face_areas = np.full(self.num_faces, float(self.dy[0]) * float(self.dz[0]))
            self.face_centroids = np.zeros((self.num_faces, 3), dtype=float)
            for i in range(self.num_faces):
                self.face_centroids[i, 0] = xc[i]
            self.face_normals = np.zeros((self.num_faces, 3), dtype=float)
            self.face_normals[:, 0] = [-1 if i == 0 else 1 for i in range(self.num_faces)]

            self.face_cells = np.zeros((self.num_faces, 2), dtype=int)
            self.face_cells[0] = [-1, 0]
            for i in range(1, self.nx):
                self.face_cells[i] = [i - 1, i]
            self.face_cells[-1] = [self.nx - 1, -1]

        elif self.dim == 2:
            num_faces_x = (self.nx + 1) * self.ny
            num_faces_y = self.nx * (self.ny + 1)
            self.num_faces = num_faces_x + num_faces_y
            self.face_areas = np.zeros(self.num_faces, dtype=float)
            self.face_centroids = np.zeros((self.num_faces, 3), dtype=float)
            self.face_normals = np.zeros((self.num_faces, 3), dtype=float)

            # x-faces (normal in x): each has area = dy  (one per row * nx+1 cols)
            for j in range(self.ny):
                base = j * (self.nx + 1)
                self.face_areas[base : base + self.nx + 1] = self.dy[j]

            # y-faces (normal in y): each has area = dx  (one per col * ny+1 rows)
            for j in range(self.ny + 1):
                base = num_faces_x + j * self.nx
                self.face_areas[base : base + self.nx] = self.dx

            # X-face centroids / normals
            f = 0
            for j in range(self.ny):
                for i in range(self.nx + 1):
                    self.face_centroids[f, 0] = xc[i]
                    self.face_centroids[f, 1] = yc_c[j]
                    self.face_normals[f, 0] = -1 if i == 0 else 1
                    f += 1

            # Y-face centroids / normals
            for j in range(self.ny + 1):
                for i in range(self.nx):
                    self.face_centroids[f, 0] = xc_c[i]
                    self.face_centroids[f, 1] = yc[j]
                    self.face_normals[f, 1] = -1 if j == 0 else 1
                    f += 1

            # Face connectivity (-1 = boundary)
            self.face_cells = np.full((self.num_faces, 2), -1, dtype=int)
            f = 0
            for j in range(self.ny):
                for i in range(self.nx + 1):
                    if i < self.nx:
                        self.face_cells[f, 0] = i + j * self.nx
                    if i > 0:
                        self.face_cells[f, 1] = (i - 1) + j * self.nx
                    f += 1
            for j in range(self.ny + 1):
                for i in range(self.nx):
                    if j < self.ny:
                        self.face_cells[f, 0] = i + j * self.nx
                    if j > 0:
                        self.face_cells[f, 1] = i + (j - 1) * self.nx
                    f += 1

        else:  # 3D
            num_faces_x = (self.nx + 1) * self.ny * self.nz
            num_faces_y = self.nx * (self.ny + 1) * self.nz
            num_faces_z = self.nx * self.ny * (self.nz + 1)
            self.num_faces = num_faces_x + num_faces_y + num_faces_z
            self.face_areas = np.zeros(self.num_faces, dtype=float)
            self.face_centroids = np.zeros((self.num_faces, 3), dtype=float)
            self.face_normals = np.zeros((self.num_faces, 3), dtype=float)

            # Build dy*dz lookup per (j,k)
            dy_dz = np.outer(self.dy, self.dz).ravel()
            dx_dz = np.outer(self.dx, self.dz).ravel()
            dx_dy = np.outer(self.dx, self.dy).ravel()

            # X-faces: area = dy*dz
            f = 0
            for k in range(self.nz):
                for j in range(self.ny):
                    base = f
                    self.face_areas[base : base + self.nx + 1] = dy_dz[j + k * self.ny]
                    for i in range(self.nx + 1):
                        self.face_centroids[f, 0] = xc[i]
                        self.face_centroids[f, 1] = yc_c[j]
                        self.face_centroids[f, 2] = zc_c[k]
                        self.face_normals[f, 0] = -1 if i == 0 else 1
                        f += 1

            # Y-faces: area = dx*dz
            for k in range(self.nz):
                for j in range(self.ny + 1):
                    base = f
                    self.face_areas[base : base + self.nx] = dx_dz[k :: self.nz] if self.nz > 1 else dx_dz
                    for i in range(self.nx):
                        self.face_centroids[f, 0] = xc_c[i]
                        self.face_centroids[f, 1] = yc[j]
                        self.face_normals[f, 1] = -1 if j == 0 else 1
                        f += 1

            # Z-faces: area = dx*dy
            for k in range(self.nz + 1):
                base = f
                self.face_areas[base : base + self.nx * self.ny] = dx_dy
                for j in range(self.ny):
                    for i in range(self.nx):
                        self.face_centroids[f, 0] = xc_c[i]
                        self.face_centroids[f, 1] = yc_c[j]
                        self.face_centroids[f, 2] = zc[k]
                        self.face_normals[f, 2] = -1 if k == 0 else 1
                        f += 1

            # Face connectivity
            self.face_cells = np.full((self.num_faces, 2), -1, dtype=int)
            f = 0
            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx + 1):
                        if i < self.nx:
                            self.face_cells[f, 0] = i + j * self.nx + k * self.nx * self.ny
                        if i > 0:
                            self.face_cells[f, 1] = (i - 1) + j * self.nx + k * self.nx * self.ny
                        f += 1
            for k in range(self.nz):
                for j in range(self.ny + 1):
                    for i in range(self.nx):
                        if j < self.ny:
                            self.face_cells[f, 0] = i + j * self.nx + k * self.nx * self.ny
                        if j > 0:
                            self.face_cells[f, 1] = i + (j - 1) * self.nx + k * self.nx * self.ny
                        f += 1
            for k in range(self.nz + 1):
                for j in range(self.ny):
                    for i in range(self.nx):
                        if k < self.nz:
                            self.face_cells[f, 0] = i + j * self.nx + k * self.nx * self.ny
                        if k > 0:
                            self.face_cells[f, 1] = i + j * self.nx + (k - 1) * self.nx * self.ny
                        f += 1

    # ------------------------------------------------------------------
    # Cell-face connectivity
    # ------------------------------------------------------------------
    def _generate_cell_face_connectivity(self):
        """cell_faces[cell_id, local_face] = global_face_id"""
        self.cell_faces = np.full((self.num_cells, 2 * self.dim), -1, dtype=int)

        if self.dim == 1:
            for i in range(self.nx):
                self.cell_faces[i, 0] = i  # left
                self.cell_faces[i, 1] = i + 1  # right

        elif self.dim == 2:
            num_faces_x = (self.nx + 1) * self.ny
            for j in range(self.ny):
                for i in range(self.nx):
                    cid = i + j * self.nx
                    self.cell_faces[cid, 0] = i + j * (self.nx + 1)  # west
                    self.cell_faces[cid, 1] = i + 1 + j * (self.nx + 1)  # east
                    self.cell_faces[cid, 2] = num_faces_x + i + j * self.nx  # south
                    self.cell_faces[cid, 3] = num_faces_x + i + (j + 1) * self.nx  # north

        else:  # 3D
            num_faces_x = (self.nx + 1) * self.ny * self.nz
            num_faces_y = self.nx * (self.ny + 1) * self.nz
            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx):
                        cid = i + j * self.nx + k * self.nx * self.ny
                        self.cell_faces[cid, 0] = i + j * (self.nx + 1) + k * (self.nx + 1) * self.ny
                        self.cell_faces[cid, 1] = i + 1 + j * (self.nx + 1) + k * (self.nx + 1) * self.ny
                        self.cell_faces[cid, 2] = num_faces_x + i + j * self.nx + k * self.nx * (self.ny + 1)
                        self.cell_faces[cid, 3] = num_faces_x + i + (j + 1) * self.nx + k * self.nx * (self.ny + 1)
                        self.cell_faces[cid, 4] = num_faces_x + num_faces_y + i + j * self.nx + k * self.nx * self.ny
                        self.cell_faces[cid, 5] = (
                            num_faces_x + num_faces_y + i + j * self.nx + (k + 1) * self.nx * self.ny
                        )

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------
    def get_cell_index(self, ix: int, iy: int = 0, iz: int = 0) -> int:
        """Convert (ix, iy, iz) to linear cell index."""
        return ix + iy * self.nx + iz * self.nx * self.ny

    def get_ijk(self, cell_index: int) -> tuple[int, int, int]:
        """Convert linear cell index to (ix, iy, iz)."""
        iz = cell_index // (self.nx * self.ny)
        rem = cell_index % (self.nx * self.ny)
        iy = rem // self.nx
        ix = rem % self.nx
        return ix, iy, iz

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------
    def set_permeability(self, perm, unit="m2"):
        """Set permeability tensor for all cells.

        Parameters
        ----------
        perm : array-like
            Permeability values. Can be:
            - scalar: isotropic, same for all cells
            - (num_cells,): isotropic, heterogeneous
            - (num_cells, 3): anisotropic (kx, ky, kz)
            - (num_cells, 3, 3): full tensor
        unit : str
            Unit of input ('m2' for m², 'md' for millidarcy)

        """
        md_to_m2 = 9.869233e-16
        factor = md_to_m2 if unit.lower() == "md" else 1.0
        perm = np.atleast_1d(perm) * factor

        if perm.size == 1:
            self.permeability = np.tile(np.eye(3) * float(perm[0]), (self.num_cells, 1, 1))
        elif perm.shape == (self.num_cells,):
            self.permeability = np.zeros((self.num_cells, 3, 3))
            for d in range(3):
                self.permeability[:, d, d] = perm
        elif perm.shape == (self.num_cells, 3):
            self.permeability = np.zeros((self.num_cells, 3, 3))
            for d in range(3):
                self.permeability[:, d, d] = perm[:, d]
        elif perm.shape == (self.num_cells, 3, 3):
            self.permeability = perm
        else:
            raise ValueError(f"Invalid permeability shape: {perm.shape}")

        # Backward-compat alias for old code/tests
        self.permiability = self.permeability

    # Backward-compatible alias
    set_permiability = set_permeability

    def set_porosity(self, porosity: np.ndarray):
        """Set porosity for all cells."""
        porosity = np.atleast_1d(porosity)
        if porosity.size == 1:
            self.porosity = np.full(self.num_cells, float(porosity[0]))
        elif porosity.shape == (self.num_cells,):
            self.porosity = porosity.astype(float)
        else:
            raise ValueError(f"Invalid porosity shape {porosity.shape}, expected scalar or ({self.num_cells},)")

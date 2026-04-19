"""
Grid module - Mesh generation and management for reservoir simulation.

Supports structured and unstructured grids with full 3D capability.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Grid:
    """Base class for reservoir simulation grids."""
    
    dim: int = 3  # Spatial dimensions (1, 2, or 3)
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
    cell_faces: np.ndarray = field(default_factory=lambda: np.array([]))  # cells x faces
    face_cells: np.ndarray = field(default_factory=lambda: np.array([]))  # faces x 2 (left, right)
    
    def __post_init__(self):
        """Validate grid properties."""
        if self.dim not in [1, 2, 3]:
            raise ValueError(f"Grid dimension must be 1, 2, or 3, got {self.dim}")
    
    @property
    def num_boundaries(self) -> int:
        """Number of boundary faces."""
        if self.face_cells.size == 0:
            return 0
        return np.sum(np.any(self.face_cells < 0, axis=1))


@dataclass
class StructuredGrid(Grid):
    """
    Structured Cartesian grid for reservoir simulation.
    
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
    
    def __post_init__(self):
        """Initialize grid geometry."""
        super().__post_init__()
        self.dim = 3 if self.nz > 1 else (2 if self.ny > 1 else 1)
        self.num_cells = self.nx * self.ny * self.nz
        
        # Handle heterogeneous cell sizes
        self.dx = np.atleast_1d(self.dx) if not isinstance(self.dx, np.ndarray) else self.dx
        self.dy = np.atleast_1d(self.dy) if not isinstance(self.dy, np.ndarray) else self.dy
        self.dz = np.atleast_1d(self.dz) if not isinstance(self.dz, np.ndarray) else self.dz
        
        self._generate_grid()
    
    def _generate_grid(self):
        """Generate grid geometry (vertices, centroids, volumes, faces)."""
        # Cell centroids
        xc = np.cumsum(np.r_[0, self.dx]) if isinstance(self.dx, np.ndarray) and len(self.dx) > 1 else np.linspace(0, self.nx * self.dx, self.nx + 1)
        yc = np.cumsum(np.r_[0, self.dy]) if isinstance(self.dy, np.ndarray) and len(self.dy) > 1 else np.linspace(0, self.ny * self.dy, self.ny + 1)
        zc = np.cumsum(np.r_[0, self.dz]) if isinstance(self.dz, np.ndarray) and len(self.dz) > 1 else np.linspace(0, self.nz * self.dz, self.nz + 1)
        
        # Cell center coordinates
        xc_centers = (xc[:-1] + xc[1:]) / 2 if len(xc) > 1 else np.array([self.dx / 2])
        yc_centers = (yc[:-1] + yc[1:]) / 2 if len(yc) > 1 else np.array([self.dy / 2])
        zc_centers = (zc[:-1] + zc[1:]) / 2 if len(zc) > 1 else np.array([self.dz / 2])
        
        # Create meshgrid for centroids
        if self.dim == 3:
            xx, yy, zz = np.meshgrid(xc_centers, yc_centers, zc_centers, indexing='ij')
        elif self.dim == 2:
            xx, yy = np.meshgrid(xc_centers, yc_centers, indexing='ij')
            zz = np.zeros_like(xx)
        else:  # 1D
            xx = xc_centers
            yy = np.zeros_like(xx)
            zz = np.zeros_like(xx)
        
        self.cell_centroids = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        
        # Cell volumes
        dx_arr = np.full(self.nx, self.dx) if np.isscalar(self.dx) else self.dx
        dy_arr = np.full(self.ny, self.dy) if np.isscalar(self.dy) else self.dy
        dz_arr = np.full(self.nz, self.dz) if np.isscalar(self.dz) else self.dz
        
        if self.dim == 3:
            dv = np.outer(np.outer(dx_arr, dy_arr), dz_arr).ravel()
        elif self.dim == 2:
            dv = np.outer(dx_arr, dy_arr).ravel()
        else:
            dv = dx_arr
        
        self.cell_volumes = dv
        
        # Generate faces and connectivity
        self._generate_faces()
    
    def _generate_faces(self):
        """Generate face geometry and connectivity."""
        # For structured grid, faces are between cells
        # Each interior cell has 6 faces (in 3D), 4 faces (in 2D), 2 faces (in 1D)
        
        if self.dim == 1:
            self.num_faces = self.nx + 1
            self.face_areas = np.ones(self.num_faces)
            self.face_centroids = np.zeros((self.num_faces, 3))
            self.face_normals = np.zeros((self.num_faces, 3))
            self.face_normals[:, 0] = [-1 if i == 0 else 1 for i in range(self.num_faces)]
            
            # Face connectivity
            self.face_cells = np.zeros((self.num_faces, 2), dtype=int)
            self.face_cells[0, 1] = 0  # Left boundary
            self.face_cells[1:, 0] = np.arange(self.nx)  # Right face of each cell
            self.face_cells[:-1, 1] = np.arange(self.nx)  # Left face of each cell
            self.face_cells[0, 0] = -1  # Boundary marker
            self.face_cells[-1, 1] = -1  # Boundary marker
            
        elif self.dim == 2:
            # 2D: faces in x and y directions
            num_faces_x = (self.nx + 1) * self.ny
            num_faces_y = self.nx * (self.ny + 1)
            self.num_faces = num_faces_x + num_faces_y
            
            # Face areas (length in 2D)
            dx_arr = np.full(self.nx, self.dx) if np.isscalar(self.dx) else self.dx
            dy_arr = np.full(self.ny, self.dy) if np.isscalar(self.dy) else self.dy
            
            self.face_areas = np.zeros(self.num_faces)
            self.face_areas[:num_faces_x] = np.tile(dy_arr, self.nx + 1)  # x-faces (dy length)
            self.face_areas[num_faces_x:] = np.tile(dx_arr, self.ny + 1)  # y-faces (dx length)
            
            self.face_centroids = np.zeros((self.num_faces, 3))
            self.face_normals = np.zeros((self.num_faces, 3))
            
            # X-faces (normal in x-direction)
            xc = (np.cumsum(np.r_[0, dx_arr])[:-1] + np.cumsum(np.r_[0, dx_arr])[1:]) / 2
            yc = np.cumsum(np.r_[0, dy_arr]) if len(dy_arr) > 1 else np.linspace(0, self.ny * self.dy, self.ny + 1)
            yc_centers = (yc[:-1] + yc[1:]) / 2 if len(yc) > 1 else np.array([self.dy / 2])
            
            face_idx = 0
            for j in range(self.ny):
                for i in range(self.nx + 1):
                    self.face_centroids[face_idx, 0] = 0 if i == 0 else np.sum(dx_arr[:i])
                    self.face_centroids[face_idx, 1] = yc_centers[j]
                    self.face_normals[face_idx, 0] = -1 if i == 0 else 1
                    face_idx += 1
            
            # Y-faces (normal in y-direction)
            xc_edges = np.cumsum(np.r_[0, dx_arr])
            yc_edges = np.cumsum(np.r_[0, dy_arr])
            
            for j in range(self.ny + 1):
                for i in range(self.nx):
                    self.face_centroids[face_idx, 0] = xc_edges[i] + dx_arr[i] / 2
                    self.face_centroids[face_idx, 1] = 0 if j == 0 else np.sum(dy_arr[:j])
                    self.face_normals[face_idx, 1] = -1 if j == 0 else 1
                    face_idx += 1
            
            # Face connectivity: face_cells[face_id] = [left_cell, right_cell]
            # -1 indicates boundary face
            self.face_cells = np.full((self.num_faces, 2), -1, dtype=int)
            
            # X-faces connectivity
            face_idx = 0
            for j in range(self.ny):
                for i in range(self.nx + 1):
                    if i < self.nx:
                        self.face_cells[face_idx, 0] = i + j * self.nx  # left cell
                    if i > 0:
                        self.face_cells[face_idx, 1] = (i - 1) + j * self.nx  # right cell
                    face_idx += 1
            
            # Y-faces connectivity
            for j in range(self.ny + 1):
                for i in range(self.nx):
                    if j > 0:
                        self.face_cells[face_idx, 1] = i + (j - 1) * self.nx  # bottom cell
                    if j < self.ny:
                        self.face_cells[face_idx, 0] = i + j * self.nx  # top cell
                    face_idx += 1
            
        else:  # 3D
            # 3D: faces in x, y, z directions
            num_faces_x = (self.nx + 1) * self.ny * self.nz
            num_faces_y = self.nx * (self.ny + 1) * self.nz
            num_faces_z = self.nx * self.ny * (self.nz + 1)
            self.num_faces = num_faces_x + num_faces_y + num_faces_z
            
            # Face areas
            dx_arr = np.full(self.nx, self.dx) if np.isscalar(self.dx) else self.dx
            dy_arr = np.full(self.ny, self.dy) if np.isscalar(self.dy) else self.dy
            dz_arr = np.full(self.nz, self.dz) if np.isscalar(self.dz) else self.dz
            
            self.face_areas = np.zeros(self.num_faces)
            self.face_areas[:num_faces_x] = np.tile(np.repeat(dy_arr * dz_arr, self.nz), self.nx + 1)  # x-faces (dy*dz)
            self.face_areas[num_faces_x:num_faces_x + num_faces_y] = np.tile(np.repeat(dx_arr * dz_arr, self.nz), self.ny + 1)  # y-faces (dx*dz)
            self.face_areas[num_faces_x + num_faces_y:] = np.tile(np.repeat(dx_arr * dy_arr, self.nz + 1), 1)  # z-faces (dx*dy)
            
            self.face_centroids = np.zeros((self.num_faces, 3))
            self.face_normals = np.zeros((self.num_faces, 3))
            
            # X-faces (normal in x-direction)
            xc_edges = np.cumsum(np.r_[0, dx_arr])
            yc = (np.cumsum(np.r_[0, dy_arr])[:-1] + np.cumsum(np.r_[0, dy_arr])[1:]) / 2 if len(dy_arr) > 1 else np.array([self.dy / 2])
            zc = (np.cumsum(np.r_[0, dz_arr])[:-1] + np.cumsum(np.r_[0, dz_arr])[1:]) / 2 if len(dz_arr) > 1 else np.array([self.dz / 2])
            
            face_idx = 0
            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx + 1):
                        self.face_centroids[face_idx, 0] = xc_edges[i]
                        self.face_centroids[face_idx, 1] = yc[j]
                        self.face_centroids[face_idx, 2] = zc[k]
                        self.face_normals[face_idx, 0] = -1 if i == 0 else 1
                        face_idx += 1
            
            # Y-faces (normal in y-direction)
            xc = (np.cumsum(np.r_[0, dx_arr])[:-1] + np.cumsum(np.r_[0, dx_arr])[1:]) / 2 if len(dx_arr) > 1 else np.array([self.dx / 2])
            yc_edges = np.cumsum(np.r_[0, dy_arr])
            zc = (np.cumsum(np.r_[0, dz_arr])[:-1] + np.cumsum(np.r_[0, dz_arr])[1:]) / 2 if len(dz_arr) > 1 else np.array([self.dz / 2])
            
            for k in range(self.nz):
                for j in range(self.ny + 1):
                    for i in range(self.nx):
                        self.face_centroids[face_idx, 0] = xc[i]
                        self.face_centroids[face_idx, 1] = yc_edges[j]
                        self.face_centroids[face_idx, 2] = zc[k]
                        self.face_normals[face_idx, 1] = -1 if j == 0 else 1
                        face_idx += 1
            
            # Z-faces (normal in z-direction)
            xc = (np.cumsum(np.r_[0, dx_arr])[:-1] + np.cumsum(np.r_[0, dx_arr])[1:]) / 2 if len(dx_arr) > 1 else np.array([self.dx / 2])
            yc = (np.cumsum(np.r_[0, dy_arr])[:-1] + np.cumsum(np.r_[0, dy_arr])[1:]) / 2 if len(dy_arr) > 1 else np.array([self.dy / 2])
            zc_edges = np.cumsum(np.r_[0, dz_arr])
            
            for k in range(self.nz + 1):
                for j in range(self.ny):
                    for i in range(self.nx):
                        self.face_centroids[face_idx, 0] = xc[i]
                        self.face_centroids[face_idx, 1] = yc[j]
                        self.face_centroids[face_idx, 2] = zc_edges[k]
                        self.face_normals[face_idx, 2] = -1 if k == 0 else 1
                        face_idx += 1
            
            # Face connectivity: face_cells[face_id] = [left_cell, right_cell]
            # -1 indicates boundary face
            self.face_cells = np.full((self.num_faces, 2), -1, dtype=int)
            
            # X-faces connectivity
            face_idx = 0
            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx + 1):
                        if i < self.nx:
                            self.face_cells[face_idx, 0] = i + j * self.nx + k * self.nx * self.ny
                        if i > 0:
                            self.face_cells[face_idx, 1] = (i - 1) + j * self.nx + k * self.nx * self.ny
                        face_idx += 1
            
            # Y-faces connectivity
            for k in range(self.nz):
                for j in range(self.ny + 1):
                    for i in range(self.nx):
                        if j > 0:
                            self.face_cells[face_idx, 1] = i + (j - 1) * self.nx + k * self.nx * self.ny
                        if j < self.ny:
                            self.face_cells[face_idx, 0] = i + j * self.nx + k * self.nx * self.ny
                        face_idx += 1
            
            # Z-faces connectivity
            for k in range(self.nz + 1):
                for j in range(self.ny):
                    for i in range(self.nx):
                        if k > 0:
                            self.face_cells[face_idx, 1] = i + j * self.nx + (k - 1) * self.nx * self.ny
                        if k < self.nz:
                            self.face_cells[face_idx, 0] = i + j * self.nx + k * self.nx * self.ny
                        face_idx += 1
        
        # Cell-face connectivity: cell_faces[cell_id, face_local_idx] = global_face_id
        # Each cell has 2*dim faces (2 in 1D, 4 in 2D, 6 in 3D)
        self.cell_faces = np.full((self.num_cells, 2 * self.dim), -1, dtype=int)
        
        if self.dim == 1:
            for i in range(self.nx):
                self.cell_faces[i, 0] = i  # left face
                self.cell_faces[i, 1] = i + 1  # right face
        
        elif self.dim == 2:
            num_faces_x = (self.nx + 1) * self.ny
            for j in range(self.ny):
                for i in range(self.nx):
                    cell_id = i + j * self.nx
                    # West face (x-direction, left)
                    self.cell_faces[cell_id, 0] = i + j * (self.nx + 1)
                    # East face (x-direction, right)
                    self.cell_faces[cell_id, 1] = i + 1 + j * (self.nx + 1)
                    # South face (y-direction, bottom)
                    self.cell_faces[cell_id, 2] = num_faces_x + i + j * self.nx
                    # North face (y-direction, top)
                    self.cell_faces[cell_id, 3] = num_faces_x + i + (j + 1) * self.nx
        
        else:  # 3D
            num_faces_x = (self.nx + 1) * self.ny * self.nz
            num_faces_y = self.nx * (self.ny + 1) * self.nz
            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx):
                        cell_id = i + j * self.nx + k * self.nx * self.ny
                        # West face (x-direction, left)
                        self.cell_faces[cell_id, 0] = i + j * (self.nx + 1) + k * (self.nx + 1) * self.ny
                        # East face (x-direction, right)
                        self.cell_faces[cell_id, 1] = i + 1 + j * (self.nx + 1) + k * (self.nx + 1) * self.ny
                        # South face (y-direction, bottom)
                        self.cell_faces[cell_id, 2] = num_faces_x + i + j * self.nx + k * self.nx * (self.ny + 1)
                        # North face (y-direction, top)
                        self.cell_faces[cell_id, 3] = num_faces_x + i + (j + 1) * self.nx + k * self.nx * (self.ny + 1)
                        # Bottom face (z-direction, down)
                        self.cell_faces[cell_id, 4] = num_faces_x + num_faces_y + i + j * self.nx + k * self.nx * self.ny
                        # Top face (z-direction, up)
                        self.cell_faces[cell_id, 5] = num_faces_x + num_faces_y + i + j * self.nx + (k + 1) * self.nx * self.ny
    
    def get_cell_index(self, ix: int, iy: int = 0, iz: int = 0) -> int:
        """Convert (ix, iy, iz) to linear cell index."""
        return ix + iy * self.nx + iz * self.nx * self.ny
    
    def get_ijk(self, cell_index: int) -> Tuple[int, int, int]:
        """Convert linear cell index to (ix, iy, iz)."""
        iz = cell_index // (self.nx * self.ny)
        iy = (cell_index % (self.nx * self.ny)) // self.nx
        ix = cell_index % self.nx
        return ix, iy, iz
    
    def set_permiability(self, perm: np.ndarray):
        """
        Set permeability tensor for all cells.
        
        Parameters
        ----------
        perm : array-like
            Permeability values. Can be:
            - scalar: isotropic, same for all cells
            - (num_cells,): isotropic, heterogeneous
            - (num_cells, 3): anisotropic (kx, ky, kz)
            - (num_cells, 3, 3): full tensor
        """
        perm = np.atleast_1d(perm)
        
        if perm.size == 1:
            # Isotropic, homogeneous
            self.permiability = np.tile(np.eye(3) * perm[0], (self.num_cells, 1, 1))
        elif perm.shape == (self.num_cells,):
            # Isotropic, heterogeneous
            self.permiability = np.zeros((self.num_cells, 3, 3))
            for i in range(3):
                self.permiability[:, i, i] = perm
        elif perm.shape == (self.num_cells, 3):
            # Anisotropic diagonal
            self.permiability = np.zeros((self.num_cells, 3, 3))
            for i in range(3):
                self.permiability[:, i, i] = perm[:, i]
        elif perm.shape == (self.num_cells, 3, 3):
            # Full tensor
            self.permiability = perm
        else:
            raise ValueError(f"Invalid permeability shape: {perm.shape}")
    
    def set_porosity(self, porosity: np.ndarray):
        """Set porosity for all cells."""
        porosity = np.atleast_1d(porosity)
        if porosity.size == 1:
            self.porosity = np.full(self.num_cells, porosity[0])
        elif porosity.shape == (self.num_cells,):
            self.porosity = porosity
        else:
            raise ValueError(f"Invalid porosity shape: {porosity.shape}")

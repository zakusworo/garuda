"""Solvers package — optional high-performance backends.

Provides PETSc-based linear and non-linear solvers for large-scale
reservoir simulation on multi-core and distributed-memory systems.
"""

try:
    from .petsc_solver import (
        PETScDMSolver,
        PETScTPFASolver,
        has_petsc,
    )

    __all__ = ["PETScTPFASolver", "PETScDMSolver", "has_petsc"]
except ImportError:
    has_petsc = False
    __all__ = ["has_petsc"]

GARUDA Documentation
========================

**Geothermal And Reservoir Understanding with Data-driven Analytics**

.. image:: https://img.shields.io/badge/python-3.9%2B-blue
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: MIT License

.. image:: https://img.shields.io/badge/code%20style-black-black
   :alt: Code style: black

GARUDA is an open-source reservoir simulator for both **petroleum** and **geothermal** systems,
written in pure Python with NumPy/SciPy acceleration. It features a modular architecture
that supports future AI/ML integration via multi-agent LLM systems.

Quick Start
-----------

.. code-block:: bash

   pip install garuda-sim

.. code-block:: python

   from garuda import StructuredGrid, TPFASolver, FluidProperties, RockProperties

   # Create a 2D grid
   grid = StructuredGrid(nx=50, ny=50, nz=1, dx=100, dy=100, dz=10)

   # Assign properties
   grid.set_permeability(100, unit='md')
   grid.set_porosity(0.2)

   # Solve single-phase flow
   solver = TPFASolver(grid, mu=1e-3, rho=1000)
   pressure = solver.solve(source_terms=np.zeros(grid.num_cells),
                           bc_type='dirichlet',
                           bc_values=[200e5, 100e5])

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/grid
   user_guide/solver
   user_guide/physics
   user_guide/thermal
   user_guide/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/grid
   api/tpfa_solver
   api/fluid_properties
   api/rock_properties
   api/single_phase
   api/well_models
   api/thermal

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Citation
--------

If you use GARUDA in your research, please cite:

.. code-block:: bibtex

   @software{garuda2025,
     author = {Kusworo, Zulfikar Aji},
     title  = {GARUDA: Geothermal And Reservoir Understanding with Data-driven Analytics},
     url    = {https://github.com/zakusworo/garuda},
     year   = {2025}
   }

License
-------

GARUDA is released under the MIT License.

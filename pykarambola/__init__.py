"""
pykarambola - Python implementation of the Karambola package for computing
Minkowski tensors on 3D triangulated surfaces.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pykarambola")
except PackageNotFoundError:
    # Package is not installed (e.g. running from source without pip install)
    __version__ = "unknown"

from .triangulation import Triangulation, LABEL_UNASSIGNED, NEIGHBOUR_UNASSIGNED
from .io_poly import parse_poly_file
from .io_off import parse_off_file
from .io_obj import parse_obj_file
from .io_glb import parse_glb_file
from .minkowski import (
    calculate_w000, calculate_w100, calculate_w200, calculate_w300,
    calculate_w010, calculate_w110, calculate_w210, calculate_w310,
    calculate_w020, calculate_w120, calculate_w220, calculate_w320,
    calculate_w102, calculate_w202, calculate_w103, calculate_w104,
)
from .spherical import calculate_sphmink
from .eigensystem import calculate_eigensystem
from .results import CalcOptions, SurfaceStatistics
from .surface import check_surface
from .api import minkowski_tensors, minkowski_tensors_from_label_image
from .invariants import compute_invariants, decompose_all
from .spharm_invariants import compute_spharm_invariants

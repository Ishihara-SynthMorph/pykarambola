"""
Output writers to produce files matching the C++ karambola format.
"""

import os
import math
import numpy as np
from .results import LABEL_UNASSIGNED
from .tensor import SymmetricRank4Tensor


VERSION = "pykarambola 2.0"
SW = 20  # column width


def _is_nan(v):
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _format_value(v):
    if _is_nan(v):
        return f"{'ERROR':>{SW}}"
    return f"{v:{SW}.12f}"


def _format_label(label):
    if label == LABEL_UNASSIGNED:
        return f"{'ALL':>{SW}}"
    return f"{label:{SW}}"


def _print_explanations(f, infilename):
    f.write(f"# input_file = {infilename}\n")
    f.write("#\n")
    f.write("# keyword explanations: (not all keywords have to show up)\n")
    f.write("# closed = the labeled surface is closed\n")
    f.write("# shared = the labeled surface has edges whose neighbours are surfaces with other labels\n")
    f.write("# open   = the labeled surface is open, there are no neighbours to some of the edges\n")
    f.write("# calculation_forced = stubbornly applied the formulae even though the surface is shared or open\n")
    f.write("# cant_be_calculated = can't be calculated because surface is shared or open\n")
    f.write("# reference_origin   = the tensors are calculated with respect to the origin\n")
    f.write("# reference_centroid = the tensors are calculated with respect to the minkowski-vectors\n")
    f.write("#\n")
    f.write(f"# calculated with {VERSION}\n")
    f.write("# contact: karambola@physik.uni-erlangen.de\n")


def _write_scalar_row(f, label, result):
    f.write(_format_label(label))
    f.write(_format_value(result.result))
    f.write(f"{result.name:>{SW}}")
    for kw in result.keywords:
        f.write(f"{kw:>{SW}} ")
    f.write("\n")


def write_scalar_file(co, w000, w100, w200, w300, append=False):
    """Write w000_w100_w200_w300 output file."""
    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, "w000_w100_w200_w300")
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        if not append:
            _print_explanations(f, co.infilename)
            f.write("#\n")
            f.write(f"#{'#1 label':>{SW-1}}")
            f.write(f"{'#2 w':>{SW}}")
            f.write(f"{'#3 name':>{SW}}")
            f.write(f"{'#4 Keywords':>{SW}}")
            f.write("\n")
        for w in [w000, w100, w200, w300]:
            for label in sorted(w.keys()):
                _write_scalar_row(f, label, w[label])


def write_vector_file(co, w010, w110, w210, w310, append=False):
    """Write w010_w110_w210_w310 output file."""
    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, "w010_w110_w210_w310")
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        if not append:
            _print_explanations(f, co.infilename)
            f.write("#\n")
            f.write(f"#{'#1 label':>{SW-1}}")
            f.write(f"{'#2 v[x]':>{SW}}")
            f.write(f"{'#3 v[y]':>{SW}}")
            f.write(f"{'#4 v[z]':>{SW}}")
            f.write(f"{'#5 name':>{SW}}")
            f.write(f"{'#6 Keywords':>{SW}}")
            f.write("\n")
        for w in [w010, w110, w210, w310]:
            for label in sorted(w.keys()):
                result = w[label]
                f.write(_format_label(label))
                for i in range(3):
                    f.write(_format_value(result.result[i]))
                f.write(f"{result.name:>{SW}}")
                for kw in result.keywords:
                    f.write(f"{kw:>{SW}} ")
                f.write("\n")


def write_matrix_file(co, w, append=False):
    """Write a matrix output file (e.g., w020, w120, etc.)."""
    if not w:
        return
    first_label = next(iter(sorted(w.keys())))
    name = w[first_label].name
    if not co.get_compute(name) and not co.get_force(name):
        return

    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, name)
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        if not append:
            _print_explanations(f, co.infilename)
            f.write("#\n")
            f.write(f"#{'#1 label':>{SW-1}}")
            for m in range(3):
                for n in range(3):
                    col = m * 3 + n + 2
                    f.write(f"{'#' + str(col) + ' m(' + str(m) + ',' + str(n) + ')':>{SW}}")
            f.write(f"{'#11 name':>{SW}}")
            f.write(f"{'#12 Keywords':>{SW}}")
            f.write("\n")
        for label in sorted(w.keys()):
            result = w[label]
            f.write(_format_label(label))
            for m in range(3):
                for n in range(3):
                    f.write(_format_value(result.result[m, n]))
            f.write(f"{result.name:>{SW}}")
            for kw in result.keywords:
                f.write(f"{kw:>{SW}} ")
            f.write("\n")


def write_eigensystem_file(co, w_eigsys, append=False):
    """Write eigensystem output file."""
    if not w_eigsys:
        return
    first_label = next(iter(sorted(w_eigsys.keys())))
    name = w_eigsys[first_label].name
    w_name = name[:4]
    if not co.get_compute(w_name) and not co.get_force(w_name):
        return

    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, name)
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        if not append:
            _print_explanations(f, co.infilename)
            f.write("#\n")
            f.write(f"#{'#1 label':>{SW-1}}")
            f.write(f"{'#2 EVal1':>{SW}}")
            f.write(f"{'#3 EVec1[x]':>{SW}}")
            f.write(f"{'#4 EVec1[y]':>{SW}}")
            f.write(f"{'#5 EVec1[z]':>{SW}}")
            f.write(f"{'#6 EVal2':>{SW}}")
            f.write(f"{'#7 EVec2[x]':>{SW}}")
            f.write(f"{'#8 EVec2[y]':>{SW}}")
            f.write(f"{'#9 EVec2[z]':>{SW}}")
            f.write(f"{'#10 EVal3':>{SW}}")
            f.write(f"{'#11 EVec3[x]':>{SW}}")
            f.write(f"{'#12 EVec3[y]':>{SW}}")
            f.write(f"{'#13 EVec3[z]':>{SW}}")
            f.write(f"{'#14 name':>{SW}}")
            f.write(f"{'#15 Keywords':>{SW}}")
            f.write("\n")
        for label in sorted(w_eigsys.keys()):
            result = w_eigsys[label]
            f.write(_format_label(label))
            for m in range(3):
                f.write(_format_value(result.result.eigen_values[m]))
                for n in range(3):
                    f.write(_format_value(result.result.eigen_vectors[m][n]))
            f.write(f"{result.name:>{SW}}")
            for kw in result.keywords:
                f.write(f"{kw:>{SW}} ")
            f.write("\n")


def write_tensor3_file(co, w):
    """Write rank-3 tensor output file."""
    if not w:
        return
    first_label = next(iter(sorted(w.keys())))
    name = w[first_label].name
    if not co.get_compute(name) and not co.get_force(name):
        return

    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, name)
    with open(filename, "w") as f:
        f.write(f"#{'#1 label':>{SW-1}}")
        col = 2
        for m in range(3):
            for n in range(3):
                for p in range(3):
                    f.write(f"{'#' + str(col) + '  m(' + str(m) + ',' + str(n) + ',' + str(p) + ')':>{SW}}")
                    col += 1
        f.write(f"{'#29 name':>{SW}}")
        f.write(f"{'#30 Keywords':>{SW}}")
        f.write("\n")
        for label in sorted(w.keys()):
            result = w[label]
            f.write(_format_label(label))
            for m in range(3):
                for n in range(3):
                    for p in range(3):
                        f.write(_format_value(result.result[m, n, p]))
            f.write(f"{result.name:>{SW}}")
            for kw in result.keywords:
                f.write(f"{kw:>{SW}} ")
            f.write("\n")
        _print_explanations(f, co.infilename)


def write_tensor4_file(co, w):
    """Write rank-4 tensor eigenvalues output file."""
    if not w:
        return
    first_label = next(iter(sorted(w.keys())))
    name = w[first_label].name
    if not co.get_compute(name) and not co.get_force(name):
        return

    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, name + "_eigval")
    with open(filename, "w") as f:
        f.write(f"#{'#1 label':>{SW-1}}")
        for i in range(6):
            f.write(f"{'#' + str(i + 2) + ' ev' + str(i):>{SW}}")
        f.write(f"{'#8 name':>{SW}}")
        f.write(f"{'#9 keywords':>{SW}}")
        f.write("\n")
        for label in sorted(w.keys()):
            result = w[label]
            f.write(_format_label(label))
            # Compute eigenvalues of 6x6 symmetric matrix
            mat = result.result.to_numpy()
            if np.any(np.isnan(mat)):
                for _ in range(6):
                    f.write(f"{'ERROR':>{SW}}")
            else:
                ev = np.linalg.eigvalsh(mat)
                ev = ev[np.argsort(np.abs(ev))]
                for val in ev:
                    f.write(_format_value(val))
            f.write(f"{result.name:>{SW}}")
            for kw in result.keywords:
                f.write(f"{kw:>{SW}} ")
            f.write("\n")
        _print_explanations(f, co.infilename)


def write_sphmink_file(co, sphmink, append=False):
    """Write spherical Minkowski tensor output file."""
    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, "msm_ql")
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        if not append:
            _print_explanations(f, co.infilename)
            f.write("#\n")
            f.write(f"#{'#1 label':>{SW-1}}")
            for i in range(9):
                f.write(f"{'#' + str(2 + 2*i) + ' q(' + str(i) + ')':>{SW}}")
                f.write(f"{'#' + str(3 + 2*i) + ' w(' + str(i) + ')':>{SW}}")
            f.write(f"{'#20 Keywords':>{SW}}")
            f.write("\n")
        for label in sorted(sphmink.keys()):
            result = sphmink[label]
            f.write(_format_label(label))
            for i in range(9):
                f.write(_format_value(result.result.ql[i]))
                f.write(_format_value(result.result.wl[i]))
            f.write(f"{result.name:>{SW}}")
            for kw in result.keywords:
                f.write(f"{kw:>{SW}} ")
            f.write("\n")


def write_surface_props_file(co, surf_stat):
    """Write surface_props output file."""
    os.makedirs(co.outfoldername, exist_ok=True)
    filename = os.path.join(co.outfoldername, "surface_props")
    with open(filename, "w") as f:
        f.write(f"shortest edge = {surf_stat.shortest_edge}\n")
        f.write(f"longest edge  = {surf_stat.longest_edge}\n")
        f.write(f"smallest area = {surf_stat.smallest_area}\n")
        f.write(f"largest area  = {surf_stat.largest_area}\n")
        f.write("\n")
        f.write(f"{'label':>15}{'  ':>15}\n")
        for label in sorted(co.labeled_surfaces_closed.keys()):
            status = co.labeled_surfaces_closed[label]
            if label == LABEL_UNASSIGNED:
                f.write(f"{'ALL':>15}")
            else:
                f.write(f"{label:>15}")
            f.write(f"{'':>15}")
            if status == 0:
                f.write("closed\n")
            elif status == 1:
                f.write("shared\n")
            elif status == 2:
                f.write("open\n")

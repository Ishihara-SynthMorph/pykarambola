"""
Command-line interface for pykarambola.
"""

import argparse
import sys
import numpy as np

from .triangulation import Triangulation
from .io_poly import parse_poly_file
from .io_off import parse_off_file, is_off_file
from .io_obj import parse_obj_file, is_obj_file
from .io_glb import parse_glb_file, is_glb_file
from .results import CalcOptions, CALC_FORCED, CANT_CALC, REFERENCE_ORIGIN, REFERENCE_CENTROID
from .surface import check_surface
from .minkowski import (
    calculate_w000, calculate_w100, calculate_w200, calculate_w300,
    calculate_w010, calculate_w110, calculate_w210, calculate_w310,
    calculate_w020, calculate_w120, calculate_w220, calculate_w320,
    calculate_w102, calculate_w202, calculate_w103, calculate_w104,
)
from .spherical import calculate_sphmink
from .eigensystem import calculate_eigensystem
from . import output


def _set_result_metadata(w, name, co):
    """Set name, keywords, and surface status on all results."""
    for label, result in w.items():
        result.name = name
        status = co.get_label_closed_status(label)
        if status == 0:
            result.append_keyword("closed")
        elif status == 1:
            result.append_keyword("shared")
        elif status == 2:
            result.append_keyword("open")
        if co.reference_origin:
            result.append_keyword(REFERENCE_ORIGIN[0])
        else:
            result.append_keyword(REFERENCE_CENTROID[0])


def _apply_surface_check(w, name, co):
    """Check if computation is allowed for each label; set NaN if not."""
    for label, result in w.items():
        status = co.get_label_closed_status(label)
        allowed = co.get_allowed_to_calc(name)
        if status > allowed:
            if co.get_force(name):
                result.append_keyword(CALC_FORCED[0])
            else:
                _set_nan(result)
                result.append_keyword(CANT_CALC[0])


def _set_nan(result):
    """Set the result to NaN."""
    r = result.result
    if isinstance(r, (int, float)):
        result.result = float('nan')
    elif isinstance(r, np.ndarray):
        result.result = np.full_like(r, float('nan'))
    elif hasattr(r, 'set_nan'):
        r.set_nan()
    elif hasattr(r, 'ql'):
        r.ql = [float('nan')] * len(r.ql)
        r.wl = [float('nan')] * len(r.wl)


def _calculate_if_needed(name, co, surface, calc_func, *args):
    """Calculate a functional if it is requested or forced."""
    if not co.get_compute(name) and not co.get_force(name):
        return {}

    if co.reference_origin:
        w = calc_func(surface)
    else:
        w = calc_func(surface, *args)

    _set_result_metadata(w, name, co)
    _apply_surface_check(w, name, co)
    return w


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute Minkowski tensors on triangulated surfaces.",
        prog="pykarambola",
    )
    parser.add_argument("-i", "--input", dest="infile", required=True,
                        help="Input file (.poly, .off, .obj, .glb, or .gltf)")
    parser.add_argument("-o", "--output", dest="outfolder", default=None,
                        help="Output directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--labels", action="store_true",
                       help="Read labels from color alpha channel")
    group.add_argument("--nolabels", "--no-labels", action="store_true",
                       help="Do not use labels")
    parser.add_argument("--compute", action="append", default=[],
                        help="Compute specific functional (can be repeated)")
    parser.add_argument("--force", action="append", default=[],
                        help="Force computation (can be repeated)")
    parser.add_argument("--reference-centroid", "--reference_centroid",
                        action="store_true",
                        help="Use centroid instead of origin as reference")

    args = parser.parse_args(argv)

    co = CalcOptions()
    co.infilename = args.infile
    co.labels_set = args.labels
    co.reference_origin = not args.reference_centroid

    # Set output folder
    if args.outfolder:
        co.outfoldername = args.outfolder
    elif co.infilename.endswith(".poly"):
        co.outfoldername = co.infilename[:-5] + "_mink_val"
    elif co.infilename.endswith(".off"):
        co.outfoldername = co.infilename[:-4] + "_mink_val"
    elif co.infilename.endswith(".obj"):
        co.outfoldername = co.infilename[:-4] + "_mink_val"
    elif co.infilename.endswith(".glb"):
        co.outfoldername = co.infilename[:-4] + "_mink_val"
    elif co.infilename.endswith(".gltf"):
        co.outfoldername = co.infilename[:-5] + "_mink_val"
    else:
        print("No output folder can be defined", file=sys.stderr)
        sys.exit(1)

    # Set computes
    if args.compute:
        for w in args.compute:
            co.set_compute(w, True)
    else:
        co.set_default_computes()

    for w in args.force:
        co.set_force(w, True)

    # Parse input
    print(f"Converting file to Minkowski triangulation format ...", end="", flush=True)

    with_labels = co.labels_set
    if is_off_file(co.infilename):
        surface = parse_off_file(co.infilename, with_labels)
    elif is_obj_file(co.infilename):
        surface = parse_obj_file(co.infilename, with_labels)
    elif is_glb_file(co.infilename):
        surface = parse_glb_file(co.infilename, with_labels)
    else:
        surface = parse_poly_file(co.infilename, with_labels)

    surface.create_vertex_polygon_lookup_table()
    surface.create_polygon_polygon_lookup_table()
    print("     done")

    # Check surface
    print("check surface ...")
    surf_stat = check_surface(co, surface)
    print(f"shortest edge = {surf_stat.shortest_edge}")
    print(f"longest edge  = {surf_stat.longest_edge}")
    print(f"smallest area = {surf_stat.smallest_area}")
    print(f"largest area  = {surf_stat.largest_area}")
    print("                       done")

    # Calculate scalars
    print("calculate w000 ...", end="", flush=True)
    w000 = _calculate_if_needed("w000", co, surface, calculate_w000)
    print("done")

    print("calculate w100 ...", end="", flush=True)
    w100 = _calculate_if_needed("w100", co, surface, calculate_w100)
    print("done")

    print("calculate w200 ...", end="", flush=True)
    w200 = _calculate_if_needed("w200", co, surface, calculate_w200)
    print("done")

    print("calculate w300 ...", end="", flush=True)
    w300 = _calculate_if_needed("w300", co, surface, calculate_w300)
    print("done")

    # Calculate vectors
    print("calculate w010 ...", end="", flush=True)
    w010 = _calculate_if_needed("w010", co, surface, calculate_w010)
    print("done")

    print("calculate w110 ...", end="", flush=True)
    w110 = _calculate_if_needed("w110", co, surface, calculate_w110)
    print("done")

    print("calculate w210 ...", end="", flush=True)
    w210 = _calculate_if_needed("w210", co, surface, calculate_w210)
    print("done")

    print("calculate w310 ...", end="", flush=True)
    w310 = _calculate_if_needed("w310", co, surface, calculate_w310)
    print("done")

    # Calculate matrices - with centroid reference when not using origin
    print("calculate w020 ...", end="", flush=True)
    if co.get_compute("w020") or co.get_force("w020"):
        if co.reference_origin:
            w020 = calculate_w020(surface)
        else:
            w020 = calculate_w020(surface, w000, w010)
        _set_result_metadata(w020, "w020", co)
        _apply_surface_check(w020, "w020", co)
    else:
        w020 = {}
    print("done")

    print("calculate w120 ...", end="", flush=True)
    if co.get_compute("w120") or co.get_force("w120"):
        if co.reference_origin:
            w120 = calculate_w120(surface)
        else:
            w120 = calculate_w120(surface, w100, w110)
        _set_result_metadata(w120, "w120", co)
        _apply_surface_check(w120, "w120", co)
    else:
        w120 = {}
    print("done")

    print("calculate w220 ...", end="", flush=True)
    if co.get_compute("w220") or co.get_force("w220"):
        if co.reference_origin:
            w220 = calculate_w220(surface)
        else:
            w220 = calculate_w220(surface, w200, w210)
        _set_result_metadata(w220, "w220", co)
        _apply_surface_check(w220, "w220", co)
    else:
        w220 = {}
    print("done")

    print("calculate w320 ...", end="", flush=True)
    if co.get_compute("w320") or co.get_force("w320"):
        if co.reference_origin:
            w320 = calculate_w320(surface)
        else:
            w320 = calculate_w320(surface, w300, w310)
        _set_result_metadata(w320, "w320", co)
        _apply_surface_check(w320, "w320", co)
    else:
        w320 = {}
    print("done")

    print("calculate w102 ...", end="", flush=True)
    w102 = _calculate_if_needed("w102", co, surface, calculate_w102)
    print("done")

    print("calculate w202 ...", end="", flush=True)
    w202 = _calculate_if_needed("w202", co, surface, calculate_w202)
    print("done")

    # Higher-order tensors
    if co.get_compute("w104") or co.get_force("w104"):
        w104 = calculate_w104(surface)
        _set_result_metadata(w104, "w104", co)
        _apply_surface_check(w104, "w104", co)
    else:
        w104 = {}

    if co.get_compute("w103") or co.get_force("w103"):
        w103 = calculate_w103(surface)
        _set_result_metadata(w103, "w103", co)
        _apply_surface_check(w103, "w103", co)
    else:
        w103 = {}

    # Spherical Minkowski tensors
    print("calculate msm ...", end="", flush=True)
    if co.get_compute("msm") or co.get_force("msm"):
        sphmink = calculate_sphmink(surface)
        _set_result_metadata(sphmink, "msm", co)
        _apply_surface_check(sphmink, "msm", co)
    else:
        sphmink = {}
    print("done")

    print()

    # Write results
    print("write results to files ...", end="", flush=True)
    output.write_surface_props_file(co, surf_stat)
    output.write_scalar_file(co, w000, w100, w200, w300)
    output.write_vector_file(co, w010, w110, w210, w310)
    output.write_matrix_file(co, w020)
    output.write_matrix_file(co, w120)
    output.write_matrix_file(co, w220)
    output.write_matrix_file(co, w320)
    output.write_matrix_file(co, w102)
    output.write_matrix_file(co, w202)
    output.write_tensor3_file(co, w103)
    output.write_tensor4_file(co, w104)
    output.write_sphmink_file(co, sphmink)
    print("     done")

    # Eigensystems
    print("calculate and write eigensystems to file ...", end="", flush=True)
    for w_mat in [w020, w120, w220, w320, w102, w202]:
        if w_mat:
            eigsys = calculate_eigensystem(w_mat)
            output.write_eigensystem_file(co, eigsys)
    print("     done")

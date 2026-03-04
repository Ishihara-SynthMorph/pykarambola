"""
Result types and calculation options for Minkowski tensors.
"""

from dataclasses import dataclass, field
from typing import Any
import math


LABEL_UNASSIGNED = -300


@dataclass
class MinkValResult:
    """Stores a single Minkowski tensor result."""
    result: Any = None
    name: str = "name_not_yet_assigned"
    keywords: list = field(default_factory=list)
    comments: list = field(default_factory=list)
    computation_successful: bool = True

    def append_keyword(self, keyword):
        if keyword not in self.keywords:
            self.keywords.append(keyword)

    def append_comment(self, comment):
        if comment not in self.comments:
            self.comments.append(comment)


@dataclass
class SurfaceStatistics:
    """Statistics about the surface mesh."""
    shortest_edge: float = math.inf
    longest_edge: float = 0.0
    smallest_area: float = math.inf
    largest_area: float = 0.0


# Keyword constants
CALC_FORCED = ("calculation_forced",
               "stubbornly applied the formulae even though the surface is shared or open")
CANT_CALC = ("cant_be_calculated",
             "can't be calculated because surface is shared or open")
REFERENCE_ORIGIN = ("reference_origin",
                    "the tensors are calculated with respect to the origin")
REFERENCE_CENTROID = ("reference_centroid",
                      "the tensors are calculated with respect to the minkowski-vectors")

COMPUTABLE_W = [
    "w000", "w100", "w200", "w300",
    "w010", "w110", "w210", "w310",
    "w020", "w120", "w220", "w320",
    "w102", "w202", "w103", "w104", "msm",
]


class CalcOptions:
    """Configuration for which Minkowski tensors to compute."""

    def __init__(self):
        self.infilename = ""
        self.outfoldername = ""
        self.reference_origin = True
        self.labels_set = False

        self.compute = {w: False for w in COMPUTABLE_W}
        self.force = {w: False for w in COMPUTABLE_W}

        # Surface type required: 0=closed only, 1=closed+shared, 2=all
        self.allowed_to_calc = {
            "w000": 0, "w100": 2, "w200": 1, "w300": 1,
            "w010": 0, "w110": 2, "w210": 1, "w310": 1,
            "w020": 0, "w120": 2, "w220": 1, "w320": 1,
            "w102": 2, "w202": 1, "w103": 2, "w104": 2, "msm": 2,
        }

        # label -> surface status (0=closed, 1=shared, 2=open)
        self.labeled_surfaces_closed = {}

    def set_compute(self, w, status):
        if w not in self.compute:
            raise ValueError(f"Unknown functional: {w}")
        self.compute[w] = status

    def set_force(self, w, status):
        if w not in self.force:
            raise ValueError(f"Unknown functional: {w}")
        self.force[w] = status

    def get_compute(self, w):
        return self.compute[w]

    def get_force(self, w):
        return self.force[w]

    def get_allowed_to_calc(self, w):
        return self.allowed_to_calc[w]

    def get_label_closed_status(self, label):
        return self.labeled_surfaces_closed[label]

    def create_label(self, label, status):
        self.labeled_surfaces_closed[label] = status

    def set_default_computes(self):
        """Set default functionals to compute when no explicit --compute given."""
        for w in ["w000", "w100", "w200", "w300",
                   "w010", "w110", "w210", "w310",
                   "w020", "w120", "w220", "w320",
                   "w102", "w202", "msm"]:
            self.compute[w] = True

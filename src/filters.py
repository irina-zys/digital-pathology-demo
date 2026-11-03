# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from CODAIT/deep-histopath at c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33:
# deephistopath/wsi/filter.py and deephistopath/preprocessing.py.
# The latter includes software licensed to the Apache Software Foundation
# under contributor agreements; see third_party/CODAIT-NOTICE.txt.
# Modifications: compact threshold tables, RGB validation, bool-only masks,
# no Spark/timing dependencies, configurable optical-density background.

"""CODAIT colour filters. True means KEEP; invert to obtain an artefact mask."""

from collections.abc import Mapping

import numpy as np

# Re-export for callers using the original filter API.
from src.image_utils import validate_rgb

PEN_THRESHOLDS = {
    "red": ((150, 80, 90), (110, 20, 30), (185, 65, 105), (195, 85, 125),
            (220, 115, 145), (125, 40, 70), (200, 120, 150), (100, 50, 65), (85, 25, 45)),
    "green": ((150, 160, 140), (70, 110, 110), (45, 115, 100), (30, 75, 60),
              (195, 220, 210), (225, 230, 225), (170, 210, 200), (20, 30, 20),
              (50, 60, 40), (30, 50, 35), (65, 70, 60), (100, 110, 105),
              (165, 180, 180), (140, 140, 150), (185, 195, 195)),
    "blue": ((60, 120, 190), (120, 170, 200), (175, 210, 230), (145, 180, 210),
             (37, 95, 160), (30, 65, 130), (130, 155, 180), (40, 35, 85),
             (30, 20, 65), (90, 90, 140), (60, 60, 120), (110, 110, 175)),
}


def resolve_pen_thresholds(overrides=None):
    """Return validated RGB rule triples; omitted colours retain CODAIT defaults.

    Rules are OR'ed per colour. Red: R>r & G<g & B<b; green: R<r & G>g
    & B>b; blue: R<r & G<g & B>b. A supplied colour replaces all its rules.
    Empty rules disable that colour. Values are integers in 0-255.
    """
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("pen_thresholds must be a mapping of colour to RGB rule triples")
    rules = dict(PEN_THRESHOLDS)
    for color, values in (overrides or {}).items():
        if color not in rules:
            raise ValueError(f"Unknown pen colour: {color}")
        array = np.asarray(values)
        if array.size == 0 and array.shape == (0,):
            rules[color] = ()
            continue
        if array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "iu" or np.any((array < 0) | (array > 255)):
            raise ValueError(f"{color} pen rules must be N x 3 integer RGB thresholds in 0-255")
        rules[color] = tuple(tuple(map(int, row)) for row in array)
    return rules


def filter_pen(rgb, color, thresholds=None):
    """Return a KEEP mask using CODAIT defaults or custom rules for this colour."""
    validate_rgb(rgb)
    if color not in PEN_THRESHOLDS:
        raise ValueError(f"Unknown pen colour: {color}")
    red, green, blue = rgb.transpose(2, 0, 1)
    keep = np.ones(rgb.shape[:2], dtype=bool)
    rules = resolve_pen_thresholds({color: thresholds} if thresholds is not None else None)
    for r, g, b in rules[color]:
        if color == "red":
            suspect = (red > r) & (green < g) & (blue < b)
        elif color == "green":
            suspect = (red < r) & (green > g) & (blue > b)
        else:
            suspect = (red < r) & (green < g) & (blue > b)
        keep &= ~suspect
    return keep


def filter_grays(rgb, tolerance=15):
    """Keep pixels whose RGB channel range exceeds the CODAIT tolerance."""
    validate_rgb(rgb)
    if not 0 <= tolerance <= 255:
        raise ValueError("Gray tolerance must be 0-255")
    return np.ptp(rgb.astype(np.int16), axis=2) > tolerance


def optical_density(rgb, background_intensity=240):
    """CODAIT OD convention: -ln((I + 1) / background); bright pixels may be negative."""
    validate_rgb(rgb)
    if not 0 < background_intensity <= 255:
        raise ValueError("Background intensity must be in (0, 255]")
    return -np.log((rgb.astype(np.float64) + 1) / background_intensity)

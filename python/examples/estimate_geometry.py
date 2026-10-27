"""Estimate the aerodynamics of a glider and print the report:
    python python/examples/estimate_geometry.py
Edit the geometry below, or build it from mesh extents with GliderGeometry.from_bounding_box.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glider_cae.estimator import LiftingLineEstimator
from glider_cae.geometry import GliderGeometry

geometry = GliderGeometry.from_bounding_box(span=2.0, length=1.1, height=0.2, mass=1.2)
report = LiftingLineEstimator(geometry).report()

for key, value in report.items():
    print(f"{key:22s} {value:.4g}" if isinstance(value, float) else f"{key:22s} {value}")

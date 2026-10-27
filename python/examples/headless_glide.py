"""Headless sanity run: python python/examples/headless_glide.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from glider_cae.simulation import GliderSimulation

sim = GliderSimulation()
sim.reset(altitude=200.0)
t = sim.trim
print(f"Trim: alpha={np.degrees(t.alpha):.2f} deg  V={t.airspeed:.2f} m/s  "
      f"gamma={np.degrees(t.gamma):.2f} deg  L/D={t.glide_ratio:.1f}")

out = sim.run(60.0, record_every=100)
for ti, s in zip(out["t"][::10], out["state"][::10]):
    print(f"t={ti:5.1f}s  alt={-s[2]:7.2f} m  range={np.hypot(s[0], s[1]):7.1f} m  "
          f"V={np.linalg.norm(s[7:10]):5.2f} m/s")
print("status:", sim.status)

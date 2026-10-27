"""Write two sample glider models to examples/models/ to try the web app and the parser:
    python python/examples/make_demo_mesh.py
  demo_glider_mm.stl   millimetres, Z up, nose toward -Y   (typical CAD export)
  demo_glider_yup.obj  metres, Y up, nose toward +Z        (typical game-engine export)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glider_cae.demo_mesh import make_demo_glider, reorient, write_obj, write_stl_binary

out = Path(__file__).resolve().parents[2] / "examples" / "models"
out.mkdir(parents=True, exist_ok=True)

vertices, faces, truth = make_demo_glider()          # body axes: x forward, y right, z down
write_stl_binary(out / "demo_glider_mm.stl",
                 reorient(vertices, perm=(1, 0, 2), signs=(1, -1, -1)) * 1000.0, faces)
write_obj(out / "demo_glider_yup.obj", reorient(vertices, perm=(1, 2, 0), signs=(1, -1, 1)), faces)
print("wrote", *sorted(p.name for p in out.iterdir()))
print("ground truth:", {k: round(v, 3) for k, v in truth.items() if k in ("span", "area", "sweep_deg", "dihedral_deg")})

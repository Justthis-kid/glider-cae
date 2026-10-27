"""Stage 3 tests: mesh loading, orientation, measurement, mass properties, full pipeline."""
import base64
import itertools
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from glider_cae.demo_mesh import (make_demo_glider, reorient, write_obj, write_stl_ascii,
                                  write_stl_binary)
from glider_cae.estimator import LiftingLineEstimator
from glider_cae.mesh_analysis import MeshOptions, analyze_mesh
from glider_cae.mesh_geometry import TriangleMesh, detect_orientation
from glider_cae.mesh_io import (MeshData, MeshError, decimate, load_mesh, pack_preview, read_obj,
                                read_stl, suggest_units)
from glider_cae.mesh_mass import is_watertight, mass_properties
from glider_cae.mesh_measure import measure_tails_and_fuselage, measure_wing
from glider_cae.simulation import GliderSimulation


# ---------------------------------------------------------------- helpers
def cube(side=1.0):
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * side
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def uv_sphere(radius=1.0, n_lat=60, n_lon=120):
    lat = np.linspace(0, np.pi, n_lat + 1)[1:-1]
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    ring = np.stack([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)], -1).reshape(-1, 3)
    v = np.vstack([ring, [[0, 0, 1], [0, 0, -1]]]) * radius
    n_top, n_bot = len(ring), len(ring) + 1
    faces = []
    idx = np.arange(n_lat - 1)[:, None] * n_lon + np.arange(n_lon)[None, :]
    nxt = np.arange(n_lat - 1)[:, None] * n_lon + (np.arange(n_lon)[None, :] + 1) % n_lon
    faces += [np.stack([idx[:-1].ravel(), nxt[:-1].ravel(), idx[1:].ravel()], 1),
              np.stack([nxt[:-1].ravel(), nxt[1:].ravel(), idx[1:].ravel()], 1)]
    first, last = np.arange(n_lon), (n_lat - 2) * n_lon + np.arange(n_lon)
    faces += [np.stack([np.full(n_lon, n_top), (first + 1) % n_lon, first], 1),
              np.stack([np.full(n_lon, n_bot), last, (last - last[0] + 1) % n_lon + last[0]], 1)]
    return v, np.concatenate(faces)


def cad_export(v, units=1000.0):
    """What a CAD export often looks like: millimetres, Z up, nose toward -Y, span along X."""
    return reorient(v, perm=(1, 0, 2), signs=(1, -1, -1)) * units


# ---------------------------------------------------------------- loading
def test_builtin_readers_round_trip_all_formats():
    v, f, _ = make_demo_glider()
    with tempfile.TemporaryDirectory() as d:
        write_stl_binary(f"{d}/a.stl", v, f)
        write_stl_ascii(f"{d}/b.stl", v, f)
        write_obj(f"{d}/c.obj", v, f)
        for reader, name in ((read_stl, "a.stl"), (read_stl, "b.stl"), (read_obj, "c.obj")):
            rv, rf = reader(Path(f"{d}/{name}").read_bytes())
            assert len(rf) == len(f)
            assert np.allclose(np.ptp(rv, axis=0), np.ptp(v, axis=0), atol=1e-5)
        for name in ("a.stl", "b.stl", "c.obj"):                       # whichever loader is installed
            m = load_mesh(f"{d}/{name}")
            assert len(m.faces) == len(f) and np.allclose(np.ptp(m.vertices, axis=0), np.ptp(v, axis=0), atol=1e-5)


def test_loader_rejects_bad_files():
    with tempfile.TemporaryDirectory() as d:
        Path(f"{d}/empty.obj").write_text("# nothing here")
        with pytest.raises(MeshError):
            load_mesh(f"{d}/empty.obj")
        Path(f"{d}/nan.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 nan\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4")
        with pytest.raises(MeshError):
            load_mesh(f"{d}/nan.obj")


def test_unit_guess():
    v, _, _ = make_demo_glider()
    assert (suggest_units(v), suggest_units(v * 100), suggest_units(v * 1000)) == ("m", "cm", "mm")


def test_preview_pack_round_trip():
    v, f, _ = make_demo_glider()
    pk = pack_preview(v, f, max_faces=900)
    raw = base64.b64decode(pk["data"])
    assert len(raw) == pk["vertices"] * 12 + pk["faces"] * 12
    verts = np.frombuffer(raw, "<f4", pk["vertices"] * 3).reshape(-1, 3)
    faces = np.frombuffer(raw, "<u4", offset=pk["vertices"] * 12).reshape(-1, 3)
    assert faces.max() < pk["vertices"] and pk["faces"] <= 900
    assert np.allclose(np.ptp(verts, axis=0), np.ptp(v, axis=0), rtol=0.05)
    assert decimate(v, f, 10 ** 6)[1] is f                                # small meshes pass through untouched


# ---------------------------------------------------------------- orientation
@pytest.mark.parametrize("kwargs", [
    {}, {"dihedral_deg": 0.0, "vtail_height": 0.12}, {"sweep_deg": 20.0},
    {"span": 3.0, "root_chord": 0.22, "tip_chord": 0.10, "htail_span": 0.55,
     "fuselage_length": 1.4, "tail_arm": 0.9},
])
def test_orientation_recovered_for_all_48_axis_arrangements(kwargs):
    v, f, _ = make_demo_glider(**kwargs)
    c0 = v - v.mean(0)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            vv = reorient(v, perm, signs)
            o = detect_orientation(TriangleMesh(vv, f))
            back = vv @ o.rotation.T
            c = back - back.mean(0)
            assert np.allclose(c, c0, atol=1e-5) or np.allclose(c * [1, -1, 1], c0, atol=1e-5)
            assert np.linalg.det(o.rotation) == pytest.approx(1.0)


def test_manual_orientation_overrides_detection():
    v, f, _ = make_demo_glider()
    vv = cad_export(v, 1.0)
    auto = detect_orientation(TriangleMesh(vv, f))
    manual = detect_orientation(TriangleMesh(vv, f), forward=auto.forward, up=auto.up)
    assert np.allclose(manual.rotation, auto.rotation)
    flipped = detect_orientation(TriangleMesh(vv, f), forward="+Y", up="+Z")      # tail first, deliberately wrong
    assert flipped.forward == "+Y"
    with pytest.raises(ValueError):
        detect_orientation(TriangleMesh(vv, f), forward="+X", up="-X")
    with pytest.raises(ValueError):
        detect_orientation(TriangleMesh(vv, f), forward="north")


# ---------------------------------------------------------------- measurement against ground truth
def test_measurements_match_the_synthetic_glider():
    v, f, truth = make_demo_glider()
    body = TriangleMesh(v - 0.5 * (v.min(0) + v.max(0)), f)
    w = measure_wing(body)
    t = measure_tails_and_fuselage(body, w)
    rel = lambda a, b: abs(a - b) / abs(b)                                        # noqa: E731
    assert rel(w.span, truth["span"]) < 0.005
    assert rel(w.root_chord, truth["root_chord"]) < 0.03 and rel(w.tip_chord, truth["tip_chord"]) < 0.03
    assert rel(w.area, truth["area"]) < 0.02 and rel(w.mac, truth["mac"]) < 0.02
    assert abs(w.sweep_deg - truth["sweep_deg"]) < 0.5
    assert abs(w.dihedral_deg - truth["dihedral_deg"]) < 0.5
    assert abs(w.incidence_deg - truth["incidence_deg"]) < 0.5
    assert abs(w.camber - truth["camber"]) < 0.006 and rel(w.thickness, truth["thickness"]) < 0.10
    assert rel(t.fuselage_length, truth["fuselage_length"]) < 0.02
    assert rel(t.fuselage_diameter, truth["fuselage_diameter"]) < 0.15
    assert rel(t.htail["span"], truth["htail_span"]) < 0.02 and rel(t.htail["chord"], truth["htail_chord"]) < 0.05
    assert rel(t.vtail["height"], truth["vtail_height"]) < 0.03 and rel(t.vtail["chord"], truth["vtail_chord"]) < 0.06
    assert abs(t.htail["arm"] - truth["htail_arm"]) < 0.03
    assert abs(t.vtail["arm"] - truth["vtail_arm"]) < 0.03


def test_other_gliders_are_measured_too():
    v, f, truth = make_demo_glider(span=3.0, root_chord=0.25, tip_chord=0.09, sweep_deg=-3.0,
                                   dihedral_deg=2.0, htail_span=0.6, htail_chord=0.11,
                                   fuselage_length=1.5, tail_arm=0.95)
    a = analyze_mesh(MeshData(cad_export(v, 1000.0), f), MeshOptions(mass=2.0))
    g = a.geometry
    assert g.span == pytest.approx(3.0, rel=0.01) and g.area == pytest.approx(truth["area"], rel=0.03)
    assert g.sweep_deg == pytest.approx(-3.0, abs=0.7) and g.dihedral_deg == pytest.approx(2.0, abs=0.7)
    assert g.htail_span == pytest.approx(0.6, rel=0.03)


def test_no_wing_is_a_clear_error():
    v, f = cube(0.5)
    with pytest.raises(MeshError):
        analyze_mesh(MeshData(v, f), MeshOptions())


def test_missing_tails_fall_back_to_sized_tails():
    v, f, _ = make_demo_glider()
    keep = (v[f][:, :, 0] > -0.35).all(axis=1)                          # chop off everything aft of x = -0.35
    a = analyze_mesh(MeshData(v, f[keep]), MeshOptions(mass=1.2))
    assert any("horizontal tail" in n for n in a.notes) and any("fin" in n for n in a.notes)
    assert a.geometry.validate() == []
    assert LiftingLineEstimator(a.geometry).report()["trim"] is not None


# ---------------------------------------------------------------- mass properties
def test_cube_properties():
    v, f = cube(2.0)
    assert is_watertight(f)
    solid, shell = mass_properties(v, f, "volume"), mass_properties(v, f, "shell")
    assert solid.volume == pytest.approx(8.0) and np.allclose(solid.centroid, 1.0)
    assert np.allclose(np.diag(solid.inertia_unit), 4.0 / 6.0)                    # a^2 / 6
    assert np.allclose(np.diag(shell.inertia_unit), 5.0 / 18.0 * 4.0)             # hollow cube: 5 a^2 / 18
    assert np.allclose(solid.inertia_unit - np.diag(np.diag(solid.inertia_unit)), 0.0, atol=1e-12)


def test_sphere_properties_and_translation_invariance():
    v, f = uv_sphere(1.5)
    for shift in (np.zeros(3), np.array([3.0, -2.0, 7.0])):
        vol = mass_properties(v + shift, f, "volume")
        shell = mass_properties(v + shift, f, "shell")
        assert np.allclose(vol.centroid, shift, atol=1e-6) and np.allclose(shell.centroid, shift, atol=1e-6)
        assert np.diag(vol.inertia_unit).mean() == pytest.approx(0.4 * 1.5 ** 2, rel=0.01)
        assert np.diag(shell.inertia_unit).mean() == pytest.approx(2.0 / 3.0 * 1.5 ** 2, rel=0.01)
    assert mass_properties(v, f, "volume").volume == pytest.approx(4 / 3 * math.pi * 1.5 ** 3, rel=0.01)


def test_open_mesh_uses_shell_model():
    v, f = cube()
    mp = mass_properties(v, f[:-2], "volume")
    assert not mp.watertight and mp.model == "shell"


def test_inertia_is_a_valid_tensor_in_the_body_frame():
    v, f, _ = make_demo_glider()
    a = analyze_mesh(MeshData(cad_export(v), f), MeshOptions(mass=1.2))
    inertia = a.geometry.inertia
    assert np.allclose(inertia, inertia.T) and np.all(np.linalg.eigvalsh(inertia) > 0)
    ixx, iyy, izz = np.diag(inertia)
    assert izz < ixx + iyy + 1e-9 and ixx < iyy + izz and iyy < ixx + izz       # triangle inequalities


# ---------------------------------------------------------------- full pipeline
def test_pipeline_is_invariant_to_axes_and_units():
    v, f, _ = make_demo_glider()
    ref = analyze_mesh(MeshData(v, f), MeshOptions(mass=1.2, units="m")).geometry
    for perm, signs, scale, units in [((1, 0, 2), (1, -1, -1), 1000.0, "auto"), ((2, 1, 0), (-1, 1, 1), 100.0, "auto"),
                                      ((0, 2, 1), (1, -1, 1), 39.3701, "in")]:
        g = analyze_mesh(MeshData(reorient(v, perm, signs) * scale, f), MeshOptions(mass=1.2, units=units)).geometry
        assert g.span == pytest.approx(ref.span, rel=0.002) and g.area == pytest.approx(ref.area, rel=0.005)
        assert g.htail_arm == pytest.approx(ref.htail_arm, abs=0.01)
        assert g.vtail_arm == pytest.approx(ref.vtail_arm, abs=0.01)
        assert np.allclose(np.sort(np.diag(g.inertia)), np.sort(np.diag(ref.inertia)), rtol=0.02)


def test_cg_modes():
    v, f, _ = make_demo_glider()
    mesh = MeshData(cad_export(v), f)
    auto = analyze_mesh(mesh, MeshOptions(mass=1.2)).geometry
    assert auto.cg_frac_mac is None
    manual = analyze_mesh(mesh, MeshOptions(mass=1.2, cg_mode="manual", cg_manual=0.33)).geometry
    assert manual.cg_frac_mac == pytest.approx(0.33)
    from_mesh = analyze_mesh(mesh, MeshOptions(mass=1.2, cg_mode="mesh"))
    assert from_mesh.geometry.cg_frac_mac is not None and 0.0 <= from_mesh.geometry.cg_frac_mac <= 0.8
    assert from_mesh.cg_mesh_frac > 0.8                                          # surface-mass centroid sits far aft
    with pytest.raises(MeshError):
        analyze_mesh(mesh, MeshOptions(cg_mode="sideways"))


def test_mass_is_estimated_when_missing_and_scales_inertia():
    v, f, _ = make_demo_glider()
    a = analyze_mesh(MeshData(v, f), MeshOptions(units="m"))
    assert a.mass_assumed and a.geometry.mass == pytest.approx(3.5 * a.geometry.area)
    b = analyze_mesh(MeshData(v, f), MeshOptions(units="m", mass=2 * a.geometry.mass))
    assert np.allclose(b.geometry.inertia, 2 * a.geometry.inertia)


def test_mesh_glider_flies():
    v, f, _ = make_demo_glider()
    a = analyze_mesh(MeshData(cad_export(v), f), MeshOptions(mass=1.2))
    sim = GliderSimulation.from_geometry(a.geometry)
    sim.reset(altitude=300.0)
    state = sim.run(30.0)["state"]
    assert sim.status == "flying" and np.all(np.isfinite(state)) and sim.trim is not None
    alpha = np.degrees(np.arctan2(state[:, 9], state[:, 7]))
    assert alpha.min() > -3.0 and alpha.max() < 9.0


def test_wrong_units_are_flagged():
    v, f, _ = make_demo_glider()
    a = analyze_mesh(MeshData(v * 1000.0, f), MeshOptions(units="m"))             # claims metres, is really mm
    assert any("span" in w for w in a.warnings)


def test_measurements_tolerate_vertex_noise():
    v, f, truth = make_demo_glider()
    noisy = v + np.random.default_rng(3).normal(0.0, 0.001, v.shape)          # 1 mm of scan noise
    g = analyze_mesh(MeshData(cad_export(noisy), f), MeshOptions(mass=1.2)).geometry
    assert g.span == pytest.approx(truth["span"], rel=0.01) and g.area == pytest.approx(truth["area"], rel=0.03)
    assert g.sweep_deg == pytest.approx(truth["sweep_deg"], abs=1.0)
    assert g.dihedral_deg == pytest.approx(truth["dihedral_deg"], abs=1.0)


def test_bridge_mesh_workflow():
    import json
    from glider_cae import bridge

    v, f, _ = make_demo_glider()
    with tempfile.TemporaryDirectory() as d:
        write_stl_binary(f"{d}/up.stl", cad_export(v), f)
        assert json.loads(bridge.init())["analysis"]["mesh"] is None
        raw = bridge.load_mesh(f"{d}/up.stl", json.dumps({"name": "demo.stl", "mass": 1.2}))
        out = json.loads(json.dumps(json.loads(raw), allow_nan=False))        # strictly valid JSON for JSON.parse
        mesh = out["analysis"]["mesh"]
        assert out["analysis"]["source"] == "mesh" and mesh["name"] == "demo.stl"
        assert mesh["orientation"]["confidence"] == "high" and mesh["preview"]["faces"] == len(f)
        assert len(mesh["cg_body"]) == 3 and out["state"]["status"] == "flying"

        edited = json.loads(bridge.configure(json.dumps({"mass": 2.4})))      # edits keep the model, scale inertia
        assert edited["analysis"]["geometry"]["mass"] == 2.4 and "preview" not in edited["analysis"]["mesh"]

        wrong = json.loads(bridge.reanalyze(json.dumps({"units": "parsecs"})))
        assert "error" in wrong and json.loads(bridge.reanalyze("{}"))["analysis"]["mesh"]["name"] == "demo.stl"
        assert "error" in json.loads(bridge.load_mesh(f"{d}/missing.stl", "{}"))
        assert json.loads(bridge.use_default())["analysis"]["source"] == "default"

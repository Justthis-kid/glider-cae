# Glider flight lab

Modular glider CAE and flight-physics suite. The Python package in `python/glider_cae` is the single
source of truth for the physics. It runs natively (tests, CLI) and, through Pyodide in a web worker,
inside the React app, which is why the whole thing can be hosted on GitHub Pages.

## Run it (PowerShell, VS Code terminal)

```powershell
npm install
npm run dev            # http://localhost:5173
```

The first dev or build run downloads the Pyodide runtime into `public/pyodide`. Later runs load it
locally, so the simulator works without internet when served from this machine. Loading extra mesh
formats through `trimesh` still requires internet; STL and OBJ use the built-in offline readers.

Python tests and a headless run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python/requirements.txt
python -m pytest -q
python python/examples/headless_glide.py
python python/examples/estimate_geometry.py
python python/examples/make_demo_mesh.py
```

## Desktop apps (Windows, macOS Apple Silicon and Intel)

The same app is also packaged with Electron. The desktop builds bundle the Python runtime, so
they run without an internet connection (installing `trimesh` for extra mesh formats is the one
thing that needs the network; STL and OBJ work offline).

Run it as a desktop window from VS Code:

```powershell
npm install
npm run electron:dev       # live-reloading window; the web runtime loads from the CDN
npm run electron:start     # downloads the Python runtime once, builds, opens the offline app
```

In VS Code you can also use Run and Debug, "Desktop app (built)".

Build installers on your own machine (each OS builds its own):

```powershell
npm run dist:win           # on Windows -> release\  Setup .exe and Portable .exe
npm run dist:mac           # on a Mac   -> release/  .dmg and .zip for Apple Silicon (arm64) and Intel (x64)
```

Or let GitHub build all of them. Push a version tag and the "Desktop apps" workflow publishes the
files on the repository's Releases page:

```powershell
git tag v0.3.0
git push origin v0.3.0
```

First launch of an unsigned app:

- Windows: SmartScreen says "Windows protected your PC". Choose "More info", then "Run anyway".
- macOS: the app is ad-hoc signed, not notarised. Open the .dmg, drag the app to Applications,
  then right-click it and choose Open the first time. If macOS still refuses ("damaged"), run
  `xattr -cr "/Applications/Glider Flight Lab.app"` in Terminal once.
- If `npm run dist:win` fails with a symbolic-link error, turn on Developer Mode in Windows
  Settings (For developers) or run the terminal as Administrator, then try again.

Removing those warnings needs paid certificates (Apple Developer ID with notarisation, a Windows
code-signing certificate), supplied to electron-builder through its standard `CSC_*` variables.

## Deploy to GitHub Pages

```powershell
git init
git add .
git commit -m "Stage 1: 6DOF physics engine and web UI"
git branch -M main
git remote add origin https://github.com/<your-username>/glider-cae.git
git push -u origin main
```

Then in the repo: Settings, Pages, Source = GitHub Actions. Every push to `main` runs the Python
tests, builds the site, and publishes it to `https://<your-username>.github.io/glider-cae/`.

## Layout

| Path | Role |
| --- | --- |
| `python/glider_cae/rigid_body.py` | Quaternion math and the 13-state 6DOF equations of motion |
| `python/glider_cae/dynamics.py` | Gravity + aerodynamic forces and moments |
| `python/glider_cae/aero.py` | `AeroModel` interface and the stage 1 placeholder model |
| `python/glider_cae/geometry.py` | `GliderGeometry`: the descriptor stage 3 (mesh parser) will fill in |
| `python/glider_cae/lifting_line.py` | Glauert lifting-line solver (lift slope, span efficiency, loading) |
| `python/glider_cae/estimator.py` | Stage 2: component build-up aero model, polar, stability report |
| `python/glider_cae/mesh_io.py` | Stage 3: loads STL / OBJ (trimesh, with a built-in fallback), units, 3D preview packing |
| `python/glider_cae/mesh_geometry.py` | Triangle mesh sections and automatic orientation (span, nose, up) |
| `python/glider_cae/mesh_measure.py` | Wing planform, airfoil, fuselage and tails measured from sections |
| `python/glider_cae/mesh_mass.py` | Centroid and inertia tensor by exact integration over triangles |
| `python/glider_cae/mesh_analysis.py` | Mesh in, `GliderGeometry` out, with notes and warnings |
| `python/glider_cae/demo_mesh.py` | Synthetic glider with known dimensions, STL / OBJ writers |
| `examples/models/` | Two sample models to load in the app (CAD-style mm STL, game-style OBJ) |
| `python/glider_cae/trim.py` | Steady-glide trim solver for clean initial conditions |
| `python/glider_cae/simulation.py` | Fixed-step loop (RK4), telemetry snapshot, batch `run()` |
| `python/glider_cae/bridge.py` | JSON bridge called from the browser |
| `src/workers/physics.worker.js` | Runs Pyodide and the Python package off the UI thread |
| `src/hooks/useGliderSim.js` | Real-time loop, control smoothing, telemetry throttling |
| `electron/` | Desktop shell: main process, preload, app:// file server |
| `scripts/` | `fetch-pyodide.mjs` (bundles the runtime), `adhoc-sign.cjs` (macOS signing hook) |
| `src/components/` | Three.js scene, HUD, Model panel (mesh upload), aerodynamics panel and drag polar |

## Frames and signs

Inertial frame NED (x North, y East, z Down). Body frame FRD (x forward, y right, z down).
Controls: W/S elevator (S = nose up), A/D aileron (D = roll right), Q/E rudder (E = yaw right),
Space pause, R reset.

## Using your own model

Open the app, press "Choose a model file" (or drop an STL / OBJ anywhere on the page). Any axis
arrangement works: the span, nose direction and up direction are detected and shown, and you can
override them. Check the units; they are guessed from the size. Enter the real mass. The centre of
gravity is placed for a 10% static margin by default, because a surface-mass centroid is usually
far too far aft (real gliders carry nose ballast); you can also use the model's own centroid or
type a CG position.

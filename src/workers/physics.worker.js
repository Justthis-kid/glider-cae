// Classic (non-module) web worker: hosts Pyodide and the glider_cae package so the physics and
// the mesh analysis never block the UI thread. Protocol (main <-> worker):
//   -> init {files, pyodideVersion, pyodideBase}  <- progress {label} ... ready {state, analysis}
//        (pyodideBase = folder holding a bundled runtime, or null to use the CDN)
//   -> advance {dt, timeScale, controls, wind}   <- state {state}
//   -> reset {altitude, heading, airspeed}       <- state {state}
//   -> configure {params}                   <- configured {state, analysis} | configure-error {message}
//   -> load-mesh {name, data, options}      <- progress ... configured {state, analysis} | mesh-error {message}
//   -> reanalyze-mesh {options} / use-default    (same replies as load-mesh)
//   any unexpected failure                       <- error {message}

let pyodide = null
let bridge = null
let trimesh = 'untried' // 'ready' | 'unavailable'

async function init({ files, pyodideVersion, pyodideBase }) {
  const indexURL = pyodideBase || `https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/`
  self.postMessage({ type: 'progress', label: pyodideBase ? 'Loading the Python runtime' : 'Downloading the Python runtime' })
  importScripts(`${indexURL}pyodide.js`)
  pyodide = await self.loadPyodide({ indexURL })

  self.postMessage({ type: 'progress', label: 'Loading NumPy' })
  await pyodide.loadPackage('numpy')

  self.postMessage({ type: 'progress', label: 'Starting the flight model' })
  const dir = '/home/pyodide/glider_cae'
  pyodide.FS.mkdirTree(dir)
  for (const [name, source] of Object.entries(files)) {
    pyodide.FS.writeFile(`${dir}/${name}`, source)
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')")
  bridge = pyodide.pyimport('glider_cae.bridge')
  return JSON.parse(bridge.init())
}

// trimesh is installed on first use. If that fails (offline, blocked CDN) the Python package falls
// back to its built-in STL / OBJ readers, so the feature degrades instead of breaking.
async function ensureTrimesh() {
  if (trimesh !== 'untried') return
  try {
    self.postMessage({ type: 'progress', label: 'Installing the mesh parser (trimesh)' })
    await pyodide.loadPackage('micropip')
    await pyodide.pyimport('micropip').install('trimesh>=4,<5')
    trimesh = 'ready'
  } catch (err) {
    trimesh = 'unavailable'
  }
}

function replyForModel(json) {
  const out = JSON.parse(json)
  if (out.error) self.postMessage({ type: 'mesh-error', message: out.error })
  else self.postMessage({ type: 'configured', state: out.state, analysis: out.analysis })
}

self.onmessage = async (event) => {
  const msg = event.data
  try {
    switch (msg.type) {
      case 'init': {
        const { state, analysis } = await init(msg)
        self.postMessage({ type: 'ready', state, analysis })
        break
      }
      case 'advance': {
        const { elevator, aileron, rudder } = msg.controls
        const [wn, we, wd] = msg.wind
        const json = bridge.advance(msg.dt, msg.timeScale, elevator, aileron, rudder, wn, we, wd)
        self.postMessage({ type: 'state', state: JSON.parse(json) })
        break
      }
      case 'reset': {
        const json = bridge.reset(msg.altitude, msg.heading, msg.airspeed)
        self.postMessage({ type: 'state', state: JSON.parse(json) })
        break
      }
      case 'configure': {
        const out = JSON.parse(bridge.configure(JSON.stringify(msg.params)))
        if (out.error) self.postMessage({ type: 'configure-error', message: out.error })
        else self.postMessage({ type: 'configured', state: out.state, analysis: out.analysis })
        break
      }
      case 'load-mesh': {
        await ensureTrimesh()
        self.postMessage({ type: 'progress', label: 'Analysing the mesh' })
        const ext = (msg.name.split('.').pop() || 'stl').toLowerCase().replace(/[^a-z0-9]/g, '')
        const path = `/tmp/upload.${ext}`
        pyodide.FS.writeFile(path, new Uint8Array(msg.data))
        replyForModel(bridge.load_mesh(path, JSON.stringify({ ...msg.options, name: msg.name })))
        break
      }
      case 'reanalyze-mesh':
        self.postMessage({ type: 'progress', label: 'Analysing the mesh' })
        replyForModel(bridge.reanalyze(JSON.stringify(msg.options)))
        break
      case 'use-default':
        replyForModel(bridge.use_default())
        break
      default:
        break
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) })
  }
}

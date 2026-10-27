import { useCallback, useEffect, useRef, useState } from 'react'
import { pythonFiles } from '../pythonSources'
import { decodePreview } from '../lib/preview'
import pyodideConfig from '../../pyodide.config.json'

const TELEMETRY_INTERVAL_MS = 100   // React re-render rate for the HUD numbers
const EASE_RATE = 8                 // 1/s, smoothing of keyboard control inputs
const AXES = ['elevator', 'aileron', 'rudder']
const zeroControls = () => ({ elevator: 0, aileron: 0, rudder: 0 })

export const DEFAULT_MESH_OPTIONS = {
  units: 'auto', forward: '', up: '', cgMode: 'static_margin', cgManual: '30', mass: '',
}

// UI form values -> the option names the Python side expects (blank / NaN become null there)
function toPythonOptions(o) {
  return {
    units: o.units,
    forward: o.forward || null,
    up: o.up || null,
    cg_mode: o.cgMode,
    cg_manual: Number(o.cgManual) / 100,
    mass: o.mass === '' ? null : Number(o.mass),
  }
}

/**
 * Owns the physics worker and the real-time loop.
 *  - stateRef: newest full state; the 3D scene reads it every frame (no re-render).
 *  - telemetry: throttled copy of the state for the HUD.
 *  - analysis / preview: aerodynamic report and the decoded 3D model of an uploaded mesh.
 * Frames are never queued: a new 'advance' is only sent once the previous one returned,
 * so a slow frame can't build up a backlog.
 */
export function useGliderSim() {
  const workerRef = useRef(null)
  const stateRef = useRef(null)
  const pendingRef = useRef(false)
  const runningRef = useRef(true)
  const meshOptionsRef = useRef(DEFAULT_MESH_OPTIONS)
  const inputsRef = useRef({
    target: zeroControls(),
    current: zeroControls(),
    wind: [0, 0, 0],
    timeScale: 1,
  })

  const [status, setStatus] = useState('loading') // loading | ready | error
  const [bootStage, setBootStage] = useState('Starting')
  const [error, setError] = useState(null)
  const [telemetry, setTelemetry] = useState(null)
  const [analysis, setAnalysis] = useState(null)          // aero estimator output
  const [configuring, setConfiguring] = useState(false)
  const [configureError, setConfigureError] = useState(null)
  const [preview, setPreview] = useState(null)            // decoded mesh of an uploaded model
  const [meshOptions, setMeshOptionsState] = useState(DEFAULT_MESH_OPTIONS)
  const [meshLoading, setMeshLoading] = useState(false)
  const [meshError, setMeshError] = useState(null)
  const [running, setRunning] = useState(true)
  const [timeScale, setTimeScaleState] = useState(1)
  const [updraft, setUpdraftState] = useState(0)

  // ---- worker lifecycle
  useEffect(() => {
    const worker = new Worker(new URL('../workers/physics.worker.js', import.meta.url))
    workerRef.current = worker
    let lastTelemetry = 0

    // a new analysis arrived (initial, edited geometry, or a model): keep the 3D model in step
    const takeAnalysis = (state, next) => {
      pendingRef.current = false
      stateRef.current = state
      setTelemetry(state)
      setAnalysis(next)
      if (!next.mesh) setPreview(null)
      else if (next.mesh.preview) setPreview(decodePreview(next.mesh.preview))
    }

    worker.onmessage = (event) => {
      const msg = event.data
      if (msg.type === 'progress') {
        setBootStage(msg.label)
      } else if (msg.type === 'ready') {
        takeAnalysis(msg.state, msg.analysis)
        setStatus('ready')
      } else if (msg.type === 'configured') {
        takeAnalysis(msg.state, msg.analysis)
        setConfigureError(null)
        setMeshError(null)
        setConfiguring(false)
        setMeshLoading(false)
      } else if (msg.type === 'configure-error') {
        setConfigureError(msg.message)
        setConfiguring(false)
      } else if (msg.type === 'mesh-error') {
        setMeshError(msg.message)
        setMeshLoading(false)
      } else if (msg.type === 'state') {
        pendingRef.current = false
        stateRef.current = msg.state
        const now = performance.now()
        if (now - lastTelemetry > TELEMETRY_INTERVAL_MS) {
          lastTelemetry = now
          setTelemetry(msg.state)
        }
      } else if (msg.type === 'error') {
        setError(msg.message)
        setStatus('error')
      }
    }
    worker.onerror = (event) => {
      setError(event.message || 'The physics worker crashed')
      setStatus('error')
    }
    // The runtime is copied from public/pyodide into the web build for offline use.
    worker.postMessage({
      type: 'init',
      files: pythonFiles,
      pyodideVersion: pyodideConfig.version,
      pyodideBase: new URL('pyodide/', document.baseURI).href,
    })

    return () => worker.terminate()
  }, [])

  // ---- real-time loop
  useEffect(() => {
    if (status !== 'ready') return undefined
    let raf = 0
    let last = performance.now()

    const loop = (now) => {
      raf = requestAnimationFrame(loop)
      if (!runningRef.current) {
        last = now
        return
      }
      if (pendingRef.current) return // previous frame still computing

      const dt = Math.max(0, (now - last) / 1000)
      last = now
      const inputs = inputsRef.current
      const k = 1 - Math.exp(-EASE_RATE * dt)
      for (const axis of AXES) {
        inputs.current[axis] += (inputs.target[axis] - inputs.current[axis]) * k
      }
      pendingRef.current = true
      workerRef.current.postMessage({
        type: 'advance',
        dt,
        timeScale: inputs.timeScale,
        controls: inputs.current,
        wind: inputs.wind,
      })
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [status])

  // ---- actions
  const setControlTarget = useCallback((axis, value) => {
    inputsRef.current.target[axis] = value
  }, [])

  const setTimeScale = useCallback((value) => {
    inputsRef.current.timeScale = value
    setTimeScaleState(value)
  }, [])

  const setUpdraft = useCallback((metersPerSecond) => {
    inputsRef.current.wind = [0, 0, -metersPerSecond] // NED: negative D is up
    setUpdraftState(metersPerSecond)
  }, [])

  const toggleRun = useCallback(() => {
    runningRef.current = !runningRef.current
    setRunning(runningRef.current)
  }, [])

  const reset = useCallback(({ altitude = 120, heading = 0, airspeed = 0 } = {}) => {
    inputsRef.current.current = zeroControls()
    workerRef.current?.postMessage({ type: 'reset', altitude, heading, airspeed })
  }, [])

  /** Re-estimate the aerodynamics for edited geometry; the simulation restarts in a trimmed glide. */
  const configure = useCallback((params) => {
    inputsRef.current.current = zeroControls()
    setConfiguring(true)
    setConfigureError(null)
    workerRef.current?.postMessage({ type: 'configure', params })
  }, [])

  const setMeshOptions = useCallback((patch) => {
    meshOptionsRef.current = { ...meshOptionsRef.current, ...patch }
    setMeshOptionsState(meshOptionsRef.current)
  }, [])

  /** Read a model file in the browser and hand its bytes to the worker (transferred, not copied). */
  const loadMeshFile = useCallback(async (file) => {
    if (!file || !workerRef.current) return
    inputsRef.current.current = zeroControls()
    setMeshLoading(true)
    setMeshError(null)
    const data = await file.arrayBuffer()
    workerRef.current.postMessage(
      { type: 'load-mesh', name: file.name, data, options: toPythonOptions(meshOptionsRef.current) },
      [data],
    )
  }, [])

  const reanalyzeMesh = useCallback(() => {
    inputsRef.current.current = zeroControls()
    setMeshLoading(true)
    setMeshError(null)
    workerRef.current?.postMessage({ type: 'reanalyze-mesh', options: toPythonOptions(meshOptionsRef.current) })
  }, [])

  const useDefaultGlider = useCallback(() => {
    inputsRef.current.current = zeroControls()
    setMeshLoading(true)
    setMeshError(null)
    workerRef.current?.postMessage({ type: 'use-default' })
  }, [])

  return {
    status, bootStage, error, telemetry, stateRef,
    analysis, configuring, configureError, configure,
    preview, meshOptions, setMeshOptions, meshLoading, meshError,
    loadMeshFile, reanalyzeMesh, useDefaultGlider,
    running, timeScale, updraft,
    setControlTarget, setTimeScale, setUpdraft, toggleRun, reset,
  }
}

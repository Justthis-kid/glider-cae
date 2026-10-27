import { useCallback, useEffect } from 'react'
import Scene from './components/Scene.jsx'
import Hud from './components/Hud.jsx'
import { useGliderSim } from './hooks/useGliderSim.js'
import { useKeyboardControls } from './hooks/useKeyboardControls.js'

function BootOverlay({ status, stage, error }) {
  if (status === 'error') {
    return (
      <div className="boot" role="alert">
        <h1>The physics engine did not start</h1>
        <p>{error}</p>
        <p>
          The Python runtime loads from cdn.jsdelivr.net. Check your connection and reload the page (the desktop app bundles it instead).
        </p>
      </div>
    )
  }
  return (
    <div className="boot" role="status">
      <h1>Starting the flight model</h1>
      <p>{stage}…</p>
      <p className="boot-note">
        {window.glider?.offlineRuntime
          ? 'The Python runtime is bundled with this app, so no internet connection is needed.'
          : 'The first visit downloads about 15 MB for the Python runtime. Later visits load from cache.'}
      </p>
    </div>
  )
}

export default function App() {
  const sim = useGliderSim()
  const { reset, toggleRun, setControlTarget } = sim
  const onReset = useCallback(() => reset(), [reset])

  useKeyboardControls(setControlTarget, { onReset, onToggleRun: toggleRun })

  // drop a model file anywhere on the page
  const { loadMeshFile, status } = sim
  useEffect(() => {
    if (status !== 'ready') return undefined
    const over = (e) => { if (e.dataTransfer?.types?.includes('Files')) e.preventDefault() }
    const drop = (e) => {
      const file = e.dataTransfer?.files?.[0]
      if (!file) return
      e.preventDefault()
      loadMeshFile(file)
    }
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragover', over)
      window.removeEventListener('drop', drop)
    }
  }, [status, loadMeshFile])

  return (
    <div className="app">
      <Scene
        stateRef={sim.stateRef}
        preview={sim.preview}
        cgBody={sim.analysis?.mesh?.cg_body}
        span={sim.analysis?.geometry.span}
      />
      {sim.telemetry && <Hud sim={sim} />}
      {sim.status !== 'ready' && (
        <BootOverlay status={sim.status} stage={sim.bootStage} error={sim.error} />
      )}
    </div>
  )
}

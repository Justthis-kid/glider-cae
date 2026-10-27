import AeroPanel from './AeroPanel.jsx'
import MeshPanel from './MeshPanel.jsx'

const num = (v, digits = 1) => (Number.isFinite(v) ? v.toFixed(digits) : '–')

function signed(v, digits = 1) {
  if (!Number.isFinite(v)) return '–'
  const text = Math.abs(v).toFixed(digits)
  if (Number(text) === 0) return text
  return `${v < 0 ? '−' : '+'}${text}`
}

const heading = (yawDeg) => (((yawDeg % 360) + 360) % 360).toFixed(0).padStart(3, '0')

const STATUS_COPY = {
  landed: 'Landed. Press R to launch again.',
  crashed: 'Touched down too fast. Press R to launch again.',
  diverged: 'The solver went unstable. Press R to reset.',
}

const TIME_SCALES = [0.5, 1, 2, 4]

function Readout({ label, value, unit, accent = false }) {
  return (
    <div className={accent ? 'readout accent' : 'readout'}>
      <span className="readout-label">{label}</span>
      <span className="readout-value">
        {value}
        {unit && <span className="readout-unit">{unit}</span>}
      </span>
    </div>
  )
}

function Row({ label, value, unit }) {
  return (
    <div className="row">
      <dt>{label}</dt>
      <dd>
        {value}
        {unit && <span className="unit">{unit}</span>}
      </dd>
    </div>
  )
}

function Surface({ label, value }) {
  const half = Math.min(Math.abs(value), 1) * 50
  return (
    <div
      className="surface"
      role="meter"
      aria-label={label}
      aria-valuemin={-1}
      aria-valuemax={1}
      aria-valuenow={Number(value.toFixed(2))}
    >
      <span className="surface-label">{label}</span>
      <span className="track">
        <span className="fill" style={{ left: value >= 0 ? '50%' : `${50 - half}%`, width: `${half}%` }} />
      </span>
    </div>
  )
}

export default function Hud({ sim }) {
  const t = sim.telemetry
  const [roll, pitch, yaw] = t.euler_deg

  return (
    <>
      <aside className="strip" aria-label="Flight data">
        <div className="lead">
          <Readout label="Altitude" value={num(t.altitude, 1)} unit="m" />
          <Readout label="Airspeed" value={num(t.airspeed, 1)} unit="m/s" />
          <Readout label="Vertical speed" value={signed(t.climb_rate, 2)} unit="m/s" />
          <Readout label="Glide ratio" value={num(t.ld, 1)} accent />
        </div>
        <dl className="rows">
          <Row label="Angle of attack" value={signed(t.alpha_deg, 1)} unit="°" />
          <Row label="Sideslip" value={signed(t.beta_deg, 1)} unit="°" />
          <Row label="Roll" value={signed(roll, 0)} unit="°" />
          <Row label="Pitch" value={signed(pitch, 0)} unit="°" />
          <Row label="Heading" value={heading(yaw)} unit="°" />
          <Row label="Load factor" value={num(t.load_factor, 2)} unit="g" />
        </dl>
        {t.status !== 'flying' && (
          <p className="banner" role="status">{STATUS_COPY[t.status] ?? t.status}</p>
        )}
      </aside>

      <div className="rail">
      <section className="actions" aria-label="Simulation controls">
        <div className="button-row">
          <button type="button" onClick={sim.toggleRun}>{sim.running ? 'Pause' : 'Resume'}</button>
          <button type="button" onClick={() => sim.reset()}>Reset</button>
        </div>
        <div className="field">
          <span id="speed-label">Simulation speed</span>
          <div className="segmented" role="group" aria-labelledby="speed-label">
            {TIME_SCALES.map((s) => (
              <button
                key={s}
                type="button"
                aria-pressed={sim.timeScale === s}
                onClick={() => sim.setTimeScale(s)}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>
        <label className="field">
          <span>Updraft {num(sim.updraft, 1)} m/s</span>
          <input
            type="range"
            min="0"
            max="3"
            step="0.1"
            value={sim.updraft}
            onChange={(e) => sim.setUpdraft(Number(e.target.value))}
          />
        </label>
      </section>

      <MeshPanel sim={sim} />

      {sim.analysis && (
        <AeroPanel
          analysis={sim.analysis}
          configuring={sim.configuring}
          error={sim.configureError}
          onConfigure={sim.configure}
        />
      )}
      </div>

      <section className="surfaces" aria-label="Control surfaces">
        <Surface label="Elevator" value={t.controls.elevator} />
        <Surface label="Aileron" value={t.controls.aileron} />
        <Surface label="Rudder" value={t.controls.rudder} />
        <p className="hint">
          Pitch with W and S, roll with A and D, yaw with Q and E. Space pauses, R resets.
        </p>
      </section>
    </>
  )
}

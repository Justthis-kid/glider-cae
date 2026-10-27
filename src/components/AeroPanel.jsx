import { useEffect, useState } from 'react'
import Polar from './Polar.jsx'
import Stat from './Stat.jsx'

// scale: display value = stored value * scale (percentages are stored as fractions)
const BASE_FIELDS = [
  { key: 'span', label: 'Wingspan', unit: 'm', step: 0.05 },
  { key: 'root_chord', label: 'Root chord', unit: 'm', step: 0.01 },
  { key: 'tip_chord', label: 'Tip chord', unit: 'm', step: 0.01 },
  { key: 'dihedral_deg', label: 'Dihedral', unit: '°', step: 0.5 },
  { key: 'sweep_deg', label: 'Sweep', unit: '°', step: 1 },
  { key: 'mass', label: 'Mass', unit: 'kg', step: 0.1 },
  { key: 'fuselage_length', label: 'Fuselage', unit: 'm', step: 0.05 },
]
const MARGIN_FIELD = { key: 'static_margin_target', label: 'Static margin', unit: '%', step: 1, scale: 100 }
const CG_FIELD = { key: 'cg_frac_mac', label: 'CG on the chord', unit: '%', step: 1, scale: 100 }

const num = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–')
const pct = (v) => `${(v * 100).toFixed(0)}%`

function toForm(geometry, fields) {
  const form = {}
  for (const f of fields) form[f.key] = String(Number((geometry[f.key] * (f.scale ?? 1)).toFixed(4)))
  return form
}

export default function AeroPanel({ analysis, configuring, error, onConfigure }) {
  // a model with a fixed CG shows that CG; otherwise the CG is placed for a static margin
  const fields = [...BASE_FIELDS, analysis.geometry.cg_frac_mac != null ? CG_FIELD : MARGIN_FIELD]
  const [form, setForm] = useState(() => toForm(analysis.geometry, fields))
  const [autoTail, setAutoTail] = useState(analysis.source !== 'mesh')

  // show the geometry that was actually used (the estimator may have sized the tails)
  useEffect(() => setForm(toForm(analysis.geometry, fields)), [analysis]) // eslint-disable-line react-hooks/exhaustive-deps
  // a measured model keeps its own tails by default; the built-in glider re-sizes them
  useEffect(() => setAutoTail(analysis.source !== 'mesh'), [analysis.source])

  const submit = (event) => {
    event.preventDefault()
    const params = { auto_tail: autoTail }
    for (const f of fields) {
      const raw = (form[f.key] ?? '').trim()
      params[f.key] = raw === '' ? null : Number(raw) / (f.scale ?? 1)
    }
    onConfigure(params)
  }

  const r = analysis.report
  return (
    <details className="panel aero" open>
      <summary>Aerodynamics</summary>

      <form onSubmit={submit} className="geometry">
        {fields.map((f) => (
          <label key={f.key} className="input">
            <span>{f.label}</span>
            <span className="input-box">
              <input
                type="number"
                inputMode="decimal"
                step={f.step}
                value={form[f.key] ?? ''}
                onChange={(e) => setForm({ ...form, [f.key]: e.target.value })}
              />
              <span className="unit">{f.unit}</span>
            </span>
          </label>
        ))}
        <label className="check">
          <input type="checkbox" checked={autoTail} onChange={(e) => setAutoTail(e.target.checked)} />
          <span>Resize the tails to suit the wing</span>
        </label>
        <button type="submit" className="apply" disabled={configuring}>
          {configuring ? 'Estimating…' : 'Estimate and restart'}
        </button>
      </form>
      {error && <p className="problem" role="alert">{error}</p>}

      {r.warnings.length > 0 && (
        <ul className="warnings">{r.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
      )}

      <dl className="rows stats">
        <Stat label="Aspect ratio" value={num(r.aspect_ratio, 1)} />
        <Stat label="Wing area" value={num(r.area, 3)} unit="m²" />
        <Stat label="Span efficiency" value={num(r.oswald, 3)} />
        <Stat label="Maximum lift coefficient" value={num(r.cl_max, 2)} />
        <Stat label="Stall speed" value={num(r.stall_speed, 1)} unit="m/s" />
        <Stat label="Best glide ratio" value={num(r.best_ld, 1)} />
        <Stat label="Speed for best glide" value={num(r.best_ld_speed, 1)} unit="m/s" />
        <Stat label="Minimum sink" value={num(r.min_sink, 2)} unit="m/s" />
        {r.trim && <Stat label="Trim speed" value={num(r.trim.speed, 1)} unit="m/s" />}
        <Stat label="Static margin" value={pct(r.static_margin)} />
        <Stat label="CG on the chord" value={pct(r.cg_frac_mac)} />
      </dl>

      <Polar polar={analysis.polar} report={r} />
      <p className="legend">
        <span className="swatch best" /> best glide ratio
        <span className="swatch trim" /> trim point
      </p>
    </details>
  )
}

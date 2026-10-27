import Stat from './Stat.jsx'

const AXES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
const minus = (s) => (s ? s.replace('-', '−') : s)

const UNITS = [
  ['auto', 'Detect'], ['m', 'Metres'], ['cm', 'Centimetres'],
  ['mm', 'Millimetres'], ['in', 'Inches'], ['ft', 'Feet'],
]
const CG_MODES = [
  ['static_margin', 'Place for a stable glide'],
  ['mesh', "Use the model's centroid"],
  ['manual', 'Set by hand'],
]

function Select({ label, value, onChange, children }) {
  return (
    <label className="input">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>{children}</select>
    </label>
  )
}

function AxisSelect({ label, value, onChange }) {
  return (
    <Select label={label} value={value} onChange={onChange}>
      <option value="">Detect</option>
      {AXES.map((a) => <option key={a} value={a}>{minus(a)}</option>)}
    </Select>
  )
}

export default function MeshPanel({ sim }) {
  const { meshOptions: o, setMeshOptions, meshLoading, meshError, bootStage } = sim
  const mesh = sim.analysis?.mesh
  const info = mesh?.orientation

  const choose = (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''                       // lets the same file be chosen again
    if (file) sim.loadMeshFile(file)
  }

  return (
    <details className="panel model" open>
      <summary>Model</summary>

      {!mesh && (
        <p className="lede">
          Load an STL or OBJ model of a glider, or drop the file anywhere on the page. Axes and
          units are detected; you can correct them below.
        </p>
      )}

      <label className={meshLoading ? 'file-button busy' : 'file-button'}>
        <input type="file" accept=".stl,.obj,.ply,.off,.glb,.gltf" onChange={choose} disabled={meshLoading} />
        <span>{meshLoading ? `${bootStage}…` : mesh ? 'Load another model' : 'Choose a model file'}</span>
      </label>
      {meshError && <p className="problem" role="alert">{meshError}</p>}

      <div className="geometry options">
        <Select label="Units" value={o.units} onChange={(v) => setMeshOptions({ units: v })}>
          {UNITS.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
        </Select>
        <Select label="Centre of gravity" value={o.cgMode} onChange={(v) => setMeshOptions({ cgMode: v })}>
          {CG_MODES.map(([v, t]) => <option key={v} value={v}>{t}</option>)}
        </Select>
        <AxisSelect label="Nose points toward" value={o.forward} onChange={(v) => setMeshOptions({ forward: v })} />
        <AxisSelect label="Up is" value={o.up} onChange={(v) => setMeshOptions({ up: v })} />
        <label className="input">
          <span>Mass</span>
          <span className="input-box">
            <input type="number" inputMode="decimal" step="0.1" min="0" placeholder="estimate"
                   value={o.mass} onChange={(e) => setMeshOptions({ mass: e.target.value })} />
            <span className="unit">kg</span>
          </span>
        </label>
        {o.cgMode === 'manual' && (
          <label className="input">
            <span>CG on the chord</span>
            <span className="input-box">
              <input type="number" inputMode="decimal" step="1" value={o.cgManual}
                     onChange={(e) => setMeshOptions({ cgManual: e.target.value })} />
              <span className="unit">%</span>
            </span>
          </label>
        )}
      </div>

      {mesh && (
        <>
          <div className="button-row">
            <button type="button" disabled={meshLoading} onClick={sim.reanalyzeMesh}>Apply options</button>
            <button type="button" disabled={meshLoading} onClick={sim.useDefaultGlider}>Built-in glider</button>
          </div>

          <dl className="rows stats">
            <Stat label="File" value={mesh.name} />
            <Stat label="Triangles" value={mesh.faces.toLocaleString()} />
            <Stat label="Closed surface" value={mesh.watertight ? 'yes' : 'no'} />
            <Stat label="Units" value={mesh.units_assumed ? `${mesh.units} (assumed)` : mesh.units} />
            <Stat label="Nose toward" value={minus(info.forward)} />
            <Stat label="Up" value={minus(info.up)} />
            <Stat label="Orientation certainty" value={info.confidence} />
            <Stat label="Mass" value={mesh.mass_assumed ? 'estimated' : 'as entered'} />
          </dl>

          {(mesh.warnings.length > 0 || mesh.cg_note) && (
            <ul className="warnings">
              {mesh.warnings.map((w) => <li key={w}>{w}</li>)}
              {mesh.cg_note && <li>{mesh.cg_note}</li>}
            </ul>
          )}
          {mesh.notes.length > 0 && <ul className="notes">{mesh.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        </>
      )}
    </details>
  )
}

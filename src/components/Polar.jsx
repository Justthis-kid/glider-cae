// Drag polar (CL against CD) drawn as plain SVG: the standard glider performance chart.
// The tangent from the origin touches the curve at best glide ratio.
const W = 250
const H = 170
const M = { left: 34, right: 10, top: 10, bottom: 28 }
const CD_MAX = 0.12

export default function Polar({ polar, report }) {
  const iPeak = polar.cl.indexOf(Math.max(...polar.cl))
  const clTop = Math.ceil((report.cl_max + 0.05) * 5) / 5
  const x = (cd) => M.left + (Math.min(cd, CD_MAX) / CD_MAX) * (W - M.left - M.right)
  const y = (cl) => H - M.bottom - (cl / clTop) * (H - M.top - M.bottom)

  // attached-flow branch only (up to the CL peak), positive lift, inside the drag range
  const points = []
  for (let i = 0; i <= iPeak; i++) {
    if (polar.cl[i] >= 0 && polar.cd[i] <= CD_MAX) points.push(`${x(polar.cd[i]).toFixed(1)},${y(polar.cl[i]).toFixed(1)}`)
  }

  const bestCd = report.best_ld_cl / report.best_ld
  const xTicks = [0, 0.04, 0.08, 0.12]
  const yTicks = []
  for (let v = 0; v <= clTop + 1e-9; v += 0.5) yTicks.push(v)

  return (
    <svg className="polar" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="Drag polar: lift coefficient against drag coefficient">
      {yTicks.map((v) => (
        <g key={`y${v}`}>
          <line className="grid" x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} />
          <text className="tick" x={M.left - 5} y={y(v) + 3} textAnchor="end">{v.toFixed(1)}</text>
        </g>
      ))}
      {xTicks.map((v) => (
        <g key={`x${v}`}>
          <line className="grid" x1={x(v)} x2={x(v)} y1={M.top} y2={H - M.bottom} />
          <text className="tick" x={x(v)} y={H - M.bottom + 12} textAnchor="middle">{v.toFixed(2)}</text>
        </g>
      ))}
      <text className="axis" x={(M.left + W - M.right) / 2} y={H - 3} textAnchor="middle">Drag coefficient</text>
      <text className="axis" transform={`translate(9 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">Lift coefficient</text>

      <line className="tangent" x1={x(0)} y1={y(0)} x2={x(bestCd * 1.25)} y2={y(report.best_ld_cl * 1.25)} />
      <polyline className="curve" points={points.join(' ')} />
      <circle className="mark-best" cx={x(bestCd)} cy={y(report.best_ld_cl)} r="3.5" />
      {report.trim && (
        <circle className="mark-trim" cx={x(report.trim.cd)} cy={y(report.trim.cl)} r="3.5" />
      )}
    </svg>
  )
}

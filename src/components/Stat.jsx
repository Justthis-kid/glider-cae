export default function Stat({ label, value, unit }) {
  return (
    <div className="row">
      <dt>{label}</dt>
      <dd>{value}{unit && <span className="unit">{unit}</span>}</dd>
    </div>
  )
}

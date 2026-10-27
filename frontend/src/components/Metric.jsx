export default function Metric({ label, value }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value ?? "-"}</div>
    </div>
  );
}

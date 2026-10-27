export const DEFAULT_TH = { accept: 0.7, newThr: 0.45, margin: 0.03 };

export default function ThresholdControls({ value, onChange }) {
  const set = (k) => (e) => onChange({ ...value, [k]: parseFloat(e.target.value) });
  const rows = [
    ["accept", "Accept threshold (same animal)", 0.3, 0.99],
    ["newThr", "New-animal threshold", 0.1, 0.99],
    ["margin", "Margin vs 2nd-best animal", 0, 0.3],
  ];
  return (
    <div className="grid3">
      {rows.map(([k, label, min, max]) => (
        <label key={k} className="slider">
          <span>{label}: <b>{value[k].toFixed(2)}</b></span>
          <input type="range" min={min} max={max} step="0.01" value={value[k]} onChange={set(k)} />
        </label>
      ))}
      {value.newThr > value.accept && <p className="error full">New-animal threshold must not exceed the accept threshold.</p>}
    </div>
  );
}

export const thFields = (t) => ({ accept: t.accept, new_thr: t.newThr, margin: t.margin });

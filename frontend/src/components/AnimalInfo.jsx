const fmtDate = (d) => (d ? d.split("-").reverse().join("/") : "-");
const kg = (v) => (v == null ? "-" : `${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })} kg`);

/** Read-only summary of one animal's registered data. */
export default function AnimalInfo({ a }) {
  if (!a) return null;
  const rows = [
    ["Tag", a.tag], ["Name", a.name || "-"], ["Species / breed", `${a.species}${a.breed ? " / " + a.breed : ""}`],
    ["Date of birth", fmtDate(a.dob)],
    ...(a.age ? [["Age", a.age]] : []),
    ["Owner / farm", a.owner || "-"],
    ["Dam ID", a.dam_id || "-"], ["Dam's 305-day milk", kg(a.dam_milk_305)],
    ["Sire ID", a.sire_id || "-"], ["Sire's dam milk", kg(a.sire_dam_milk)],
  ];
  return (
    <table><tbody>{rows.map(([k, v]) => <tr key={k}><th style={{ width: "42%" }}>{k}</th><td>{v}</td></tr>)}</tbody></table>
  );
}
export { fmtDate, kg };
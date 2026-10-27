import { useEffect, useState } from "react";
import ThresholdControls, { DEFAULT_TH, thFields } from "../components/ThresholdControls.jsx";
import AnimalInfo from "../components/AnimalInfo.jsx";
import { errMsg, listAnimals, verify } from "../api/client.js";

const COLORS = { match: "ok", verified: "ok", new: "info", unsure: "warn", not_verified: "bad", no: "bad", none: "bad" };

export default function Verify() {
  const [animals, setAnimals] = useState([]);
  const [mode, setMode] = useState("search");
  const [claimed, setClaimed] = useState("");
  const [files, setFiles] = useState([]);
  const [same, setSame] = useState(false);
  const [th, setTh] = useState(DEFAULT_TH);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [out, setOut] = useState(null);
  useEffect(() => { listAnimals().then((a) => { setAnimals(a); if (a[0]) setClaimed(String(a[0].id)); }); }, []);

  const run = async () => {
    setErr(""); setOut(null); setBusy(true);
    try {
      setOut(await verify({ files, ...thFields(th), claimed_id: mode === "claim" ? claimed : undefined, same_animal: mode === "search" && same }));
    } catch (e) { setErr(errMsg(e)); }
    setBusy(false);
  };

  return (
    <>
      <h1>✅ Verify / Search an animal</h1>
      {!animals.length ? <p className="info">No animals registered yet. Register one first.</p> : (
        <div className="card">
          <div className="row">
            <label className="check"><input type="radio" checked={mode === "search"} onChange={() => setMode("search")} /> 🔎 Who is this? (compare with ALL)</label>
            <label className="check"><input type="radio" checked={mode === "claim"} onChange={() => setMode("claim")} /> 🎯 Is this animal X?</label>
          </div>
          {mode === "claim" && (
            <select value={claimed} onChange={(e) => setClaimed(e.target.value)}>
              {animals.map((a) => <option key={a.id} value={a.id}>{a.tag} ({a.species}) {a.name}</option>)}
            </select>
          )}
          {mode === "search" && (
            <label className="check gap-top"><input type="checkbox" checked={same} onChange={(e) => setSame(e.target.checked)} />
              All photos show ONE animal (combine them - more accurate). Leave off to identify every cow in every photo separately.</label>
          )}
          <input className="gap-top" type="file" multiple accept="image/*,video/*" onChange={(e) => setFiles([...e.target.files])} />
          <details className="gap-top"><summary>⚙️ Advanced settings</summary><ThresholdControls value={th} onChange={setTh} /></details>
          <button className="btn primary gap-top" disabled={!files.length || busy} onClick={run}>{busy ? "Working..." : "Run"}</button>
        </div>
      )}
      {err && <p className="error">{err}</p>}
      {out?.notes.map((n, i) => <p key={i} className="warn-text">{n}</p>)}
      {out?.results.map((r, i) => (
        <div className="card row" key={i}>
          <img className="thumb-lg" src={r.crop} alt="" />
          <div className="grow">
            <div className="muted">{r.source} - {r.label}</div>
            <p className={`badge ${COLORS[r.verdict]}`}>{r.message}{r.similarity != null && ` - similarity ${r.similarity}`}</p>
            {r.animal && <><h4 style={{ margin: "6px 0" }}>{r.verdict === "unsure" ? "Closest registered animal" : "Registered data"}</h4><AnimalInfo a={r.animal} /></>}
            <details className="gap-top"><summary>Top candidates</summary>
              <table><thead><tr><th>ID</th><th>Name</th><th>Similarity</th></tr></thead>
                <tbody>{r.ranked.map((x) => <tr key={x.id}><td>{x.tag}</td><td>{x.name}</td><td>{x.score}</td></tr>)}</tbody></table></details>
          </div>
        </div>
      ))}
    </>
  );
}

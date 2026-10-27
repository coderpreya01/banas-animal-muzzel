import { useEffect, useState } from "react";
import AnimalInfo, { fmtDate } from "../components/AnimalInfo.jsx";
import { errMsg, getAnimal, listAnimals, thumbUrl } from "../api/client.js";

const when = (v) => (v ? String(v).slice(0, 16).replace("T", " ") : "");

export default function Records() {
  const [animals, setAnimals] = useState([]);
  const [sel, setSel] = useState(null);
  const [detail, setDetail] = useState(null);
  const [msg, setMsg] = useState("");

  useEffect(() => { listAnimals().then(setAnimals).catch((e) => setMsg(errMsg(e))); }, []);
  useEffect(() => {
    if (!sel) return setDetail(null);
    getAnimal(sel).then(setDetail).catch((e) => setMsg(errMsg(e)));
  }, [sel]);

  return (
    <>
      <h1>📋 Registered animals</h1>
      <p className="muted">Animals are added and edited in the BBBRC app. Muzzle embeddings are created automatically from their images.</p>
      {msg && <p className="info">{msg}</p>}
      {!animals.length ? <p className="info">No animals found in the database yet.</p> : (
        <>
          <div className="grid6">
            {animals.map((a) => (
              <div key={a.id} className={`card pick ${sel === a.id ? "selected" : ""}`} onClick={() => setSel(a.id)}>
                <img className="fit" src={thumbUrl(a.id)} alt="" onError={(e) => (e.target.style.display = "none")} />
                <b>{a.tag}</b>
                <div className="muted">{a.breed || "-"} · {a.visits} visit(s)</div>
              </div>
            ))}
          </div>
          <table className="gap-top"><thead><tr><th>ID</th><th>Tag</th><th>Gender</th><th>Breed</th><th>DOB</th><th>Dam ID</th><th>Dam 305d kg</th><th>Sire</th><th>Sire's dam kg</th><th>Embeddings</th><th>Visits</th><th>Last seen</th></tr></thead>
            <tbody>{animals.map((a) => (
              <tr key={a.id}><td>{a.id}</td><td>{a.tag}</td><td>{a.gender}</td><td>{a.breed}</td><td>{fmtDate(a.dob)}</td><td>{a.dam_id}</td><td>{a.dam_milk_305 ?? ""}</td><td>{a.sire_id}</td><td>{a.sire_dam_milk ?? ""}</td>
                <td>{a.samples}/{a.images.length}</td><td>{a.visits}</td><td>{when(a.last_seen)}</td></tr>))}</tbody></table>
        </>
      )}

      {detail && (
        <div className="card gap-top">
          <h2>{detail.tag}</h2>
          <AnimalInfo a={detail} />
          {detail.images.length > 0 && <><h3>Images</h3>
            <div className="thumbs">{detail.images.map((u) => <img key={u} src={u} alt="" />)}</div></>}
          <h3>Embeddings</h3>
          {detail.embeddings.length ? (
            <table><thead><tr><th>Image</th><th>Status</th><th>Model</th><th>Updated</th><th>Problem</th></tr></thead>
              <tbody>{detail.embeddings.map((e) => (
                <tr key={e.slot}><td>Image{e.slot}</td><td>{e.status}</td><td>{e.model}</td><td>{when(e.updated_at)}</td><td>{e.error}</td></tr>))}</tbody></table>
          ) : <p className="muted">Not embedded yet (runs automatically every minute).</p>}
          <h3>Visit history</h3>
          {detail.sightings.length ? (
            <table><thead><tr><th>Source</th><th>From (s)</th><th>To (s)</th><th>Score</th><th>Seen at</th></tr></thead>
              <tbody>{detail.sightings.map((s, i) => (
                <tr key={i}><td>{s.video}</td><td>{s.start_s?.toFixed(1)}</td><td>{s.end_s?.toFixed(1)}</td>
                  <td>{s.score?.toFixed(2) ?? "-"}</td><td>{when(s.seen_at)}</td></tr>))}</tbody></table>
          ) : <p className="muted">No visits recorded.</p>}
        </div>
      )}
    </>
  );
}
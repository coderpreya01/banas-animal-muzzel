import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import AnimalInfo from "../components/AnimalInfo.jsx";
import Metric from "../components/Metric.jsx";
import ThresholdControls, { DEFAULT_TH, thFields } from "../components/ThresholdControls.jsx";
import { errMsg, identifyMedia } from "../api/client.js";

const MAX_FILES = 20;
const IMG = /\.(jpe?g|png|webp)$/i;
const VID = /\.(mp4|mov|avi|mkv)$/i;
const isPhoto = (f) => IMG.test(f.name);
const isMedia = (f) => IMG.test(f.name) || VID.test(f.name);
const fileKey = (f) => `${f.name}|${f.size}|${f.lastModified}`;
const BADGE = { found: "ok", not_found: "bad", unsure: "warn" };

export default function IdentifyVideo() {
  const [files, setFiles] = useState([]);
  const [th, setTh] = useState(DEFAULT_TH);
  const [o, setO] = useState({ fps: 5, max_seconds: 0, save: true, species: "auto", single_animal: false });
  const [busy, setBusy] = useState(false);
  const [up, setUp] = useState(0);
  const [err, setErr] = useState("");
  const [r, setR] = useState(null);
  const pick = useRef(null);

  // files ACCUMULATE; copy the FileList BEFORE clearing the input, otherwise the list becomes empty
  const addFiles = (e) => {
    const picked = Array.from(e.target.files || []);
    e.target.value = "";
    if (!picked.length) return;
    const good = picked.filter(isMedia);
    setErr(good.length < picked.length
      ? `${picked.length - good.length} file(s) ignored - use JPG, PNG, WEBP photos or MP4, MOV, AVI, MKV videos.` : "");
    setFiles((old) => {
      const seen = new Set(old.map(fileKey));
      return [...old, ...good.filter((f) => !seen.has(fileKey(f)))].slice(0, MAX_FILES);
    });
  };
  const removeFile = (i) => setFiles((old) => old.filter((_, k) => k !== i));

  const nPhotos = files.filter(isPhoto).length;
  const nVideos = files.length - nPhotos;

  const run = async () => {
    setErr(""); setR(null); setBusy(true); setUp(0);
    try {
      setR(await identifyMedia({ files, ...thFields(th), ...o }, (p) => setUp(p.total ? p.loaded / p.total : 0)));
    } catch (e) { setErr(errMsg(e)); }
    setBusy(false);
  };

  return (
    <>
      <h1>🎬 Identify from Photos &amp; Videos</h1>
      <p className="muted">Upload photos, videos or both. A registered cow is recognised and its saved data is shown.
        A cow that is not registered shows "not found" - nothing is registered here (use Register Animal for that).</p>
      <div className="card">
        <div className="row wrap">
          <button type="button" className="btn" disabled={busy} onClick={() => pick.current?.click()}>📂 Add photos / videos</button>
          <input ref={pick} type="file" multiple accept="image/jpeg,image/png,image/webp,video/*" style={{ display: "none" }} onChange={addFiles} />
          <span className="muted">{nPhotos} photo(s), {nVideos} video(s) (max {MAX_FILES})</span>
          {files.length > 0 && <button type="button" className="btn" disabled={busy} onClick={() => setFiles([])}>Clear all</button>}
        </div>
        {files.length > 0 && (
          <table className="gap-top"><tbody>
            {files.map((f, i) => (
              <tr key={fileKey(f)}>
                <td>{isPhoto(f) ? "🖼️" : "🎬"}</td>
                <td>{f.name}</td>
                <td>{(f.size / 1048576).toFixed(1)} MB</td>
                <td><button type="button" className="btn danger" disabled={busy} onClick={() => removeFile(i)}>✕</button></td>
              </tr>))}
          </tbody></table>
        )}
        <details className="gap-top">
          <summary>⚙️ Advanced settings</summary>
          <ThresholdControls value={th} onChange={setTh} />
          <div className="grid3 gap-top">
            <label>Video analysis FPS<input type="number" min="1" max="30" value={o.fps} onChange={(e) => setO({ ...o, fps: +e.target.value })} /></label>
            <label>Videos: only first N seconds (0 = all)<input type="number" min="0" value={o.max_seconds} onChange={(e) => setO({ ...o, max_seconds: +e.target.value })} /></label>
            <label>Species
              <select value={o.species} onChange={(e) => setO({ ...o, species: e.target.value })}>
                {["auto", "cow", "dog", "sheep", "horse", "cat"].map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <label className="check"><input type="checkbox" checked={o.save} onChange={(e) => setO({ ...o, save: e.target.checked })} /> Log a visit for each registered animal found</label>
            <label className="check"><input type="checkbox" checked={o.single_animal} onChange={(e) => setO({ ...o, single_animal: e.target.checked })} /> Only ONE animal in each photo / video</label>
          </div>
        </details>
        <button className="btn primary gap-top" disabled={!files.length || busy || th.newThr > th.accept} onClick={run}>
          {busy ? (up < 1 ? `Uploading ${Math.round(up * 100)}%` : "Detecting & matching...") : "🔍 Identify animals"}
        </button>
      </div>
      {err && <p className="error">{err}</p>}
      {r && <Result r={r} />}
    </>
  );
}

function Result({ r }) {
  const d = r.diag || {};
  return (
    <>
      <div className="grid4 gap-top">
        <Metric label="Registered animals found" value={r.n_unique_found} />
        <Metric label="Not found (not registered)" value={r.n_not_found} />
        <Metric label="Unsure" value={r.n_unsure} />
        <Metric label="Files checked" value={r.n_photos + r.n_videos} />
      </div>
      {!r.saved && <p className="info">Visits were not logged (option is off).</p>}
      {r.notes?.map((n, i) => <p key={i} className="warn-text">{n}</p>)}

      <h2>🔎 Results</h2>
      {!r.results.length ? <p className="warn-text">No animal with a sharp, large-enough muzzle was found.</p> :
        r.results.map((x, i) => (
          <div className="card row" key={i}>
            <img className="thumb-lg" src={x.crop} alt="" />
            <div className="grow">
              <div className="muted">{x.kind === "photo" ? "🖼️" : "🎬"} {x.source} - {x.label}</div>
              <p className={`badge ${BADGE[x.verdict]}`}>{x.message}{x.similarity != null && ` - similarity ${x.similarity}`}</p>
              {x.animal && <><h4 style={{ margin: "6px 0" }}>Registered data</h4><AnimalInfo a={x.animal} />
                <p className="muted">Visits logged so far: {x.animal.visits}</p></>}
              {x.verdict === "not_found" && <p className="muted">This animal is not in the database. <Link to="/register">Register it first →</Link></p>}
              {x.verdict === "unsure" && <p className="muted">Take a closer, sharper, front-facing muzzle photo and try again.</p>}
              {x.candidates.length > 0 && (
                <details className="gap-top"><summary>Top candidates</summary>
                  <table><thead><tr><th>ID</th><th>Name</th><th>Similarity</th></tr></thead>
                    <tbody>{x.candidates.map((c) => <tr key={c.id}><td>{c.tag}</td><td>{c.name}</td><td>{c.score}</td></tr>)}</tbody></table>
                </details>)}
            </div>
          </div>))}

      {r.timeline.length > 0 && (<>
        <h2>🕒 Visit timeline</h2>
        <table><thead><tr><th>ID</th><th>File</th><th>From (s)</th><th>To (s)</th><th>Tracks merged</th><th>Match score</th></tr></thead>
          <tbody>{r.timeline.map((t, i) => (
            <tr key={i}><td>{t.label}</td><td>{t.kind === "photo" ? "🖼️" : "🎬"} {t.source}</td>
              <td>{t.kind === "photo" ? "photo" : t.start}</td><td>{t.kind === "photo" ? "-" : t.end}</td>
              <td>{t.tracks_merged}</td><td>{t.score ?? "-"}</td></tr>))}</tbody></table>
      </>)}

      {(d.neg_max != null || d.pos_median != null) && (
        <details className="card gap-top"><summary>🔬 Calibration help (from videos)</summary>
          {d.neg_max != null && <p>Different animals filmed together ({d.n_neg} pairs): mean <b>{d.neg_mean.toFixed(2)}</b>, highest <b>{d.neg_max.toFixed(2)}</b>.</p>}
          {d.pos_median != null && <p>Same animal (first vs second half of a track): median <b>{d.pos_median.toFixed(2)}</b>.</p>}
          {d.suggest != null && <p className="ok-text">Suggested accept threshold: about <b>{d.suggest.toFixed(2)}</b>.</p>}
        </details>)}
    </>
  );
}
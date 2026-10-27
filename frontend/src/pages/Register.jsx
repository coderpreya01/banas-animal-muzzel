import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { errMsg, registerAnimal } from "../api/client.js";

const SPECIES = ["cow", "dog", "sheep", "horse", "cat"];
const BREEDS = ["Gir", "Sahiwal", "Kankrej", "Red Sindhi", "Tharparkar", "Rathi", "Holstein Friesian", "Jersey",
  "HF Cross", "Jersey Cross", "Murrah (buffalo)", "Other"];
const MIN_PHOTOS = 3;
const MAX_PHOTOS = 12;
const EMPTY = { species: "cow", breed: "Gir", name: "", owner: "", dob: "", dam_id: "", dam_milk_305: "", sire_id: "", sire_dam_milk: "" };
const today = new Date().toISOString().slice(0, 10);

const isSupported = (f) => /^image\/(jpeg|png|webp)$/.test(f.type) || /\.(jpe?g|png|webp)$/i.test(f.name);
const fileKey = (f) => `${f.name}|${f.size}|${f.lastModified}`;

export default function Register() {
  const [form, setForm] = useState(EMPTY);
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [review, setReview] = useState(null);   // 409: duplicate animal OR mixed-up photos
  const [done, setDone] = useState(null);
  const camRef = useRef(null);
  const galRef = useRef(null);

  // thumbnails for the chosen photos
  useEffect(() => {
    const urls = files.map((f) => URL.createObjectURL(f));
    setPreviews(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  // Photos ACCUMULATE: take/choose some, then add more, remove a bad one.
  // IMPORTANT: copy the FileList BEFORE clearing the input, otherwise the list becomes empty.
  const addFiles = (e) => {
    const picked = Array.from(e.target.files || []);
    e.target.value = "";
    if (!picked.length) return;
    const good = picked.filter(isSupported);
    setErr(good.length < picked.length
      ? `${picked.length - good.length} file(s) ignored - only JPG, PNG or WEBP photos are supported (HEIC is not).`
      : "");
    setFiles((old) => {
      const seen = new Set(old.map(fileKey));
      return [...old, ...good.filter((f) => !seen.has(fileKey(f)))].slice(0, MAX_PHOTOS);
    });
  };
  const removeFile = (i) => setFiles((old) => old.filter((_, k) => k !== i));

  const submit = async (force = false) => {
    setErr(""); setReview(null); setDone(null); setBusy(true);
    try {
      const r = await registerAnimal({ ...form, force, photos: files });
      if (r.status === 409) setReview(r.data);
      else {
        setDone(r.data);
        setFiles([]);
        setForm({ ...EMPTY, species: form.species, breed: form.breed, owner: form.owner });
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    } catch (e) { setErr(errMsg(e)); }
    setBusy(false);
  };

  const enough = files.length >= MIN_PHOTOS;
  return (
    <>
      <h1>📝 Register a new animal</h1>
      <p className="muted">Enter the animal's data and upload at least {MIN_PHOTOS} sharp, close, front-facing muzzle photos (5 is better).
        Next time any photo of this animal is scanned, it is identified automatically and this data is shown.</p>

      {done && (
        <div className="card ok">
          <h3>✅ Registered {done.animal.tag}{done.animal.breed && ` · ${done.animal.breed}`}</h3>
          <div className="row">
            <img className="thumb-lg" src={done.thumb} alt="muzzle" />
            <div>
              <p>{done.embeddings_saved} muzzle embedding(s) and {done.photos_saved} photo(s) saved to PostgreSQL.</p>
              {done.skipped.map((s, i) => <p key={i} className="warn-text">Skipped: {s}</p>)}
              {done.warnings.map((s, i) => <p key={i} className="muted">Note: {s}</p>)}
              <Link to="/records">Open Animal Records →</Link>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <h3>Animal</h3>
        <div className="grid3">
          <label>Species<select value={form.species} onChange={set("species")}>{SPECIES.map((s) => <option key={s}>{s}</option>)}</select></label>
          <label>Breed<select value={form.breed} onChange={set("breed")}>{BREEDS.map((b) => <option key={b}>{b}</option>)}</select></label>
          <label>Date of birth<input type="date" max={today} value={form.dob} onChange={set("dob")} /></label>
          <label>Name (optional)<input value={form.name} onChange={set("name")} placeholder="e.g. Lakshmi" /></label>
          <label>Owner / farm (optional)<input value={form.owner} onChange={set("owner")} placeholder="e.g. Patel farm" /></label>
        </div>

        <h3>Dam (mother)</h3>
        <div className="grid3">
          <label>Dam ID<input inputMode="numeric" value={form.dam_id} onChange={set("dam_id")} placeholder="e.g. 101513575330" /></label>
          <label>Dam's 305-day milk yield (kg)<input type="number" min="0" step="0.01" value={form.dam_milk_305} onChange={set("dam_milk_305")} placeholder="e.g. 3573.83" /></label>
        </div>

        <h3>Sire (father)</h3>
        <div className="grid3">
          <label>Sire ID / name<input value={form.sire_id} onChange={set("sire_id")} placeholder="e.g. NDD ESPECTACULO" /></label>
          <label>Sire's dam milk yield (kg)<input type="number" min="0" step="0.01" value={form.sire_dam_milk} onChange={set("sire_dam_milk")} placeholder="e.g. 11309" /></label>
        </div>

        <h3>Muzzle photos <span className={enough ? "ok-text" : "warn-text"}>({files.length} added, minimum {MIN_PHOTOS}, maximum {MAX_PHOTOS})</span></h3>
        <div className="row wrap">
          <button type="button" className="btn" disabled={busy} onClick={() => camRef.current?.click()}>📷 Take photo</button>
          <button type="button" className="btn" disabled={busy} onClick={() => galRef.current?.click()}>🖼️ Choose from gallery (many)</button>
          <input ref={camRef} type="file" accept="image/*" capture="environment" style={{ display: "none" }} onChange={addFiles} />
          <input ref={galRef} type="file" accept="image/jpeg,image/png,image/webp" multiple style={{ display: "none" }} onChange={addFiles} />
        </div>

        {previews.length > 0 && (
          <div className="thumbs">
            {previews.map((p, i) => (
              <div key={p} style={{ position: "relative" }}>
                <img src={p} alt={`photo ${i + 1}`} />
                <button type="button" className="btn danger" style={{ position: "absolute", top: 2, right: 2, padding: "0 6px" }}
                  onClick={() => removeFile(i)} disabled={busy} title="Remove">✕</button>
              </div>
            ))}
          </div>
        )}

        <button type="button" className="btn primary gap-top" disabled={!enough || busy} onClick={() => submit(false)}>
          {busy ? "Reading muzzles... please wait" : "Register animal"}
        </button>
        {!enough && <p className="muted">Add {MIN_PHOTOS - files.length} more photo(s) to enable registration.</p>}
      </div>

      {err && <p className="error">{err}</p>}

      {review && (
        <div className="card warn">
          <b>⚠️ {review.detail}</b>
          {review.match && <p>Closest match: <b>{review.match.tag}</b>{review.match.name && ` (${review.match.name})`} - similarity {review.match.score}.</p>}
          {review.inconsistent?.map((m, i) => <p key={i}>• {m}</p>)}
          <button type="button" className="btn" onClick={() => submit(true)} disabled={busy}>
            {review.match ? "Register as a NEW animal anyway" : "They are the same animal - register anyway"}
          </button>
        </div>
      )}
    </>
  );
}
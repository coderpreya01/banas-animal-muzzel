import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Metric from "../components/Metric.jsx";
import { errMsg, getStats } from "../api/client.js";

export default function Dashboard() {
  const [s, setS] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { getStats().then(setS).catch((e) => setErr(errMsg(e))); }, []);
  return (
    <>
      <div className="hero">
        <h1>🐄🐕 Animal Muzzle Re-ID</h1>
        <p>Same muzzle = same ID. A new animal gets a new ID automatically.</p>
      </div>
      {err && <p className="error">Backend not reachable: {err}</p>}
      <div className="grid4">
        <Metric label="Registered animals" value={s?.animals} />
        <Metric label="Total visits logged" value={s?.visits} />
        <Metric label="Species" value={s?.species} />
        <Metric label="Next ID" value={s ? `#${s.next_id}` : null} />
      </div>
      <div className="grid3 gap-top">
        <Link className="card tile" to="/register"><h3>📝 Register Animal</h3><p>Add a new animal with its photos.</p></Link>
        <Link className="card tile" to="/identify"><h3>🎬 Identify from Video</h3><p>Upload a video: every animal is tracked and identified.</p></Link>
        <Link className="card tile" to="/live"><h3>📡 Live Camera</h3><p>Phone or webcam, real-time IDs.</p></Link>
      </div>
    </>
  );
}

import { useEffect, useRef, useState } from "react";
import ThresholdControls, { DEFAULT_TH, thFields } from "../components/ThresholdControls.jsx";
import { errMsg, liveFrame, liveStop } from "../api/client.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default function LiveCamera() {
  const video = useRef(null), overlay = useRef(null);
  const grab = useRef(document.createElement("canvas"));
  const running = useRef(false), session = useRef("");
  const [on, setOn] = useState(false);
  const [facing, setFacing] = useState("environment");
  const [opt, setOpt] = useState({ register_new: true, single_animal: false });
  const [th, setTh] = useState(DEFAULT_TH);
  const [stats, setStats] = useState([]);
  const [fps, setFps] = useState(0);
  const [err, setErr] = useState("");
  const live = useRef({});
  live.current = { th, opt };

  const draw = (res) => {
    const c = overlay.current, v = video.current;
    c.width = v.videoWidth; c.height = v.videoHeight;
    const g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height);
    const lw = Math.max(2, c.width / 250), fs = Math.max(14, c.width / 45);
    g.font = `bold ${fs}px sans-serif`; g.lineWidth = lw;
    res.tracks.forEach((t) => {
      const [x1, y1, x2, y2] = [t.box[0] * c.width, t.box[1] * c.height, t.box[2] * c.width, t.box[3] * c.height];
      g.strokeStyle = t.color; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const tw = g.measureText(t.text).width + 10, th = fs + 8, ty = y1 - th >= 0 ? y1 - th : y1;
      g.fillStyle = t.color; g.fillRect(x1, ty, tw, th);
      g.fillStyle = "#fff"; g.fillText(t.text, x1 + 5, ty + fs);
    });
  };

  const sendFrame = async () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const k = Math.min(1, 640 / v.videoWidth), cv = grab.current;
    cv.width = v.videoWidth * k; cv.height = v.videoHeight * k;
    cv.getContext("2d").drawImage(v, 0, 0, cv.width, cv.height);
    const blob = await new Promise((r) => cv.toBlob(r, "image/jpeg", 0.8));
    const { th: t, opt: o } = live.current;
    const res = await liveFrame({ session_id: session.current, image: blob, ...thFields(t), ...o });
    draw(res); setStats(res.stats); setFps(res.fps);
  };

  const start = async () => {
    setErr("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.current.srcObject = stream; await video.current.play();
      session.current = Math.random().toString(36).slice(2);
      running.current = true; setOn(true);
      while (running.current) {
        const t0 = performance.now();
        try { await sendFrame(); } catch (e) { setErr(errMsg(e)); await sleep(1000); }
        await sleep(Math.max(0, 120 - (performance.now() - t0)));
      }
    } catch (e) { setErr(e.name === "NotAllowedError" ? "Camera permission denied." : errMsg(e)); setOn(false); }
  };

  const stop = () => {
    running.current = false; setOn(false);
    video.current?.srcObject?.getTracks().forEach((t) => t.stop());
    if (video.current) video.current.srcObject = null;
    overlay.current?.getContext("2d").clearRect(0, 0, 9999, 9999);
    liveStop(session.current);
  };
  useEffect(() => () => stop(), []); // eslint-disable-line

  return (
    <>
      <h1>📡 Live Camera</h1>
      <p className="muted">Point the camera at an animal. A box appears with its ID. New animals are registered automatically.</p>
      <details className="card"><summary>📱 Using a phone? (read once)</summary>
        <ol>
          <li>Start Vite with <code>npm run dev</code> (it already listens on your network).</li>
          <li>Phone browsers only allow the camera on <b>HTTPS</b> (or localhost): run <code>cloudflared tunnel --url http://localhost:5173</code> (or <code>ngrok http 5173</code>) and open the https link on the phone.</li>
          <li>Choose <b>Rear camera</b>, press <b>Start</b>, allow camera access. A GPU gives a much smoother view.</li>
        </ol>
      </details>
      <div className="card row wrap">
        <label>Camera <select value={facing} disabled={on} onChange={(e) => setFacing(e.target.value)}>
          <option value="environment">Rear camera</option><option value="user">Front camera</option></select></label>
        <label className="check"><input type="checkbox" checked={opt.register_new} onChange={(e) => setOpt({ ...opt, register_new: e.target.checked })} /> Auto-register new animals</label>
        <label className="check"><input type="checkbox" checked={opt.single_animal} onChange={(e) => setOpt({ ...opt, single_animal: e.target.checked })} /> Only ONE animal in view</label>
        {!on ? <button className="btn primary" onClick={start}>▶ Start</button> : <button className="btn danger" onClick={stop}>■ Stop</button>}
        {on && <span className="muted">{fps} fps</span>}
      </div>
      <details className="card"><summary>⚙️ Advanced settings</summary><ThresholdControls value={th} onChange={setTh} /></details>
      {err && <p className="error">{err}</p>}
      <div className="stage" style={{ display: on ? "block" : "none" }}>
        <video ref={video} playsInline muted />
        <canvas ref={overlay} />
      </div>
      <h2>Animals seen in this session</h2>
      {stats.length ? (
        <table><thead><tr><th>ID</th><th>Species</th><th>Visits</th><th>In view</th><th>Last seen (s ago)</th></tr></thead>
          <tbody>{stats.map((s) => <tr key={s.id}><td>{s.id}</td><td>{s.species}</td><td>{s.visits}</td><td>{s.in_view ? "yes" : "no"}</td><td>{s.last_seen_s}</td></tr>)}</tbody></table>
      ) : <p className="muted">{on ? "Looking for animals... keep the face close, sharp and facing the camera." : "Press Start to begin."}</p>}
    </>
  );
}

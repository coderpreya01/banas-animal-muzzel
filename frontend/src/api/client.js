import axios from "axios";

// same origin in Docker/Render (FastAPI serves the React build); set VITE_API_URL for local dev
const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || "" });

// always returns a STRING (FastAPI validation errors come as a list -> would crash React)
export const errMsg = (e) => {
  const d = e?.response?.data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x?.msg || JSON.stringify(x)).join("; ");
  if (d) return JSON.stringify(d);
  return e?.message || "Something went wrong";
};
export const thumbUrl = (id) => `${api.defaults.baseURL}/api/animals/${id}/thumb`;

const form = (obj) => {
  const f = new FormData();
  Object.entries(obj).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    if (Array.isArray(v)) v.forEach((x) => f.append(k, x));
    else f.append(k, v);
  });
  return f;
};

export const getStats = () => api.get("/api/stats").then((r) => r.data);
export const getConfig = () => api.get("/api/config").then((r) => r.data);
// animals are READ-ONLY here: they are added / edited in the BBBRC app
export const listAnimals = () => api.get("/api/animals").then((r) => r.data);
export const getAnimal = (id) => api.get(`/api/animals/${id}`).then((r) => r.data);

// photos AND videos together: fields.files = array of File. Never registers anything.
export const identifyMedia = (fields, onUpload) =>
  api.post("/api/identify/media", form(fields), { onUploadProgress: onUpload }).then((r) => r.data);

// send an image LINK (e.g. Cloudinary URL) instead of a file -> same result as the photo upload
export const identifyUrl = (image_url, opts = {}) =>
  api.post("/api/identify/url", { image_url, ...opts }).then((r) => r.data);

export const verify = (fields) => api.post("/api/verify", form(fields)).then((r) => r.data);
export const liveFrame = (fields) => api.post("/api/live/frame", form(fields)).then((r) => r.data);
export const liveStop = (sid) => api.delete(`/api/live/${sid}`).catch(() => {});
"""End-to-end API test with stubbed AI models (SQLite). Run:  python -m pytest tests -q   (or python tests/test_registration_flow.py)"""
import io, os, sys
from datetime import date

os.environ["DATABASE_URL"] = "sqlite:///./_test.db"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
if os.path.exists("_test.db"): os.remove("_test.db")

import cv2, numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.session import Base, engine
from app.db import models  # noqa
from app.routers import animals, verify
from app.services import pipeline, state
from app.services.metadata import MetadataError, parse_dob, parse_milk

Base.metadata.create_all(engine)
rng = np.random.default_rng(0)
DIM = 64
BASE = {k: rng.normal(size=DIM) for k in "ABC"}
class FakeEmb: name = "fake"
state.get_models = lambda: (object(), FakeEmb())
animals.get_models = verify.get_models = state.get_models
QUEUE = []
def fake_analyze(frame, det, emb, fallback_whole=False, species=None):
    return QUEUE.pop(0)
pipeline.analyze_image = fake_analyze

def item(cow, sharp=200.0, noise=0.25, n_extra=0):
    v = BASE[cow] + rng.normal(size=DIM) * noise
    crop = np.full((120, 120, 3), 128, np.uint8)
    d = dict(species="cow", crop=crop, vec=(v / np.linalg.norm(v)).astype(np.float32), sharp=sharp, bbox=(0, 0, 1, 1))
    return [d] + [dict(d) for _ in range(n_extra)]

PNG = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tobytes()
def photos(n): return [("photos", (f"p{i}.png", io.BytesIO(PNG), "image/png")) for i in range(n)]
app = FastAPI(); app.include_router(animals.router); app.include_router(verify.router)
c = TestClient(app)

GIR = dict(species="cow", breed="Gir", dob="21/08/26", dam_id="101513575330", dam_milk_305="3573.83",
           sire_id="NDD ESPECTACULO", sire_dam_milk="11309")

# ---- metadata parsing
assert parse_dob("21/08/26") == date(2026, 8, 21) and parse_dob("21-08-2026") == date(2026, 8, 21)
for bad in ("31/02/2026", "21/08/2030", "abc"):
    try: parse_dob(bad); raise SystemExit(f"should fail: {bad}")
    except MetadataError: pass
assert parse_milk("3,573.83 kg", "x") == 3573.83 and parse_milk("", "x") is None

# ---- register cow A with your exact example data
QUEUE[:] = [item("A") for _ in range(4)]
r = c.post("/api/animals/register", data=GIR, files=photos(4)); assert r.status_code == 200, r.text
a = r.json()["animal"]
assert a["breed"] == "Gir" and a["dob"] == "2026-08-21" and a["dam_id"] == "101513575330"
assert a["dam_milk_305"] == 3573.83 and a["sire_id"] == "NDD ESPECTACULO" and a["sire_dam_milk"] == 11309.0
assert a["tag"] == "COW-0001" and r.json()["embeddings_saved"] == 4
print("register A ok:", a["tag"], a["breed"], a["dob"])

# ---- too few usable photos / blurry photos
QUEUE[:] = [item("B"), item("B")]
r = c.post("/api/animals/register", data=GIR, files=photos(2)); assert r.status_code == 422 and "at least 3" in r.text
QUEUE[:] = [item("B"), item("B", sharp=5.0), item("B"), item("B"), item("B")]
r = c.post("/api/animals/register", data={**GIR, "dam_id": "999"}, files=photos(5)); assert r.status_code == 200, r.text
assert any("blurry" in s for s in r.json()["skipped"]); b = r.json()["animal"]; assert b["tag"] == "COW-0002"
print("quality gate ok (blurry skipped, B registered as", b["tag"] + ")")

# ---- invalid metadata is rejected before any AI work
r = c.post("/api/animals/register", data={**GIR, "dob": "99/99/99"}, files=photos(3)); assert r.status_code == 422 and "Date of birth" in r.text

# ---- duplicate: same cow A again -> 409, force overrides
QUEUE[:] = [item("A") for _ in range(3)]
r = c.post("/api/animals/register", data=GIR, files=photos(3)); assert r.status_code == 409 and r.json()["match"]["tag"] == "COW-0001"
print("duplicate detection ok:", r.json()["match"])

# ---- inconsistent photo warning (one photo of a different cow)
QUEUE[:] = [item("C"), item("C"), item("C"), item("B", noise=0.0)]
r = c.post("/api/animals/register", data=GIR, files=photos(4))
assert r.status_code == 409 and r.json()["inconsistent"], r.text
print("mixed-cow registration blocked ->", r.json()["inconsistent"])
QUEUE[:] = [item("C"), item("C"), item("C"), item("B", noise=0.0)]
r = c.post("/api/animals/register", data={**GIR, "force": "true"}, files=photos(4)); assert r.status_code == 200
QUEUE[:] = [item("C") for _ in range(3)]
r = c.post("/api/animals/register", data={**GIR, "force": "true"}, files=photos(3)); assert r.status_code == 200

# ---- one-to-many search: photo with cow A + cow C in it
QUEUE[:] = [item("A", n_extra=0) + item("C")]
r = c.post("/api/verify", files=[("files", ("herd.png", io.BytesIO(PNG), "image/png"))]); assert r.status_code == 200, r.text
res = r.json()["results"]; print([(x["verdict"], x["message"]) for x in res])
print(res[1]["verdict"]); assert res[0]["verdict"] == "match" and res[0]["animal"]["dam_id"] == "101513575330" and res[0]["animal"]["breed"] == "Gir"
assert res[1]["verdict"] in ("new", "unsure") or res[1]["message"]

# ---- same_animal fusion of 3 photos of B
QUEUE[:] = [item("B", noise=0.6), item("B", noise=0.6), item("B", noise=0.6)]
r = c.post("/api/verify", data={"same_animal": "true"}, files=[("files", (f"q{i}.png", io.BytesIO(PNG), "image/png")) for i in range(3)])
res = r.json()["results"]; assert len(res) == 1 and res[0]["verdict"] == "match" and res[0]["animal"]["tag"] == "COW-0002", res
print("fused search ok ->", res[0]["message"])

# ---- edit a record's data
r = c.put(f"/api/animals/{a['id']}", json={"dam_milk_305": "3600", "name": "Gauri"}); assert r.status_code == 200, r.text
j = r.json(); assert j["dam_milk_305"] == 3600.0 and j["name"] == "Gauri" and j["sire_id"] == "NDD ESPECTACULO" and j["dob"] == "2026-08-21"
r = c.put(f"/api/animals/{a['id']}", json={"dob": "bad"}); assert r.status_code == 422
print("edit ok\nALL TESTS PASSED")
os.remove("_test.db")

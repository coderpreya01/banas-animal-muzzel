# 🐄 Animal Muzzle Re-ID

**Same muzzle = same ID. A new animal gets a new ID automatically.**

Animal Muzzle Re-ID recognises individual cattle (and dogs, sheep, horses, cats) from their muzzle/face.
Upload a video, film live from a phone or webcam, or register an animal from photos. The system
detects each animal, tracks it, reads its muzzle and gives it a permanent ID (`COW-0001`, `COW-0002` ...).
When the same animal shows up again it keeps the same ID and its visit counter increases.
No model training is required: everything uses pretrained models.

This project is the FastAPI + React + PostgreSQL version of the original Streamlit app.

## Features

- 📝 **Animal registration form**: species, name, owner and several photos. Muzzle embeddings are stored in PostgreSQL.
- 🎬 **Identify from video**: detect, track, merge fragments, one animal = one ID, visits and timeline.
- ✅ **Verify / search**: "Who is this?" (compare with all IDs) or "Is this animal X?" (1:1 check).
- 📡 **Live camera**: phone or webcam, boxes and IDs in real time.
- 📋 **Animal records**: thumbnails, rename, delete, visit history, registered photos, reset.
- 🔬 **Calibration help**: suggests accept/new thresholds from your own herd.

## How it works

```
photo / video / live frame
  -> YOLOv8 (COCO) detects cow / dog / sheep / horse / cat, ByteTrack follows each animal
  -> species decided by a confidence-weighted vote over the whole track
  -> muzzle crop (custom YOLO if models/muzzle_detector.pt exists, else head estimate)
  -> best sharp crops -> DINOv2 ViT-S/14 embedding (grayscale + CLAHE, grid features)
  -> "average cow" centering removes what all cows share (coat colour, head shape)
  -> open-set decision: match (same ID) / new (new ID) / uncertain
  -> tracks of the same animal are merged, animals seen together never share an ID
  -> PostgreSQL: animals, embeddings, photos, sightings (visits)
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, React Router, Axios |
| Backend | FastAPI, Uvicorn, SQLAlchemy 2, Pydantic Settings |
| Database | PostgreSQL (plain tables, no pgvector) |
| AI | Ultralytics YOLOv8 + ByteTrack, DINOv2 (PyTorch), OpenCV |

## Folder structure

```
animal-muzzle-reid/
├── README.md
├── COMMANDS.md                  # Windows step-by-step commands
├── backend/
│   ├── requirements.txt
│   ├── .env                     # DATABASE_URL etc.
│   ├── data/                    # "average cow" statistics (auto-created)
│   ├── models/                  # optional muzzle_detector.pt
│   └── app/
│       ├── main.py              # FastAPI app, CORS, startup
│       ├── schemas.py
│       ├── core/config.py       # settings + thresholds
│       ├── db/
│       │   ├── session.py       # engine, auto-create database animal_muzzel
│       │   ├── models.py        # Animal, Embedding, Photo, Sighting
│       │   └── crud.py          # all SQL access
│       ├── routers/
│       │   ├── animals.py       # register form, list, edit, delete, thumbnails
│       │   ├── identify.py      # identify from video
│       │   ├── verify.py        # who is this / is this X
│       │   └── live.py          # live camera frames
│       ├── services/
│       │   ├── detector.py      # YOLOv8 + ByteTrack + muzzle localisation
│       │   ├── embedder.py      # DINOv2 / ResNet50 embedding
│       │   ├── population.py    # average-cow centering
│       │   ├── identity.py      # open-set matching
│       │   ├── pipeline.py      # video pipeline, track merging
│       │   ├── live.py          # real-time engine (one per session)
│       │   └── state.py         # shared model instances
│       └── utils/images.py
└── frontend/
    ├── package.json, vite.config.js, index.html
    └── src/
        ├── main.jsx, App.jsx, styles.css
        ├── api/client.js        # every backend call
        ├── components/          # Layout, ThresholdControls, Metric
        └── pages/               # Dashboard, Register, IdentifyVideo, Verify, LiveCamera, Records
```

## Database (PostgreSQL `animal_muzzel`)

| Table | Content |
|-------|---------|
| `animals` | id, tag (`COW-0001`), species, **breed, dob, dam_id, dam_milk_305, sire_id, sire_dam_milk**, name, owner, thumb (BYTEA), created_at |
| `embeddings` | animal_id, model, dim, **vector (BYTEA = float32 array)**, created_at |
| `photos` | animal_id, filename, image (BYTEA, the registered photo) |
| `sightings` | animal_id, video, start_s, end_s, score, seen_at (one row = one visit) |

Embeddings are stored without any extension. To read one back:
`np.frombuffer(row.vector, dtype=np.float32)`. Matching runs in NumPy (cosine similarity), which is
fast enough for thousands of animals. For very large herds you can switch to the `pgvector` extension later.
The database and tables are created automatically on the first start.

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health`, `/api/stats`, `/api/config` | status, dashboard numbers, default thresholds |
| POST | `/api/animals/register` | form-data: `species, breed, dob, dam_id, dam_milk_305, sire_id, sire_dam_milk, name, owner, force, photos[]` (min 3 usable photos) |
| GET | `/api/animals` · `/api/animals/{id}` | list · detail with photos and visits |
| GET | `/api/animals/{id}/thumb` · `/api/animals/photos/{photo_id}` | images |
| PUT / DELETE | `/api/animals/{id}` | rename · delete |
| DELETE | `/api/animals?confirm=RESET` | wipe all data |
| POST | `/api/identify/video` | video file + thresholds, returns animals, timeline, review list |
| POST | `/api/verify` | `files[]` (+ optional `claimed_id`, `same_animal`). Matches return the animal's full registered data |
| POST | `/api/live/frame` · DELETE `/api/live/{session}` | live camera |

Interactive docs: http://localhost:8000/docs

## Configuration

`backend/.env`

```
DATABASE_URL=postgresql+psycopg2://postgres:admin123@localhost:5432/animal_muzzel
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Thresholds live in `backend/app/core/config.py` (`ACCEPT_THRESHOLD`, `NEW_THRESHOLD`, `MATCH_MARGIN`).
Different cows sharing one ID → raise the accept threshold. One cow split into several IDs → lower it.
On a dairy farm set `FORCE_SPECIES=cow` in `.env` to stop cow/dog mix-ups.

## Registration data & accuracy rules

Registered per animal: breed, date of birth (`dd/mm/yyyy` or `dd/mm/yy`), dam ID, dam's 305-day milk (kg),
sire ID/name, sire's dam milk (kg). All are validated (no future DOB, milk 0-30000 kg). Existing databases are
upgraded automatically on start (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`), no data is lost.

- At least **3 usable muzzle photos** are required (`REG_MIN_PHOTOS` in `config.py`); blurry / tiny muzzles are skipped with the reason.
- Double registration is caught: the mean of all photos **and** a vote of the single photos are compared with every registered animal.
- A photo that looks like a different cow than the rest is flagged and registration waits for confirmation (`REG_OUTLIER_SIM`).
- Verify / Search identifies every cow in every photo (1 : many). Tick "All photos show ONE animal" to combine photos into one query.

Tests (models stubbed, SQLite): `cd backend && python tests/test_registration_flow.py`

## Tips for good results

- Sharp, close, front-facing muzzle photos (3-5 per animal) when registering.
- For best accuracy fine-tune a muzzle detector and save it as `backend/models/muzzle_detector.pt`.
- A GPU makes the live view much smoother.
- Phone camera needs HTTPS (or localhost): use `cloudflared` or `ngrok`.

## Not included (compared to the Streamlit version)

The annotated output video (boxes burnt into an MP4) is not rendered. Results are returned as data
(animals, timeline, thumbnails) and the live page draws boxes in the browser.

See **COMMANDS.md** for the full Windows setup.

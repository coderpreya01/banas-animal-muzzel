# Windows commands (run in order)

## 0. Install once
- Python 3.10  (tick "Add Python to PATH")
- Node.js 18 or newer
- PostgreSQL (user `postgres`, password `admin123`, port `5432`)

## 1. Unzip
Unzip `animal-muzzle-reid.zip`, then open **two** Command Prompt / PowerShell windows.

## 2. Database (optional - the backend creates it automatically)
```
psql -U postgres -c "CREATE DATABASE animal_muzzel;"
```
Connection string used: `postgresql://postgres:admin123@localhost:5432/animal_muzzel`
(already set in `backend\.env`).

## 3. Backend  (window 1)
```
cd animal-muzzle-reid\backend
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
First start downloads YOLOv8 and DINOv2 (a few minutes). Wait for `[startup] ready`.
API docs: http://localhost:8000/docs

PowerShell blocks activation? Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 4. Frontend  (window 2)
```
cd animal-muzzle-reid\frontend
npm install
npm run dev
```
Open http://localhost:5173

## 5. Daily use
Backend:
```
cd animal-muzzle-reid\backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```
Frontend:
```
cd animal-muzzle-reid\frontend
npm run dev
```

## 6. Phone camera (needs HTTPS)
```
cloudflared tunnel --url http://localhost:5173
```
Open the https link on the phone, go to **Live Camera**, choose Rear camera, press Start.

## 7. Production build (optional)
```
cd animal-muzzle-reid\frontend
npm run build
```

## 8. Handy checks
```
psql -U postgres -d animal_muzzel -c "SELECT id, tag, species, name FROM animals;"
psql -U postgres -d animal_muzzel -c "SELECT animal_id, model, dim, length(vector) AS bytes FROM embeddings LIMIT 5;"
```
Reset everything: Animal Records page → Danger zone, or
`psql -U postgres -d animal_muzzel -c "TRUNCATE sightings, embeddings, photos, animals RESTART IDENTITY CASCADE;"`

## Troubleshooting
- `password authentication failed`: fix the password in `backend\.env`.
- Old Streamlit data is not migrated: embeddings are rebuilt when you register / identify again.
- Slow on CPU: lower "Analysis FPS" on the video page.

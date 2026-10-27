"""Run the embedding sync ONCE from the command line (no web server needed) and print what happened.

    cd backend
    python sync_now.py            # only new / changed images
    python sync_now.py --force    # re-embed every image
"""
import json
import sys

from app.db.session import init_db
from app.services import embedding_sync
from app.db import crud

init_db()
print("columns in animals table:", crud.columns_seen())
embedding_sync.seed_population()
result = embedding_sync.sync_all(force="--force" in sys.argv)
print(json.dumps(result, indent=2, default=str))
print("embedding table summary:", crud.embedding_summary())
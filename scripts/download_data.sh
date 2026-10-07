#!/usr/bin/env bash
# Fetches both public datasets used by this project into data/external/.
# Does NOT fetch or fabricate any proprietary club data (see METHODOLOGY.md).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$ROOT_DIR/data/external"

mkdir -p "$EXTERNAL_DIR"

# --- Metrica Sports open sample tracking data (3 full matches) ---
METRICA_DIR="$EXTERNAL_DIR/metrica"
if [ ! -d "$METRICA_DIR" ]; then
  echo "Cloning Metrica Sports sample-data..."
  git clone --depth 1 https://github.com/metrica-sports/sample-data.git "$METRICA_DIR"
else
  echo "Metrica sample-data already present at $METRICA_DIR, skipping."
fi

# --- StatsBomb open data: 2023 NWSL only (competition 49, season 107) ---
# Sparse, blob-filtered clone: fetches only README.md (terms of use),
# competitions.json, the NWSL match lists, and the event files for the 2023
# season's matches (~390 MB), not the full multi-competition repo. src/data/statsbomb_loader.py reads these local
# files directly. No "360" files exist for this season; per-shot freeze
# frames are inside the shot events themselves.
STATSBOMB_DIR="$EXTERNAL_DIR/statsbomb"
if [ ! -d "$STATSBOMB_DIR" ]; then
  echo "Sparse-cloning StatsBomb open-data (2023 NWSL only)..."
  git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/statsbomb/open-data.git "$STATSBOMB_DIR"
  git -C "$STATSBOMB_DIR" sparse-checkout init --no-cone
  printf '/README.md\n/data/competitions.json\n/data/matches/49/\n' \
    > "$STATSBOMB_DIR/.git/info/sparse-checkout"
  git -C "$STATSBOMB_DIR" checkout
  # Add the season's event files, using the match list just checked out.
  python3 - "$STATSBOMB_DIR" <<'PY'
import json, sys
root = sys.argv[1]
with open(f"{root}/data/matches/49/107.json") as f:
    match_ids = [m["match_id"] for m in json.load(f)]
with open(f"{root}/.git/info/sparse-checkout", "a") as f:
    f.writelines(f"/data/events/{m}.json\n" for m in match_ids)
print(f"{len(match_ids)} matches")
PY
  git -C "$STATSBOMB_DIR" checkout
else
  echo "StatsBomb open-data already present at $STATSBOMB_DIR, skipping."
fi

echo "Done. Data available under $EXTERNAL_DIR/{metrica,statsbomb}."
echo "Next: run 'python -m src.pipeline' (see README.md, Setup)."

"""Loads Metrica Sports open sample tracking + event data.

This is the ONLY data source in this project with real, simultaneous
multi-player (x, y) positions for all 22 players + ball, frame by frame.
Anything requiring true positional entropy or Voronoi territory control at a
given instant must come from here, not from StatsBomb event data.
See METHODOLOGY.md, principle 3, before using this for anything else.

Source: https://github.com/metrica-sports/sample-data
Games 1-2 are Metrica's own CSV format (read directly).
Game 3 is the newer EPTS/JSON format: tracking is read via kloppy (which
Metrica itself recommends for that format); events are parsed from JSON here.

Coordinate system: Metrica's raw CSVs use normalized (0, 1) coordinates for
both axes, with (0, 0) at the top-left and (0.5, 0.5) at kickoff. Convert to
meters (pitch is 105m x 68m, per the Metrica sample-data README) before computing any
distance-based feature; see src/data/cleaning.py:metrica_to_meters().
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

CSV_GAME_IDS = (1, 2)
EPTS_GAME_IDS = (3,)
FRAME_RATE_HZ = 25.0
TRACKING_COLUMNS = ["frame", "period", "time_s", "team", "player_id", "x", "y"]
BALL_ID = "ball"
BALL_TEAM = "Ball"

_EVENT_COLUMN_NAMES = {
    "Team": "team",
    "Type": "type",
    "Subtype": "subtype",
    "Period": "period",
    "Start Frame": "start_frame",
    "Start Time [s]": "start_time_s",
    "End Frame": "end_frame",
    "End Time [s]": "end_time_s",
    "From": "from_player",
    "To": "to_player",
    "Start X": "start_x",
    "Start Y": "start_y",
    "End X": "end_x",
    "End Y": "end_y",
}


def _metrica_data_dir() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["metrica_dir"]) / "data"


def _require_known_game(game_id: int) -> None:
    if game_id not in CSV_GAME_IDS + EPTS_GAME_IDS:
        raise ValueError(
            f"Unknown Metrica sample game {game_id}; expected one of "
            f"{CSV_GAME_IDS + EPTS_GAME_IDS}."
        )


def _epts_paths() -> dict[str, Path]:
    game_dir = _metrica_data_dir() / "Sample_Game_3"
    return {
        "metadata": game_dir / "Sample_Game_3_metadata.xml",
        "tracking": game_dir / "Sample_Game_3_tracking.txt",
        "events": game_dir / "Sample_Game_3_events.json",
    }


def _load_epts_metadata_dataset():
    """Parse only game 3's EPTS metadata (one frame) via kloppy."""
    from kloppy import metrica

    paths = _epts_paths()
    return metrica.load_tracking_epts(
        meta_data=str(paths["metadata"]), raw_data=str(paths["tracking"]),
        limit=1, coordinates="metrica",
    )


@lru_cache(maxsize=1)
def _epts_team_grounds() -> dict[str, str]:
    """{EPTS team id: "Home"/"Away"}, e.g. {"FIFATMA": "Home", ...}, from the metadata."""
    teams = _load_epts_metadata_dataset().metadata.teams
    return {t.team_id: str(t.ground).split(".")[-1].capitalize() for t in teams}


@lru_cache(maxsize=1)
def _load_epts_tracking_wide() -> pd.DataFrame:
    """Game 3 tracking (both teams + ball) in the same wide '<id>|x' layout as the CSVs.

    Read with kloppy, which Metrica recommends for its EPTS format, keeping
    Metrica's own normalized coordinates (coordinates="metrica") so
    cleaning.metrica_to_meters() applies unchanged. Cached because parsing
    the full file takes ~40 s and both teams come from the same file.

    kloppy's timestamps restart at 0 each period, whereas games 1-2 use a
    match clock; time_s is therefore rebuilt as frame / 25 Hz, which matches
    the CSV games' convention and the event file's own start/end times.
    """
    from kloppy import metrica

    paths = _epts_paths()
    dataset = metrica.load_tracking_epts(
        meta_data=str(paths["metadata"]), raw_data=str(paths["tracking"]),
        coordinates="metrica",
    )
    if dataset.metadata.frame_rate != FRAME_RATE_HZ:
        raise ValueError(f"Expected {FRAME_RATE_HZ} Hz, got {dataset.metadata.frame_rate}.")
    raw = dataset.to_df()
    wide = pd.DataFrame(
        {
            "frame": raw["frame_id"].astype(int),
            "period": raw["period_id"].astype(int),
            "time_s": raw["frame_id"] / FRAME_RATE_HZ,
        }
    )
    for team in dataset.metadata.teams:
        for player in team.players:
            pid = player.player_id
            if f"{pid}_x" in raw.columns:
                wide[f"{pid}|x"] = raw[f"{pid}_x"].astype(float)
                wide[f"{pid}|y"] = raw[f"{pid}_y"].astype(float)
    wide[f"{BALL_ID}|x"] = raw["ball_x"].astype(float)
    wide[f"{BALL_ID}|y"] = raw["ball_y"].astype(float)
    wide.attrs["team_players"] = {
        str(t.ground).split(".")[-1].capitalize(): [p.player_id for p in t.players]
        for t in dataset.metadata.teams
    }
    return wide


def _subtype_name(subtypes: dict | list | None) -> str | None:
    """EPTS-JSON subtypes -> the CSV games' convention, e.g. 'ON TARGET-SAVED'."""
    if subtypes is None:
        return None
    if isinstance(subtypes, dict):
        return subtypes["name"]
    return "-".join(st["name"] for st in subtypes)


def _parse_epts_events(raw: dict, team_grounds: dict[str, str]) -> pd.DataFrame:
    """Flatten Metrica's game-3 JSON events into the same schema as the CSV games."""
    rows = []
    for ev in raw["data"]:
        rows.append(
            {
                "team": team_grounds[ev["team"]["id"]],
                "type": ev["type"]["name"],
                "subtype": _subtype_name(ev["subtypes"]),
                "period": ev["period"],
                "start_frame": ev["start"]["frame"],
                "start_time_s": ev["start"]["time"],
                "end_frame": ev["end"]["frame"],
                "end_time_s": ev["end"]["time"],
                "from_player": ev["from"]["id"] if ev["from"] else None,
                "to_player": ev["to"]["id"] if ev["to"] else None,
                "start_x": ev["start"]["x"],
                "start_y": ev["start"]["y"],
                "end_x": ev["end"]["x"],
                "end_y": ev["end"]["y"],
            }
        )
    events = pd.DataFrame(rows, columns=list(_EVENT_COLUMN_NAMES.values()))
    for col in ("start_x", "start_y", "end_x", "end_y"):
        events[col] = events[col].astype(float)  # JSON nulls -> NaN, as in the CSVs
    return events


def _normalize_player_id(raw: str) -> str:
    """'Player 26' -> 'Player26'; already-normalized IDs pass through unchanged."""
    return re.sub(r"^Player\s+(\d+)$", r"Player\1", raw.strip())


def _parse_tracking_columns(columns: list[str]) -> list[str]:
    """Map Metrica's raw tracking header to explicit '<object>|x' / '<object>|y' names.

    Metrica's third header row names each player once ("Player11"), followed
    by an unnamed column: the named column is x, the unnamed one after it is
    y. The ball follows the same pattern under the name "Ball". At least one
    file (game 2, Away) spells a player "Player 26"; IDs are normalized to
    the space-free form so they match across files.
    """
    fixed = {"Period": "period", "Frame": "frame", "Time [s]": "time_s"}
    names: list[str] = []
    current: str | None = None
    for col in columns:
        if col in fixed:
            names.append(fixed[col])
        elif re.fullmatch(r"Player ?\d+|Ball", col):
            current = BALL_ID if col == "Ball" else _normalize_player_id(col)
            names.append(f"{current}|x")
        elif col.startswith("Unnamed") and current is not None:
            names.append(f"{current}|y")
            current = None
        else:
            raise ValueError(f"Unexpected Metrica tracking column '{col}'.")
    return names


def _wide_to_long(wide: pd.DataFrame, team: str, include_ball: bool) -> pd.DataFrame:
    """Reshape one-row-per-frame tracking into one row per object per frame."""
    object_ids = [c.split("|")[0] for c in wide.columns if c.endswith("|x")]
    if not include_ball:
        object_ids = [o for o in object_ids if o != BALL_ID]
    pieces = [
        pd.DataFrame(
            {
                "frame": wide["frame"],
                "period": wide["period"],
                "time_s": wide["time_s"],
                "team": BALL_TEAM if obj == BALL_ID else team,
                "player_id": obj,
                "x": wide[f"{obj}|x"],
                "y": wide[f"{obj}|y"],
            }
        )
        for obj in object_ids
    ]
    long = pd.concat(pieces, ignore_index=True)
    return long[TRACKING_COLUMNS].sort_values(["frame", "player_id"], ignore_index=True)


def load_tracking(game_id: int, team: str, include_ball: bool = True) -> pd.DataFrame:
    """Load raw tracking data for one team in one Metrica sample game.

    Parameters
    ----------
    game_id : int
        1, 2, or 3 (see config.yaml: metrica.game_ids).
    team : str
        "Home" or "Away".
    include_ball : bool
        Whether to include the ball's rows (team "Ball", player_id "ball").
        The ball is duplicated in both teams' files, so pass False for one
        side when combining; or use load_match_tracking().

    Returns
    -------
    pd.DataFrame
        Long format, one row per (frame, object), with columns
        frame, period, time_s, team, player_id, x, y; in Metrica's raw
        normalized (0-1) coordinates. Convert with
        cleaning.metrica_to_meters() before feature calculation.
        Players not on the pitch (unused substitutes, or players already
        substituted off) have NaN x/y; they are kept, not dropped, so the
        frame grid stays complete.

    Notes
    -----
    Games 1-2 ship as `Sample_Game_{game_id}_RawTrackingData_{team}_Team.csv`
    with a three-row header (team / jersey number / column name). Only the
    third row is needed; it's parsed here so nothing downstream has to know
    about it. Player IDs are "Player<N>".

    Game 3 is FIFA EPTS (metadata XML + tracking text), read via kloppy.
    Its player IDs are the EPTS IDs (e.g. "P3578"), which are also what the
    game-3 event file uses, so tracking and events still join on player_id.
    """
    _require_known_game(game_id)
    if team not in ("Home", "Away"):
        raise ValueError(f"team must be 'Home' or 'Away', got '{team}'.")
    if game_id in EPTS_GAME_IDS:
        wide = _load_epts_tracking_wide()
        keep = set(wide.attrs["team_players"][team])
        cols = [
            c for c in wide.columns
            if "|" not in c or c.split("|")[0] in keep or c.startswith(f"{BALL_ID}|")
        ]
        return _wide_to_long(wide[cols], team, include_ball)
    path = (
        _metrica_data_dir()
        / f"Sample_Game_{game_id}"
        / f"Sample_Game_{game_id}_RawTrackingData_{team}_Team.csv"
    )
    wide = pd.read_csv(path, skiprows=2)
    wide.columns = _parse_tracking_columns(list(wide.columns))
    return _wide_to_long(wide, team, include_ball)


def load_match_tracking(game_id: int) -> pd.DataFrame:
    """Load both teams plus the ball (once) for one game, same schema as load_tracking."""
    home = load_tracking(game_id, "Home", include_ball=True)
    away = load_tracking(game_id, "Away", include_ball=False)
    return pd.concat([home, away], ignore_index=True).sort_values(
        ["frame", "player_id"], ignore_index=True
    )


def load_events(game_id: int) -> pd.DataFrame:
    """Load Metrica's event data for one sample game (synced with tracking).

    Parameters
    ----------
    game_id : int
        1, 2, or 3.

    Returns
    -------
    pd.DataFrame
        One row per event, with snake_case columns: team, type, subtype,
        period, start_frame, start_time_s, end_frame, end_time_s,
        from_player, to_player, start_x, start_y, end_x, end_y. Coordinates
        are Metrica's raw normalized (0-1) values; convert with
        cleaning.metrica_to_meters(). start_frame/end_frame index directly
        into load_tracking()'s 'frame' column.

    Notes
    -----
    Game 3's JSON is parsed directly rather than via kloppy's event reader:
    kloppy maps events onto its own provider-neutral taxonomy, whereas
    src/data/sequences.py relies on Metrica's own type names (e.g. BALL LOST,
    RECOVERY), which the JSON shares verbatim with the game 1-2 CSVs. Game 3
    also contains CARRY events, which games 1-2 do not.
    """
    _require_known_game(game_id)
    if game_id in EPTS_GAME_IDS:
        with open(_epts_paths()["events"]) as f:
            return _parse_epts_events(json.load(f), _epts_team_grounds())
    game_dir = _metrica_data_dir() / f"Sample_Game_{game_id}"
    path = game_dir / f"Sample_Game_{game_id}_RawEventsData.csv"
    events = pd.read_csv(path)
    unexpected = set(events.columns) - set(_EVENT_COLUMN_NAMES)
    if unexpected:
        raise ValueError(f"Unexpected Metrica event columns: {sorted(unexpected)}.")
    events = events.rename(columns=_EVENT_COLUMN_NAMES)
    for col in ("from_player", "to_player"):
        events[col] = events[col].map(_normalize_player_id, na_action="ignore")
    return events


def list_available_games() -> list[int]:
    """Return the game_ids actually present under data/external/metrica.

    Useful as a sanity check after running scripts/download_data.sh;
    don't assume all 3 games downloaded successfully without checking.
    A game counts as present if its Sample_Game_<id> directory exists and
    contains at least one file.
    """
    data_dir = _metrica_data_dir()
    if not data_dir.exists():
        return []
    found = []
    for game_dir in data_dir.glob("Sample_Game_*"):
        match = re.fullmatch(r"Sample_Game_(\d+)", game_dir.name)
        if match and game_dir.is_dir() and any(game_dir.iterdir()):
            found.append(int(match.group(1)))
    return sorted(found)

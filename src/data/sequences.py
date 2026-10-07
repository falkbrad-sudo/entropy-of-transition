"""Rule-based identification of counter-attack sequences from Metrica event data.

A "counter-attack" here is defined operationally, so it can be checked and
argued with rather than taken on faith:

1. Team T makes a RECOVERY in its own half (the ball is won deep, so the
   opponent is caught in an attacking shape with the pitch behind it).
2. Team T then takes a SHOT within `min_seconds`..`max_seconds` of the
   recovery.
3. Possession is not broken in between: no SET PIECE or BALL OUT (dead
   ball, so both teams can reorganize), no RECOVERY by the opponent, and no
   BALL LOST by T.

The time window and the own-half condition are tunable thresholds, not
established definitions; vary them and check how many sequences survive
before relying on a particular set. The output's start/end frames index
directly into Metrica tracking data, so each sequence can be fed to
src/features/entropy.py and voronoi.py.

`find_control_recoveries` returns the comparison group for a baseline:
own-half recoveries by the same rule whose possession ended *without* a
shot. Without it, nothing distinguishes "what a defense does during a
counter-attack" from "what a defense does whenever it loses the ball deep in
the opponent's half".

Event selection is kept separate from the feature modules on purpose: this
decides *which* frames are interesting; the features measure what happens in
them.
"""
from __future__ import annotations

import pandas as pd

SEQUENCE_COLUMNS = [
    "attacking_team",
    "defending_team",
    "period",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "recovery_x_m",
    "shot_subtype",
]
CONTROL_COLUMNS = [c for c in SEQUENCE_COLUMNS if c != "shot_subtype"] + ["end_reason"]


def find_counter_attacks(
    events: pd.DataFrame,
    directions: dict[str, dict[int, int]],
    min_seconds: float = 3.0,
    max_seconds: float = 15.0,
    max_recovery_x_m: float = 0.0,
) -> pd.DataFrame:
    """Find recovery -> shot sequences matching the module's counter-attack rules.

    Parameters
    ----------
    events : pd.DataFrame
        Event data from metrica_loader.load_events(), with start_x/start_y
        already converted to standard meters (cleaning.metrica_to_meters),
        in original row order (Metrica's event order).
    directions : dict[str, dict[int, int]]
        {team: {period: +1/-1}} from cleaning.attack_direction_by_period(),
        for every team in `events`. Used to express each recovery's x in the
        recovering team's own attacking frame (negative = own half).
    min_seconds, max_seconds : float
        Allowed recovery-to-shot duration. The lower bound excludes
        second-ball scrambles where the recovery and shot are essentially
        the same action.
    max_recovery_x_m : float
        Recoveries further upfield than this (in the attacking team's frame)
        are excluded. 0.0 means "own half only".

    Returns
    -------
    pd.DataFrame
        One row per sequence, columns as in SEQUENCE_COLUMNS. If several
        qualifying recoveries precede the same shot, only the latest one is
        kept (the earlier ones are by definition followed by a possession
        break, so in practice this only matters for consecutive recoveries).
    """
    events = events.reset_index(drop=True)
    teams = list(directions)
    rows = []
    for shot_idx in events.index[events["type"] == "SHOT"]:
        shot = events.loc[shot_idx]
        team = shot["team"]
        opponent = next((t for t in teams if t != team), None)
        # Walk backwards from the shot to the most recent possession change.
        for idx in range(shot_idx - 1, -1, -1):
            ev = events.loc[idx]
            if ev["period"] != shot["period"]:
                break
            if ev["type"] == "RECOVERY" and ev["team"] == team:
                duration = shot["start_time_s"] - ev["start_time_s"]
                sign = directions[team][int(ev["period"])]
                recovery_x = ev["start_x"] * sign
                if (
                    min_seconds <= duration <= max_seconds
                    and recovery_x <= max_recovery_x_m
                ):
                    rows.append(
                        {
                            "attacking_team": team,
                            "defending_team": opponent,
                            "period": int(shot["period"]),
                            "start_frame": int(ev["start_frame"]),
                            "end_frame": int(shot["start_frame"]),
                            "start_time_s": float(ev["start_time_s"]),
                            "end_time_s": float(shot["start_time_s"]),
                            "duration_s": float(duration),
                            "recovery_x_m": float(recovery_x),
                            "shot_subtype": shot["subtype"],
                        }
                    )
                break
            if _breaks_possession(ev, team):
                break
    return pd.DataFrame(rows, columns=SEQUENCE_COLUMNS)


def _breaks_possession(event: pd.Series, team: str) -> bool:
    """True if this event ends `team`'s uninterrupted possession."""
    if event["type"] in ("SET PIECE", "BALL OUT"):
        return True
    if event["type"] == "RECOVERY" and event["team"] != team:
        return True
    return event["type"] == "BALL LOST" and event["team"] == team


def find_control_recoveries(
    events: pd.DataFrame,
    directions: dict[str, dict[int, int]],
    min_seconds: float = 3.0,
    max_seconds: float = 15.0,
    max_recovery_x_m: float = 0.0,
    frame_rate_hz: float = 25.0,
) -> pd.DataFrame:
    """Find own-half recoveries whose possession ended without a shot.

    The baseline group for find_counter_attacks: the same starting condition
    (team T recovers the ball in its own half), but T does not shoot before
    its possession is broken (same break rules as the counter-attacks).

    Each control's window runs from the recovery to the first of: the
    possession break, `max_seconds` after the recovery, or the period's last
    event. Windows shorter than `min_seconds` are dropped, matching the
    counter-attacks' lower bound. If T makes another RECOVERY before the
    window ends, the earlier one is dropped (as in find_counter_attacks, the
    latest recovery starts the sequence), so control windows never overlap.

    Parameters
    ----------
    events, directions, min_seconds, max_seconds, max_recovery_x_m
        As in find_counter_attacks.
    frame_rate_hz : float
        Tracking frame rate, to convert a `max_seconds` cut-off into a frame.

    Returns
    -------
    pd.DataFrame
        One row per control, columns as in CONTROL_COLUMNS. end_frame /
        end_time_s mark the window end; end_reason is "possession_break",
        "max_seconds" or "period_end".
    """
    events = events.reset_index(drop=True)
    teams = list(directions)
    rows = []
    for rec_idx in events.index[events["type"] == "RECOVERY"]:
        rec = events.loc[rec_idx]
        team = rec["team"]
        sign = directions[team][int(rec["period"])]
        recovery_x = rec["start_x"] * sign
        if recovery_x > max_recovery_x_m:
            continue
        cap_time = rec["start_time_s"] + max_seconds
        end, reason = None, "period_end"
        for idx in range(rec_idx + 1, len(events)):
            ev = events.loc[idx]
            if ev["period"] != rec["period"]:
                break
            if ev["start_time_s"] > cap_time:
                reason = "max_seconds"
                break
            if ev["team"] == team and ev["type"] in ("SHOT", "RECOVERY"):
                reason = "shot" if ev["type"] == "SHOT" else "later_recovery"
                break
            end = ev
            if _breaks_possession(ev, team):
                reason = "possession_break"
                break
        if reason in ("shot", "later_recovery"):
            continue
        if reason == "max_seconds":
            end_time = float(cap_time)
            end_frame = int(rec["start_frame"]) + round(max_seconds * frame_rate_hz)
        elif end is None:
            continue
        else:
            end_time = float(end["start_time_s"])
            end_frame = int(end["start_frame"])
        duration = end_time - float(rec["start_time_s"])
        if duration < min_seconds:
            continue
        rows.append(
            {
                "attacking_team": team,
                "defending_team": next((t for t in teams if t != team), None),
                "period": int(rec["period"]),
                "start_frame": int(rec["start_frame"]),
                "end_frame": end_frame,
                "start_time_s": float(rec["start_time_s"]),
                "end_time_s": end_time,
                "duration_s": duration,
                "recovery_x_m": float(recovery_x),
                "end_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=CONTROL_COLUMNS)

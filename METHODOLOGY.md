# Methodology and data principles

This project treats a soccer defense's shape as a particle system and measures how its organization (entropy, dispersion, Voronoi territory, pitch control) changes during transitions, especially counter-attacks. Its value depends on every claim being checkable, so this document sets out the principles the analysis follows and the conventions that keep it that way. The Roadmap in the README records what is built.

## Principles

1. **Public data only, nothing fabricated.** Only the two public datasets below are used. No club's proprietary GPS, tracking or medical data is claimed or simulated. If a task seems to need data the project doesn't have, say so instead of synthesizing a placeholder. `src/data/club_loader.py` is a stub that raises an error by design. Synthetic data appears only in unit tests, where it is labelled as such.

2. **Only the configured season.** The StatsBomb sample is the 2023 NWSL season (competition_id=49, season_id=107). `tests/test_loaders.py::test_statsbomb_matches_are_all_from_2023` fails if that ever changes. Teams or seasons that are not in the free release are not added without a real, current data source.

3. **Per-instant, multi-player shape metrics come from Metrica tracking only.** StatsBomb's free NWSL release has no "360" files. Each event gives the position of the player on the ball, and shot events also carry a freeze frame: the positions of the players in camera view at the instant of the shot. Freeze frames are single snapshots with no velocities, limited to the camera's view and only at shots, so they cannot follow a defense through a counter-attack. Entropy, Voronoi and pitch control (`src/features/entropy.py`, `voronoi.py`, `pitch_control.py`) therefore run on Metrica tracking data only. Season-scale shape from StatsBomb is a rolling or windowed aggregate of event positions (`src/features/formation.py`) and is described as such.

4. **Cite real sources.** Methods adapted from published work are named and linked in the docstring and the README. The pitch-control model follows Spearman (2018) and Laurie Shaw's Friends of Tracking implementation.

5. **Tests and baselines, not impressions.** Each finding states its rule, a baseline or significance test, and the command that reproduces it. Null results are reported as null results.

## Data sources

See the README for full detail.

- **Metrica Sports open sample data** (`data/external/metrica/`): 3 full anonymized matches, 25 fps tracking of all players and the ball. Use for anything that needs simultaneous multi-player positions.
- **StatsBomb open data, 2023 NWSL** (`data/external/statsbomb/`): full event data, read from a local sparse clone of `statsbomb/open-data` made by `scripts/download_data.sh` (not fetched with `statsbombpy` at runtime). Use for season-level, real-team analysis where a windowed aggregate is enough.

Run `scripts/download_data.sh` to populate both before anything else.

## Code conventions

- Python 3.11+, type hints on all public functions, numpy-style docstrings.
- All loaders return `pandas.DataFrame`. Raw-format quirks (Metrica's 0 to 1 normalized coordinates, StatsBomb's 120 x 80 pitch units) stay inside the conversion functions in `src/data/cleaning.py`; nothing downstream needs to know which source a frame came from.
- One function, one job. Feature calculations in `src/features/` are pure functions on cleaned DataFrames and do not load data.
- Every feature function has a test in `tests/` built on a small, hand-constructable example with a known answer (for example, four points in a square give a specific, checkable dispersion), not only real match data.
- `pytest` and `ruff check .` must pass before a change is finished.

## Pitfalls

- **Pooled vs. per-match shape.** A player's position averaged over many matches is pulled toward their long-run mean, so pooled multi-match windows report a much smaller shape than the average single match. Compare pooled with pooled and per-match with per-match (see `src/features/formation.py`).
- **Entropy saturates.** Grid entropy tops out at ln 10 ≈ 2.30 nats for 10 players on the default grid, so it mostly counts occupied cells.
- **Voronoi share is partly mechanical.** A team pushed back toward its own goal is nearest to less of the pitch by construction, so a falling share shows where play went, not disorganization by itself.

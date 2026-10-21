"""Write the scored pitch data to data/processed/ (one file per season)."""
from pathlib import Path

import pandas as pd

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
COLS = ["game_pk", "game_date", "game_year", "at_bat_number", "pitch_number", "pitcher", "player_name", "p_throws",
        "stand", "pitch_type", "velo", "spin", "ivb", "hb", "rel_x", "rel_z", "ext", "arm_angle", "d_velo", "d_ivb",
        "d_hb", "loc_x", "loc_z", "balls", "strikes", "rv_cn", "rv_x", "x_stuff", "x_location", "x_pitching"]
d = pd.read_pickle("data/scored.pkl.gz")[COLS]
for c in d.select_dtypes("float").columns:
    d[c] = d[c].round(4)
for y in (2025, 2026):
    d[d.game_year == y].to_csv(OUT / f"pitches_scored_{y}.csv.gz", index=False)
print("wrote", sorted(p.name for p in OUT.iterdir()))

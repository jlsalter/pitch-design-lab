"""
Stuff+, Location+ and Pitching+.

Each model predicts the denoised, count-neutral run value of a pitch (pitcher's perspective, + =
good for the pitcher; batted balls valued by exit velocity and launch angle) from a different set of inputs:
  Stuff     - physics only: velocity, spin, movement, release, extension, arm angle, and how the
              pitch differs from the pitcher's own fastball. Nothing about location or count.
  Location  - where the pitch crossed the plate, the count, batter side, and the pitch class.
  Pitching  - everything.

Models are cross-fit by season: the 2025 model scores 2026 pitches and vice versa, so every grade
is out-of-sample. Same-season ("xs_") scores are also kept: next-season validation must use a
2025 grade from a model that never saw 2026. Both fitted Stuff models are saved; the design lab scores 2026 pitches
with the 2025 model, so its numbers match the out-of-sample grades.

    python src/models.py   ->  data/scored.pkl.gz, models/stuff_2026.pkl, results/model_fit.json
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

Path("models").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)

STUFF = ["velo", "spin", "ivb", "hb", "rel_x", "rel_z", "ext", "axis_sin", "axis_cos", "arm_angle",
         "d_velo", "d_ivb", "d_hb", "same_side"]
CLASS_CODE = {"fastball": 0, "cutter": 1, "breaking": 2, "offspeed": 3}
LOCATION = ["loc_x", "loc_z", "balls", "strikes", "same_side", "class_code"]
PITCHING = STUFF + ["loc_x", "loc_z", "balls", "strikes", "class_code"]
SETS = {"stuff": STUFF, "location": LOCATION, "pitching": PITCHING}


def gbm():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                         min_samples_leaf=400, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=30, random_state=0)


def main():
    df = pd.read_pickle("data/pitches.pkl.gz")
    df["class_code"] = df.pitch_class.map(CLASS_CODE)
    fit = {}
    for name, feats in SETS.items():
        df["x_" + name] = np.nan
        df["xs_" + name] = np.nan    # same-season model (for next-season prediction: no future data)
        for train_yr, test_yr in [(2025, 2026), (2026, 2025)]:
            tr, te = df.game_year == train_yr, df.game_year == test_yr
            m = gbm().fit(df.loc[tr, feats], df.loc[tr, "rv_x"])
            df.loc[te, "x_" + name] = m.predict(df.loc[te, feats])
            df.loc[tr, "xs_" + name] = m.predict(df.loc[tr, feats])
            y, p = df.loc[te, "rv_x"].values, df.loc[te, "x_" + name].values
            # pitch-level R^2 is small by nature (one pitch's outcome is mostly noise); the useful
            # check is whether predicted deciles line up with realized run value
            dec = pd.qcut(p, 10, labels=False, duplicates="drop")
            cal = pd.DataFrame({"pred": p, "real": y, "dec": dec}).groupby("dec").mean()
            fit[f"{name}_{train_yr}to{test_yr}"] = {
                "iters": int(m.n_iter_), "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                "corr": float(np.corrcoef(y, p)[0, 1]),
                "decile_pred_runs_per100": (100 * cal.pred).round(2).tolist(),
                "decile_real_runs_per100": (100 * cal.real).round(2).tolist()}
            print(name, train_yr, "->", test_yr, {k: (round(v, 4) if isinstance(v, float) else v)
                                                   for k, v in fit[f"{name}_{train_yr}to{test_yr}"].items()
                                                   if not k.startswith("decile")}, flush=True)
            if name == "stuff":
                pickle.dump(m, open(f"models/stuff_{train_yr}.pkl", "wb"))
    json.dump(fit, open("results/model_fit.json", "w"), indent=1)
    keep = ["game_pk", "game_date", "game_year", "at_bat_number", "pitch_number", "pitcher", "player_name",
            "fielding_team", "p_throws", "stand", "same_side", "pitch_type", "pitch_class", "velo", "spin", "ivb",
            "hb", "rel_x", "rel_z", "ext", "axis", "arm_angle", "fb_velo", "fb_ivb", "fb_hb", "d_velo", "d_ivb",
            "d_hb", "axis_sin", "axis_cos", "loc_x", "loc_z", "balls", "strikes", "description", "events",
            "rv", "rv_cn", "rv_x", "x_stuff", "x_location", "x_pitching", "xs_stuff", "xs_location", "xs_pitching"]
    df[keep].to_pickle("data/scored.pkl.gz")
    print("saved data/scored.pkl.gz")


if __name__ == "__main__":
    main()

"""
Turn pitch-level predictions into Stuff+, Location+ and Pitching+ grades, and validate them.

Scale (one conversion for every grade): 100 = league-average pitch; 10 points = one standard
deviation of pitcher-season averages among pitchers with 500+ pitches. Pitch-type grades use the
same conversion, so they spread wider than pitcher grades (as in public models).

Validation:
  1. Reliability: split each pitcher-season into odd/even games and correlate the halves;
     Spearman-Brown gives the number of pitches needed to reach a reliability of 0.7.
  2. Next-season prediction: 2025 grades (from a model that never saw 2026) vs. 2026 run value
     per 100 pitches, compared with 2025 run value, K-BB% and a FIP-style rate.

    python src/grades.py   ->  results/pitcher_grades.csv, results/pitch_type_grades.csv, results/validation.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results"); R.mkdir(exist_ok=True)
MIN_P, MIN_PT = 500, 100
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
         "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}


def nice(name):
    if isinstance(name, str) and ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


def main():
    d = pd.read_pickle("data/scored.pkl.gz")
    d["name"] = d.player_name.map(nice)
    d["swing"] = d.description.isin(SWING)
    d["whiff"] = d.description.isin(WHIFF)
    pa = d[d.events.notna()]

    # ---- pitcher-season table -------------------------------------------------------------
    g = d.groupby(["pitcher", "game_year"])
    P = g.agg(name=("name", "last"), throws=("p_throws", "first"), team=("fielding_team", "last"),
              pitches=("rv", "size"), rv100=("rv", "mean"), rv_cn100=("rv_cn", "mean"),
              x_stuff=("x_stuff", "mean"), x_location=("x_location", "mean"), x_pitching=("x_pitching", "mean"),
              xs_stuff=("xs_stuff", "mean"), xs_location=("xs_location", "mean"), xs_pitching=("xs_pitching", "mean"),
              whiffs=("whiff", "sum"), swings=("swing", "sum"))
    P["rv100"] *= 100; P["rv_cn100"] *= 100
    e = pa.groupby(["pitcher", "game_year"]).events
    P["pa"] = e.size()
    P["k"] = e.apply(lambda s: s.isin(["strikeout", "strikeout_double_play"]).sum())
    P["bb"] = e.apply(lambda s: s.isin(["walk", "intent_walk"]).sum())
    P["hbp"] = e.apply(lambda s: (s == "hit_by_pitch").sum())
    P["hr"] = e.apply(lambda s: (s == "home_run").sum())
    P["k_bb"] = (P.k - P.bb) / P.pa
    P["fip_rate"] = (13 * P.hr + 3 * (P.bb + P.hbp) - 2 * P.k) / P.pa      # FIP numerator per PA
    P["whiff_rate"] = P.whiffs / P.swings

    # ---- scaling --------------------------------------------------------------------------
    scale = {}
    q = P[P.pitches >= MIN_P]
    for m in ("stuff", "location", "pitching"):
        for pre in ("x_", "xs_"):
            mu = d[pre + m].mean()                       # league-average pitch
            sd = q[pre + m].std()                        # spread of qualified pitcher-seasons
            scale[pre + m] = (mu, sd)
            P[pre.replace("x", "").replace("_", "") + ("" if pre == "x_" else "s_") + m] = 0  # placeholder
    P = P.drop(columns=[c for c in P.columns if c in ("stuff", "location", "pitching", "s_stuff", "s_location", "s_pitching")])
    for m in ("stuff", "location", "pitching"):
        mu, sd = scale["x_" + m]
        P[m.capitalize() + "+"] = 100 + 10 * (P["x_" + m] - mu) / sd
        mu, sd = scale["xs_" + m]
        P[m.capitalize() + "+ (same-season model)"] = 100 + 10 * (P["xs_" + m] - mu) / sd
    P.reset_index().to_csv(R / "pitcher_grades.csv", index=False)

    # ---- pitch-type table -----------------------------------------------------------------
    t = d.groupby(["pitcher", "game_year", "pitch_type"])
    T = t.agg(name=("name", "last"), throws=("p_throws", "first"), pitches=("rv", "size"),
              velo=("velo", "mean"), spin=("spin", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean"),
              rel_z=("rel_z", "mean"), ext=("ext", "mean"), arm_angle=("arm_angle", "mean"),
              x_stuff=("x_stuff", "mean"), x_location=("x_location", "mean"), x_pitching=("x_pitching", "mean"),
              rv_cn=("rv_cn", "mean"), whiffs=("whiff", "sum"), swings=("swing", "sum"))
    T["usage"] = T.pitches / T.groupby(level=[0, 1]).pitches.transform("sum")
    T["whiff_rate"] = T.whiffs / T.swings
    for m in ("stuff", "location", "pitching"):
        mu, sd = scale["x_" + m]
        T[m.capitalize() + "+"] = 100 + 10 * (T["x_" + m] - mu) / sd
    T["runs_per100_vs_avg"] = 100 * (T.x_stuff - scale["x_stuff"][0])
    T.reset_index().to_csv(R / "pitch_type_grades.csv", index=False)

    # ---- 1. reliability ---------------------------------------------------------------------
    d["half"] = d.game_pk % 2
    H = d.groupby(["pitcher", "game_year", "half"]).agg(n=("rv", "size"), stuff=("x_stuff", "mean"),
                                                       location=("x_location", "mean"), pitching=("x_pitching", "mean"),
                                                       rv=("rv_cn", "mean")).unstack("half")
    H = H[(H["n"][0] >= 250) & (H["n"][1] >= 250)]
    n_half = float(H["n"].mean(axis=1).mean())
    rel = {}
    for m in ("stuff", "location", "pitching", "rv"):
        r = float(np.corrcoef(H[m][0], H[m][1])[0, 1])
        r_pitch = r / (n_half - (n_half - 1) * r)                      # Spearman-Brown, per-pitch reliability
        n70 = 0.7 * (1 - r_pitch) / (0.3 * r_pitch)                    # pitches for reliability 0.7
        rel[m] = {"split_half_r": round(r, 3), "pitches_for_r_0.7": int(round(n70))}
    rel["pitchers"] = int(len(H)); rel["avg_pitches_per_half"] = round(n_half)

    # ---- 2. next-season prediction ------------------------------------------------------------
    a = P.reset_index()
    y25 = a[(a.game_year == 2025) & (a.pitches >= MIN_P)].set_index("pitcher")
    y26 = a[(a.game_year == 2026) & (a.pitches >= MIN_P)].set_index("pitcher")
    both = y25.join(y26[["rv_cn100", "rv100", "k_bb"]], rsuffix="_2026", how="inner")
    preds = {"Stuff+ 2025": "Stuff+ (same-season model)", "Location+ 2025": "Location+ (same-season model)",
             "Pitching+ 2025": "Pitching+ (same-season model)", "Run value/100 2025": "rv_cn100",
             "K-BB% 2025": "k_bb", "FIP-style rate 2025 (negated)": "fip_rate"}
    pred = {}
    for label, col in preds.items():
        x = -both[col] if "negated" in label else both[col]
        pred[label] = {"r_with_2026_run_value": round(float(np.corrcoef(x, both.rv_cn100_2026)[0, 1]), 3),
                       "r_with_2026_k_bb": round(float(np.corrcoef(x, both.k_bb_2026)[0, 1]), 3)}
    out = {"scale": {k: [float(v[0]), float(v[1])] for k, v in scale.items()}, "reliability": rel,
           "next_season": pred, "n_pitchers_both_seasons": int(len(both))}
    json.dump(out, open(R / "validation.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

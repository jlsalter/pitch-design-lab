"""
Pitch design lab.

For every 2026 pitcher with enough pitches, use the Stuff model to value two kinds of change:

  1. Tweaks to an existing pitch (e.g. +1 mph, +2 in of ride, more sweep). A tweak only counts if
     it is realistic: at least 10% of pitchers with a similar arm angle and release height already
     throw that pitch type with that much of the changed trait. Fastball tweaks re-score the whole
     arsenal, because every other pitch is measured partly by its separation from the fastball.
  2. A new pitch, designed from the 15 most mechanically similar pitchers who already throw it
     (arm angle, release height, extension, fastball velocity and shape). Their median shape,
     relative to their own fastball, is applied to this pitcher's fastball and release. Its value
     is the gain from replacing the pitcher's weakest secondary pitch with it at the same usage
     (fastballs are never replaced, because per-pitch value is not comparable across pitch roles).

Values are in runs per season at the pitcher's 2026 pitch counts. The model scores physics only,
so these are estimates of the pitch's quality, not of command, health or how hitters adjust.

    python src/lab.py   ->  results/lab_tweaks.csv, results/lab_new_pitch.csv, results/lab_opportunities.csv
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models import STUFF

R = Path("results")
MIN_PITCHES = 800        # pitcher-season
MIN_TYPE = 150           # pitches of a type to evaluate it
PEER_PCT = 0.90          # a tweak must not exceed what the top 10% of similar-slot peers achieve

TWEAKS = {  # pitch type -> list of (label, feature, delta)
    "FF": [("+1 mph", "velo", 1.0), ("+2 in ride", "ivb", 2.0), ("+2 in arm-side run", "hb", 2.0)],
    "SI": [("+1 mph", "velo", 1.0), ("+2 in sink", "ivb", -2.0), ("+2 in arm-side run", "hb", 2.0)],
    "FC": [("+1 mph", "velo", 1.0), ("+2 in cut", "hb", -2.0), ("+2 in ride", "ivb", 2.0)],
    "SL": [("+1.5 mph", "velo", 1.5), ("+3 in sweep", "hb", -3.0), ("+3 in depth", "ivb", -3.0)],
    "ST": [("+1.5 mph", "velo", 1.5), ("+3 in sweep", "hb", -3.0), ("+3 in depth", "ivb", -3.0)],
    "CU": [("+1.5 mph", "velo", 1.5), ("+3 in depth", "ivb", -3.0), ("+3 in sweep", "hb", -3.0)],
    "KC": [("+1.5 mph", "velo", 1.5), ("+3 in depth", "ivb", -3.0)],
    "CH": [("-1.5 mph (more separation)", "velo", -1.5), ("+2 in drop", "ivb", -2.0), ("+2 in fade", "hb", 2.0)],
    "FS": [("-1.5 mph (more separation)", "velo", -1.5), ("+2 in drop", "ivb", -2.0), ("+2 in fade", "hb", 2.0)],
}
NEW_TYPES = ["FC", "SL", "ST", "CU", "CH", "FS"]      # secondary pitches only
NAMES = {"FF": "four-seamer", "SI": "sinker", "FC": "cutter", "SL": "slider", "ST": "sweeper", "CU": "curveball",
         "KC": "knuckle curve", "CH": "changeup", "FS": "splitter", "SV": "slurve", "FO": "forkball"}
SHAPE = ["d_velo", "d_ivb", "d_hb", "spin", "axis_sin", "axis_cos"]


def overlaps(new_type, throws):
    """Skip suggestions that duplicate a pitch the pitcher already has in another name."""
    if new_type == "SL" and throws & {"ST", "SV"}:          # a sweeper/slurve thrower has a slider-family pitch
        return True
    if new_type == "CU" and throws & {"KC", "SV"}:
        return True
    if new_type in ("CH", "FS") and throws & {"CH", "FS", "FO"}:   # one offspeed pitch is enough to start
        return True
    return False


def main():
    model = pickle.load(open("models/stuff_2025.pkl", "rb"))   # out-of-sample for 2026, same as the grades
    mu, sd = json.load(open(R / "validation.json"))["scale"]["x_stuff"]
    to_plus = lambda x: 100 + 10 * (x - mu) / sd

    d = pd.read_pickle("data/scored.pkl.gz")
    d = d[d.game_year == 2026].copy()
    names = pd.read_csv(R / "pitcher_grades.csv").query("game_year == 2026").set_index("pitcher")
    counts = d.groupby("pitcher").size()
    pitchers = counts[counts >= MIN_PITCHES].index

    # pitch-type summaries over both seasons for peer matching
    allp = pd.read_pickle("data/scored.pkl.gz")
    units = allp.groupby(["pitcher", "game_year", "pitch_type"]).agg(
        n=("velo", "size"), velo=("velo", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean"), spin=("spin", "mean"),
        axis_sin=("axis_sin", "mean"), axis_cos=("axis_cos", "mean"), arm_angle=("arm_angle", "mean"),
        rel_z=("rel_z", "mean"), ext=("ext", "mean"), fb_velo=("fb_velo", "mean"), fb_ivb=("fb_ivb", "mean"),
        fb_hb=("fb_hb", "mean"), d_velo=("d_velo", "mean"), d_ivb=("d_ivb", "mean"), d_hb=("d_hb", "mean")).reset_index()
    units = units[units.n >= 100]
    del allp

    def score(rows):
        return model.predict(rows[STUFF])

    tweak_rows, new_rows = [], []
    for p in pitchers:
        g = d[d.pitcher == p].copy()
        base_all = score(g)
        g["x"] = base_all
        arm, relz, ext = g.arm_angle.mean(), g.rel_z.mean(), g.ext.mean()
        types = g.pitch_type.value_counts()
        types = types[types >= MIN_TYPE]
        fb_types = [t for t in ("FF", "SI") if t in types.index]
        primary_fb = max(fb_types, key=lambda t: types[t]) if fb_types else None

        # ---------- 1. tweaks ----------
        for pt in types.index:
            for label, feat, delta in TWEAKS.get(pt, []):
                cur = g.loc[g.pitch_type == pt, feat].mean()
                peers = units[(units.pitch_type == pt) & (units.pitcher != p) &
                              (units.arm_angle.sub(arm).abs() <= 5) & (units.rel_z.sub(relz).abs() <= 0.3)]
                if len(peers) < 20:
                    continue
                lim = peers[feat].quantile(PEER_PCT if delta > 0 else 1 - PEER_PCT)
                target = cur + delta
                feasible = target <= lim if delta > 0 else target >= lim
                h = g.copy()
                m = h.pitch_type == pt
                h.loc[m, feat] += delta
                if pt == primary_fb and feat in ("velo", "ivb", "hb"):
                    h["d_" + feat] = h[feat] - h.loc[m, feat].mean()       # every pitch's separation moves
                elif feat in ("velo", "ivb", "hb"):
                    h.loc[m, "d_" + feat] += delta
                new = score(h)
                gain_all = float((new - base_all).sum())                  # runs over the season
                tweak_rows.append({
                    "pitcher": p, "pitch_type": pt, "pitch": NAMES.get(pt, pt), "change": label, "feature": feat,
                    "current": round(cur, 1), "target": round(target, 1), "peer_limit": round(lim, 1),
                    "n_peers": len(peers), "feasible": bool(feasible), "pitches": int(m.sum()),
                    "stuff_plus_before": round(float(to_plus(base_all[m.values].mean())), 1),
                    "stuff_plus_after": round(float(to_plus(new[m.values].mean())), 1),
                    "runs_per_season": round(gain_all, 2)})

        # ---------- 2. new pitch ----------
        if primary_fb is None:
            continue
        fbm = g[g.pitch_type == primary_fb][["velo", "ivb", "hb"]].mean()
        # a new secondary pitch replaces the weakest existing SECONDARY pitch at the same usage.
        # Fastballs are never replaced: per-pitch value is not comparable across roles (fastballs are
        # thrown in the zone to get strikes and always grade lower per pitch than chase pitches).
        secondary = [t for t in types.index if t not in ("FF", "SI")]
        if not secondary:
            continue
        weakest = g[g.pitch_type.isin(secondary)].groupby("pitch_type").x.mean().idxmin()
        weak_x = g.loc[g.pitch_type == weakest, "x"].mean()
        weak_n = int(types[weakest])
        throws = set(g.pitch_type.value_counts(normalize=True).loc[lambda s: s >= 0.05].index)
        me = np.array([arm, relz, ext, fbm.velo, fbm.ivb, fbm.hb])
        for nt in NEW_TYPES:
            if nt in throws or overlaps(nt, throws):
                continue
            cand = units[(units.pitch_type == nt) & (units.pitcher != p)].dropna(subset=["fb_velo"])
            if len(cand) < 30:
                continue
            X = cand[["arm_angle", "rel_z", "ext", "fb_velo", "fb_ivb", "fb_hb"]].values
            scale = X.std(0)
            dist = np.sqrt((((X - me) / scale) ** 2).sum(1))
            near = cand.iloc[np.argsort(dist)[:15]]
            shape = near[SHAPE].median()
            # synthetic pitches: this pitcher's release, the peers' shape relative to his fastball
            base = g[g.pitch_type == primary_fb].sample(min(400, int(types[primary_fb])), random_state=0).copy()
            syn = base.copy()
            syn["velo"] = fbm.velo + shape.d_velo
            syn["ivb"] = fbm.ivb + shape.d_ivb
            syn["hb"] = fbm.hb + shape.d_hb
            syn["d_velo"], syn["d_ivb"], syn["d_hb"] = shape.d_velo, shape.d_ivb, shape.d_hb
            syn["spin"], syn["axis_sin"], syn["axis_cos"] = shape.spin, shape.axis_sin, shape.axis_cos
            nx = float(score(syn).mean())
            new_rows.append({
                "pitcher": p, "new_pitch": NAMES[nt], "new_type": nt,
                "velo": round(syn.velo.iloc[0], 1), "ivb": round(syn.ivb.iloc[0], 1), "hb": round(syn.hb.iloc[0], 1),
                "spin": int(shape.spin), "stuff_plus": round(float(to_plus(nx)), 1),
                "replaces": NAMES.get(weakest, weakest), "replaces_type": weakest,
                "replaced_stuff_plus": round(float(to_plus(weak_x)), 1), "replaced_pitches": weak_n,
                "runs_per_season": round((nx - weak_x) * weak_n, 2),
                "peer_examples": ", ".join(names.name.reindex(near.pitcher.unique()[:4]).dropna().tolist())})

    T = pd.DataFrame(tweak_rows)
    N = pd.DataFrame(new_rows)
    for df in (T, N):
        df.insert(1, "name", names.name.reindex(df.pitcher).values)
        df.insert(2, "team", names.team.reindex(df.pitcher).values)
        df.insert(3, "throws", names.throws.reindex(df.pitcher).values)
    T.to_csv(R / "lab_tweaks.csv", index=False)
    N.to_csv(R / "lab_new_pitch.csv", index=False)

    # best feasible opportunity per pitcher
    best_t = T[T.feasible].sort_values("runs_per_season").groupby("pitcher").tail(1).set_index("pitcher")
    best_n = N.sort_values("runs_per_season").groupby("pitcher").tail(1).set_index("pitcher")
    opp = pd.DataFrame({"name": names.name.reindex(pitchers), "team": names.team.reindex(pitchers),
                        "throws": names.throws.reindex(pitchers), "pitches": counts.reindex(pitchers),
                        "stuff_plus": names["Stuff+"].reindex(pitchers).round(1),
                        "best_tweak": best_t.pitch.str.cat(best_t.change, sep=": ").reindex(pitchers),
                        "tweak_runs": best_t.runs_per_season.reindex(pitchers),
                        "best_new_pitch": best_n.new_pitch.reindex(pitchers),
                        "new_pitch_runs": best_n.runs_per_season.reindex(pitchers)})
    opp["best_runs"] = opp[["tweak_runs", "new_pitch_runs"]].max(axis=1)
    opp.sort_values("best_runs", ascending=False).to_csv(R / "lab_opportunities.csv")
    print("tweaks:", len(T), "feasible:", int(T.feasible.sum()), " new-pitch options:", len(N), " pitchers:", len(pitchers))
    print(opp.sort_values("best_runs", ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()

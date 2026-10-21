"""
Backtest: when pitchers really changed their pitches between 2025 and 2026, did their results
move the way the model predicted?

For each pitcher with 500+ pitches in both seasons:
  predicted change = mean leak-free Stuff prediction in 2026 - in 2025 (src/backtest_preds.py; the
                     model never saw 2026 or the pitcher's own 2025 outcomes, so the prediction
                     reflects only how his pitches physically changed)
  actual change    = mean denoised run value in 2026 - in 2025
Both are in runs per 100 pitches, pitcher's perspective (+ = better).

Real changes are flagged two ways:
  new pitch      - a pitch type thrown <2% in 2025 and >=5% in 2026, whose shape is at least 4 in of
                   movement or 3 mph away from every pitch he threw >=5% in 2025 (this drops cases
                   where Statcast simply relabeled a slider as a sweeper)
  fastball change- same primary fastball both years, with >=1.5 in change in vertical or horizontal
                   break or >=1 mph change in velocity

    python src/backtest.py   ->  results/backtest.json, results/backtest_pitchers.csv, figures/05_backtest.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, F = Path("results"), Path("figures")
MIN_N, B, SEED = 500, 4000, 0
rng = np.random.default_rng(SEED)

d = pd.read_pickle("data/backtest_preds.pkl.gz")
names = pd.read_csv(R / "pitcher_grades.csv").drop_duplicates("pitcher", keep="last").set_index("pitcher")

# ---------- pitcher-season summaries ----------
ps = d.groupby(["pitcher", "game_year"]).agg(n=("b_stuff", "size"), pred=("b_stuff", "mean"),
                                             act=("rv_x", "mean"), act_cn=("rv_cn", "mean")).unstack()
ps = ps[(ps["n"][2025] >= MIN_N) & (ps["n"][2026] >= MIN_N)]
P = pd.DataFrame({
    "n25": ps["n"][2025], "n26": ps["n"][2026],
    "pred_change": 100 * (ps["pred"][2026] - ps["pred"][2025]),
    "actual_change": 100 * (ps["act"][2026] - ps["act"][2025]),
    "actual_change_raw": 100 * (ps["act_cn"][2026] - ps["act_cn"][2025]),
    "actual_2025": 100 * ps["act"][2025]})
P["w"] = 1 / (1 / P.n25 + 1 / P.n26)          # noise in the actual change shrinks with pitches thrown

# ---------- flag real arsenal changes ----------
pt = d[d.pitcher.isin(P.index)].groupby(["pitcher", "game_year", "pitch_type"]).agg(
    n=("velo", "size"), velo=("velo", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean")).reset_index()
pt["usage"] = pt.n / pt.groupby(["pitcher", "game_year"]).n.transform("sum")
new_rows, fb_rows = [], []
for p, g in pt.groupby("pitcher"):
    a, b = g[g.game_year == 2025].set_index("pitch_type"), g[g.game_year == 2026].set_index("pitch_type")
    a_main = a[a.usage >= 0.05]
    for t, r in b[b.usage >= 0.05].iterrows():
        if t in a.index and a.loc[t, "usage"] >= 0.02:
            continue
        mv = np.hypot(a_main.ivb - r.ivb, a_main.hb - r.hb)
        dv = (a_main.velo - r.velo).abs()
        if ((mv >= 4) | (dv >= 3)).all():
            new_rows.append({"pitcher": p, "new_type": t, "usage_2026": round(r.usage, 3)})
    fa = [t for t in ("FF", "SI") if t in a.index]
    fb = [t for t in ("FF", "SI") if t in b.index]
    if fa and fb:
        ta, tb = max(fa, key=lambda t: a.loc[t, "n"]), max(fb, key=lambda t: b.loc[t, "n"])
        if ta == tb:
            dz, dx, dvv = b.loc[tb, "ivb"] - a.loc[ta, "ivb"], b.loc[tb, "hb"] - a.loc[ta, "hb"], b.loc[tb, "velo"] - a.loc[ta, "velo"]
            if abs(dz) >= 1.5 or abs(dx) >= 1.5 or abs(dvv) >= 1.0:
                fb_rows.append({"pitcher": p, "fb_type": tb, "d_ivb": round(dz, 1), "d_hb": round(dx, 1), "d_velo": round(dvv, 1)})
NEW, FB = pd.DataFrame(new_rows), pd.DataFrame(fb_rows)
P["new_pitch"] = P.index.isin(NEW.pitcher)
P["fb_change"] = P.index.isin(FB.pitcher)
P["changed"] = P.new_pitch | P.fb_change


# ---------- statistics ----------
def wls(x, y, w, z=None):
    X = np.column_stack([np.ones_like(x), x] + ([z] if z is not None else []))
    W = np.sqrt(w)
    return np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0][1]


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def boot(df, f):
    idx = np.arange(len(df)); out = []
    for _ in range(B):
        out.append(f(df.iloc[rng.choice(idx, len(idx))]))
    return [round(float(np.quantile(out, q)), 3) for q in (0.05, 0.95)], float(np.mean(np.array(out) <= 0))


def summarize(df, label):
    x, y, w = df.pred_change.values, df.actual_change.values, df.w.values
    slope = wls(x, y, w)
    s_ci, s_p = boot(df, lambda s: wls(s.pred_change.values, s.actual_change.values, s.w.values))
    ctrl = wls(x, y, w, df.actual_2025.values)
    c_ci, _ = boot(df, lambda s: wls(s.pred_change.values, s.actual_change.values, s.w.values, s.actual_2025.values))
    r = wcorr(x, y, w)
    r_ci, r_p = boot(df, lambda s: wcorr(s.pred_change.values, s.actual_change.values, s.w.values))
    raw = wls(x, df.actual_change_raw.values, w)
    raw_ci, _ = boot(df, lambda s: wls(s.pred_change.values, s.actual_change_raw.values, s.w.values))
    up, dn = df[df.pred_change > 0], df[df.pred_change <= 0]
    diff = lambda s: np.average(s[s.pred_change > 0].actual_change, weights=s[s.pred_change > 0].w) - \
        np.average(s[s.pred_change <= 0].actual_change, weights=s[s.pred_change <= 0].w)
    d_ci, d_p = boot(df, diff)
    return {"group": label, "pitchers": len(df),
            "slope": round(slope, 3), "slope_90ci": s_ci, "slope_boot_p_le_0": round(s_p, 4),
            "slope_controlling_2025_results": round(ctrl, 3), "slope_controlling_90ci": c_ci,
            "slope_actual_not_denoised": round(raw, 3), "slope_actual_not_denoised_90ci": raw_ci,
            "corr": round(r, 3), "corr_90ci": r_ci,
            "predicted_better_n": len(up), "predicted_better_actual_change": round(float(np.average(up.actual_change, weights=up.w)), 3),
            "predicted_worse_n": len(dn), "predicted_worse_actual_change": round(float(np.average(dn.actual_change, weights=dn.w)), 3),
            "better_minus_worse": round(float(diff(df)), 3), "better_minus_worse_90ci": d_ci,
            "pred_change_sd": round(float(np.sqrt(np.cov(x, aweights=w))), 3)}


res = [summarize(P, "all pitchers"), summarize(P[P.changed], "made a real change"),
       summarize(P[P.new_pitch], "added a new pitch"), summarize(P[P.fb_change], "changed fastball"),
       summarize(P[~P.changed], "no major change")]
out = {"note": "runs per 100 pitches, pitcher's perspective; slope = actual change per 1 run/100 of predicted change "
               "(1 = perfectly calibrated); weighted by 1/(1/n25 + 1/n26); 90% CIs from a pitcher bootstrap",
       "min_pitches_each_season": MIN_N, "bootstrap": B, "results": res}
json.dump(out, open(R / "backtest.json", "w"), indent=1)
P2 = P.join(names[["name"]]).merge(NEW.groupby("pitcher").new_type.agg(", ".join), left_index=True, right_index=True, how="left") \
      .merge(FB.set_index("pitcher"), left_index=True, right_index=True, how="left")
P2.round(3).sort_values("pred_change", ascending=False).to_csv(R / "backtest_pitchers.csv")
print(pd.DataFrame(res).set_index("group").T.to_string())

# ---------- figure ----------
INK, INK2, GRID, SURF, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b8b6b0", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})

# left: binned calibration, all pitchers
C = P.copy(); C["bin"] = pd.qcut(C.pred_change, 6, labels=False)
rows = []
for b_, g in C.groupby("bin"):
    m = lambda s: np.average(s.actual_change, weights=s.w)
    bs = [m(g.sample(len(g), replace=True, random_state=i)) for i in range(1000)]
    rows.append((np.average(g.pred_change, weights=g.w), m(g), *np.quantile(bs, [0.05, 0.95])))
rows = np.array(rows)
a1.scatter(P[~P.changed].pred_change, P[~P.changed].actual_change, s=10, color=MUTED, alpha=0.5, linewidths=0, label="no major change")
a1.scatter(P[P.changed].pred_change, P[P.changed].actual_change, s=12, color=ORANGE, alpha=0.55, linewidths=0, label="made a real change")
lim = max(np.abs(rows[:, 0]).max(), 0.5) * 1.3
a1.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--")
a1.text(lim * 0.95, lim * 0.95 + 0.12, "predicted = actual", ha="right", va="bottom", fontsize=8, color=INK2)
a1.errorbar(rows[:, 0], rows[:, 1], yerr=[rows[:, 1] - rows[:, 2], rows[:, 3] - rows[:, 1]], fmt="o", color=BLUE,
            ms=7, lw=1.8, capsize=0, zorder=4, label="average of each sixth (90% CI)")
a1.set_xlim(-lim, lim); a1.set_ylim(-2, 2)
a1.set_xlabel("Predicted change, runs per 100 pitches")
a1.set_ylabel("Actual change, runs per 100 pitches")
s0 = res[0]
a1.set_title(f"All {s0['pitchers']} pitchers: results moved with the predictions", fontsize=10)
a1.legend(loc="lower right", fontsize=7.5, frameon=False)

# right: among pitchers who made a real change, model said better vs worse
s1 = res[1]
vals = [s1["predicted_worse_actual_change"], s1["predicted_better_actual_change"]]
labs = [f"Model said worse\n({s1['predicted_worse_n']} pitchers)", f"Model said better\n({s1['predicted_better_n']} pitchers)"]
ch = P[P.changed]
ci = []
for cond in (ch.pred_change <= 0, ch.pred_change > 0):
    g = ch[cond]
    bs = [np.average((s := g.sample(len(g), replace=True, random_state=i)).actual_change, weights=s.w) for i in range(2000)]
    ci.append(np.quantile(bs, [0.05, 0.95]))
a2.bar([0, 1], vals, color=[MUTED, BLUE], width=0.55)
a2.errorbar([0, 1], vals, yerr=[[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]],
            fmt="none", color=INK, lw=1.2, capsize=4)
a2.axhline(0, color=INK2, lw=1)
for i, v in enumerate(vals):
    a2.text(i + 0.3, v, f"{v:+.2f}", ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")
a2.set_xticks([0, 1], labs); a2.grid(axis="x", visible=False); a2.set_xlim(-0.5, 1.75)
a2.set_ylabel("Actual change, runs per 100 pitches")
a2.set_title(f"{s1['pitchers']} pitchers who added a pitch or changed their fastball", fontsize=10)
a2.text(0.5, -0.27, f"Gap: {s1['better_minus_worse']:+.2f} runs per 100 pitches (90% CI {s1['better_minus_worse_90ci'][0]:.2f} to {s1['better_minus_worse_90ci'][1]:.2f})", transform=a2.transAxes, ha="center", fontsize=8.5, color=INK)
fig.text(0.01, -0.04, "Pitchers with 500+ pitches in both 2025 and 2026. Predictions come from models trained on 2025 only, never on the pitcher being scored. "
         "Actual change uses denoised run value. Error bars: 90% bootstrap intervals.", fontsize=7.5, color=INK2)
fig.tight_layout()
fig.savefig(F / "05_backtest.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figure saved")

"""
Figures for the README.

    python src/figures.py   ->  figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, F = Path("results"), Path("figures"); F.mkdir(exist_ok=True)
INK, INK2, GRID, SURF, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b8b6b0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# one fixed color per pitch type (categorical slots, never reassigned)
TYPE_COLOR = {"FF": "#2a78d6", "SI": "#eb6834", "FC": "#1baf7a", "SL": "#eda100", "ST": "#e87ba4",
              "SV": "#e87ba4", "CU": "#008300", "KC": "#008300", "CH": "#6250d6", "FS": "#e34948", "FO": "#e34948"}
NAMES = {"FF": "Four-seamer", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
         "KC": "Knuckle curve", "CH": "Changeup", "FS": "Splitter", "SV": "Slurve", "FO": "Forkball"}
CASES = ["Jacob Misiorowski", "MacKenzie Gore", "Michael King", "Paul Skenes"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left"})


def save(fig, name):
    fig.savefig(F / name, dpi=200, bbox_inches="tight"); plt.close(fig)


V = json.load(open(R / "validation.json"))

# ---- 1. stability -------------------------------------------------------------------------
rel = V["reliability"]
items = [("Stuff+", rel["stuff"]["pitches_for_r_0.7"]), ("Pitching+", rel["pitching"]["pitches_for_r_0.7"]),
         ("Location+", rel["location"]["pitches_for_r_0.7"]), ("Actual run value", rel["rv"]["pitches_for_r_0.7"])]
fig, ax = plt.subplots(figsize=(7.5, 2.9))
y = np.arange(len(items))[::-1]
for (lab, v), yy in zip(items, y):
    ax.hlines(yy, 30, v, color=BLUE if lab == "Stuff+" else MUTED, lw=2)
    ax.scatter(v, yy, s=60, color=BLUE if lab == "Stuff+" else MUTED, zorder=3)
    ax.text(v * 1.12, yy, f"{v:,} pitches", va="center", fontsize=9, color=INK2)
ax.set_xscale("log"); ax.set_xlim(30, 30000)
ax.set_xticks([50, 100, 500, 1000, 5000, 10000], ["50", "100", "500", "1,000", "5,000", "10,000"])
ax.minorticks_off()
ax.set_yticks(y, [i[0] for i in items]); ax.grid(axis="y", visible=False)
ax.set_xlabel("Pitches needed for a split-half reliability of 0.7 (log scale)")
ax.set_title("Stuff+ stabilizes in about 76 pitches; results take ~6,300")
save(fig, "02_stability.png")

# ---- 2. next-season prediction --------------------------------------------------------------
ns = V["next_season"]
lab = {"Stuff+ 2025": "Stuff+", "Pitching+ 2025": "Pitching+", "K-BB% 2025": "K-BB%",
       "FIP-style rate 2025 (negated)": "FIP-style rate", "Run value/100 2025": "Run value (actual)",
       "Location+ 2025": "Location+"}
rows = sorted([(lab[k], v["r_with_2026_run_value"]) for k, v in ns.items()], key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.barh([r[0] for r in rows], [r[1] for r in rows], height=0.6,
        color=[BLUE if r[0] == "Stuff+" else MUTED for r in rows])
for i, (_, v) in enumerate(rows):
    ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
ax.set_xlabel(f"Correlation of the 2025 metric with 2026 run value per pitch ({V['n_pitchers_both_seasons']} pitchers)")
ax.set_title("2025 Stuff+ predicts 2026 results better than 2025 results do")
ax.grid(axis="y", visible=False); ax.set_xlim(0, 0.4)
save(fig, "03_next_season.png")

# ---- 3. league opportunities ---------------------------------------------------------------
O = pd.read_csv(R / "lab_opportunities.csv").dropna(subset=["best_runs"]).head(15).iloc[::-1]
kind = np.where(O.new_pitch_runs >= O.tweak_runs, "new", "tweak")
label = np.where(kind == "new", "Add a " + O.best_new_pitch.fillna(""), O.best_tweak.fillna("").str.capitalize())
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh(np.arange(len(O)), O.best_runs, height=0.65, color=[BLUE if k == "new" else ORANGE for k in kind])
for i, (v, l) in enumerate(zip(O.best_runs, label)):
    ax.text(v + 0.1, i, l, va="center", fontsize=8.5, color=INK2)
ax.set_yticks(np.arange(len(O)), [f"{n} ({t})" for n, t in zip(O.name, O.team)])
ax.set_xlim(0, O.best_runs.max() * 1.75); ax.grid(axis="y", visible=False)
ax.set_xlabel("Runs saved per season from the best realistic change")
ax.set_title("The 15 biggest pitch-design opportunities of 2026")
ax.text(0.98, 0.03, "■ new pitch   ", color=BLUE, transform=ax.transAxes, ha="right", fontsize=9)
ax.text(0.98, 0.08, "■ change to an existing pitch", color=ORANGE, transform=ax.transAxes, ha="right", fontsize=9)
save(fig, "04_top_opportunities.png")

# ---- 4. case studies -------------------------------------------------------------------------
d = pd.read_pickle("data/scored.pkl.gz")
d = d[d.game_year == 2026]
G = pd.read_csv(R / "pitch_type_grades.csv").query("game_year == 2026")
T = pd.read_csv(R / "lab_tweaks.csv"); N = pd.read_csv(R / "lab_new_pitch.csv")
Opp = pd.read_csv(R / "lab_opportunities.csv").set_index("name")
case_rows = []
for k, nm in enumerate(CASES, start=1):
    g = G[(G.name == nm) & (G.pitches >= 50)]
    pid = int(g.pitcher.iloc[0]); team = Opp.loc[nm, "team"]
    pts = d[d.pitcher == pid]
    bt = T[(T.name == nm) & T.feasible].sort_values("runs_per_season").tail(1)
    bn = N[N.name == nm].sort_values("runs_per_season").tail(1)
    use_new = len(bn) and (not len(bt) or bn.runs_per_season.iloc[0] >= bt.runs_per_season.iloc[0])
    fig, ax = plt.subplots(figsize=(7, 6))
    placed = []
    for _, r in g.sort_values("ivb", ascending=False).iterrows():
        sample = pts[pts.pitch_type == r.pitch_type].sample(min(250, int(r.pitches)), random_state=0)
        c = TYPE_COLOR.get(r.pitch_type, MUTED)
        ax.scatter(sample.hb, sample.ivb, s=6, color=c, alpha=0.18, linewidths=0)
        ax.scatter(r.hb, r.ivb, s=60 + 900 * r.usage, color=c, edgecolors=SURF, linewidths=2, zorder=3)
        crowded = any(abs(r.hb - x) < 6 and abs(r.ivb - y) < 4 for x, y in placed)
        ax.annotate(f"{NAMES.get(r.pitch_type, r.pitch_type)}  {r['Stuff+']:.0f}", (r.hb, r.ivb),
                    xytext=(10, -16) if crowded else (10, 8), textcoords="offset points", fontsize=9,
                    color=INK, zorder=4)
        placed.append((r.hb, r.ivb))
    if use_new:
        b = bn.iloc[0]
        ax.scatter(b.hb, b.ivb, s=320, marker="*", facecolor=SURF, edgecolor=INK, linewidths=1.6, zorder=5)
        # default label sits below-right of the star; move it above if an existing pitch is in the way
        blocked = any(b.hb < x < b.hb + 13 and b.ivb - 6 < y < b.ivb + 1 for x, y in placed)
        ax.annotate(f"New {b.new_pitch}  {b.stuff_plus:.0f}\n{b.velo:.0f} mph · {b.spin:,.0f} rpm", (b.hb, b.ivb),
                    xytext=(0, 14) if blocked else (12, -26), ha="center" if blocked else "left",
                    textcoords="offset points", fontsize=9, color=INK, fontweight="bold", zorder=6)
        rep = g[g.pitch_type == b.replaces_type].iloc[0]
        ax.scatter(rep.hb, rep.ivb, s=140, marker="x", color=INK, linewidths=1.5, zorder=5)
        title = f"{nm} ({team}): add a {b.new_pitch}, +{b.runs_per_season:.1f} runs/season"
        sub = (f"A {b.new_pitch} built from pitchers with a similar arm slot, replacing his {b.replaces} "
               f"(Stuff+ {b.replaced_stuff_plus:.0f} → {b.stuff_plus:.0f}).")
        rec = {"change": f"Add a {b.new_pitch} ({b.velo:.0f} mph, {b.ivb:+.0f} in vertical, {b.hb:+.0f} in horizontal break) "
                         f"in place of his {b.replaces}", "stuff_before": b.replaced_stuff_plus, "stuff_after": b.stuff_plus,
               "runs": b.runs_per_season, "peers": b.peer_examples}
    else:
        b = bt.iloc[0]
        r0 = g[g.pitch_type == b.pitch_type].iloc[0]
        dx = 0 if b.feature != "hb" else b.target - b.current
        dy = 0 if b.feature != "ivb" else b.target - b.current
        ax.annotate("", xy=(r0.hb + dx, r0.ivb + dy), xytext=(r0.hb, r0.ivb),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8), zorder=5)
        ax.annotate(f"{b.change}: Stuff+ {b.stuff_plus_before:.0f} → {b.stuff_plus_after:.0f}",
                    (r0.hb + dx, r0.ivb + dy), xytext=(12, 6), textcoords="offset points",
                    fontsize=9, color=INK, fontweight="bold", zorder=6)
        title = f"{nm} ({team}): {b.pitch} {b.change.replace('+', '+')}, +{b.runs_per_season:.1f} runs/season"
        sub = (f"{b.current:.1f} → {b.target:.1f} in; the top 10% of pitchers with his arm slot reach {b.peer_limit:.1f} in. "
               "Every other pitch is graded partly against the fastball, so all of them gain.")
        rec = {"change": f"{b.pitch.capitalize()}: {b.change} ({b.current:.1f} → {b.target:.1f} in)",
               "stuff_before": b.stuff_plus_before, "stuff_after": b.stuff_plus_after, "runs": b.runs_per_season,
               "peers": ""}
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    lim = max(22, np.abs(pts[["hb", "ivb"]]).quantile(0.99).max() + 3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Horizontal break, inches  (← glove side   ·   arm side →)")
    ax.set_ylabel("Induced vertical break, inches")
    ax.set_title(title, fontsize=11.5, pad=30)
    ax.text(0, 1.015, "\n".join(__import__("textwrap").wrap(sub, 100)), transform=ax.transAxes, fontsize=8.5,
            color=INK2, va="bottom")
    fig.text(0.125, 0.02, "2026 arsenal from the pitcher's view, lefties mirrored. Dot size = usage; label = Stuff+ "
             "(100 = league average)." + (" Marker x = pitch replaced." if use_new else ""), fontsize=7.5, color=INK2)
    save(fig, f"case_{k}_{nm.lower().replace(' ', '_').replace('.', '')}.png")
    p26 = pd.read_csv(R / "pitcher_grades.csv").query("game_year == 2026 and name == @nm").iloc[0]
    case_rows.append({"pitcher": nm, "team": team, "stuff_plus_2026": round(p26["Stuff+"], 1),
                      "pitching_plus_2026": round(p26["Pitching+"], 1), **rec})
pd.DataFrame(case_rows).round(2).to_csv(R / "case_studies.csv", index=False)
print(pd.DataFrame(case_rows).round(1).to_string())

"""
Leak-free Stuff predictions for the backtest.

The backtest asks: when a pitcher's pitches physically changed from 2025 to 2026, did his results
move the way the model said they would? For that, the predicted change must come from a model that
saw neither 2026 nor the pitcher's own 2025 outcomes (otherwise a pitcher who was unlucky in 2025
gets a slightly lower in-sample 2025 grade, and regression to the mean looks like model skill).

So the 2025 data is split into 5 folds by pitcher. Each fold's pitchers are scored, in both seasons,
by a Stuff model trained on 2025 pitches from the other four folds only.

    python src/backtest_preds.py   ->  data/backtest_preds.pkl.gz
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models import STUFF, gbm

d = pd.read_pickle("data/scored.pkl.gz")[["game_year", "pitcher", "pitch_type", "rv_x", "rv_cn"] + STUFF]
d["b_stuff"] = np.nan
d25 = d[d.game_year == 2025]
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(d25, groups=d25.pitcher)):
    held = d25.pitcher.iloc[te].unique()
    m = gbm().fit(d25.iloc[tr][STUFF], d25.iloc[tr]["rv_x"])
    mask = d.pitcher.isin(held)
    d.loc[mask, "b_stuff"] = m.predict(d.loc[mask, STUFF])
    print(f"fold {k + 1}/5: {len(held)} pitchers, {m.n_iter_} trees", flush=True)

d = d.dropna(subset=["b_stuff"])
d[["game_year", "pitcher", "pitch_type", "rv_x", "rv_cn", "velo", "ivb", "hb", "arm_angle", "b_stuff"]] \
    .to_pickle("data/backtest_preds.pkl.gz")
print("saved", len(d), "pitches")

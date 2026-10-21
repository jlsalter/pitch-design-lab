"""
Build the pitch-level modeling dataset for 2025-26.

Reads the processed pitch table from the ABS project (which already aligns 2025 and 2026
pitch locations to the middle of the plate), and writes data/pitches.parquet-like pickle with:
  - physics features, mirrored so every pitcher is right-handed (arm side = positive x)
  - arsenal features: each pitch relative to the pitcher's primary fastball that season
  - location features relative to the batter's zone
  - targets: count-neutral run value from the pitcher's perspective (rv_cn), and a denoised version
    (rv_x) in which each batted ball's result is replaced by the average for its exit velocity and launch angle

    python src/build_pitches.py [path/to/abs-project/data/pitches.pkl.gz]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../abs-project/data/pitches.pkl.gz")
OUT = Path("data"); OUT.mkdir(exist_ok=True)

KEEP = ["game_pk", "game_date", "game_year", "at_bat_number", "pitch_number", "pitcher", "player_name",
        "batter", "p_throws", "stand", "pitch_type", "release_speed", "release_spin_rate", "spin_axis",
        "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
        "px", "pz", "sz_top", "sz_bot", "abs_top", "abs_bot", "balls", "strikes", "description",
        "events", "delta_run_exp", "home_team", "away_team", "inning_topbot", "launch_speed", "launch_angle"]

# pitch classes (FA = position player pitching, EP eephus, PO pitchout, CS slow curve, KN knuckleball)
CLASS = {"FF": "fastball", "SI": "fastball", "FC": "cutter",
         "SL": "breaking", "ST": "breaking", "SV": "breaking", "CU": "breaking", "KC": "breaking",
         "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed"}


def main():
    df = pd.read_pickle(SRC)
    df = df[[c for c in KEEP if c in df.columns]]
    df = df[df.pitch_type.isin(CLASS)].copy()
    df["pitch_class"] = df.pitch_type.map(CLASS)
    df["is_lhp"] = (df.p_throws == "L").astype(int)
    sign = np.where(df.is_lhp == 1, -1.0, 1.0)

    # --- physics, normalized to a right-handed pitcher (inches for movement) ---
    df["velo"] = df.release_speed
    df["spin"] = df.release_spin_rate
    df["ivb"] = df.pfx_z * 12                       # induced vertical break
    df["hb"] = df.pfx_x * 12 * sign * -1            # arm-side horizontal break (+ = arm side)
    # Statcast pfx_x is catcher's view: a RHP's arm side is negative x, so flip to make arm side positive
    df["rel_x"] = df.release_pos_x * sign * -1      # + = toward arm side (away from the plate's center)
    df["rel_z"] = df.release_pos_z
    df["ext"] = df.release_extension
    df["axis"] = np.where(df.is_lhp == 1, 360 - df.spin_axis, df.spin_axis)
    df["axis_sin"], df["axis_cos"] = np.sin(np.radians(df.axis)), np.cos(np.radians(df.axis))
    df["same_side"] = (df.stand == df.p_throws).astype(int)

    # --- arsenal context: each pitch vs. the pitcher's primary fastball in that season ---
    fb = df[df.pitch_type.isin(["FF", "SI"])]
    primary = (fb.groupby(["pitcher", "game_year", "pitch_type"]).size().rename("n").reset_index()
               .sort_values("n").groupby(["pitcher", "game_year"]).tail(1)[["pitcher", "game_year", "pitch_type"]])
    fb_mean = (fb.merge(primary, on=["pitcher", "game_year", "pitch_type"])
               .groupby(["pitcher", "game_year"])[["velo", "ivb", "hb"]].mean()
               .rename(columns=lambda c: "fb_" + c))
    df = df.join(fb_mean, on=["pitcher", "game_year"])
    for c in ("velo", "ivb", "hb"):
        df["d_" + c] = df[c] - df["fb_" + c]
    # pitchers with no FF/SI (rare): compare against their own hardest pitch type
    no_fb = df.fb_velo.isna()
    if no_fb.any():
        hard = df[no_fb].groupby(["pitcher", "game_year"])[["velo", "ivb", "hb"]].transform("max")
        for c in ("velo", "ivb", "hb"):
            df.loc[no_fb, "d_" + c] = df.loc[no_fb, c] - hard[c]

    # --- location (mid-plate, both seasons) relative to the batter's zone ---
    top = df.abs_top.fillna(df.sz_top); bot = df.abs_bot.fillna(df.sz_bot)
    df["loc_x"] = np.where(df.stand == "R", df.px, -df.px) * 12     # inches; + = away from the batter
    df["loc_z"] = (df.pz - bot) / (top - bot)                       # 0 = bottom of zone, 1 = top

    # --- target: count-neutral run value, pitcher's perspective (+ = good for the pitcher) ---
    df["rv"] = -df.delta_run_exp
    df["count"] = df.balls.astype(str) + "-" + df.strikes.astype(str)
    df["rv_cn"] = df.rv - df.groupby(["game_year", "count"])["rv"].transform("mean")

    # --- denoised target: a batted ball's actual result depends on fielders and luck, so replace it
    # with the average result of balls hit at the same exit velocity and launch angle that season.
    # Swings and misses, fouls, balls and called strikes keep their actual (count-neutral) value.
    bip = (df.description == "hit_into_play") & df.launch_speed.notna() & df.launch_angle.notna()
    df["ls_bin"] = (df.launch_speed // 3).clip(upper=40)
    df["la_bin"] = (df.launch_angle.clip(-60, 70) // 4)
    bin_mean = df[bip].groupby(["game_year", "ls_bin", "la_bin"])["rv_cn"].transform("mean")
    df["rv_x"] = df.rv_cn
    df.loc[bip, "rv_x"] = bin_mean

    need = ["velo", "spin", "ivb", "hb", "rel_x", "rel_z", "ext", "axis_sin", "arm_angle", "d_velo", "d_ivb",
            "d_hb", "loc_x", "loc_z", "rv_cn", "rv_x"]
    before = len(df)
    df = df.dropna(subset=need)
    df["fielding_team"] = np.where(df.inning_topbot == "Top", df.home_team, df.away_team)
    cols = ["game_pk", "game_date", "game_year", "at_bat_number", "pitch_number", "pitcher", "player_name",
            "fielding_team", "p_throws", "is_lhp", "stand", "same_side", "pitch_type", "pitch_class", "velo",
            "spin", "ivb", "hb", "rel_x", "rel_z", "ext", "axis", "axis_sin", "axis_cos", "arm_angle",
            "fb_velo", "fb_ivb", "fb_hb", "d_velo", "d_ivb", "d_hb", "loc_x", "loc_z", "balls", "strikes",
            "count", "description", "events", "launch_speed", "launch_angle", "rv", "rv_cn", "rv_x"]
    df[cols].to_pickle(OUT / "pitches.pkl.gz")
    print(f"kept {len(df):,} of {before:,} pitches; by year:", df.groupby("game_year").size().to_dict())
    print("pitchers:", df.pitcher.nunique(), " pitch types:", df.pitch_type.value_counts().to_dict())


if __name__ == "__main__":
    main()

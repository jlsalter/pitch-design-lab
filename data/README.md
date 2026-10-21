# Data

`processed/pitches_scored_{2025,2026}.csv.gz`: every regular-season pitch (1,413,233 in total), with features and model scores. Source: Baseball Savant / MLB Advanced Media; not affiliated with MLB.

| Column | Meaning |
|---|---|
| `game_pk`, `at_bat_number`, `pitch_number` | Statcast pitch key |
| `pitcher`, `player_name`, `p_throws`, `stand` | Pitcher id and name, pitcher hand, batter side |
| `pitch_type` | Statcast pitch type (FF four-seamer, SI sinker, FC cutter, SL slider, ST sweeper, CU curveball, KC knuckle curve, CH changeup, FS splitter, ...) |
| `velo`, `spin` | Release speed (mph), spin rate (rpm) |
| `ivb`, `hb` | Induced vertical break and horizontal break (inches). `hb` is mirrored so arm side is positive for both hands |
| `rel_x`, `rel_z`, `ext`, `arm_angle` | Release side (arm side +) and height (ft), extension (ft), arm angle (degrees) |
| `d_velo`, `d_ivb`, `d_hb` | Difference from the pitcher's primary fastball that season |
| `loc_x`, `loc_z` | Location at mid-plate: inches from center (+ = away from the batter), and height as a fraction of the batter's zone (0 = bottom, 1 = top) |
| `balls`, `strikes` | Count before the pitch |
| `rv_cn` | Actual run value, pitcher's perspective, count-neutral |
| `rv_x` | Denoised target: like `rv_cn`, but batted balls valued at the average for their exit velocity and launch angle |
| `x_stuff`, `x_location`, `x_pitching` | Out-of-sample model predictions (runs per pitch). `results/validation.json` has the scale used to convert them to 100-based grades |

`data/pitches.pkl.gz` and `data/scored.pkl.gz` are intermediate files, rebuilt by `run_all.sh` and not committed.

# Methods and decisions

Each section gives the choice, why it was made, and the alternatives considered.

## 1. Data
All 2025 and 2026 regular-season pitches from Baseball Savant: 1,413,233 pitches and 1,059 pitchers after removing position players pitching, eephus pitches and pitchouts. Pitch locations come from the [abs-challenge-analysis](https://github.com/jlsalter/abs-challenge-analysis) pipeline, which moves 2025 locations to the middle of the plate along each pitch's tracked trajectory so both seasons are measured the same way (Statcast changed its reference point in 2026).

## 2. Features
- **Physics.** Velocity, spin rate, spin axis (as sine and cosine, so 359° and 1° are close), induced vertical break, horizontal break, release side and height, extension, arm angle.
- **Handedness.** Left-handers are mirrored so arm side is always positive. Check: sinkers and changeups break toward the arm side, and sweepers toward the glove side, by the same amounts for both hands.
- **Arsenal context.** Each pitch's velocity, vertical break and horizontal break minus the pitcher's primary fastball that season (the more-used of his four-seamer and sinker). Separation from the fastball is a large part of why a breaking ball or changeup works. Without these features a model can't tell a 92-mph changeup from a 92-mph fastball.
- **Batter side.** A same-side indicator; platoon matchups change a pitch's value.

## 3. Target
- **Run value, count-neutral.** Statcast's change in run expectancy, from the pitcher's perspective, minus that season's average for the count. A pitch thrown 3-0 and a pitch thrown 0-2 are then judged on the same footing.
- **Denoised batted balls.** A ball in play is valued at the season average for balls in its 3-mph exit-velocity and 4° launch-angle bin, not its actual result. Whether a 100-mph liner finds a glove is mostly about fielders and luck, not the pitch. This cut the standard deviation of batted-ball values from 0.50 to 0.32 runs, and roughly doubled the Stuff model's out-of-sample signal (pitch-level correlation 0.025 → 0.046).
- **Alternative considered:** separate models for whiff, called strike and contact quality, combined afterward. That is more flexible but adds several modeling stages. A single regression on the denoised target is simpler to validate and explain.

## 4. Models
- **Algorithm.** Gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`): up to 600 trees, learning rate 0.05, 63 leaves, at least 400 pitches per leaf, L2 = 1, early stopping on a 10% validation split.
- **Feature sets.** Stuff = physics, arsenal context and batter side. Location = location relative to the batter's zone, count, batter side and pitch class. Pitching = all of them.
- **Cross-fitting.** The model trained on 2025 scores 2026, and vice versa, so no pitch is graded by a model that saw it. Pitch-level fit is small by nature (out-of-sample r: Stuff 0.045, Location 0.26, Pitching 0.27), because one pitch's outcome is mostly noise. The meaningful tests are at the pitcher level (section 5).
- **Scale.** Grade = 100 + 10 × (mean prediction − league mean) / SD of pitcher-season means (pitchers with 500+ pitches). Pitch-type grades use the same conversion, so they spread wider than pitcher grades, as in public models.

## 5. Validation
- **Reliability.** Each pitcher-season is split into odd and even games (852 pitcher-seasons with 250+ pitches in each half). The correlation of the halves is converted with Spearman-Brown to the number of pitches needed for a reliability of 0.7. Results: Stuff+ 76, Pitching+ 464, Location+ 965, actual run value 6,347.
- **Next-season prediction.** For 324 pitchers with 500+ pitches in both seasons, 2025 metrics are correlated with 2026 count-neutral run value per pitch. The 2025 grades here come from the model trained on 2025 only (in-sample for 2025, never exposed to 2026), because the cross-fit 2025 grades come from the 2026 model and would leak future data. Results: Stuff+ 0.34, Pitching+ 0.33, K-BB% 0.32, FIP-style rate 0.26, actual run value 0.23, Location+ 0.09.
- **Face validity.** The 2026 leaders include Misiorowski, Burns, Sasaki and deGrom. Mason Miller's slider and Tyler Rogers' submarine sinker are among the top individual pitches.

## 6. Design lab
- **Scoring.** The 2025 Stuff model scores 2026 pitches, so lab numbers match the out-of-sample grades exactly.
- **Tweaks.** Shift one trait of every pitch of a type (for example +2 in of vertical break) and re-score the pitcher's whole 2026 arsenal. Changing the primary fastball also changes every other pitch's separation from it. Value = summed change in predicted run value over his actual 2026 pitches.
- **Feasibility.** Peers are pitcher-seasons throwing the same pitch type (100+ pitches) with an arm angle within 5° and release height within 0.3 ft. A tweak is feasible only if the target value is within the peers' 90th percentile in that direction. 2,008 of 3,174 candidate tweaks pass.
- **New pitches.**
  1. Find the 15 pitchers who throw the pitch and are closest in standardized arm angle, release height, extension and fastball velocity and shape.
  2. Take their median shape relative to their own fastball (velocity gap, vertical and horizontal break gaps, spin, spin axis).
  3. Apply that shape to this pitcher's fastball, keeping his release.
  4. The new pitch replaces his weakest secondary pitch at the same usage; the value is the difference in predicted run value times that usage.
- **Restrictions.**
  - Only secondary pitches can be suggested or replaced. Per-pitch run value isn't comparable across roles: fastballs are thrown in the zone to get strikes and always grade lower per pitch than chase pitches. An earlier version that allowed replacing the primary fastball produced unrealistic suggestions, such as a curveball replacing a 51%-usage sinker.
  - Suggestions that duplicate an existing pitch family are skipped (for example, a slider for a pitcher who already throws a sweeper).
- **Selection.** Each pitcher's best feasible option is ranked by runs per season. The four case studies all rank in the top 20 of 333, chosen to cover both kinds of change and well-known pitchers across the range of current Stuff+.

## 7. Backtest: did predicted changes match real ones?
- **Question.** The lab assumes that when a pitch's physical traits change, the pitcher's results change by the amount the Stuff model predicts. The 2025 to 2026 offseason allows a direct check.
- **Leak-free predictions** (`src/backtest_preds.py`). 2025 pitches are split into 5 folds by pitcher. Each fold's pitchers are scored, in both seasons, by a Stuff model trained on 2025 pitches from the other folds only. So a pitcher's predicted change depends only on how his pitches physically changed, never on his own outcomes or on 2026. This matters: an in-sample 2025 grade partly reflects a pitcher's 2025 luck, which would make ordinary regression to the mean look like model skill.
- **Sample.** 324 pitchers with 500+ pitches in both seasons.
  - Predicted change = mean 2026 prediction − mean 2025 prediction.
  - Actual change = mean 2026 denoised run value − mean 2025, in runs per 100 pitches.
  - Pitchers are weighted by 1/(1/n₂₀₂₅ + 1/n₂₀₂₆), since the actual change is noisier with fewer pitches.
- **Real changes.**
  - *New pitch:* a type thrown under 2% in 2025 and at least 5% in 2026, at least 4 in of movement or 3 mph from every pitch he threw 5%+ in 2025. The shape rule removes Statcast relabels, such as a slider reclassified as a sweeper.
  - *Fastball change:* the same primary fastball both years, with at least 1.5 in of change in vertical or horizontal break, or 1 mph of velocity.
  - 85 pitchers added a pitch, 129 changed their fastball, and 174 did at least one.
- **Results** (90% intervals from 4,000 pitcher bootstrap resamples).
  - Slope of actual on predicted change: 1.35 (0.92 to 1.75), or 1.11 (0.74 to 1.48) controlling for 2025 run value. A slope of 1 is perfect calibration.
  - Weighted correlation: 0.27 (0.18 to 0.34).
  - Among the 174 changers, actual change was +0.11 runs/100 where the model predicted improvement and −0.14 where it predicted decline. The gap is 0.26 (0.07 to 0.44).
  - Subgroups: new pitch slope 1.65 (0.91 to 2.26); fastball change 1.42 (0.73 to 2.08); no major change 1.31 (0.73 to 1.95).
  - With raw count-neutral run value instead of the denoised target: slope 0.66 (0.04 to 1.28).
- **Caveats.**
  - The changes were self-selected, not assigned, so the test supports the model's calibration but is not a causal estimate.
  - The average actual change is slightly negative (−0.08 runs/100) for this group of returning pitchers. That shows up as an intercept and does not affect the slope.

## 8. Limitations
- **Physics, not execution.** The model has no notion of command, health, deception from tunneling beyond fastball separation, or whether changing one pitch disturbs the others.
- **Correlation, not causation.** Counterfactual predictions assume the model's relationships hold when a pitcher changes his pitch. That is plausible for modest changes inside the observed peer range, which the feasibility rule enforces.
- **Usage held fixed.** The lab keeps usage at 2026 levels. A better pitch would probably be thrown more, so gains may be understated.
- **Two seasons.** The next-season comparison and the backtest each rest on one year-over-year pair.

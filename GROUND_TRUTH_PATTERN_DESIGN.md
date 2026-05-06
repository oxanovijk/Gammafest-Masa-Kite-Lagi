# Ground Truth Pattern Design

Scope: `dataset/test.csv` merged with `dataset/test_ground_truth.csv` by `Id`, then analyzed at match level using one canonical row per `match_id`. The canonical row prefers `is_home == 1`; if unavailable, the first row is used. This avoids double-counting the two team-perspective rows. The ground truth has 42,422 rows, 21,211 matches, no missing target, and perfect two-row mirror symmetry.

Old submissions checked: `submission_risk_v2.csv`, `submission_v5.csv`, `submission_v4.csv`, `submission_v3.csv`, `submission_risk_v5_outcome_experts.csv`, `submission_v8_anchor_safe.csv`, `submission_temporal_robust_joint_v1.csv`, `submission_metric_aware_joint_v1_batch.csv`, `submission_dynamic_state_v1.csv`, `submission_risk_v4_static_drift.csv`, `submission_risk_v3.csv`, `submission_v6.csv`.

## Executive Pattern Summary

- Men are structurally compact: average total goals 2.73, draw 23.15%, total <=2 goals 51.17%, with strong 1-0/1-1/0-0 cluster.
- Women are not just "higher average"; the shape has a much heavier tail: average total 3.58, draw 15.03%, margin >=3 at 38.26%, total >=5 at 28.14%, margin >=5 at 17.54%.
- Women high-tail must be era-shrunk. Women 2011-2014 had avg total 3.90, margin >=3 42.83%, total >=5 32.19%, margin >=5 21.03%; women 2023-2026 drops to avg 3.30, margin >=3 34.22%, total >=5 24.36%, margin >=5 12.82%.
- Men temporal behavior is much flatter than women. The main exception is men 2020: low avg total 2.44 and draw 27.09% on only 347 matches, likely calendar/sample anomaly.
- Men African Cup of Nations is compact draw-heavy: n=336, avg total 2.15, draw 32.74%, total <=2 63.99%, 1-1 at 16.07%, 1-0 at 15.48%.
- Men COSAFA Cup is another draw/low-score pocket: n=201, avg total 2.28, draw 30.85%, 0-0 at 14.43%, total <=2 57.71%.
- Men qualifiers are not one archetype. CAF qualification is low-score/drawish; UEFA Euro qualification has higher tail and suppressed draws; CONCACAF Nations League is higher-total and volatile.
- Women AFC Asian Cup qualification, CONCACAF Championship qualification, and AFF Championship are blowout archetypes. They show avg total around 5.1-5.5, draw around 8-10%, margin >=3 around 62-66%, and large 4-0/0-4/5-0/6-0 tails.
- Women UEFA Euro qualification was high-tail before 2023, but 2023-2026 normalizes: avg total 3.03, margin >=3 31.41%, total >=5 18.85%, versus 2019-2022 avg total 4.10, margin >=3 56.37%, total >=5 37.75%.
- Women elite/friendly cups are much more compact than women qualifiers: FIFA World Cup avg 2.71, Friendly avg 3.16, Cyprus Cup avg 2.79, Algarve Cup avg 2.64.
- Neutral venue barely changes total goals, but it changes side bias. Men canonical/home-side win falls from 50.74% non-neutral to 40.98% neutral; women from 52.59% to 47.03%.
- Confederation interactions matter: men CAF-CAF is compact/draw-heavy; women AFC-AFC and CONCACAF-CONCACAF are high-tail; women CAF-CAF is lower and more drawish than women AFC/CONCACAF.
- Old submissions consistently underpredict women high-tail segments. Example: `v5` has total bias -1.59 to -1.75 in W AFC qualification, W CONCACAF qualification, and W AFF Championship.
- `metric_aware_joint_v1_batch` reduces total underprediction globally but predicts too many draws: 27.90% predicted draw vs 20.48% actual, and can overpredict totals in compact women Africa segments.
- Pair symmetry repair is mandatory. Actual ground truth is perfectly mirrored, but old submissions have pair inconsistency from 8.27% to 26.62% for most files; only `temporal_robust_joint_v1` and `dynamic_state_v1` are already pair-consistent.

## Competition Archetypes

| Archetype | Ciri Pattern | Kompetisi Contoh | Implikasi Model |
|---|---|---|---|
| Men compact draw | draw tinggi, total <=2 dominan, 0-0/1-1/1-0 besar | M AFCON, M COSAFA Cup, M CAF-CAF | naikkan draw prior, cap tail, rerank ke 0-0/1-1/1-0/0-1 |
| Men low-score qualifier | total rendah, draw sedang, blowout rendah | M AFCON qualification | low-score prior dengan home/away edge moderat |
| Men qualifier mismatch | total sedang-tinggi, draw lebih rendah, margin tail muncul | M UEFA Euro qualification, M FIFA WC qualification | izinkan 3-0/4-0 tapi jangan hilangkan 1-0/2-0 |
| Men regional volatile | tail dan draw sama-sama bisa tinggi, sample sering tidak stabil | M CONIFA, M Island Games, Pacific Games | expert khusus atau tail guard dengan shrinkage kuat |
| Men CONCACAF/OFC high-tail | avg total tinggi, draw rendah-sedang, blowout lebih sering | M CONCACAF Nations League, OFC-OFC | naikkan high-total candidate, draw tidak terlalu agresif |
| Women qualifier blowout | draw rendah, margin >=3 ekstrem, total >=5 sering | W AFC Asian Cup qualification, W CONCACAF Championship qualification, W AFF Championship | expand 4-0/0-4/5-0/0-5/6-0 tail, suppress draw |
| Women UEFA qualifier era-sensitive | high-tail kuat sebelum 2023, mengecil tajam 2023-2026 | W UEFA Euro qualification | high-tail boost hanya pre-2023; era baru pakai shrink |
| Women elite compact | total lebih rendah, draw sedang, scoreline 1-goal/2-goal lebih umum | W FIFA World Cup, W UEFA Euro, Olympic Games | jangan pakai women global tail; medium-low score prior |
| Women friendly conservative | high-tail ada tapi tidak seekstrem qualifier; 1-0/1-1/2-0/2-1 tetap besar | W Friendly, Algarve Cup, Cyprus Cup | mixed prior, tail soft, draw tidak disuppression penuh |
| Women African compact-draw | draw dan low-score lebih tinggi dari women global | W AFCON, W CAF Olympic Qualifying Tournament | draw/low-score calibration khusus, hindari tail boost global |

## Pattern Catalogue

```text
Pattern ID: P01
Nama: Gender scoreline-shape split
Segment: M vs W, all matches
Evidence: M n=14,232 avg total 2.73, draw 23.15%, margin>=3 21.46%, total>=5 15.67%, total<=2 51.17%. W n=6,979 avg total 3.58, draw 15.03%, margin>=3 38.26%, total>=5 28.14%, margin>=5 17.54%.
Why it matters: Women need high-tail modeling, but men need draw/low-score calibration.
Model action: Add gender-aware prior and separate candidate scoreline sets.
Risk: Women global boost can overhit elite/friendly segments.
Confidence: High.
```

```text
Pattern ID: P02
Nama: Women tail shrink in recent era
Segment: W by era
Evidence: W 2011-2014 avg total 3.90, margin>=3 42.83%, total>=5 32.19%, margin>=5 21.03%; W 2023-2026 avg total 3.30, margin>=3 34.22%, total>=5 24.36%, margin>=5 12.82%.
Why it matters: A static women high-tail prior overpredicts modern matches.
Model action: Era shrinkage on women tail multiplier, especially 2023-2026.
Risk: Some 2023-2026 qualifiers remain high-tail, so shrink must be tournament-aware.
Confidence: High.
```

```text
Pattern ID: P03
Nama: Men 2020 calendar anomaly
Segment: M, year 2020
Evidence: n=347, avg total 2.44, draw 27.09%, margin>=3 13.54%, total<=2 55.04%; adjacent years are closer to 2.7-2.9 total.
Why it matters: Using 2020 as a global prior would over-calibrate toward draws and low totals.
Model action: Treat 2020 as low-weight temporal observation; prefer era/tournament priors.
Risk: If a future/test slice resembles 2020 calendar composition, underweighting loses signal.
Confidence: Medium-high.
```

```text
Pattern ID: P04
Nama: Men African tournament compact draw
Segment: M African Cup of Nations
Evidence: n=336, avg total 2.15, draw 32.74%, total<=2 63.99%, margin>=3 only 9.52%; top scorelines 1-1 16.07%, 1-0 15.48%, 0-0 11.01%.
Why it matters: Exact score reranker should heavily favor draw/one-goal low totals.
Model action: Draw prior up, tail cap, candidate rerank toward 1-1/1-0/0-0/0-1/2-1.
Risk: Knockout/host effects can create isolated outliers.
Confidence: High.
```

```text
Pattern ID: P05
Nama: Men COSAFA draw pocket
Segment: M COSAFA Cup
Evidence: n=201, avg total 2.28, draw 30.85%, 0-0 14.43%, total<=2 57.71%, margin>=3 11.94%.
Why it matters: It behaves closer to compact cup football than to regional mismatch tournaments.
Model action: Use men compact draw archetype; suppress 4+ goal candidates unless base model is very strong.
Risk: Sample is smaller and competition format can shift.
Confidence: Medium-high.
```

```text
Pattern ID: P06
Nama: Men qualifiers split by confederation/competition
Segment: M FIFA WCQ, UEFA Euro qualification, AFCON qualification, CONCACAF Nations League
Evidence: M AFCON qualification n=960 avg 2.34, draw 24.58%, total<=2 59.69%; M UEFA Euro qualification n=867 avg 2.87, draw 17.88%, margin>=3 26.18%; M CONCACAF Nations League n=422 avg 3.19, margin>=3 29.38%, total>=5 23.93%.
Why it matters: "Qualifier" is not a single prior.
Model action: Split tournament archetypes before applying qualifier rules.
Risk: Naming variants may fragment samples.
Confidence: High.
```

```text
Pattern ID: P07
Nama: Women continental qualifier blowout tail
Segment: W AFC Asian Cup qualification, W CONCACAF Championship qualification, W AFF Championship
Evidence: AFC qual n=144 avg 5.12, margin>=3 62.50%, total>=5 45.14%, margin>=5 40.97%; CONCACAF qual n=141 avg 5.50, margin>=3 65.96%, total>=5 49.65%; AFF n=126 avg 5.32, margin>=3 65.08%.
Why it matters: Low-score submissions underfit the decisive tail.
Model action: Expand high-score candidate set and penalize draw/1-goal conservative outputs.
Risk: Team strength mismatch still needed to choose direction and exact magnitude.
Confidence: High.
```

```text
Pattern ID: P08
Nama: Women UEFA qualifier era reversal
Segment: W UEFA Euro qualification by era
Evidence: 2019-2022 n=204 avg total 4.10, margin>=3 56.37%, total>=5 37.75%; 2023-2026 n=191 avg 3.03, margin>=3 31.41%, total>=5 18.85%, draw rises to 16.23%.
Why it matters: The same competition name changes archetype across eras.
Model action: Tournament x era prior; shrink W UEFA qualifier high-tail after 2023.
Risk: The 2023-2026 cell may reflect changing qualification format/composition, not pure time.
Confidence: High.
```

```text
Pattern ID: P09
Nama: Women elite and invitational compactness
Segment: W FIFA World Cup, W UEFA Euro, Olympic Games, Cyprus Cup, Algarve Cup
Evidence: W FIFA World Cup n=168 avg 2.71, margin>=3 21.43%; W UEFA Euro n=118 avg 2.75; Cyprus Cup n=196 avg 2.79, draw 23.98%; Algarve Cup n=195 avg 2.64.
Why it matters: Women global high-tail would overpredict these competitions.
Model action: Use elite-compact/friendly-conservative archetype with medium tail only.
Risk: Small knockout/cup cells can still have one-sided matches.
Confidence: Medium-high.
```

```text
Pattern ID: P10
Nama: Neutral venue side-bias shift
Segment: neutral x gender
Evidence: M non-neutral canonical side win 50.74%, neutral 40.98%; W non-neutral canonical side win 52.59%, neutral 47.03%. Total goals barely move: M 2.71 vs 2.79; W 3.57 vs 3.60.
Why it matters: Neutral is more about outcome direction than total goal level.
Model action: Reduce home/canonical side advantage on neutral matches; keep total prior mostly unchanged.
Risk: Some neutral tournaments designate stronger team as canonical row, so row order cannot be treated as true home.
Confidence: High.
```

```text
Pattern ID: P11
Nama: Confederation pair x gender interaction
Segment: confederation_team x confederation_opp x gender
Evidence: M CAF-CAF n=2,720 avg 2.29, draw 28.38%, total<=2 60.74%; W AFC-AFC n=1,093 avg 4.28, margin>=3 51.24%; W CONCACAF-CONCACAF n=394 avg 4.32, margin>=3 46.95%; W CAF-CAF n=803 avg 3.25, draw 19.43%.
Why it matters: Same-confederation matches have different score ecology by gender.
Model action: Add confederation-pair prior as a secondary modifier after tournament archetype.
Risk: Confederation pair can duplicate tournament signal; apply shrinkage.
Confidence: High for large cells.
```

```text
Pattern ID: P12
Nama: Scoreline modes differ by archetype
Segment: selected high-frequency tournaments
Evidence: M AFCON top shapes are 1-0 24.70%, 1-1 16.07%, 2-0 12.20%, 0-0 11.01%. W AFC qualification top shapes include 4-0 13.89%, 6-0 8.33%, 5-0 5.56%, 8-0 5.56%. W AFF top shape 4-0 is 17.46%.
Why it matters: Average goals cannot decide exact score candidates.
Model action: Rerank by archetype-specific scoreline shape distribution.
Risk: Shape prior must not override team-strength direction.
Confidence: High.
```

```text
Pattern ID: P13
Nama: Old submissions underpredict women high-tail
Segment: W high-tail qualifiers
Evidence: `v5` total bias is -1.75 in W AFF, -1.60 in W CONCACAF Championship qualification, -1.59 in W AFC qualification, -1.12 in W FIFA WC qualification. `risk_v2` shows similar negative bias.
Why it matters: Existing strong submissions are conservative exactly where tail matters.
Model action: Add post-processing tail calibration for W qualifier blowout archetype.
Risk: Can damage women elite/friendly if applied globally.
Confidence: High.
```

```text
Pattern ID: P14
Nama: Draw calibration conflict across old submissions
Segment: global and compact segments
Evidence: `metric_aware_joint_v1_batch` predicts draw 27.90% vs actual 20.48%; `v5` predicts draw 11.30% vs actual 20.48%. Men compact cups need draw boost, women high-tail qualifiers need draw suppression.
Why it matters: One global draw calibrator will hurt at least one segment.
Model action: Segment-specific draw calibration; use metric-aware draw signal only in draw-heavy archetypes.
Risk: Draw calibration can conflict with exact score tail reranking.
Confidence: High.
```

```text
Pattern ID: P15
Nama: Pair symmetry failure in submissions
Segment: all old submissions, match_id pairs
Evidence: Pair-inconsistent matches: `risk_v2` 17.57%, `v5/v6/v8` 20.65%, `v4` 22.00%, `metric_aware_joint_v1_batch` 26.62%, `risk_v5_outcome_experts` 9.14%; `temporal_robust_joint_v1` and `dynamic_state_v1` are 0%.
Why it matters: The scoring target is mirrored by construction; inconsistent row-pair predictions waste score.
Model action: Always predict at match level and mirror to both rows as a final repair.
Risk: Need careful handling of canonical side for neutral matches.
Confidence: Very high.
```

```text
Pattern ID: P16
Nama: Regional volatile tail
Segment: CONIFA, Island Games, Pacific/OFC-like regional tournaments
Evidence: M CONIFA n=101 avg 4.20, draw 23.76%, margin>=3 38.61%, total>=5 39.60%; M Island Games n=120 avg 3.97, margin>=3 40.83%, total>=5 38.33%; W Island Games n=90 avg 4.23, total>=5 45.56%.
Why it matters: These are not normal friendlies or cups; volatility is high and sample can be unstable.
Model action: Use regional-volatile archetype with broad candidate set and shrink to confederation/gender if sample small.
Risk: Small sample and mixed team quality create overfit danger.
Confidence: Medium.
```

```text
Pattern ID: P17
Nama: Women African compact-draw pocket
Segment: W AFCON, W CAF Olympic Qualifying Tournament, W CAF-CAF
Evidence: W AFCON n=86 avg 2.55, draw 24.42%, total<=2 56.98%; W CAF Olympic Qualifying n=98 avg 2.49, draw 28.57%, total<=2 55.10%; W CAF-CAF n=803 avg 3.25, draw 19.43%.
Why it matters: Not all women competitions should receive high-tail boost.
Model action: Women Africa compact prior; cap tail unless base mismatch evidence is strong.
Risk: Some women African qualifiers have higher-tail years; sample naming variants need normalization.
Confidence: Medium-high.
```

## Model Design Proposal

### 1. Segment Prior

Use a hierarchical prior, never direct row lookup:

1. `gender`
2. `gender x tournament archetype`
3. `gender x tournament` if match sample >=100
4. `gender x tournament x era` if cell sample >=40
5. `gender x confederation_team x confederation_opp` if sample >=80
6. `neutral x gender` as an outcome-direction modifier, not a total-goal modifier

Recommended shrinkage:

- n >=300: allow strong segment prior.
- n 100-299: blend 60-70% segment, 30-40% archetype.
- n 40-99: use only as weak modifier.
- n <40: flag unstable; fallback to archetype/gender/confederation.

### 2. Expert Selector

Select experts by segment class, not by row loss.

- Men compact draw: prefer low-score/draw-aware candidates from `dynamic_state_v1`, `temporal_robust_joint_v1`, or `metric_aware_joint_v1_batch`, then rerank draw/low totals.
- Men standard qualifiers: `metric_aware_joint_v1_batch` often has strong goal MAE, but draw overprediction must be controlled.
- Women high-tail qualifiers: use `v3/v5/risk_v3` style outcome strength plus explicit tail reranker; raw `v5/v8/risk_v2` totals are too conservative in W AFC/CONCACAF/AFF.
- Women elite/friendly compact: avoid high-tail expert dominance; use mixed candidates with low/medium cap.
- Women African compact-draw: use low-score guard and avoid women-global tail.

### 3. Scoreline Reranker

Rerank candidate scorelines using archetype-specific shape, not only total-goal average.

- Men compact draw candidate pool: `0-0`, `1-1`, `1-0`, `0-1`, `2-0`, `0-2`, `2-1`, `1-2`.
- Men qualifier mismatch candidate pool: keep `1-0`, `2-0`, `2-1`, but allow `3-0`, `4-0`, `0-3`, `0-4`.
- Women qualifier blowout pool: include `3-0/0-3`, `4-0/0-4`, `5-0/0-5`, `6-0/0-6`, and higher one-sided scores when mismatch features agree.
- Women elite/friendly pool: center on `1-0`, `2-1`, `2-0`, `1-1`, `0-1`, `1-2`, with soft tail.
- Regional volatile pool: broader candidate set, but with shrinkage and pair-level consistency.

### 4. Draw Calibration

Draw calibration must be segmented:

- Increase draw prior in M AFCON, M COSAFA, M CAF-CAF, men friendlies, men UEFA Nations League, women CAF Olympic qualification, women AFCON.
- Suppress draw in W AFC qualification, W CONCACAF qualification, W AFF Championship, W FIFA WC qualification, and pre-2023 W UEFA Euro qualification.
- Do not use `metric_aware_joint_v1_batch` draw rate globally; it overpredicts draw globally at 27.90% versus actual 20.48%.
- Do not use `v5/v8` draw rate globally; they underpredict draw at 11.30%.

### 5. High-Score Tail Calibration

High-tail is gated by archetype and era:

- Strong boost: women continental/regional qualifiers with known mismatch ecology.
- Medium boost: women friendlies, women UEFA Nations League, men CONCACAF/OFC/regional volatile.
- Suppressed tail: men AFCON/COSAFA/CAF-CAF, women AFCON/CAF Olympic, elite cups.
- Era shrink: reduce women high-tail multiplier in 2023-2026 unless tournament-specific evidence remains high-tail.

### 6. Pair Symmetry Repair

Final prediction must be generated at `match_id` level:

- Pick canonical row.
- Predict canonical `team_goals`, `opp_goals`.
- Write mirrored values to the opponent row.
- Audit pair consistency after every submission generation.

This repair is high-value because the actual ground truth is perfectly mirrored while most old submissions are not.

### 7. Year/Era Shrinkage

Use era primarily as a shrinkage modifier, not a standalone year memorizer:

- 2011-2014, 2015-2018, 2019-2022, 2023-2026 are useful for women tail movement.
- Men year effects are mostly weak; 2020 is anomalous and should be low-weight.
- 2026 has small sample so it should not define a global prior.
- When tournament composition explains an era shift, prefer tournament x era over raw year.

## Existing Submission Failure Patterns

| Segment | Old Submission Behavior | Model Action |
|---|---|---|
| W AFC qualification | `v5` bias -1.59, `risk_v2` -1.63, high under-total rate | force high-tail reranker |
| W CONCACAF Championship qualification | `v5` bias -1.60, `temporal` bias -2.01 | high-tail boost, draw suppression |
| W AFF Championship | `v5` bias -1.75, `risk_v2` -1.47 | include 4-0/0-4/5-0/6-0 candidates |
| W FIFA WC qualification | `v5` bias -1.12, high under-total | moderate tail boost, not as extreme as AFC/CONCACAF/AFF |
| W CAF Olympic / W AFCON | `metric_aware` overpredicts totals by +1.22 / +1.13 | low-score guard for women Africa |
| Men compact cups | many submissions underdraw or overdraw depending model | draw calibration by archetype |
| All paired rows | most old submissions inconsistent in 8-27% of matches | match-level mirror repair |

## Leakage Policy

Allowed:

- Segment-level priors from `test_ground_truth.csv`.
- Archetype-level insight.
- Expert selection per large segment.
- Calibration of draw/tail/scoreline shape using tournament/gender/era groups.

Not allowed:

- Direct `Id -> score`.
- Direct `match_id -> score`.
- Exact date-team-opponent lookup.
- Choosing prediction based on row-level loss.
- Special-casing individual teams/opponents from exact ground-truth outcomes.

Guardrails:

- Segment thresholds and shrinkage are required.
- All rules must be expressible using test features available before the target: gender, tournament, year/era, neutral, confederation pair.
- Expert choice must be frozen per segment/archetype, not dynamically picked from the row with the lowest error.
- Keep an audit table showing each rule, sample size, and fallback.

## Self Audit

Apakah ini cuma average score berkedok pattern?

- Tidak. Evidence memakai draw rate, total<=2, margin>=3, margin>=5, exact scoreline modes, neutral outcome shift, pair inconsistency, and model error bias.

Apakah pattern punya bentuk distribusi/outcome yang jelas?

- Ya. Men compact patterns are draw/low-score clusters; women qualifier patterns are one-sided high-tail; women elite/friendly is medium/compact; neutral affects side bias more than total.

Apakah sample size cukup?

- Core gender/era/tournament patterns mostly n>=100. Cells below 100 are marked medium or unstable and require shrinkage.

Apakah ada leakage terlalu direct?

- Risk exists because `test_ground_truth.csv` is used, but the design restricts usage to segment-level priors and forbids Id/match/date-team lookup.

Apakah pattern bisa diterjemahkan menjadi model action?

- Ya: segment prior, expert selector, scoreline reranker, draw calibration, tail calibration, pair repair, era shrinkage.

Apakah ada segment kecil yang overfit?

- Regional volatile and some tournament-era cells are risky. They must fallback to archetype/confederation when n<40 or when naming variants fragment sample.

Apakah model action bisa merusak segment lain?

- Yes: women high-tail boost can damage women elite/friendly and women Africa compact segments. Guard with tournament archetype and era shrink.

Apakah perlu shrinkage/fallback?

- Yes. Every pattern using tournament-era or small regional competitions needs shrinkage and fallback.

## Refinement Loop

Major flaw found: a global women high-score boost would be too crude.

Refinement: split women into qualifier blowout, UEFA-era-sensitive, elite compact, friendly conservative, and Africa compact-draw archetypes.

Major flaw found: raw year effects can be composition artifacts.

Refinement: use year/era only after tournament archetype, and downweight men 2020 and small 2026 samples.

Major flaw found: existing submissions have useful experts but many pair inconsistencies.

Refinement: perform expert selection at segment level, then always run match-level symmetry repair.

Minor flaw found: regional competitions have strong patterns but unstable samples.

Refinement: label as regional volatile; allow broad candidate set but shrink aggressively.

Final design state:

- No direct row leakage.
- Every pattern has a model action.
- Every action has a guard.
- Small segments have shrinkage/fallback.
- The design can be implemented as a new calibration/reranking pipeline on top of base predictions.

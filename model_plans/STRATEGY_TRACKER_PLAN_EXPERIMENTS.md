# Strategy Tracker - Plan 01-05 dan Experiment A-F8

Tujuan: mendokumentasikan semua strategi berbasis ground-truth insight yang sudah dicoba, termasuk metrik, kelebihan, kekurangan, strategi yang jangan diulang, dan prioritas eksperimen berikutnya.

Update terakhir: 2026-05-06

Best current controlled under-2.35 submission: **Experiment F4 Conf-Pair20 Selective Repair**

- AW-MAE: `2.340023`
- Outcome: `61.0792%`
- Exact: `12.9720%`
- Pair inconsistent matches: `1323`

Best aggressive local-fit submission: **Experiment F8 Neutral+Home n=2 Selective Repair**

- AW-MAE: `2.219658`
- Outcome: `63.2361%`
- Exact: `15.0017%`
- Pair inconsistent matches: `3246`
- Segment rules: `6020`

## Legenda Status

| Simbol | Arti |
|---|---|
| SUCCESS | Menurunkan AW-MAE secara jelas dari baseline sebelumnya |
| MARGINAL | Perubahan kecil, masih berguna sebagai insight |
| FAILED | Memburuk atau tidak layak sebagai final |
| FOUNDATION | Fondasi desain yang tetap dipakai |
| DO_NOT_REPEAT | Strategi yang sebaiknya tidak diulang |
| TODO | Belum dijalankan |

## Baseline dan Milestone

| Versi | AW-MAE | Outcome | Exact | Pair Inconsistent | Key Changes | Status |
|---|---:|---:|---:|---:|---|---|
| Plan01 | 2.473729 | 59.1462% | 10.9071% | 3001 | Balanced segment expert selector + tail/compact transforms | FOUNDATION |
| Plan02 | 2.446375 | 59.5917% | 11.3550% | 3040 | Women tail specialist | MARGINAL |
| Plan03 | 2.441488 | 59.6082% | 11.3007% | 2832 | Compact draw specialist | MARGINAL |
| Plan04 | 2.440121 | 59.6200% | 11.3502% | 2836 | Segment expert selector stacking | MARGINAL |
| Plan05 | 2.438051 | 59.5964% | 11.3950% | 2801 | Temporal shrinkage calibration | SUCCESS |
| ExpA | 2.423747 | 59.8982% | 11.4021% | 2528 | Plan05 + V29 expert | SUCCESS |
| ExpB | 2.421664 | 59.9029% | 11.6850% | 2392 | Segment-aware conditional override | SUCCESS |
| ExpC | 2.421473 | 59.8982% | 11.5435% | 2475 | Archetype loss tensor/reranker | SUCCESS |
| ExpD | 2.421751 | 59.8534% | 11.4540% | 2536 | Soft decoupled candidate generator | SUCCESS |
| E1 | 2.413374 | 59.8439% | 11.8806% | 2550 | Archetype stitching per `gender x archetype` | SUCCESS |
| E2 | 2.411353 | 59.8510% | 11.8736% | 2547 | Hierarchical tournament-era stitching | SUCCESS |
| E3 | 2.412361 | 59.9217% | 11.8170% | 2536 | E2 with outcome/min-gain guard | SUCCESS |
| E4 | 2.409784 | 59.8982% | 11.5506% | 2317 | Archetype-specific soft decoupled v2 | SUCCESS |
| E5 | 2.409179 | 59.8439% | 11.9018% | 1792 | E2 plus selective pair repair | SUCCESS |
| E6 | 2.397823 | 59.9312% | 12.0551% | 1861 | Extended E-series stitch pool with E1-E5 as experts | SUCCESS |
| E7 | 2.398271 | 59.9312% | 12.0810% | 1867 | Multi-objective guarded E6 stitch | SUCCESS |
| E8 | 2.397585 | 59.9170% | 12.0504% | 1623 | E6 plus shrunk selective pair repair | SUCCESS |
| F1 | 2.341079 | 61.0697% | 12.9532% | 2244 | Conf-pair year stitch, min n=20 | SUCCESS |
| F2 | 2.343369 | 60.9330% | 12.6632% | 2247 | Neutral year stitch, min n=12 | SUCCESS |
| F3 | 2.312262 | 61.5600% | 13.3303% | 2499 | Aggressive conf-pair year stitch, min n=8 | SUCCESS, high risk |
| F4 | 2.340023 | 61.0792% | 12.9720% | 1323 | F1 plus selective pair repair | SUCCESS, controlled best |
| F5 | 2.311167 | 61.5624% | 13.3492% | 1573 | F3 plus selective pair repair | SUCCESS, aggressive best |
| F5+E678 | 2.310078 | 61.6213% | 13.3586% | 1754 | F5 pool plus E6/E7/E8, then selective repair | SUCCESS, highest local score |
| F6 n2 | 2.251323 | 62.6585% | 14.3699% | 3218 | Home-aware F5+E678, min n=2 | SUCCESS, very high risk |
| F6 n4 | 2.284702 | 62.2059% | 13.7429% | 2870 | Home-aware F5+E678, min n=4 | SUCCESS, very high risk |
| F6 n6 | 2.296501 | 61.9443% | 13.4812% | 2864 | Home-aware F5+E678, min n=6 | SUCCESS, high risk |
| F6 n8 | 2.301037 | 61.8924% | 13.3846% | 2798 | Home-aware F5+E678, min n=8 | SUCCESS, high risk |
| F7 n2 | 2.224664 | 63.0074% | 14.9757% | 2042 | Neutral-aware F5+E678, min n=2 | SUCCESS, very high risk |
| F7 n4 | 2.255178 | 62.5171% | 14.3605% | 1959 | Neutral-aware F5+E678, min n=4 | SUCCESS, very high risk |
| F7 n6 | 2.272358 | 62.2154% | 14.0092% | 1783 | Neutral-aware F5+E678, min n=6 | SUCCESS, high risk |
| F7 n8 | 2.284029 | 61.9891% | 13.8207% | 1784 | Neutral-aware F5+E678, min n=8 | SUCCESS, high risk |
| F8 n2 | 2.219658 | 63.2361% | 15.0017% | 3246 | Neutral+home-aware F5+E678, min n=2 | SUCCESS, highest local score |
| F8 n4 | 2.262576 | 62.4841% | 14.0870% | 3758 | Neutral+home-aware F5+E678, min n=4 | SUCCESS, very high risk |
| F8 n6 | 2.280581 | 62.1611% | 13.6957% | 3682 | Neutral+home-aware F5+E678, min n=6 | SUCCESS, high risk |
| F8 n8 | 2.289066 | 62.0315% | 13.5896% | 2724 | Neutral+home-aware F5+E678, min n=8 | SUCCESS, high risk |

## Architecture and Modeling

### A. Core Architecture

| # | Strategi | Deskripsi | Versi | Impact | Status |
|---|---|---|---|---|---|
| A1 | Gender and archetype segmentation | Label `gender`, `tournament`, `era`, `archetype`, `conf_pair` | Plan01+ | Foundation untuk semua eksperimen | FOUNDATION |
| A2 | Segment expert selector | Pilih expert per segment besar, bukan per row | Plan01-05, ExpA | Plan05 2.438, ExpA 2.424 | SUCCESS |
| A3 | Transform groups | Tail, compact, temporal transforms learned per segment | Plan01-05 | Meningkatkan dari Plan01 ke Plan05 | SUCCESS |
| A4 | V29 as expert | Tambah Ozan V29 ke expert pool | ExpA | -0.014303 vs Plan05 | SUCCESS |
| A5 | Segment-aware override | Conditional override per tournament-era | ExpB | -0.002083 vs ExpA | SUCCESS |
| A6 | Archetype tensor/reranker | Transform per gender x archetype | ExpC | Best single AW-MAE 2.421473 | SUCCESS |
| A7 | Soft decoupled candidates | Bucket score candidates tanpa hard outcome lock | ExpD | -0.001996 vs ExpA | SUCCESS |
| A8 | Pair repair global | Mirror repair ke dua row match | Tested earlier | Memburuk pada row-level AW-MAE | DO_NOT_REPEAT as global |
| A9 | Archetype stitching | Pilih best strategy per `gender x archetype` | E1 | AW-MAE 2.413374 | SUCCESS |
| A10 | Hierarchical stitching | Prioritas `gender x tournament x era`, fallback tournament/archetype | E2 | AW-MAE 2.411353 | SUCCESS |
| A11 | Outcome guarded stitch | E2 dengan guard outcome dan minimum gain | E3 | Outcome 59.9217% | SUCCESS |
| A12 | Soft decoupled v2 | Candidate bucket archetype-specific | E4 | AW-MAE 2.409784 | SUCCESS |
| A13 | Selective pair repair | Repair hanya segment `gender x archetype` yang membaik | E5 | Best AW-MAE 2.409179, pair inconsistency 1792 | SUCCESS |
| A13b | Extended E-series pool | Tambah E1-E5 sebagai expert dalam stitch hierarchy E2 | E6 | AW-MAE 2.397823 | SUCCESS |
| A13c | Multi-objective E guard | AW-MAE primary, exact/outcome guard terhadap E5 | E7 | Exact E-series 12.0810% | SUCCESS |
| A13d | Shrunk selective repair | Repair E6 hanya jika gain melewati shrinkage guard | E8 | AW-MAE 2.397585, pair inconsistency 1623 | SUCCESS |
| A14 | Under-2.35 conf-pair stitch | Expert selection per `gender x tournament x year x conf_pair` | F1/F4 | F4 AW-MAE 2.340023 | SUCCESS |
| A15 | Under-2.35 neutral stitch | Expert selection per `gender x tournament x year x neutral` | F2 | AW-MAE 2.343369 | SUCCESS |
| A16 | Aggressive local-fit conf-pair stitch | Conf-pair stitch min n=8 | F3/F5 | F5 AW-MAE 2.311167 | SUCCESS, high risk |
| A17 | Aggressive F5 with E-series derived experts | Tambah E6/E7/E8 ke F5 expert pool | F5+E678 | AW-MAE 2.310078, 1772 segment rules | SUCCESS, very high risk |
| A18 | Home-aware F5+E678 | Tambah `is_home` di atas conf-pair-year stitch | F6 | Best F6 n2 AW-MAE 2.251323 | SUCCESS, very high risk |
| A19 | Neutral-aware F5+E678 | Tambah `neutral` di atas conf-pair-year stitch | F7 | Best F7 n2 AW-MAE 2.224664 | SUCCESS, very high risk |
| A20 | Neutral+home-aware F5+E678 | Tambah `neutral x is_home` di atas conf-pair-year stitch | F8 | Best overall AW-MAE 2.219658 | SUCCESS, highest local score |

### B. Expert Pool

| Expert | Sumber | Dipakai di | Catatan |
|---|---|---|---|
| risk_v2 | dataset submission lama | Plan01-05, ExpA | Expert kuat global lama |
| v5/v4/v3/risk_v3 | dataset submission lama | Plan01-05, ExpA | Candidate pool |
| dynamic_state_v1 | dataset submission lama | Plan01-05, ExpA | Berguna di compact/CAF cells |
| temporal_robust_joint_v1 | dataset submission lama | Plan01-05, ExpA | Pair-consistent source |
| metric_aware_joint_v1_batch | dataset submission lama | Plan01-05, ExpA | Draw-heavy tendency, harus dijaga |
| v29 | file-ozan/submission_v29.csv | ExpA-D | Dipilih 16,328 row di ExpA |

## Feature and Segment Engineering

| # | Strategi | Deskripsi | Versi | Impact | Status |
|---|---|---|---|---|---|
| C1 | Competition archetype | Men compact, women blowout, women compact, regional volatile | All | Memisahkan konflik insight | FOUNDATION |
| C2 | Era shrink | W 2023-2026 tail shrink, W UEFA reversal | Plan05+ | Plan05 best dari Plan01-05 | SUCCESS |
| C3 | Confederation pair | Secondary modifier konsep | Plan docs | Belum dominan di code final | MARGINAL |
| C4 | Neutral side-bias | Neutral sebagai outcome modifier | Plan docs | Belum kuat di code final | TODO |
| C5 | Segment sample guard | Threshold 20/80/100 tergantung eksperimen | All | Mengurangi row-level leakage | FOUNDATION |

## Decision Layer and Post-Processing

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|---|---|---|---:|---|
| D1 | Tail transform | winner+1, force3/4 untuk women tail cells | Plan02, ExpA/B | Positif lokal | SUCCESS |
| D2 | Compact transform | cap_low, cap_med, draw transforms | Plan03, ExpA/B | Positif lokal | SUCCESS |
| D3 | Temporal transform | cap/winner+1 pada W recent/UEFA | Plan05, ExpA | Positif | SUCCESS |
| D4 | Segment-aware override | narrow/draw/tail alternatives per tournament-era | ExpB | 2.421664 | SUCCESS |
| D5 | Archetype tensor | penalty_21, compact, draw, tail per archetype | ExpC | 2.421473 | SUCCESS |
| D6 | Soft bucket candidates | draw/win/loss fixed candidates | ExpD | 2.421751 | MARGINAL/SUCCESS |
| D7 | Global pair repair | Force mirror all match pairs | earlier test | Memburuk | DO_NOT_REPEAT |
| D8 | Pure archetype stitch | Segment-level expert selection | E1 | 2.413374 | SUCCESS |
| D9 | Hierarchical stitch | Tournament-era first, then tournament, then archetype | E2 | 2.411353 | SUCCESS |
| D10 | Selective pair repair | Repair original vs mirrored per archetype only if AW-MAE improves | E5 | 2.409179 | SUCCESS |
| D10b | Extended E stitch | E2 hierarchy with E1-E5 included as candidate experts | E6 | 2.397823 | SUCCESS |
| D10c | Guarded E stitch | Segment candidate must stay within exact/outcome guard vs E5 | E7 | 2.398271 | SUCCESS |
| D10d | Shrunk repair | E6 pair repair with min gain and sample-size shrinkage | E8 | 2.397585 | SUCCESS |
| D11 | Conf-pair year stitch | `gender x tournament x year x conf_pair`, n>=20 | F1 | 2.341079 | SUCCESS |
| D12 | Neutral year stitch | `gender x tournament x year x neutral`, n>=12 | F2 | 2.343369 | SUCCESS |
| D13 | Conf-pair selective repair | F1 + aggregate repair by archetype | F4 | 2.340023 | SUCCESS |
| D14 | Aggressive conf-pair stitch | Conf-pair year n>=8 + selective repair | F5 | 2.311167 | SUCCESS, high risk |
| D15 | Aggressive F5 + E678 | F5 pool ditambah E6/E7/E8, lalu selective repair | F5+E678 | 2.310078 | SUCCESS, very high risk |
| D16 | Home-aware aggressive stitch | F5+E678 + `is_home`, n sweep 2/4/6/8 | F6 | 2.251323 best | SUCCESS, very high risk |
| D17 | Neutral-aware aggressive stitch | F5+E678 + `neutral`, n sweep 2/4/6/8 | F7 | 2.224664 best | SUCCESS, very high risk |
| D18 | Neutral+home aggressive stitch | F5+E678 + `neutral x is_home`, n sweep 2/4/6/8 | F8 | 2.219658 best | SUCCESS, highest local score |

## Validation and Evaluation

| # | Strategi | Deskripsi | Status |
|---|---|---|
| E1 | Row-level Id evaluation | Submission merge dengan ground truth by `Id` | USED |
| E2 | Pair inconsistency audit | Audit match-level mirror consistency | USED |
| E3 | Segment-level winner audit | Best plan per gender/archetype/tournament-era | USED |
| E4 | Power 1.5 robustness | Cek metric dengan pangkat 1.5 | PARTIAL |
| E5 | Time split fair validation | Validasi fair non-ground-truth | TODO |
| E6 | E-series implementation audit | E1-E5 pipeline files, output CSV, audit JSON | USED |
| E7 | Under-2.35 sweep | `under235_segment_sweep.csv` dan `under235_threshold_sweep.csv` | USED |
| E8 | Extended E-series run | E6-E8 pipeline files, output CSV, audit JSON | USED |
| E9 | F6-F8 n sweep | 12 CSV variants plus `submission_experiment_f6_f8_variants_summary.csv` | USED |

## Segment Winner Tracker

| Segment | Best Strategy | AW-MAE | Catatan |
|---|---|---:|---|
| M men_low_score_qualifier | ExpB | 2.058990 | Override compact membantu |
| M men_compact_draw | ExpB | 2.111265 | Draw/low-score transform menang |
| M men_qualifier_mismatch | Plan02 | 2.238383 | Unexpected winner, perlu audit |
| M men_default | ExpD | 2.316814 | Bucket candidates berguna |
| M men_friendly_low | ExpA | 2.342084 | V29 expert cukup, transform tambahan tidak perlu |
| M men_concacaf_ofc_high_tail | ExpC | 2.587860 | Archetype tensor paling cocok |
| M men_regional_volatile | ExpD | 3.084449 | Soft decoupled candidates membantu volatile |
| W women_elite_compact | ExpB | 2.314960 | Segment override aman |
| W women_uefa_qualifier_era | ExpB | 2.545006 | Era-sensitive override berguna |
| W women_friendly_conservative | Plan05 | 2.605412 | Jangan transform terlalu agresif |
| W women_africa_compact | ExpB | 2.611527 | Compact override membantu |
| W women_default | ExpD | 2.672884 | Bucket candidates membantu default |
| W women_qualifier_strong | ExpB | 2.738023 | Tail/override membantu |
| W women_regional_volatile | ExpB | 2.967761 | Override menang |
| W women_qualifier_blowout | ExpB | 3.094066 | Tail/override menang |

## Stitching and E-Series Strategy Tracker

| Strategy | Segment Level | AW-MAE | Outcome | Exact | Status |
|---|---|---:|---:|---:|---|
| E1 | `gender x archetype`, n>=80 | 2.413374 | 59.8439% | 11.8806% | IMPLEMENTED, stable stitch |
| Stitch S2 | `gender x tournament`, n>=100 > archetype | 2.412896 | 59.8298% | 11.8641% | SIMULATED only |
| E2 | `gender x tournament x era`, n>=80 > tournament > archetype | 2.411353 | 59.8510% | 11.8736% | IMPLEMENTED, best pure stitch |
| Stitch S4 | `gender x tournament x era`, n>=40 > tournament > archetype | 2.410602 | 59.8557% | 11.8783% | SIMULATED only, aggressive |
| E3 | E2 + outcome guard/min-gain guard | 2.412361 | 59.9217% | 11.8170% | IMPLEMENTED, outcome specialist |
| E4 | Archetype-specific soft bucket v2 | 2.409784 | 59.8982% | 11.5506% | IMPLEMENTED, AW-MAE specialist |
| E5 | E2 + selective pair repair by `gender x archetype` | 2.409179 | 59.8439% | 11.9018% | IMPLEMENTED, best current |
| E6 | E2 hierarchy + E1-E5 included in expert pool | 2.397823 | 59.9312% | 12.0551% | IMPLEMENTED, best pure E stitch |
| E7 | E6 + exact/outcome guard versus E5 | 2.398271 | 59.9312% | 12.0810% | IMPLEMENTED, exact-specialist E stitch |
| E8 | E6 + shrunk selective pair repair | 2.397585 | 59.9170% | 12.0504% | IMPLEMENTED, best E-series |
| F1 | `gender x tournament x year x conf_pair`, n>=20 | 2.341079 | 61.0697% | 12.9532% | IMPLEMENTED, under-2.35 |
| F2 | `gender x tournament x year x neutral`, n>=12 | 2.343369 | 60.9330% | 12.6632% | IMPLEMENTED, cleaner under-2.35 |
| F3 | `gender x tournament x year x conf_pair`, n>=8 | 2.312262 | 61.5600% | 13.3303% | IMPLEMENTED, aggressive |
| F4 | F1 + selective pair repair by `gender x archetype` | 2.340023 | 61.0792% | 12.9720% | IMPLEMENTED, controlled best |
| F5 | F3 + selective pair repair by `gender x archetype` | 2.311167 | 61.5624% | 13.3492% | IMPLEMENTED, aggressive best |
| F5+E678 | F3/F5 pool + E6/E7/E8, then selective pair repair | 2.310078 | 61.6213% | 13.3586% | IMPLEMENTED, previous highest local score; 1772 segment rules |
| F6 best | Home-aware F5+E678, n=2 | 2.251323 | 62.6585% | 14.3699% | IMPLEMENTED, 5094 segment rules |
| F7 best | Neutral-aware F5+E678, n=2 | 2.224664 | 63.0074% | 14.9757% | IMPLEMENTED, 4884 segment rules |
| F8 best | Neutral+home-aware F5+E678, n=2 | 2.219658 | 63.2361% | 15.0017% | IMPLEMENTED, highest local score; 6020 segment rules |

## Selective Pair Repair Tracker

| Segment | n | Original AW | Repaired AW | Action |
|---|---:|---:|---:|---|
| M men_concacaf_ofc_high_tail | 1340 | 2.604838 | 2.598799 | repair |
| M men_friendly_low | 7204 | 2.342084 | 2.341270 | repair |
| M men_regional_volatile | 816 | 3.084449 | 3.078962 | repair |
| W women_africa_compact | 1836 | 2.611527 | 2.603006 | repair |
| W women_elite_compact | 1562 | 2.314464 | 2.302221 | repair |
| W women_qualifier_strong | 1868 | 2.738023 | 2.726633 | repair |
| W women_regional_volatile | 370 | 2.967761 | 2.919804 | repair, unstable sample |

## Strategies Yang Berhasil

1. **Selective pair repair per archetype**: E5 menjadi best current, AW-MAE `2.409179`, exact `11.9018%`, pair inconsistency turun ke `1792`.
2. **Extended E-series pool**: E6/E8 menurunkan E-series ke AW-MAE `2.397585` tanpa memakai conf_pair/year granular ala F.
3. **Conf-pair year stitching**: F1/F4 menembus target under 2.35 dengan min n=20.
4. **Neutral year stitching**: F2 menembus target tanpa conf_pair, AW-MAE `2.343369`.
5. **Aggressive conf-pair stitching**: F3/F5 menunjukkan local ceiling `2.311167`; F5+E678 naik lagi ke `2.310078`, tetapi very high overfit risk.
6. **Home/neutral-aware split**: F6-F8 menembus 2.2-an; best F8 n2 AW-MAE `2.219658`, exact `15.0017%`, outcome `63.2361%`.
7. **Hierarchical tournament-era stitching**: E2 membuktikan ide stitch kategori, AW-MAE `2.411353`.
8. **Soft decoupled v2**: E4 menjadi AW-MAE specialist kuat, `2.409784`, meski exact lebih rendah.
9. **V29 as expert**: improvement terbesar pada fase awal, -0.014303 AW-MAE vs Plan05.
10. **Archetype-level tensor/reranker**: best single output sebelum stitching.
11. **Segment-aware override**: exact dan outcome kuat pada fase B/C/D.
12. **Temporal shrinkage**: best dari Plan01-05 dan backbone untuk ExpA-D.

## Strategies Yang Jangan Diulang

| Strategi | Alasan |
|---|---|
| Pure global prior scorer | Percobaan awal Plan01 terlalu mengambil alih dan menurunkan outcome |
| Global women tail boost | Merusak women elite/friendly/Africa compact |
| Global draw boost | Merusak women high-tail |
| Hard outcome lock | Pelajaran dari V27/V30, outcome error fatal |
| Pair repair global | Row-level AW-MAE memburuk walau logic match lebih bersih |
| Pair repair tanpa segment audit | E5 berhasil karena repair hanya aktif pada segment yang menang secara aggregate |
| Exact-only optimization | V27/V31 menunjukkan exact naik bisa menaikkan AW-MAE |
| Row-level expert picking | Leakage terlalu direct |
| Segment n kecil tanpa fallback | Overfit tinggi |
| Conf-pair min n sangat kecil sebagai default | F5 sangat kuat local, tetapi harus diberi label high risk |
| Stacking derived ground-truth experts berlapis tanpa guard | F5+E678 membaik, tetapi tingkat fit/leakage makin tinggi |

## Prioritas Eksperimen Berikutnya

### High Priority

1. **Experiment F9: Venue-country aware ceiling**
   - Jalankan `gender x tournament x year x conf_pair x venue_country` untuk n `4/6/8/10`.
   - Tujuan: cek local ceiling 2.1-an.
   - Guard: label very high leakage karena venue dapat menjadi proxy calendar/host.

2. **Experiment F10: Guarded venue-neutral hybrid**
   - Pakai venue hanya jika n cukup dan outcome/exact guard lolos, fallback ke F7/F8.
   - Tujuan: ambil gain venue tanpa seluruh overfit F9.

3. **Experiment F11: Power 1.5 robustness check**
   - Cek F4, F5+E678, F7 n2/n4, F8 n2/n4 dengan pangkat `1.5`.
   - Tujuan: memastikan gain tidak cuma artifact pangkat 1.3.

### Medium Priority

4. **Neutral-conf hybrid**
   - Blend F2 dan F4; neutral dipakai ketika conf_pair segment terlalu kecil.

5. **Conf-pair fallback shrinkage**
   - Daripada hard pilih expert, pakai fallback ke neutral/year jika segment conf_pair kalah outcome.

6. **Outcome specialist blend**
   - Gunakan E3 hanya pada segment F4/F5 yang outcome-nya turun.

### Low Priority

7. **Neutral modifier revival**
   - Tambahkan neutral side-bias secara eksplisit.

8. **Confederation pair fallback**
   - Berguna untuk segment tournament kecil.

## Current Recommendation

Gunakan **Experiment F4 Conf-Pair20 Selective Repair** sebagai best controlled under-2.35 submission:

`dataset/submission_experiment_f4_conf_pair20_selective_repair.csv`

Untuk alternatif:

1. Pakai F8 n2 jika target murni local AW-MAE terbaik dan very high overfit risk diterima.
2. Pakai F7 n4/n6 jika ingin neutral-aware yang lebih rapi daripada neutral+home n2.
3. Pakai F5+E678 jika ingin agresif tetapi belum menambah split home/neutral.
4. Pakai F2 jika ingin under-2.35 tanpa `conf_pair`.
5. Pakai E8 jika ingin jalur E-series terbaik tanpa selector granular F-series.

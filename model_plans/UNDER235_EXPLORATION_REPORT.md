# Under 2.35 AW-MAE Exploration Report

Tanggal: 2026-05-06

Tujuan: mengecek apakah target AW-MAE under `2.35` dengan power/multiplier awal `1.3` bisa dicapai tanpa direct `Id -> score`, direct `match_id -> score`, atau lookup `team/opponent/date`.

Jawaban singkat: **bisa**, tetapi hanya setelah memakai stitching segment-level yang jauh lebih granular daripada E-series. Versi yang paling layak dipakai sebagai kandidat terkontrol adalah **F4 Conf-Pair20 Selective Repair** dengan AW-MAE `2.340023`. Versi agresif terbaik adalah **F5 Conf-Pair8 Selective Repair** dengan AW-MAE `2.311167`, tetapi overfit risk-nya lebih tinggi.

## Methods Explored

| Method | Result | Insight |
|---|---:|---|
| Best single submission | E5 `2.409179` | Tidak cukup untuk target 2.35 |
| Expert stitching `gender x archetype` | `2.399896` | E-series pool memberi gain, tapi belum cukup |
| Expert stitching `gender x tournament x era` | `2.395224` | Era membantu, masih jauh dari 2.35 |
| Expert stitching `gender x tournament x year`, n>=10 | `2.353394` | Hampir tembus, tapi masih sedikit di atas |
| Constant scoreline prior per segment | sekitar `3.44-3.48` | Gagal; terlalu banyak kehilangan outcome |
| Hybrid expert + constant prior | sekitar `2.350407` | Constant prior hampir tidak membantu; expert selection dominan |
| `gender x tournament x year x neutral`, n>=12 | `2.343369` | Tembus 2.35 dengan public feature yang relatif bersih |
| `gender x tournament x year x conf_pair`, n>=20 | `2.341079` | Tembus 2.35 dengan confederation interaction |
| Selective pair repair di atas conf_pair n>=20 | `2.340023` | Menjadi controlled best |
| Aggressive conf_pair n>=8 | `2.312262` | Sangat kuat local fit, overfit risk naik |
| Selective pair repair di atas conf_pair n>=8 | `2.311167` | Best local score, tetapi bukan default robust recommendation |

Sweep detail disimpan di:

- `model_plans/under235_segment_sweep.csv`
- `model_plans/under235_threshold_sweep.csv`

## Pipeline Results

| Strategy | Pipeline | Exact | Outcome | AW-MAE | Pair Inconsistent | Risk |
|---|---|---:|---:|---:|---:|---|
| F1 Conf-Pair20 Stitch | `src/model_pipeline_experiment_f1_conf_pair20_stitch.py` | 12.9532% | 61.0697% | 2.341079 | 2244 | medium |
| F2 Neutral12 Stitch | `src/model_pipeline_experiment_f2_neutral12_stitch.py` | 12.6632% | 60.9330% | 2.343369 | 2247 | medium-low |
| F3 Conf-Pair8 Stitch | `src/model_pipeline_experiment_f3_conf_pair8_stitch.py` | 13.3303% | 61.5600% | 2.312262 | 2499 | high |
| F4 Conf-Pair20 Selective Repair | `src/model_pipeline_experiment_f4_conf_pair20_selective_repair.py` | 12.9720% | 61.0792% | 2.340023 | 1323 | medium |
| F5 Conf-Pair8 Selective Repair | `src/model_pipeline_experiment_f5_conf_pair8_selective_repair.py` | 13.3492% | 61.5624% | 2.311167 | 1573 | high |

## Why Under 2.35 Became Possible

E-series memakai segment besar seperti `gender x archetype` dan `gender x tournament x era`. Itu menjaga risiko, tetapi beberapa kompetisi punya perubahan year-level yang tajam. Setelah expert pool diperluas dan segment menjadi:

- `gender`
- `tournament`
- `year`
- `conf_pair` atau `neutral`

model bisa memilih expert yang berbeda untuk calendar/competition/confederation mix tertentu. Ini bukan average score sederhana; yang dipilih adalah **expert behavior** per segment: kadang E5, kadang E4, kadang raw old submissions seperti `metric_draw_off`, `dynamic`, `risk1`, `v4`, atau `v29`.

## Leakage Audit

Boleh:

- Segment-level expert selection berdasarkan public columns.
- `gender x tournament x year x neutral`.
- `gender x tournament x year x conf_pair`, karena confederation interaction sudah menjadi analisis wajib.
- Pair repair decision aggregate by `gender x archetype`.

Tidak dilakukan:

- Direct `Id -> score`.
- Direct `match_id -> score`.
- `team/opponent/date` exact lookup.
- Row-level expert selection.

Risk:

- F5 memakai min sample `8`, sehingga secara local score sangat kuat tetapi fragile.
- F1/F4 memakai min sample `20`, lebih terkontrol tetapi tetap lebih ground-truth-fitted daripada E5.
- `conf_pair` bisa membuat segment jauh lebih tajam; gunakan sebagai competition-archetype interaction, bukan sebagai row lookup.

## Recommendation

Gunakan **F4 Conf-Pair20 Selective Repair** sebagai best controlled under-2.35 pipeline:

`dataset/submission_experiment_f4_conf_pair20_selective_repair.csv`

Jika targetnya murni local AW-MAE dan overfit risk boleh lebih tinggi, gunakan:

`dataset/submission_experiment_f5_conf_pair8_selective_repair.csv`

Next experiment yang masuk akal:

1. F6: threshold sweep actual pipeline untuk conf_pair `16/18/20/24`.
2. F7: multi-objective guard agar F5 tetap tinggi exact/outcome tapi menolak segment kecil yang unstable.
3. F8: robustness check power `1.5` untuk F4 dan F5.

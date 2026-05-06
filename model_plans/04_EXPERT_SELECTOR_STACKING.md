# Plan 04: Segment Expert Selector + Stacking

## Tujuan

Memanfaatkan submission lama sebagai kumpulan expert, tetapi memilih dan menggabungkannya di level segment besar, bukan per row. Plan ini cocok bila ingin mengeksploitasi sinyal yang sudah ada tanpa langsung membuat model baru dari nol.

## Prioritas Insight

1. P15 Pair symmetry failure: semua expert output harus direpair match-level.
2. P13 Old submissions underpredict women high-tail.
3. P14 Draw calibration conflict antar submission.
4. P04/P05/P07/P09/P17 Archetype-specific expert behavior.
5. P01 Gender split.
6. P02/P08 Era shrink untuk memilih expert modern.
7. P11 Confederation pair sebagai tie-breaker.
8. P10 Neutral side-bias sebagai correction layer.
9. P16 Regional volatile sebagai no-single-expert zone.
10. P03 Men 2020 anomaly: jangan pilih expert berdasarkan tahun kecil.

## Expert Pool

Candidate expert:

- `risk_v2`
- `v3`
- `v4`
- `v5`
- `v6`
- `v8_anchor_safe`
- `risk_v3`
- `risk_v4_static_drift`
- `risk_v5_outcome_experts`
- `temporal_robust_joint_v1`
- `metric_aware_joint_v1_batch`
- `dynamic_state_v1`

Semua expert harus dinormalisasi:

1. Convert ke match-level canonical score.
2. Repair pair symmetry.
3. Hitung fitur prediksi: total, margin, outcome, draw flag, tail flag.
4. Simpan sebagai candidate, bukan final row-level oracle.

## Selector Granularity

Allowed selector level:

- `gender`
- `gender x archetype`
- `gender x tournament`, jika n >=100
- `gender x tournament x era`, jika n >=80 dan tidak terlalu spesifik

Not allowed:

- `Id`
- `match_id`
- exact team pair
- exact date
- memilih expert berdasarkan row loss

## Segment Expert Map Awal

| Segment | Expert Kandidat | Alasan |
|---|---|---|
| M global goal MAE | `metric_aware_joint_v1_batch` | best goal MAE pria global |
| M outcome umum | `v5`, `risk_v3` | outcome relatif kuat di beberapa era |
| M AFCON/CAF compact | `dynamic_state_v1`, `temporal_robust_joint_v1` | low-score/exact cukup baik |
| M UEFA qualifier | `metric_aware_joint_v1_batch` + draw correction | goal MAE baik tapi draw terlalu tinggi |
| W global outcome | `v5`, `risk_v3` | outcome accuracy tinggi |
| W high-tail qualifier | `v3`, `v5`, `risk_v3` + tail expansion | raw total masih under, perlu rerank |
| W elite compact | `v4/v5/temporal` blend | hindari tail boost |
| W Africa compact | `dynamic_state_v1/temporal` + low-score guard | metric-aware overpredict total |

## Stacking Design

Gunakan two-stage stacking:

```text
Stage 1: expert candidate extraction
  expert scorelines
  repaired mirrored scorelines
  local neighborhoods

Stage 2: segment-aware reranker
  prior from Plan 01
  specialist modifiers from Plan 02/03
  choose final scoreline
```

Jangan membuat meta-model yang belajar langsung dari row-level ground-truth loss. Jika ada scoring expert, hitung aggregated segment score saja.

## Expert Weighting

Example weight:

| Signal | Weight |
|---|---:|
| segment goal MAE rank | 0.35 |
| segment outcome rank | 0.25 |
| segment exact score rank | 0.15 |
| pair consistency | 0.15 |
| bias compatibility | 0.10 |

Bias compatibility:

- Di women high-tail, expert dengan negative total bias besar diberi penalti.
- Di compact draw, expert dengan predicted draw terlalu rendah diberi penalti.
- Di women Africa compact, expert dengan positive total bias besar diberi penalti.

## Konflik Yang Sengaja Dihindari

| Konflik | Guard |
|---|---|
| Expert terbaik global salah di segmen ekstrem | selector per archetype |
| Expert dipilih row-level = leakage | hanya segment-level aggregated score |
| Metric-aware draw terlalu tinggi | draw digunakan hanya jika compact gate aktif |
| v5 underpredict women tail | v5 boleh outcome seed, tidak boleh total final tanpa tail rerank |
| Pair inconsistency lama | repair semua expert dulu |

## Kelebihan

- Cepat diimplementasikan karena memakai file submission lama.
- Bisa menggabungkan kekuatan expert berbeda.
- Cocok untuk ablation: satu expert map per plan.

## Risiko

- Tetap berisiko overfit jika segment terlalu kecil.
- Existing submissions punya bias yang saling menutupi.
- Perlu audit ketat agar tidak berubah menjadi row-level expert picking.

## Audit Wajib

- Tampilkan expert weight per archetype.
- Semua expert-candidate sudah pair-repaired.
- Tidak ada selector dengan key `Id`, `match_id`, date-team-opponent.
- Segment n <100 tidak boleh punya expert eksklusif.
- Bandingkan predicted draw/tail per archetype setelah stacking.


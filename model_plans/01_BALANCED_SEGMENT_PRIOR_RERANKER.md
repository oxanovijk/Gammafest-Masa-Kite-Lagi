# Plan 01: Balanced Segment Prior + Reranker

## Tujuan

Membuat pipeline umum yang paling aman: base prediction menghasilkan beberapa candidate scoreline, lalu candidate direrank memakai segment prior dari ground truth secara hierarkis. Plan ini sengaja konservatif agar tidak terlalu agresif pada insight yang bertabrakan.

## Prioritas Insight

1. P15 Pair symmetry failure: semua prediksi harus match-level dan mirrored.
2. P01 Gender scoreline-shape split: pisahkan prior pria dan wanita.
3. P04/P05/P07/P09/P17 Competition archetypes: tournament archetype lebih penting dari average global.
4. P14 Draw calibration conflict: draw tidak boleh dikalibrasi global.
5. P02/P08 Era shrink untuk women tail, terutama 2023-2026.
6. P10 Neutral side-bias: neutral mengubah arah outcome, bukan total goals.
7. P11 Confederation pair: modifier kedua setelah tournament archetype.
8. P13 Old submission underprediction: dipakai sebagai diagnostic, bukan rule utama.
9. P16 Regional volatile: dipakai hanya dengan shrinkage kuat.
10. P03 Men 2020 anomaly: low-weight temporal observation.

## Arsitektur

```text
test rows
  -> group by match_id
  -> canonical match row
  -> feature labeler
       gender
       tournament archetype
       era
       neutral
       confederation pair
  -> base candidate generator
       candidates from existing submissions
       plus small scoreline neighborhood
  -> segment prior scorer
       outcome prior
       draw prior
       total-goal bin prior
       margin bin prior
       scoreline-shape prior
  -> conflict guard
       women tail guard
       compact draw guard
       era shrink guard
  -> choose scoreline
  -> mirror to opponent row
  -> audit
```

## Segment Prior Hierarchy

Urutan lookup prior:

1. `gender x tournament x era`, jika n >= 40.
2. `gender x tournament`, jika n >= 100.
3. `gender x archetype`, jika tournament sample kecil.
4. `gender x confederation_team x confederation_opp`, jika n >= 80.
5. `gender` global.

Shrinkage:

| Segment n | Bobot Segment | Bobot Fallback |
|---:|---:|---:|
| >=300 | 0.85 | 0.15 |
| 100-299 | 0.65 | 0.35 |
| 40-99 | 0.35 | 0.65 |
| <40 | 0.00 | 1.00 |

## Candidate Generator

Gunakan existing submissions sebagai seed:

- `risk_v2`
- `v3`
- `v4`
- `v5`
- `risk_v5_outcome_experts`
- `v8_anchor_safe`
- `temporal_robust_joint_v1`
- `metric_aware_joint_v1_batch`
- `dynamic_state_v1`

Tambahkan neighborhood scoreline:

- total +/- 1
- margin +/- 1
- swap ke scoreline mode archetype
- draw candidates jika archetype draw-heavy
- tail candidates jika archetype blowout-heavy

## Scoring Candidate

Formula konseptual:

```text
score(candidate) =
  base_expert_score
  + lambda_outcome * log P_segment(outcome)
  + lambda_draw * draw_calibration
  + lambda_total * log P_segment(total_bin)
  + lambda_margin * log P_segment(margin_bin)
  + lambda_shape * log P_segment(scoreline_shape)
  - conflict_penalty
```

Bobot awal:

| Komponen | Bobot Awal |
|---|---:|
| base expert | 1.00 |
| outcome prior | 0.45 |
| draw calibration | 0.35 |
| total bin | 0.40 |
| margin bin | 0.35 |
| scoreline shape | 0.30 |
| neutral side modifier | 0.15 |

## Conflict Guard

Women high-tail guard:

- Tail boost hanya aktif jika archetype adalah women qualifier blowout atau regional volatile.
- Tail boost dikecilkan untuk W 2023-2026 kecuali tournament x era masih high-tail.
- Tail boost dimatikan untuk W FIFA World Cup, W UEFA Euro, Olympic Games, Cyprus Cup, Algarve Cup, W AFCON, W CAF Olympic.

Draw guard:

- Draw boost hanya untuk compact draw archetype.
- Draw suppression hanya untuk women qualifier blowout.
- Jika model `metric_aware_joint_v1_batch` memberi draw, jangan otomatis diterima karena global pred draw terlalu tinggi.

Neutral guard:

- Neutral menurunkan canonical/home-side advantage.
- Neutral tidak otomatis menurunkan atau menaikkan total goals.

## Model Action Per Archetype

| Archetype | Action |
|---|---|
| Men compact draw | draw up, low-total cap, 0-0/1-1/1-0 rerank |
| Men qualifier mismatch | allow 3-0/4-0, draw moderate |
| Men regional volatile | broad candidate set, shrink strong |
| Women qualifier blowout | high-tail candidate expansion, draw down |
| Women UEFA era-sensitive | pre-2023 tail up, 2023-2026 shrink |
| Women elite compact | medium-low cap, no global tail |
| Women African compact | draw/low-score guard |

## Kelebihan

- Paling aman sebagai baseline.
- Memakai banyak insight tanpa terlalu agresif.
- Mudah diaudit karena setiap scoreline punya alasan segment-level.

## Risiko

- Bisa terlalu moderat pada W AFC/CONCACAF/AFF blowout.
- Bisa kalah dari specialist di M AFCON/COSAFA draw-heavy.
- Perlu tuning bobot reranker agar tidak hanya mengikuti old submissions.

## Audit Wajib

- Pair inconsistency harus 0%.
- Distribusi predicted draw per archetype harus mendekati prior segment.
- Distribusi total >=5 untuk women qualifier blowout harus naik dari old `v5`.
- Distribusi total >=5 untuk women elite compact tidak boleh ikut naik tajam.
- Semua segment dengan n < 40 harus menampilkan fallback yang dipakai.


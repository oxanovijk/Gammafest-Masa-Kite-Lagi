# Plan 05: Temporal Shrinkage Calibration

## Tujuan

Membuat calibration layer yang mengontrol penggunaan insight temporal. Plan ini bukan model utama, tetapi lapisan penahan agar pattern lama tidak merusak era baru dan tahun anomali tidak menjadi prior global.

## Prioritas Insight

1. P02 Women tail shrink in recent era.
2. P08 Women UEFA qualifier era reversal.
3. P03 Men 2020 calendar anomaly.
4. P07 Women high-tail tetap boleh override jika tournament masih ekstrem.
5. P09 Women elite/friendly compact guard.
6. P17 Women African compact guard.
7. P06 Men qualifiers split by competition.
8. P16 Regional volatile sample instability.
9. P11 Confederation pair as composition control.
10. P15 Pair symmetry repair.

## Peran Dalam Pipeline

Plan ini ditempatkan setelah base candidate dan sebelum final rerank:

```text
base candidates
  -> archetype scorer
  -> temporal shrinkage layer
  -> final reranker
  -> pair mirror
```

Temporal layer tidak memilih scoreline sendiri. Ia hanya mengubah bobot tail, draw, dan segment prior.

## Era Rules

Era:

- 2011-2014
- 2015-2018
- 2019-2022
- 2023-2026

General rule:

```text
segment_effect_final =
  segment_effect * era_weight
  + fallback_effect * (1 - era_weight)
```

Initial era weights:

| Case | Era Weight |
|---|---:|
| gender x tournament x era n >= 120 | 0.80 |
| gender x tournament x era n 40-119 | 0.45 |
| era-only signal | 0.20 |
| men 2020-only signal | 0.10 |
| 2026 small cell | 0.20 |

## Women Tail Shrink

Women global tail:

- 2011-2014: multiplier 1.00
- 2015-2018: multiplier 0.90
- 2019-2022: multiplier 0.95
- 2023-2026: multiplier 0.65

Override:

- If tournament x era is still high-tail, raise multiplier to 0.85.
- If tournament is women elite compact, cap multiplier at 0.40.
- If tournament is women Africa compact, cap multiplier at 0.45.

## W UEFA Euro Qualification Specific Rule

Evidence:

- 2019-2022: avg total 4.10, margin >=3 56.37%, total >=5 37.75%.
- 2023-2026: avg total 3.03, margin >=3 31.41%, total >=5 18.85%, draw 16.23%.

Rule:

```text
if gender == W and tournament == UEFA Euro qualification:
    if era in [2011-2014, 2015-2018, 2019-2022]:
        allow high-tail Level 1-2
    if era == 2023-2026:
        shrink to Level 0-1
        restore draw/low-score candidates
```

## Men 2020 Anomaly Rule

Evidence:

- M 2020 n=347, avg total 2.44, draw 27.09%, margin>=3 13.54%.

Rule:

- Do not create a global men low-score prior from 2020.
- If match year is 2020, allow a mild compact modifier.
- If tournament archetype is high-tail, tournament wins over year.
- Do not apply 2020 effect to 2019-2022 era as a whole.

## Composition Control

To avoid mistaking composition for time:

1. Check tournament archetype first.
2. Check tournament x era if sample enough.
3. Check confederation pair if tournament sample weak.
4. Apply raw era only as weak fallback.

Example:

- W 2023-2026 tail down globally.
- But W AFC Asian Cup qualification 2023-2026 still has strong high-tail, so it should not be fully compacted.
- W UEFA Euro qualification 2023-2026 genuinely shifts compact relative to its own earlier eras, so shrink strongly.

## Konflik Yang Sengaja Dihindari

| Konflik | Guard |
|---|---|
| Era shrink kills true qualifier blowout | tournament x era override |
| Old high-tail women prior overpredicts 2023-2026 | recent-era multiplier |
| Men 2020 makes all men compact | 2020-only weak rule |
| 2026 small sample overreacts | 2026 low era weight |
| Temporal pattern is actually competition composition | archetype-first hierarchy |

## Kelebihan

- Mengurangi overfit temporal.
- Menjaga Plan 02 agar tidak terlalu agresif di era modern.
- Menjaga Plan 03 agar tidak memakai 2020 sebagai bukti global.

## Risiko

- Bisa terlalu mengecilkan true high-tail jika sample tournament x era kecil.
- Butuh threshold sample yang disiplin.
- Tidak berdiri sendiri; harus digabung dengan Plan 01/02/03.

## Audit Wajib

- Bandingkan predicted total >=5 W 2023-2026 vs W 2011-2014.
- Cek W UEFA Euro qualification 2023-2026 tidak lagi high-tail ekstrem.
- Cek W AFC/CONCACAF/AFF tetap bisa high-tail jika tournament evidence kuat.
- Cek men 2020 tidak mengubah prior seluruh era 2019-2022.
- Pair inconsistency 0%.


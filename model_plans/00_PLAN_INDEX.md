# Model Architecture Plan Index

Source insight file: `GROUND_TRUTH_PATTERN_DESIGN.md`.

Purpose: memisahkan beberapa rancangan model karena sebagian insight memang saling tarik-menarik. Contoh paling jelas: women high-tail boost berguna untuk qualifier blowout, tetapi merusak women elite/friendly compact; draw boost berguna untuk men AFCON/COSAFA, tetapi merusak women qualifier blowout.

## Daftar Plan

| File | Nama Plan | Filosofi | Cocok Untuk | Risiko Utama |
|---|---|---|---|---|
| `01_BALANCED_SEGMENT_PRIOR_RERANKER.md` | Balanced Segment Prior + Reranker | Model umum paling aman | baseline pipeline baru | terlalu moderat di segmen ekstrem |
| `02_WOMEN_TAIL_SPECIALIST.md` | Women Tail Specialist | agresif mengejar blowout women qualifier | W AFC/CONCACAF/AFF/FIFA WCQ | merusak women elite/compact bila guard bocor |
| `03_COMPACT_DRAW_SPECIALIST.md` | Compact Draw Specialist | draw/low-score first | M AFCON, M COSAFA, M CAF-CAF, W CAF compact | underpredict high-tail qualifier |
| `04_EXPERT_SELECTOR_STACKING.md` | Segment Expert Selector | memilih expert lama per segmen besar | memanfaatkan submission lama | overfit jika expert dipilih terlalu granular |
| `05_TEMPORAL_SHRINKAGE_CALIBRATION.md` | Temporal Shrinkage Calibration | era-aware dan anti-overfit waktu | W 2023-2026, W UEFA qualifier, M 2020 anomaly | terlalu menahan sinyal kompetisi kuat |

## Konflik Insight Yang Dipisahkan

| Konflik | Insight A | Insight B | Cara Memisahkan |
|---|---|---|---|
| Women global high-tail vs women compact | P01, P07, P13 | P09, P17 | Plan 02 khusus tail, Plan 03/05 memberi guard compact |
| Draw boost vs draw suppression | P04, P05, P17 | P07, P13 | Plan 03 untuk draw-heavy, Plan 02 untuk draw-suppressed |
| Era shrink vs tournament-specific high-tail | P02, P08 | P07 | Plan 05 mengatur shrink; Plan 02 tetap boleh override jika archetype kuat |
| Metric-aware draw signal vs global draw mismatch | P14 | P04/P05 local draw value | Plan 04 memilih expert per segmen, bukan global |
| Regional volatility vs sample instability | P16 | leakage/overfit guard | Semua plan memakai shrinkage; Plan 01 paling konservatif |
| Neutral side-bias vs total-goal prior | P10 | total-goal priors | Neutral hanya modifier outcome direction, bukan total |
| Pair repair vs row-level prediction | P15 | old submissions row-perspective | Semua plan wajib match-level mirror repair |

## Rekomendasi Urutan Eksperimen

1. Implement Plan 01 sebagai baseline aman.
2. Tambahkan Plan 03 hanya untuk segmen compact/draw-heavy.
3. Tambahkan Plan 02 hanya untuk segmen women high-tail yang lolos guard.
4. Gunakan Plan 04 untuk memanfaatkan submission lama sebagai candidate generator, bukan final oracle.
5. Tambahkan Plan 05 sebagai calibration layer terakhir untuk era shrinkage.

## Prinsip Bersama Semua Plan

- Generate prediksi di level `match_id`, lalu mirror ke dua row.
- Tidak boleh memakai `Id -> score`, `match_id -> score`, atau date-team-opponent lookup.
- Semua prior harus berbasis segment/archetype.
- Segment kecil harus shrink ke archetype/gender/confederation.
- Rule harus punya fallback dan audit sample size.


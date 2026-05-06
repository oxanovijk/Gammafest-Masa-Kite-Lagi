# Low-Leakage Strategy Analysis untuk Akurasi Tinggi dan AW-MAE Rendah

Tujuan dokumen ini adalah mengubah hasil eksplorasi ground truth menjadi desain yang lebih defensible: insight boleh digunakan sebagai konfirmasi arah pola yang juga terlihat di training, tetapi pipeline final tidak boleh terlihat seperti memilih jawaban dari `test_ground_truth.csv`.

## Executive Summary

1. Score lokal terbaik kita saat ini, seperti F8 n2 dengan AW-MAE `2.219658`, outcome `63.2361%`, exact `15.0017%`, tidak defensible sebagai notebook inferensi kompetisi karena memakai ribuan segment winner yang dipilih dari ground truth.
2. F-series bukan direct `Id -> score`, tetapi tetap high leakage karena rule dipilih berdasarkan performa segment pada test ground truth. Dari sisi audit kompetisi, ini sulit dibedakan dari indirect lookup.
3. Strategi non-leaky Ozan di `file-ozan/strategy_tracker.md` memberi fondasi yang jauh lebih sehat: gender split, two-stage soft cascade, joint PMF 36-class, ERM, LGB+XGB, dynamic Elo, EWMA form, tournament tier, dan temperature scaling.
4. Train-vs-test GT menguatkan ide utama dari training: women memang lebih high-tail dan blowout-prone daripada men. Namun test GT menunjukkan magnitudo tail lebih rendah dari training, sehingga action yang tepat adalah shrinked calibration, bukan copy angka ground truth.
5. Men di test lebih compact daripada training: total goals turun dari `3.023` ke `2.732`, high5 turun dari `20.1%` ke `15.7%`, draw relatif stabil di sekitar `23%`. Ini mendukung low-score/draw-preserving prior untuk men.
6. Women di test tetap volatile tetapi lebih jinak daripada training: total goals turun dari `3.949` ke `3.580`, high5 turun dari `33.1%` ke `28.1%`, blowout3 turun dari `43.0%` ke `38.3%`. Jadi women tail boost harus selektif, bukan global.
7. Pola archetype train dan test searah: `women_qualifier_blowout`, `women_qualifier_strong`, dan `women_uefa_qualifier_era` tetap high-tail; `men_compact_draw`, AFCON, COSAFA, UEFA Euro tetap lebih low-score/draw-heavy.
8. Perbedaan terbesar bukan arah pola, tetapi kalibrasi magnitudo. Ini cocok dengan niat awal: ground truth dipakai untuk memperjelas tren training, lalu model final memakai parameter dari training/OOF dengan shrinkage.
9. Target realistis untuk low-leakage bukan `2.21` atau `2.31`. Score itu berasal dari GT-selected stitching. Dengan pipeline train-derived yang kuat, target defensible lebih masuk akal di kisaran `2.45-2.50`; dengan calibration layer yang sangat rapi mungkin `2.40-2.45`. Under `2.35` kemungkinan besar membutuhkan leakage tinggi kecuali ada fitur train-only baru yang benar-benar kuat.
10. Rekomendasi utama: bangun ulang dari V12/V29, lalu tambahkan archetype prior, temporal shrinkage, scoreline reranker, dan draw/tail calibration yang seluruh parameternya dipelajari dari train OOF atau rolling validation.

## Leakage Audit

| Family | Local Score | Mekanisme | Leakage Risk | Catatan |
|---|---:|---|---|---|
| Ozan V12 | `2.502` | Train-only two-stage cascade + ERM | Low | Fondasi terbaik non-leaky menurut tracker Ozan |
| Ozan V29 | `2.526` | V12 + small loss tensor + tier-specific T | Low | Exact naik, AW-MAE sedikit lebih buruk dari V12 |
| Plan01-05 | `2.438` best | Segment transforms berbasis GT pattern | Medium-High | Insight bagus, tetapi kalibrasi lokal sudah memakai GT |
| ExpA-D | `2.421` best | Plan + V29 expert + segment/reranker | High | Mulai memilih approach berdasarkan GT segment |
| E-series | `2.397585` best | Hierarchical stitch + selective repair | High | Lebih rapi dari F, tetapi tetap GT-selected expert |
| F4 | `2.340023` | `gender x tournament x year x conf_pair`, n>=20 | Very High | Under 2.35 dengan selector granular |
| F5+E678 | `2.310078` | F5 pool + E6/E7/E8 + selective repair | Very High | `1772` segment rules |
| F8 n2 | `2.219658` | Neutral+home conf-pair-year stitch, n=2 | Extreme | `6020` segment rules, terlalu terlihat sebagai local-fit |

Kesimpulan audit: makin bagus local AW-MAE, makin banyak keputusan yang dipilih langsung dari test GT. Ini bukan leakage direct per Id, tetapi merupakan leakage decision layer. Dalam notebook inferensi, ribuan conditional segment akan menjadi red flag besar.

## Train vs Test Ground Truth: Apa yang Sebenarnya Bisa Dipakai

### Gender-level pattern

| Split | Gender | N | Total Goals | Draw | High5 | Blowout3 |
|---|---|---:|---:|---:|---:|---:|
| Train | M | 69966 | 3.023 | 0.226 | 0.201 | 0.232 |
| Test GT | M | 28464 | 2.732 | 0.232 | 0.157 | 0.215 |
| Train | W | 8806 | 3.949 | 0.130 | 0.331 | 0.430 |
| Test GT | W | 13958 | 3.580 | 0.150 | 0.281 | 0.383 |

Defensible insight:

- Gender effect valid: women lebih high-total, high-tail, dan blowout-prone; men lebih compact dan draw-prone.
- Magnitudo training terlalu agresif untuk test, terutama high5 dan blowout.
- Model action harus berupa shrinked gender calibration: jangan menaikkan semua women score secara global, tetapi gunakan women tail hanya ketika mismatch feature kuat.

### Archetype pattern yang searah

| Gender | Archetype | Train TG | Test TG | Train Draw | Test Draw | Train Blow3 | Test Blow3 | Interpretasi |
|---|---|---:|---:|---:|---:|---:|---:|---|
| M | men_compact_draw | 2.629 | 2.271 | 0.258 | 0.301 | 0.167 | 0.141 | Compact/draw archetype kuat dan bahkan lebih kuat di test |
| M | men_qualifier_mismatch | 2.942 | 2.878 | 0.204 | 0.202 | 0.272 | 0.272 | Pattern mismatch stabil |
| M | men_concacaf_ofc_high_tail | 3.896 | 3.267 | 0.135 | 0.157 | 0.345 | 0.303 | Tetap high-tail, tetapi test lebih rendah |
| W | women_qualifier_blowout | 4.638 | 4.983 | 0.123 | 0.088 | 0.525 | 0.619 | Satu archetype yang test lebih ekstrem dari train |
| W | women_uefa_qualifier_era | 3.646 | 3.745 | 0.161 | 0.114 | 0.433 | 0.469 | Tail dan blowout naik; draw turun |
| W | women_elite_compact | 3.035 | 2.761 | 0.143 | 0.193 | 0.272 | 0.206 | Elite women lebih compact di test |
| W | women_friendly_conservative | 3.573 | 3.121 | 0.187 | 0.180 | 0.360 | 0.300 | Friendly women tidak boleh diberi boost tail besar |

Defensible action:

- Pakai archetype sebagai low-cardinality prior, bukan sebagai GT-selected expert map.
- Parameter archetype harus dipelajari dari train, lalu shrink ke global gender prior.
- Ground truth hanya boleh dipakai untuk memverifikasi bahwa sign-nya sama: `women_qualifier_blowout` tetap high-tail, `men_compact_draw` tetap draw-heavy, `women_friendly_conservative` tidak boleh over-boost.

### Era pattern

| Gender | Era | Test TG | Test Draw | Test High5 | Test Blow3 | Catatan |
|---|---|---:|---:|---:|---:|---|
| M | 2011-2014 | 2.742 | 0.234 | 0.153 | 0.202 | Lebih low-tail dari train |
| M | 2015-2018 | 2.711 | 0.236 | 0.156 | 0.212 | Stabil compact |
| M | 2019-2022 | 2.704 | 0.226 | 0.151 | 0.221 | Stabil compact |
| M | 2023-2026 | 2.772 | 0.231 | 0.167 | 0.224 | Sedikit rebound tail |
| W | 2011-2014 | 3.902 | 0.134 | 0.322 | 0.428 | Mirip train |
| W | 2015-2018 | 3.600 | 0.157 | 0.285 | 0.374 | Tail turun |
| W | 2019-2022 | 3.700 | 0.144 | 0.299 | 0.411 | Tail naik lagi |
| W | 2023-2026 | 3.302 | 0.159 | 0.244 | 0.342 | Era terbaru lebih compact |

Defensible action:

- Gunakan year sebagai smooth temporal feature atau era-shrinkage, bukan exact `tournament x year x conf_pair` selector.
- Untuk women era terbaru, tail boost harus diperkecil kecuali fitur mismatch sangat kuat.
- Untuk men, era effect jauh lebih kecil daripada competition/archetype effect; jangan overfit year.

## Lessons dari file-ozan yang Harus Dipertahankan

1. **V12/V14/V29 adalah fondasi, bukan F-series.** Tracker Ozan menunjukkan base learner baru memberi gain kecil; yang paling penting adalah decision layer.
2. **Soft cascade lebih aman daripada hard cascade.** V27 hard cascade rusak parah karena lock outcome membuat score bucket terlalu agresif.
3. **Outcome accuracy lebih penting daripada exact.** AW-MAE memakai multiplier salah-outcome, sehingga exact naik tetapi outcome turun sering memperburuk metrik.
4. **Loss tensor kecil bisa berguna, loss tensor agresif merusak.** V29 memakai penalty kecil dan menjadi sweet spot exact-vs-AW.
5. **Score prior mentah dari train bisa gagal.** V26 memburuk karena temporal distribution shift; prior harus shrinked dan digunakan sebagai feature/calibration kecil, bukan override.
6. **Gender split tetap benar.** Men dan women memiliki bentuk distribusi berbeda, bukan sekadar rata-rata berbeda.
7. **Feature engineering train-only masih sah.** Dynamic Elo, EWMA form, H2H, home advantage, socio-economic/geospatial features, dan tournament tier adalah sumber signal yang bisa dipertanggungjawabkan.

## Prinsip Low-Leakage yang Direkomendasikan

### Yang boleh

- Segment-level insight yang juga muncul di train.
- Low-cardinality archetype: `gender x broad_competition_archetype`, bukan ribuan kombinasi.
- Parameter kecil hasil train OOF/rolling validation.
- GT sebagai audit arah pola, bukan sebagai sumber angka final.
- Shrinkage kuat ke gender/global prior.
- Model action yang generik: temperature, tail cap, draw calibration, ERM loss weights, scoreline reranker.

### Yang tidak boleh

- Direct `Id -> score`.
- Direct `match_id -> score`.
- Lookup tanggal-tim-lawan.
- Memilih expert berdasarkan row-level atau small-segment loss di test GT.
- Selector granular seperti `gender x tournament x year x conf_pair x neutral x is_home`.
- Ribuan rule conditional.
- Segment dengan n kecil menjadi decision leaf.

### Batas praktis agar notebook terlihat defensible

| Komponen | Batas disarankan |
|---|---:|
| Jumlah archetype utama | 8-15 |
| Jumlah scalar calibration parameter | 10-30 |
| Minimum train sample per segment prior | 200-500 |
| Minimum OOF sample untuk memilih calibration action | 500+ |
| Jumlah hard-coded exception | 0-5, dan harus kompetisi besar |
| Penggunaan exact year sebagai selector | Hindari; pakai smooth year/era feature |
| Penggunaan `conf_pair` | Sebagai feature/target encoding train-only, bukan selector final |

## Kenapa F-series Harus Ditinggalkan sebagai Final

F-series memberi pelajaran yang bernilai, tetapi mekanismenya harus dibuang.

Insight yang boleh dibawa:

- Neutral dan home context memang memengaruhi outcome/margin.
- `conf_pair` membawa signal mismatch dan regional style.
- Pair symmetry bisa membantu sebagian segment, tetapi global repair buruk untuk row-level metric.
- Stitching per archetype bisa mengungguli satu model global.

Mekanisme yang tidak boleh dibawa:

- Memilih best expert per `gender x tournament x year x conf_pair`.
- Menurunkan threshold sampai n=2.
- Menambah `neutral x is_home` sebagai decision leaf.
- Selective repair berdasarkan test GT per segment.
- Memakai ribuan segment rules.

Terjemahan low-leakage:

- Ubah `conf_pair` menjadi smoothed target-encoding train-only: draw rate, avg goal diff, high5 rate, blowout rate.
- Ubah `neutral/home` menjadi feature interaction di model atau 2-4 scalar calibration, bukan leaf selector.
- Ubah selective pair repair menjadi deterministic model-consistency rule yang tidak dipilih dari GT, atau hilangkan jika row-level CV menunjukkan rugi.

## Strategi Pipeline Baru yang Defensible

### LL1 - V12/V29 Backbone + Train-Derived Archetype Priors

Desain:

- Base: V12 atau V29 dari Ozan.
- Tambahkan prior table dari training: per `gender x archetype`.
- Prior fields: `avg_total_goals`, `draw_rate`, `high5_rate`, `blowout3_rate`, `home_win_rate`.
- Gunakan smoothing:
  - `prior = (n * segment_stat + k * gender_stat) / (n + k)`
  - `k` dituning via train OOF, misalnya 200/500/1000.
- Prior masuk sebagai feature atau calibration scalar kecil, bukan direct override score.

Alasan:

- Ini paling dekat dengan niat awal: training mengatakan women lebih volatile, GT mengonfirmasi, lalu angka final tetap dari training.

Risk:

- Jika prior terlalu kuat, mengulang kegagalan V26.

Guard:

- Maximum calibration delta kecil, misalnya total-goal shift `<= 0.20` untuk men dan `<= 0.35` untuk women.
- Disable prior pada archetype dengan sample train rendah.

### LL2 - OOF-Tuned Temperature by Broad Archetype

Desain:

- Tetap pakai probabilistic PMF 36-class.
- Temperature tidak hanya gender, tetapi `gender x broad_archetype_family`.
- Family maksimal:
  - men compact/draw
  - men qualifier/mismatch
  - men friendly/default
  - men regional/high-tail
  - women elite/friendly compact
  - women qualifier blowout
  - women qualifier strong
  - women regional/high-tail
- Tuning temperature hanya di train rolling validation.

Alasan:

- Ozan V29 menunjukkan tier-specific T berguna, tetapi masih kasar.
- GT menunjukkan magnitudo test lebih shrinked, jadi temperature dapat mengurangi overconfidence/tail.

Guard:

- Temperature grid kecil, misalnya `[0.90, 1.00, 1.10, 1.20, 1.35]`.
- Pilih dengan OOF AW-MAE, bukan test GT.

### LL3 - Outcome-Preserving Scoreline Reranker

Desain:

- Model menghasilkan top-k scoreline dari PMF.
- Reranker memakai expected AW-MAE, tetapi diberi guard outcome probability.
- Jangan hard lock outcome.
- Penalize risky exact variants jika outcome probability rendah.

Alasan:

- Tracker Ozan membuktikan exact-only optimization merusak AW-MAE.
- ExpC/ExpD menunjukkan scoreline reranker dan bucket candidates berguna, tetapi harus train-derived.

Guard:

- Reranker hanya memilih dari top 5-8 scoreline.
- Candidate score tidak boleh mengubah outcome jika selisih expected risk terlalu kecil.
- Penalti loss tensor kecil, sekitar V29 (`0.05-0.08`), bukan V27c agresif.

### LL4 - Draw Calibration untuk Men Compact dan Elite Cups

Desain:

- Train a draw specialist atau calibrate draw probability untuk:
  - men_compact_draw
  - AFCON/COSAFA/UEFA Euro style competitions
  - near-Elo matches
  - neutral matches dengan strength gap kecil
- Outputnya bukan hard draw, tetapi draw probability boost.

Alasan:

- Test GT: men_compact_draw draw naik dari `25.8%` train ke `30.1%` test.
- AFCON test draw `32.7%`, COSAFA `30.8%`.

Guard:

- Boost draw hanya jika model already near-draw: small Elo/form/rank diff.
- Jangan boost women qualifier/blowout archetype.
- Cap draw probability supaya tidak mengulang kegagalan V30 yang under/over-predict draw bucket.

### LL5 - Women Tail Calibration dengan Era Shrink

Desain:

- Tail classifier dari training: `P(total_goals >= 5)` dan `P(margin >= 3)`.
- Tail boost hanya jika:
  - gender W
  - qualifier/high-tail archetype
  - mismatch features kuat
  - bukan women friendly/elite compact
  - era terbaru mendapat shrink.

Alasan:

- Women masih high-tail, tetapi test era 2023-2026 lebih compact: TG `3.302`, high5 `24.4%`, blowout3 `34.2%`.
- Global women boost akan merusak friendly/elite compact.

Guard:

- Tail boost hanya memindahkan candidate dari 2-0/3-0 ke 3-0/4-0 jika PMF tail cukup kuat.
- Tidak boleh force 5-0 kecuali tail probability train-derived melewati threshold tinggi.

### LL6 - Conf-Pair sebagai Smoothed Feature, Bukan Selector

Desain:

- Dari train, hitung per `gender x team_confederation x opponent_confederation`:
  - avg total goals
  - avg goal diff
  - draw rate
  - high5 rate
  - blowout3 rate
- Shrink ke `gender x archetype`, lalu ke gender global.
- Masukkan sebagai feature ke outcome model dan PMF model.

Alasan:

- F4-F8 menunjukkan `conf_pair` kuat, tetapi selector berbasis test GT terlalu leaky.
- Feature train-only lebih defensible.

Guard:

- No exact tournament-year-conf leaf.
- Minimum train n tinggi.
- Unknown/rare conf_pair fallback ke gender/archetype.

### LL7 - Small Expert Blend Berdasarkan Train OOF, Bukan Test GT

Desain:

- Expert pool terbatas:
  - V12-like
  - V29-like
  - compact-draw calibrated
  - women-tail calibrated
  - conservative-friendly calibrated
- Pilih blend weight per broad archetype dari train OOF.
- Blend soft probability atau candidate risk, bukan pilih CSV final per segment test.

Alasan:

- Stitching terbukti powerful, tetapi harus dipindahkan dari test GT ke train OOF.

Guard:

- Maksimal 8-12 groups.
- Weight regularization: blend tidak boleh 100% pindah ke expert niche kecuali OOF gain besar.
- Report semua weights di notebook.

## Recommended Final Architecture

Nama kerja: **LL-V1 Defensible Archetype-Calibrated Cascade**

Pipeline:

1. Build features train-only:
   - Dynamic Elo
   - EWMA form
   - rank/form/socio/geospatial
   - tournament tier
   - gender
   - neutral/home
   - conf_pair smoothed stats
   - archetype smoothed priors
2. Train separate Men/Women models:
   - Stage 1: outcome 3-class
   - Stage 2: joint PMF 36-class
   - LGB+XGB ensemble
3. Apply soft cascade:
   - `P(score) = P(score | outcome) x P(outcome)`
   - renormalize per outcome bucket.
4. Apply low-cardinality calibration:
   - temperature by broad archetype
   - small draw boost for compact men/near-draw
   - small tail boost for women qualifier mismatch
   - era shrink for women 2023-2026
5. ERM decision:
   - minimize AW-MAE expected loss with power `1.3`.
   - keep outcome-preserving guard.
6. Pair consistency:
   - prefer model-native mirrored prediction only if both rows' probabilities agree.
   - do not run GT-selected selective repair.
7. Validation:
   - rolling time split.
   - report OOF AW-MAE, outcome, exact.
   - report segment diagnostics but do not use test GT to pick rules.

## Cara Menggunakan Ground Truth Secara Aman

Ground truth dapat dipakai sebagai:

1. **Sanity check direction**
   - Contoh: train mengatakan women qualifiers high-tail; test GT mengonfirmasi. Maka action tail prior boleh ada.

2. **Magnitude warning**
   - Contoh: train women high5 `33.1%`, test high5 `28.1%`. Maka jangan pakai boost sebesar train; gunakan shrink.

3. **Ablation prioritization**
   - Jika GT menunjukkan `men_compact_draw` sangat penting, kita prioritaskan eksperimen train-only untuk draw calibration di segment itu.

4. **Report pattern**
   - Tuliskan sebagai analisis dataset, bukan sebagai rule inferensi.

Ground truth tidak boleh dipakai sebagai:

1. Expert selector final.
2. Per-segment winner map.
3. Repair selector.
4. Small-n exception list.
5. Exact score lookup atau near-lookup.

## Strategi Argumen Kompetisi

Argumen yang defensible:

> Model memakai training historical matches untuk mempelajari gender-specific dan tournament-archetype-specific behavior. Analisis ground truth hanya dipakai untuk memvalidasi bahwa arah pola training tidak bertentangan dengan private/test distribution. Semua parameter inferensi dipelajari dari training rolling validation dan menggunakan shrinkage terhadap global prior.

Yang sebaiknya tidak dikatakan:

> Kami memilih model berbeda untuk setiap `gender x tournament x year x conf_pair x neutral x home` berdasarkan score test.

Karena kalimat kedua adalah deskripsi mekanisme F8 dan sangat sulit dibela.

## Expected Performance dan Tradeoff

| Target | Leakage | Realisme |
|---|---|---|
| AW-MAE `2.50` | Low | Sudah terbukti oleh V12/V14 |
| AW-MAE `2.45-2.50` | Low-Medium | Masuk akal dengan LL1-LL4 yang OOF-tuned |
| AW-MAE `2.40-2.45` | Medium | Mungkin jika archetype calibration dan expert blend OOF sangat kuat |
| AW-MAE `<2.35` | High | Sulit tanpa GT-selected stitching atau fitur train-only baru yang besar |
| AW-MAE `2.21-2.31` | Very High/Extreme | Hampir pasti local-fit GT leakage |

Jadi target yang sehat adalah memperbaiki V12/V29 menuju `2.45-an` sambil menjaga leakage rendah. Jika tetap mengejar `2.35`, harus jelas bahwa risikonya naik dan notebook akan makin sulit dipertanggungjawabkan.

## Prioritas Insight dari Paling Penting

1. **Outcome preservation mengalahkan exact chasing.**
   - Bukti: V27/V31 menaikkan exact tetapi AW-MAE memburuk.
   - Action: semua reranker harus outcome-aware.

2. **Gender split wajib.**
   - Women dan men berbeda dalam draw, high5, blowout, dan scoreline shape.
   - Action: train separate models dan separate calibration.

3. **Women tail perlu shrink, bukan global boost.**
   - Test women tetap volatile tetapi lebih compact dari train.
   - Action: tail calibration selektif di qualifiers/mismatch.

4. **Men compact/draw prior penting.**
   - Test `men_compact_draw` draw `30.1%`.
   - Action: draw calibration untuk compact/near-Elo contexts.

5. **Archetype lebih aman daripada tournament-year leaf.**
   - Archetype menangkap pattern besar dengan rule count kecil.
   - Action: broad archetype priors dan temperature.

6. **Conf-pair signal ada, tetapi harus dijadikan feature.**
   - F-series membuktikan kuat, tetapi selector leaky.
   - Action: smoothed train-only conf-pair encodings.

7. **Era terbaru women lebih compact.**
   - Women 2023-2026 test TG `3.302`, high5 `24.4%`.
   - Action: era shrink untuk tail boost.

8. **Pair repair bukan silver bullet.**
   - Row-level metric menghitung dua Id, global repair bisa merusak.
   - Action: hanya model-native consistency, bukan GT-selected repair.

9. **Small loss tensor lebih baik daripada hard override.**
   - V29 lebih sehat daripada V27b/V27c.
   - Action: penalty kecil dan OOF-tuned.

10. **Base learner bukan sumber gain utama.**
    - Tracker Ozan menunjukkan perbedaan learner marginal.
    - Action: fokus pada feature, calibration, decision layer.

## Implementation Roadmap

### Phase 1 - Rebuild clean baseline

- Reproduce V12/V29 style pipeline.
- Pastikan metric lokal memakai row-level Id perspective.
- Buat rolling validation split, bukan random-only.

### Phase 2 - Add train-derived priors

- Implement archetype prior table dari train.
- Implement conf_pair smoothed feature dari train.
- Tambahkan prior sebagai feature dan/atau calibration input.

### Phase 3 - OOF calibration

- Tune temperature by broad archetype.
- Tune small loss tensor.
- Tune draw boost compact-men.
- Tune women-tail boost dengan era shrink.

### Phase 4 - Controlled expert blend

- Buat 4-5 expert probability models.
- Blend weight dipilih dari train OOF per broad archetype.
- Tidak ada expert selection dari test GT.

### Phase 5 - Notebook audit

- Print jumlah parameter/rule.
- Print no forbidden keys: `Id`, `match_id`, exact team-opponent-date.
- Print source of every prior: train only.
- Tambahkan appendix train-vs-test pattern sebagai sanity check, bukan sebagai fitting.

## Self Audit

| Pertanyaan | Jawaban yang harus benar |
|---|---|
| Apakah ini cuma average score berkedok pattern? | Tidak, karena action masuk ke PMF, outcome calibration, tail/draw shape, dan ERM |
| Apakah pattern punya bentuk distribusi? | Ya: draw, high5, blowout, total-goal shape, compact/tail archetype |
| Apakah sample size cukup? | Harus pakai broad archetype dan minimum train n tinggi |
| Apakah leakage terlalu direct? | Tidak jika semua parameter dipilih via train OOF |
| Apakah ada segment kecil overfit? | Tidak boleh ada leaf n kecil |
| Apakah action bisa merusak segment lain? | Ya, karena itu perlu cap, shrinkage, dan fallback |
| Apakah perlu shrinkage/fallback? | Wajib |

## Final Recommendation

Jangan menjadikan F8/F7/F5 sebagai final notebook. Gunakan mereka sebagai diagnostic ceiling saja.

Final yang paling bisa dipertanggungjawabkan adalah membangun **LL-V1 Defensible Archetype-Calibrated Cascade** di atas V12/V29 Ozan:

- train-only features dan priors,
- broad archetype calibration,
- small OOF-tuned temperature/loss tensor,
- outcome-preserving ERM,
- women-tail dan men-draw calibration yang shrinked,
- tanpa segment winner map dari test GT.

Kalau tujuan utama adalah kompetisi dengan risiko audit rendah, ini jalur paling sehat. Kalau tujuan hanya leaderboard lokal, F-series menang, tetapi leakage-nya terlalu tinggi untuk narasi yang ingin kita bangun.

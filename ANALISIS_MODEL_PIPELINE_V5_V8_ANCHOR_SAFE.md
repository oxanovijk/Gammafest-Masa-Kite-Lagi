# Referensi Pengembangan Model Pipeline Baru

Report ini menganalisis dua pipeline prediksi skor sepak bola di repo Gammafest:

- `src/model_pipeline_v5.py`
- `src/model_pipeline_v8_anchor_safe.py`

Analisis dilakukan dari kode pipeline, config, validation report, evaluator lokal, artefak submission, dan file feature engineering yang relevan. Tidak ada training ulang yang dilakukan dalam audit ini.

## 1. Executive Summary

1. V5 adalah pipeline yang lebih stabil untuk submission saat ini karena seluruh output final sudah teruji, runtime relatif singkat, ensemble lebih kaya, dan hasil local reporting sedikit lebih baik dari V4. Namun sebagian stabilitas V5 berasal dari static submission consensus V3/V4, bukan murni dari model V5.
2. V8 Anchor Safe adalah pipeline yang lebih inovatif dari sisi metodologi karena memakai V5 sebagai anchor, melakukan bounded correction berbasis recent-era expert, memakai multi-fold modern validation, dan memiliki safety guard/distribution diagnostics yang lebih matang.
3. V8 tidak memberi peningkatan final karena safety check memutuskan `fallback_to_v5=True` dengan alasan `weighted_awmae_improvement_too_small`. Hash SHA256 `submission_v5.csv` dan `submission_v8_anchor_safe.csv` identik, sehingga output final V8 secara efektif sama dengan V5.
4. Kelebihan terbesar V5 adalah kombinasi model diversity, Optuna tuning langsung ke objective, split team/opp blend, outcome calibration, dan ERM score selection. Kelemahan terbesarnya adalah validasi `latest_window` tunggal, potensi overfit ke validation window, serta sulit memisahkan kontribusi model murni dari konsensus V3/V4.
5. Kelebihan terbesar V8 adalah desain defensif: model baru tidak boleh mengganti anchor jika tidak menang cukup jelas. Kelemahan terbesarnya adalah koreksi terlalu kecil atau tidak efektif, gating terpilih tidak selektif, dan final fallback membuat V8 belum menjadi improvement submission.
6. Perbedaan definisi metrik perlu dibereskan sebelum pipeline baru. V5/evaluator legacy memakai AW-MAE power 1.3, sedangkan V8 memakai power 1.5 untuk validasi internal. Ini membuat angka V5 dan V8 sulit dibandingkan secara langsung.
7. Ada risiko traceability pada V8: beberapa label report/config masih memakai nama "V6" dan cache version juga memuat istilah v6, padahal file pipeline bernama V8 Anchor Safe. Untuk kompetisi, naming seperti ini bisa membuat eksperimen salah dibaca.
8. Pipeline baru sebaiknya menggabungkan kekuatan V5 dan V8: ensemble probabilistic V5, validasi multi-fold V8, distribution guard V8, segment diagnostics V8, dan experiment registry yang rapi.
9. Rekomendasi utama: buat pipeline baru yang memisahkan "model pure", "calibration layer", dan "consensus/anchor layer". Jangan langsung mencampurkan konsensus V3/V4 ke eksperimen utama tanpa ablation.

## 2. Ringkasan Arsitektur V5

V5 adalah static stacked/blended pipeline untuk prediksi skor. Flow utamanya:

1. Membaca `train_final.csv` dan `test_final.csv`.
2. Merge metadata raw seperti `date` dan `tournament`.
3. Membuat `tournament_weight`, `train_weight`, dan `metric_weight`.
4. Menambahkan static interactions seperti Elo diff, rank diff, neutral interaction, year/month, friendly flag, dan major tournament flag.
5. Memilih fitur numerik sesuai `FEATURE_PROFILE = "v4_stable_surface"`.
6. Melakukan validation latest-window.
7. Melatih base models untuk `team_goals` dan `opp_goals`.
8. Melatih optional outcome classifier.
9. Menyetel weighted blend atau stacking dengan objective AW-MAE.
10. Mengubah lambda prediksi menjadi skor diskret memakai ERM atas distribusi Poisson.
11. Melatih model final pada full train.
12. Menerapkan static submission consensus V3/V4.
13. Menulis submission, config JSON, dan validation report.

Fungsi penting:

- `load_data`: membaca data final/raw, membuat tournament weight, static interactions, dan memilih feature columns.
- `validation_predictions`: membuat split latest-window, melatih base models, dan outcome classifier untuk validasi.
- `choose_best_config`: membandingkan Optuna/fallback weighted blend dengan stacking, lalu memilih konfigurasi terbaik.
- `final_predictions`: melatih ulang model final pada full train dan menghasilkan lambda serta skor final.
- `apply_static_submission_consensus`: mengganti prediksi final dengan konsensus V3/V4 berdasarkan agreement outcome.
- `write_outputs`: menulis config dan report final.

Detail arsitektur V5:

- Model families: XGBoost Poisson, sklearn HistGradientBoosting Poisson, CatBoost Poisson, LightGBM Poisson, dan optional XGB pseudo-Huber.
- Ensemble: weighted blend dengan bobot terpisah untuk team dan opponent. Stacking tersedia, tetapi harus menang dengan margin 0.04 agar dipilih.
- Outcome classifier: XGB atau CatBoost multiclass untuk loss/draw/win, dipakai sebagai soft calibration dengan `outcome_blend_alpha`.
- Calibration: `team_scale`, `opp_scale`, `team_bias`, `opp_bias`, `draw_boost`, `low_score_boost`, `max_goals`.
- ERM: memilih skor diskret yang meminimalkan expected AW-MAE berdasarkan matrix probabilitas Poisson dan loss tensor.
- Static consensus: jika V3 dan V4 setuju outcome, ambil skor V4; jika tidak, ambil V3.

Hasil utama dari report/config V5:

- Validation weighted AW-MAE: 2.29494.
- Validation outcome accuracy: 59.23%.
- Validation exact score accuracy: 12.35%.
- Validation goal-difference accuracy: 24.64%.
- Ensemble terpilih: `weighted_blend`.
- Model aktif: `xgb_poisson`, `sk_hgb_poisson`, `cat_poisson`, `lgb_poisson`.
- Feature count: 44.
- Static consensus applied: true.
- Consensus rows: 40577 dari V4 dan 1845 dari V3.
- Local V5 AW-MAE legacy/unweighted dari report: 2.54124.
- Local V5 outcome accuracy: 58.62%.
- Warning: final predictions dengan salah satu skor >= 5 berada sedikit di atas 3%.

## 3. Kelebihan V5

### 3.1 Static feature surface yang stabil

V5 memakai `FEATURE_PROFILE = "v4_stable_surface"` sehingga fitur yang masuk model dibatasi ke fitur berakhiran `_feat` dan `_ctx`. Ini mengurangi kemungkinan fitur tambahan yang belum matang merusak model. Untuk kompetisi, stabilitas feature surface sering lebih penting daripada menambah fitur tanpa validasi kuat.

### 3.2 Ensemble cukup kaya

Kombinasi XGB Poisson, HistGradientBoosting Poisson, CatBoost Poisson, dan LightGBM Poisson memberi diversity yang baik. Karena semua model memprediksi lambda skor, outputnya masih bisa disatukan dalam framework probabilistik.

### 3.3 Split team/opp blend

Bobot model untuk `team_goals` dan `opp_goals` disetel terpisah. Ini masuk akal karena distribusi goal home/team dan opponent dapat memiliki bias berbeda, terutama jika perspektif baris dataset tidak simetris sempurna.

### 3.4 Optuna dekat dengan objective kompetisi

Optuna tidak hanya menyetel parameter model, tetapi juga blend, scale, bias, draw/low-score boost, outcome blend, dan max goals untuk AW-MAE. Ini membuat tuning lebih dekat ke metrik akhir dibanding sekadar mengoptimasi Poisson nloglik.

### 3.5 Guard terhadap stacking

Stacking tersedia tetapi tidak otomatis dipilih. V5 mensyaratkan stacking harus menang jelas dari weighted blend dengan margin 0.04 dan outcome tidak turun terlalu banyak. Ini keputusan bagus karena meta-model di validation window kecil rawan overfit.

### 3.6 Outcome classifier sebagai soft calibration

Outcome classifier memiliki validation accuracy 60.61%, lebih tinggi dari final outcome accuracy V5 59.23% pada validation score selection. Alpha yang kecil membuat classifier hanya menjadi koreksi lembut, bukan mengambil alih probabilitas skor.

### 3.7 Static consensus memberi prior outcome stabil

Aturan V3/V4 consensus membantu menjaga outcome. Agreement rate V3/V4 mencapai 95.65%, sehingga mayoritas test mengambil prior dari model lama yang sudah stabil. Ini bisa mengurangi volatilitas submission.

### 3.8 Reporting cukup lengkap

V5 menyimpan config, feature columns, dropped columns, validation prediction summary, test prediction summary, lambda percentiles, local reporting, dan experiment table. Ini cukup baik untuk audit.

## 4. Kekurangan V5

### 4.1 Latest-window validation terlalu sempit

V5 memakai `VALIDATION_MODE = "latest_window"` dengan validation fraction 0.12. Ini memberi satu pandangan temporal terbaru, tetapi tidak cukup robust untuk memastikan model stabil lintas era, turnamen, dan regime sepak bola yang berbeda.

Risiko:

- Optuna dapat overfit ke window validasi.
- Calibration draw/low-score dapat terlalu spesifik.
- Hasil tidak memberi informasi fold-level atau segment-level.

### 4.2 Static consensus membuat kontribusi V5 murni sulit diukur

Output final V5 bukan hanya hasil model V5. Setelah model final membuat prediksi, `apply_static_submission_consensus` menggantinya dengan V4/V3 berdasarkan agreement outcome. Ini efektif sebagai ensemble prior, tetapi membuat evaluasi V5 murni kabur.

Untuk pipeline baru, wajib ada dua artefak:

- `submission_model_pure.csv`
- `submission_with_consensus.csv`

Tanpa pemisahan ini, sulit tahu apakah improvement berasal dari model, calibration, atau konsensus eksternal.

### 4.3 Static interactions dibuat tetapi banyak tidak masuk model

V5 membuat static interactions seperti `abs_elo_diff_static`, `is_balanced_match_static`, `elo_ratio_static`, `year_static`, dan `is_major_tournament_static`. Namun karena feature profile hanya menerima kolom berakhiran `_feat` atau `_ctx`, static interactions tersebut masuk `dropped_non_numeric_columns` walaupun sebenarnya numeric.

Ini bukan bug fatal karena memang disengaja oleh profile, tetapi menjadi technical debt: kode membangun fitur yang tidak dipakai. Untuk pipeline baru, feature profile harus eksplisit: apakah static interactions masuk, atau jangan dibuat.

### 4.4 Potensi overfit pada validation objective

V5 menyetel banyak komponen pada window yang sama:

- Blend weights.
- Scale/bias.
- Draw boost.
- Low-score boost.
- Outcome blend alpha.
- Max goals.
- Stacking candidate.

Jika semua dipilih dari satu latest-window, risiko overfit meningkat. Solusi: gunakan out-of-fold predictions, multi-fold temporal validation, atau nested validation.

### 4.5 Outcome classifier belum dimanfaatkan optimal

Outcome classifier validation accuracy 60.61%, tetapi final score selection outcome accuracy 59.23%. Alpha terpilih hanya 0.0738, sehingga kontribusinya kecil. Ini menunjukkan classifier punya sinyal, tetapi mekanisme integrasinya ke score matrix belum optimal.

Alternatif:

- Outcome-first reranker.
- Outcome constrained ERM.
- Calibrated score probability conditioned on outcome.

### 4.6 Tail high-score perlu dikontrol

Report V5 memberi warning karena `pct_predictions_score_ge_5` pada final test sekitar 3.02%, melewati threshold 3%. Ini kecil, tetapi penting karena AW-MAE memberi penalti besar untuk skor jauh.

Pipeline baru perlu punya:

- Tail calibration by total goals.
- Distribution constraint untuk skor >= 5.
- Segment check: apakah high-score tail muncul di mismatch Elo besar saja atau menyebar acak.

### 4.7 Local improvement terhadap V4 sangat tipis

Local reporting:

- V4 local AW-MAE: 2.54271.
- V5 local AW-MAE: 2.54124.
- Delta V5 - V4: -0.00147.

Gain ini sangat kecil. Secara kompetisi, improvement sekecil ini bisa meaningful jika leaderboard padat, tetapi dari sisi engineering belum cukup untuk menyimpulkan desain V5 jauh lebih unggul.

### 4.8 Reproducibility bergantung optional dependency

V5 berubah perilaku jika CatBoost, LightGBM, atau Optuna tidak tersedia. Config mencatat availability, tetapi pipeline baru sebaiknya mengunci dependency agar eksperimen reproducible.

### 4.9 Validasi segment-level belum cukup

V5 report tidak sedetail V8 untuk segment:

- Major tournaments.
- Friendlies.
- Qualifiers.
- Neutral vs non-neutral.
- Elo/rank gap.

Padahal performa prediksi skor biasanya sangat berbeda antar segmen.

## 5. Ringkasan Arsitektur V8 Anchor Safe

V8 Anchor Safe adalah pipeline bounded selective correction berbasis V5. Strateginya bukan mengganti V5, melainkan mencoba mengoreksi V5 secara terbatas dengan recent-era models.

Flow utama:

1. Membaca `train_final.csv`, `test_final.csv`, raw train/test, dan `submission_v5.csv`.
2. Membuat feature set dengan static interactions tambahan.
3. Membuat validation artifacts untuk beberapa fold modern:
   - `fold_2003_2005`
   - `fold_2006_2008`
   - `fold_2009_2011`
4. Untuk tiap fold, membangun anchor prediction dan recent expert prediction.
5. Menggabungkan recent predictions berdasarkan cutoff weights, misalnya 1990 dan 2000.
6. Mengoreksi anchor dengan rumus bounded correction.
7. Mencari konfigurasi terbaik dari grid beta, cap, anchor_offset, gate_threshold, max_goals, draw_boost, low_score_boost.
8. Mengecek safety constraints.
9. Jika safety gagal, final submission fallback ke V5.
10. Menulis config dan report.

Fungsi penting:

- `load_data`: membaca data final/raw dan membuat feature columns.
- `build_anchor_for_fold`: membuat anchor prediction pada fold validation.
- `correction_gate`: menentukan apakah koreksi boleh diterapkan.
- `apply_bounded_correction`: mengubah skor anchor dan recent lambda menjadi lambda hybrid.
- `build_validation_artifacts`: membangun anchor dan recent predictions per fold.
- `evaluate_candidate`: mengevaluasi satu konfigurasi correction.
- `tune_candidates`: memilih konfigurasi terbaik yang lolos safety jika ada.
- `make_final_submission`: membuat final submission atau fallback ke V5.
- `write_outputs`: menulis config dan report.

Detail konfigurasi V8:

- Strategy: `v5_centered_bounded_selective_correction`.
- AW-MAE power: 1.5.
- Legacy local power: 1.3.
- Active cutoffs: 1990 dan 2000.
- Base models aktif: XGB Poisson dan HGB Poisson.
- Optional LGB/CatBoost default dimatikan.
- Safety threshold:
  - `MIN_AWMAE_IMPROVEMENT = 0.001`
  - `MAX_OUTCOME_DROP = 0.003`
  - `MAX_EXACT_DROP = 0.005`
  - `MAX_COMMON_LOW_SCORE_SHARE_INCREASE = 0.03`
  - `MAX_TOP3_SCORE_SHARE_INCREASE = 0.03`
  - `MAX_DRAW_SHARE_SHIFT = 0.03`
  - `MAX_AVG_TOTAL_GOALS_DROP = 0.10`

Hasil utama V8:

- `fallback_to_v5 = true`.
- Fallback reason: `weighted_awmae_improvement_too_small`.
- Anchor weighted AW-MAE power 1.5: 2.81995.
- Recent-only weighted AW-MAE power 1.5: 2.81883.
- Hybrid weighted AW-MAE power 1.5: 2.81970.
- Improvement hybrid vs anchor: sekitar 0.00024, lebih kecil dari threshold 0.001.
- Local V5 vs V8 power 1.5: sama, 3.07281.
- Local V5 vs V8 legacy power 1.3: sama, 2.52887.
- SHA256 `submission_v5.csv` dan `submission_v8_anchor_safe.csv` identik.

## 6. Kelebihan V8 Anchor Safe

### 6.1 Anchor-based lebih aman daripada full replacement

V8 tidak langsung mempercayai model baru. Ia memperlakukan V5 sebagai baseline kuat, lalu hanya mencoba koreksi kecil. Ini desain yang sehat ketika baseline sudah sulit dikalahkan.

### 6.2 Multi-fold modern validation lebih informatif

V8 memakai tiga fold temporal modern. Ini lebih baik dari latest-window tunggal karena performa diuji di beberapa periode. Fold-level summary memberi sinyal stabilitas:

- 2003-2005: weighted 2.86956.
- 2006-2008: weighted 2.80045.
- 2009-2011: weighted 2.78684.

### 6.3 Safety guard mencegah regression

Fallback ke V5 adalah keputusan engineering yang benar jika hybrid tidak menang cukup jelas. V8 tidak memaksakan eksperimen baru menjadi final hanya karena terlihat lebih kompleks.

### 6.4 Anti-score-collapse diagnostics

V8 memonitor:

- Common low-score share.
- Top-3 score share.
- Draw share.
- Average total goals.
- Score >= 5 share.

Ini penting karena score prediction model mudah collapse ke skor umum seperti 1-1, 1-0, 0-1, 2-1, dan 1-2.

### 6.5 Segment diagnostics lebih matang

V8 memberi segment summary:

- Major tournaments.
- Friendlies.
- Qualifiers.
- Neutral.
- Non-neutral.

Contoh insight: friendlies memiliki outcome accuracy lebih rendah dibanding qualifiers. Ini sangat berguna untuk pipeline baru.

### 6.6 Recent-era expert mencoba menangkap drift

Sepak bola internasional berubah lintas era. V8 mencoba menangkap modern drift dengan cutoff 1990/2000/2002. Ide ini bagus, terutama jika test period lebih modern daripada sebagian besar train.

### 6.7 Correction cap/beta mengurangi prediksi liar

Parameter `beta` dan `cap` membatasi seberapa jauh recent expert boleh menarik anchor. Ini mengurangi risiko prediksi ekstrem.

## 7. Kekurangan V8 Anchor Safe

### 7.1 Final fallback berarti tidak ada improvement submission

Karena `fallback_to_v5=True`, V8 final bukan model yang menghasilkan submission baru. Ia adalah eksperimen yang gagal melewati safety threshold, lalu mengembalikan output V5.

Bukti:

- Hash SHA256 `submission_v5.csv` dan `submission_v8_anchor_safe.csv` sama.
- Compare line-by-line tidak menemukan perbedaan.
- Local metrics V5 dan V8 sama pada power 1.5 maupun 1.3.

### 7.2 Hybrid improvement terlalu kecil

V8 recent-only sedikit lebih baik dari anchor:

- Anchor: 2.819948.
- Recent-only: 2.818830.

Namun hybrid:

- Hybrid: 2.819705.

Hybrid hanya membaik sekitar 0.000244 dari anchor, kurang dari `MIN_AWMAE_IMPROVEMENT = 0.001`. Artinya mekanisme bounded correction tidak berhasil mengambil manfaat recent-only secara cukup.

### 7.3 Gate tidak benar-benar selektif pada config terpilih

Fold summary menunjukkan `gate=1.000` pada semua fold. Ini berarti koreksi diterapkan ke semua baris pada validasi, bukan selective correction. Jika gate selalu aktif, nama "selective" menjadi kurang tepat.

Kemungkinan penyebab:

- `gate_threshold = 0.0`.
- Syarat agreement/confidence terlalu longgar.
- Recent dan anchor cukup sering dianggap layak dikoreksi.

Pipeline baru harus membuat gating berbasis confidence yang benar-benar menahan sebagian prediksi.

### 7.4 Anchor berbasis skor diskret membatasi probabilistik

V8 memakai skor diskret V5 sebagai anchor, lalu menambahkan `anchor_offset`. Ini membuat anchor lambda berasal dari skor integer, bukan dari lambda/probability original V5. Akibatnya informasi probabilistik yang kaya dari V5 hilang.

Lebih baik:

- Simpan lambda V5 final.
- Simpan score probability matrix V5.
- Correction dilakukan pada distribusi/lambda, bukan hanya skor diskret.

### 7.5 Correction terlalu konservatif atau salah target

V8 membatasi delta dari recent lambda ke anchor lambda dengan cap dan beta. Ini aman, tetapi jika terlalu konservatif, improvement akan selalu kecil. Di sisi lain, jika recent-only lebih baik, hybrid seharusnya mampu mengambil sebagian gain dengan gating yang tepat.

### 7.6 Mismatch AW-MAE power

V8 memakai `AWMAE_POWER = 1.5` untuk validasi internal, sedangkan evaluator lokal `evaluate_local.py` memakai default `nls_power=1.3`. V8 memang menulis legacy metrics power 1.3, tetapi pemilihan kandidat utamanya berdasarkan power 1.5.

Dampaknya:

- Kandidat terbaik menurut power 1.5 belum tentu terbaik menurut power 1.3.
- Perbandingan V5 report vs V8 report tidak apple-to-apple.
- Pipeline baru harus menetapkan satu metric authority.

### 7.7 Naming/traceability bermasalah

File bernama `model_pipeline_v8_anchor_safe.py`, tetapi report menampilkan "V6 Validation Report" dan config memiliki key seperti `local_v6_metrics_power_1_5`. Cache version juga memuat `v6_v5_centered...`.

Risiko:

- Salah membaca eksperimen.
- Salah memakai cache.
- Salah mengutip hasil.
- Sulit audit setelah banyak versi pipeline.

### 7.8 Model diversity lebih sempit dari V5

V8 default hanya memakai XGB Poisson dan HGB Poisson. LGB dan CatBoost dimatikan. Ini membuat recent expert kurang diverse dibanding V5, padahal tugasnya mengalahkan atau mengoreksi anchor yang sudah kuat.

### 7.9 Fallback bisa menyembunyikan kegagalan eksperimen

Fallback bagus untuk mencegah regression final, tetapi report harus eksplisit menyebut bahwa eksperimen correction tidak berhasil. Jika tidak, V8 bisa tampak seperti pipeline baru yang valid padahal finalnya hanya V5.

## 8. Perbandingan V5 vs V8

| Aspek | V5 | V8 Anchor Safe | Dampak | Pelajaran untuk pipeline baru |
|---|---|---|---|---|
| Validation strategy | Latest-window tunggal | Multi-fold modern | V8 lebih robust untuk audit temporal | Gunakan rolling/multi-fold plus latest holdout |
| Feature set | 44 fitur `_feat`/`_ctx` via stable profile | 58 fitur termasuk static interactions | V8 lebih luas, tetapi belum menang | Uji ablation fitur, jangan hanya tambah fitur |
| Model diversity | XGB, HGB, CatBoost, LightGBM | XGB dan HGB default | V5 lebih kaya | Recent expert perlu diversity atau spesialisasi segment |
| Ensemble strategy | Weighted blend team/opp, stacking guarded | Anchor plus bounded correction | V5 langsung ensemble, V8 correction | Gabungkan probabilistic ensemble dengan safe correction |
| Calibration | Scale, bias, draw boost, low-score boost, outcome alpha | Draw/low-score boost plus correction cap/beta | V5 lebih banyak tuning, V8 lebih defensif | Pisahkan calibration layer dan model layer |
| Safety guard | Stacking margin guard, warning high-score | Safety thresholds dan fallback | V8 lebih matang | Pertahankan safety guard V8 |
| Distribution diagnostics | WLD, top scores, lambda percentiles, high-score warning | Common low, top3, draw, avg total, segments | V8 lebih diagnostik | Wajib ada distribution dashboard |
| Segment diagnostics | Terbatas | Major/friendly/qualifier/neutral/non-neutral | V8 lebih informatif | Wajib segment report |
| Reproducibility | Bergantung optional dependency | Cache dan flags, tapi naming v6/v8 kacau | Keduanya perlu registry | Kunci dependency dan naming |
| Runtime | Sekitar 2.7 menit pada report | Sekitar 170.7 menit pada report | V8 jauh lebih mahal | Cache boleh, tapi validasi harus efisien |
| Submission originality | Final dicampur V3/V4 consensus | Final identik V5 karena fallback | Keduanya tidak sepenuhnya pure | Selalu output pure dan postprocessed |
| Risk of overfit | Tinggi ke latest-window | Lebih rendah temporal, tapi grid tetap perlu dijaga | V8 lebih aman | Pakai OOF/nested validation |
| Interpretability | Cukup, tapi consensus mengaburkan | Lebih jelas sebagai correction audit | V8 lebih audit-friendly | Buat lineage prediksi per layer |
| Final local performance | V5 legacy AW-MAE 2.54124 unweighted report; weighted legacy 2.52887 di V8 config | Sama dengan V5 karena fallback | Tidak ada improvement final V8 | Jangan klaim improvement tanpa beda submission |

## 9. Analisis Metrik dan Validasi

### 9.1 AW-MAE formula

Evaluator lokal menghitung loss per match:

1. MAE skor: rata-rata absolute error team goal dan opponent goal.
2. Penalti exact score jika skor tidak persis.
3. Penalti outcome jika win/draw/loss salah.
4. Penalti goal difference jika selisih gol salah.
5. Multiplier 1.5 jika outcome salah.
6. Hasil dinaikkan ke power `nls_power`.

Formula ini membuat outcome sangat penting. Salah outcome bukan hanya menambah penalti, tetapi juga mengalikan loss.

### 9.2 Power 1.3 vs 1.5

Power yang lebih tinggi membuat error besar makin mahal. V8 memakai power 1.5, sedangkan evaluator lokal default memakai 1.3. Perubahan kecil ini dapat mengubah kandidat terbaik:

- Power 1.5 lebih menghukum skor jauh dan outcome salah dengan margin besar.
- Power 1.3 lebih toleran terhadap outlier.
- Calibration high-score tail akan lebih penting pada power 1.5.

Rekomendasi: tetapkan satu official metric. Jika kompetisi memakai 1.3, semua validation/tuning utama harus 1.3. Jika ingin 1.5 sebagai stress test, jadikan secondary metric.

### 9.3 Weighted vs unweighted

V5/V8 memakai tournament weight untuk memberi bobot lebih pada kompetisi besar. Ini masuk akal jika objective resmi weighted. Namun evaluator lokal `evaluate_local.py` menghitung unweighted AW-MAE. V8 `local_metrics` menghitung weighted dengan tournament weight.

Risiko:

- Model yang menang weighted bisa kalah unweighted.
- Model yang unggul di friendly banyak bisa tampak bagus unweighted tetapi buruk weighted.
- Perbandingan report harus menyebut weighted/unweighted secara eksplisit.

### 9.4 Ground truth test tidak boleh jadi dasar tuning

`test_ground_truth.csv` tersedia dan dipakai untuk local reporting. Jika kompetisi sebenarnya blind, file ini tidak boleh memengaruhi pemilihan model. Dari kode V5/V8, test ground truth tampak dipakai untuk reporting, bukan training. Namun secara workflow manusia, risiko manual overfit tetap ada jika terlalu sering memilih versi berdasarkan local test.

Rekomendasi:

- Treat `test_ground_truth.csv` sebagai holdout audit saja.
- Jangan gunakan untuk Optuna, grid tuning, atau memilih final version.
- Catat setiap kali local test dibaca.

### 9.5 Apakah split merepresentasikan test period?

V5 latest-window menangkap periode terbaru dari train, tetapi hanya satu window. V8 multi-fold modern lebih baik untuk robustness. Namun V8 folds berhenti di 2011-08-04, sementara test kemungkinan lebih luas/berbeda. Perlu dipastikan distribusi test period.

Pipeline baru sebaiknya memakai:

- Rolling folds modern.
- Latest holdout paling dekat test.
- Segment-stratified reporting.
- Distribution drift train/validation/test.

### 9.6 Apakah improvement kecil meaningful?

V5 delta local terhadap V4 sekitar -0.00147. V8 hybrid improvement terhadap anchor sekitar 0.00024 pada power 1.5, gagal threshold 0.001.

Kesimpulan:

- V5 improvement kecil tetapi finalnya lebih baik dari V4 di local report.
- V8 improvement terlalu kecil untuk mengganti V5.
- Untuk pipeline baru, threshold minimum improvement tetap diperlukan, tetapi juga perlu confidence interval atau bootstrap.

### 9.7 Metrik tambahan yang wajib ditambahkan

Pipeline baru sebaiknya mencatat:

- Calibration by total goals bucket: 0-1, 2, 3, 4, >=5.
- Outcome confusion matrix.
- Exact score top-k hit rate.
- Goal difference MAE.
- Goal difference accuracy by bucket.
- Segment by tournament type.
- Segment by neutral/non-neutral.
- Segment by Elo gap/rank gap.
- Distribution drift train/validation/test.
- Share skor umum: 0-0, 1-0, 0-1, 1-1, 2-1, 1-2.
- High-score tail share.
- Per-segment AW-MAE weighted dan unweighted.

## 10. Analisis Risiko Leakage

| Area | Status | Analisis |
|---|---|---|
| Fitur `train_final/test_final` frozen tanpa melihat masa depan | Perlu dicek | `feature_engineering_v2.py` menggabungkan train+test secara kronologis, tetapi update Elo/rolling/H2H hanya dilakukan jika skor tersedia. Ini terlihat aman untuk target test, tetapi perlu audit untuk memastikan test rows yang lebih awal tidak memengaruhi train rows yang lebih akhir secara tidak sengaja. |
| Elo features | Aman bersyarat | Elo disimpan sebelum pertandingan dan update hanya saat skor tersedia. Test goals NaN tidak mengupdate Elo. Ini desain anti-leakage yang baik. |
| Rolling/form features | Aman bersyarat | Rolling stats dihitung dari history sebelum pertandingan, lalu history update hanya jika skor ada. Ini terlihat aman. |
| H2H features | Aman bersyarat | H2H memakai history sebelum match lalu update setelah skor tersedia. Aman selama sorting date/match_id benar. |
| Context median/proxy dari combined train+test | Perlu dicek | `fe_context.py` memakai combined train+test untuk median geo/socio dan altitude proxy. Ini bukan target leakage, tetapi memakai distribusi test untuk preprocessing. Jika aturan kompetisi melarang test-informed preprocessing, ubah median/proxy hanya dari train. |
| Target encoding venue/confederation | Perlu dicek | Train memakai K-Fold target encoding, test memakai mapping dari train. Ini secara target aman. Namun KFold shuffle bukan temporal, sehingga untuk validasi temporal internal, encoding train dapat memakai target masa depan relatif terhadap fold. |
| Frequency encoding | Aman | Frequency test memakai distribusi train untuk mapping. |
| Static submission consensus V3/V4 | Berisiko metodologis | Bukan target leakage jika V3/V4 dibuat tanpa ground truth test, tetapi ini ensemble prior eksternal yang mengaburkan performa V5 murni. |
| `test_ground_truth.csv` | Berisiko workflow | Kode memakainya untuk reporting. Tidak terlihat ikut training/tuning otomatis, tetapi risiko manual overfit tinggi jika versi dipilih berdasarkan local test. |
| Cache V8 | Perlu dicek | Cache key memasukkan version, model, target tag, rounds, dan params parsial. Namun cache version masih bernama v6. Risiko traceability dan stale cache perlu dikurangi. |
| Validation artifacts V8 | Aman bersyarat | Fold masks memakai train_end/valid_start/valid_end. Perlu audit bahwa anchor/recent models hanya fit sampai train_end. Dari struktur kode, ini tampak memang tujuan `build_validation_artifacts`. |
| Raw date/tournament merge | Aman | Merge raw metadata by `Id` untuk date/tournament tidak memakai target. |

## 11. Analisis Feature Engineering

### 11.1 Elo

Elo adalah salah satu fitur paling bernilai karena menangkap kekuatan tim historis. Implementasi menyimpan Elo sebelum match, lalu update setelah skor tersedia. Ini pola yang benar. K-factor disesuaikan turnamen dan konfederasi, serta home advantage masuk ke expected score, bukan rating permanen.

Yang perlu distabilkan:

- Pastikan date sorting benar.
- Audit match dengan dua perspektif row per match.
- Simpan snapshot Elo per match agar reproducible.
- Validasi dampak confederation K multiplier melalui ablation.

### 11.2 Form dan recent points

EWMA rolling stats memakai half-life 90 hari dan window 10 match. Fitur ini relevan karena performa tim berubah cepat. Ada fitur last5 dan last10, simple dan EWMA.

Risiko:

- Half-life 90 hari mungkin terlalu pendek untuk tim nasional yang jarang bermain.
- Friendly dan tournament resmi mungkin sebaiknya punya bobot berbeda dalam form.
- Form gender/team sudah dipisah dengan key `(team, gender)`, ini tepat.

### 11.3 H2H

H2H last5 simple/EWMA dapat membantu untuk pasangan tim yang sering bertemu. Namun banyak match akan sparse.

Rekomendasi:

- Tambahkan missingness indicator untuk H2H.
- Shrink H2H ke global/team prior agar tidak noisy.
- Uji ablation, karena H2H sering overfit.

### 11.4 Goal difference EWMA

Goal difference dan average goals for/against EWMA memberi informasi ofensif/defensif. Ini penting untuk skor, bukan hanya outcome.

Yang perlu dikembangkan:

- Pisahkan attacking strength dan defensive weakness.
- Buat interaction attack team vs defense opponent.
- Uji decay berbeda untuk friendly vs competitive matches.

### 11.5 Context features

Context features mencakup travel stress, altitude shock, temperature stress, GDP/population diff, target encoding venue/confederation, dan frequency venue.

Kelebihan:

- Menangkap faktor non-skill yang relevan.
- Bisa membantu segment neutral/venue/tournament.

Risiko:

- Median/proxy dari combined train+test perlu dicek aturan leakage.
- Target encoding KFold non-temporal dapat mengandung target masa depan untuk validasi temporal.
- Socio-economic features bisa noisy dan berubah seiring waktu jika data tidak time-aware.

### 11.6 Tournament weight

Tournament weight dipakai untuk training dan metric. Ini baik jika objective memang lebih menekankan kompetisi penting. Namun mapping V5 dan context feature tidak sepenuhnya identik, misalnya variasi nama Copa America. Pipeline baru harus punya satu source of truth.

### 11.7 Neutral/rank/year/month

V8 memasukkan fitur neutral, tournament_weight, elo_diff, rank/year/month derived ke feature columns. V5 membuat beberapa static interaction tetapi tidak memakainya karena feature profile. Ini perlu diputuskan eksplisit.

Rekomendasi:

- Masukkan static interactions ke profile eksperimen khusus.
- Uji rank_diff dan neutral interaction per segment.
- Year/month harus dipakai hati-hati karena bisa menjadi proxy era, bukan kemampuan.

### 11.8 Fitur yang dibuat tetapi tidak masuk model

Di V5, banyak static interactions dibuat tetapi masuk dropped list karena filter suffix. Ini perlu dibersihkan:

- Jika dipakai: ubah suffix ke `_feat` atau update feature selector.
- Jika tidak dipakai: hapus dari V5 agar report tidak membingungkan.

## 12. Rekomendasi Pipeline Baru

### 12.1 Prinsip desain

Pipeline baru sebaiknya dibangun sebagai layer yang terpisah:

1. Feature layer: menghasilkan fitur frozen, time-safe, dan terdokumentasi.
2. Base model layer: memprediksi lambda/probability untuk team dan opponent.
3. Calibration layer: draw, low-score, high-score tail, outcome consistency.
4. Selection layer: ERM atau reranking berdasarkan official AW-MAE.
5. Safety layer: distribution guard dan segment guard.
6. Optional consensus layer: V3/V4/V5 anchor, tetapi selalu dipisah dari pure model.

### 12.2 Validation strategy

Gunakan kombinasi:

- Rolling temporal folds.
- Multi-fold modern seperti V8.
- Latest holdout seperti V5.
- Segment report pada semua fold.
- Bootstrap confidence interval untuk delta kecil.

Jangan memilih final dari satu latest-window saja.

### 12.3 Metric authority

Tetapkan satu metric utama:

- Jika official adalah AW-MAE power 1.3, semua tuning utama memakai 1.3.
- Power 1.5 boleh menjadi stress metric.
- Report harus selalu menampilkan weighted dan unweighted.

### 12.4 V5 sebagai baseline anchor, bukan fallback buta

V5 layak menjadi baseline. Namun pipeline baru tidak boleh hanya fallback ke V5 tanpa memberi informasi. Simpan:

- Pure new model output.
- New model plus calibration.
- New model plus V5 anchor.
- New model plus consensus.

### 12.5 Probabilistic anchor

Jangan memakai skor diskret V5 sebagai satu-satunya anchor. Simpan lambda/probability dari V5 agar correction tidak kehilangan informasi.

### 12.6 Gating selektif

Gating harus benar-benar menahan sebagian prediksi. Kandidat gating:

- Agreement outcome antara anchor dan recent expert.
- Margin confidence outcome.
- Entropy probability score matrix.
- Delta lambda magnitude.
- Segment-specific reliability.
- Historical density untuk team/opponent.

Jika gate_share selalu 1.0, berarti gate tidak bekerja.

### 12.7 Calibration layer

Tambahkan calibration khusus:

- Draw calibration.
- Low-score calibration.
- High-score tail suppression/allowance.
- Outcome probability calibration.
- Segment-specific calibration.

### 12.8 Experiment registry

Setiap eksperimen wajib mencatat:

- Version name.
- File output.
- Source code path.
- Metric formula dan power.
- Weighted/unweighted flag.
- Folds.
- Feature set hash/list.
- Dependency aktif.
- Seed.
- Runtime.
- Fallback status.
- Apakah submission identik dengan baseline.

## 13. Eksperimen Lanjutan yang Disarankan

### 13.1 V5 pure tanpa static consensus

- Hipotesis: V5 model murni mungkin berbeda signifikan dari final consensus dan perlu dievaluasi terpisah.
- Perubahan minimal: disable `USE_STATIC_SUBMISSION_CONSENSUS`.
- Risiko: local score bisa turun karena prior V3/V4 hilang.
- Metrik sukses: pure V5 tidak jauh lebih buruk dari consensus dan punya distribusi lebih sehat.
- Stop jika: pure V5 kalah jelas di semua folds dan segment utama.

### 13.2 V5 dengan multi-fold validation

- Hipotesis: konfigurasi V5 yang dipilih latest-window belum tentu optimal lintas fold.
- Perubahan minimal: aktifkan/implementasikan multi-fold path yang sudah ada skeleton-nya.
- Risiko: runtime naik dan hasil tuning berubah.
- Metrik sukses: average fold AW-MAE membaik dan variance antar fold turun.
- Stop jika: improvement hanya muncul pada satu fold.

### 13.3 Static interactions benar-benar dimasukkan

- Hipotesis: fitur static interactions yang sekarang dropped dapat memberi sinyal tambahan.
- Perubahan minimal: ubah feature selector agar fitur static masuk pada eksperimen khusus.
- Risiko: overfit dan leakage proxy era.
- Metrik sukses: improvement fold-level konsisten tanpa distribution collapse.
- Stop jika: gain hanya di latest-window.

### 13.4 V8 tanpa fallback untuk diagnosis

- Hipotesis: melihat output hybrid non-fallback akan memperjelas jenis prediksi yang berubah.
- Perubahan minimal: buat mode diagnostic no-fallback, bukan untuk final submission.
- Risiko: submission buruk jika dipakai final.
- Metrik sukses: ditemukan segment di mana hybrid konsisten menang.
- Stop jika: hybrid tidak menang di segment mana pun.

### 13.5 V8 dengan gate selektif

- Hipotesis: correction hanya berguna pada subset match tertentu.
- Perubahan minimal: naikkan gate threshold atau tambahkan confidence-based gate.
- Risiko: gate terlalu ketat sehingga tidak ada perubahan.
- Metrik sukses: gate_share 10%-60% dan delta AW-MAE positif.
- Stop jika: gate_share tetap 1.0 atau 0.0.

### 13.6 Recent expert per segment

- Hipotesis: recent expert tidak cocok global, tetapi berguna untuk major tournament/friendly/qualifier tertentu.
- Perubahan minimal: train/evaluate recent correction per segment.
- Risiko: sample size kecil.
- Metrik sukses: segment tertentu menang tanpa menurunkan global score.
- Stop jika: segment gain tidak stabil antar fold.

### 13.7 Calibration-only layer di atas V5

- Hipotesis: V5 lebih butuh calibration daripada model baru.
- Perubahan minimal: ambil V5 lambda/skor, tuning draw/low/high/outcome calibration.
- Risiko: overfit ke validation.
- Metrik sukses: distribution lebih sehat dan AW-MAE membaik.
- Stop jika: exact naik tetapi outcome turun besar.

### 13.8 Outcome-first reranker

- Hipotesis: outcome classifier punya sinyal yang belum dimanfaatkan.
- Perubahan minimal: score candidates di-rerank agar konsisten dengan outcome probability.
- Risiko: exact score turun.
- Metrik sukses: outcome accuracy naik tanpa AW-MAE naik.
- Stop jika: draw distribution collapse.

### 13.9 Poisson vs Negative Binomial

- Hipotesis: goal distribution overdispersed, Poisson terlalu kaku.
- Perubahan minimal: implement negative-binomial probability matrix atau variance inflation.
- Risiko: tail high-score meningkat.
- Metrik sukses: high-goal calibration membaik tanpa tail warning.
- Stop jika: score >=5 share melewati threshold.

### 13.10 Ensemble V5 pure plus recent lambda dengan OOF meta-model

- Hipotesis: meta-model OOF bisa belajar kapan recent expert mengalahkan V5.
- Perubahan minimal: generate OOF predictions dari V5 pure dan recent expert, train small meta-model.
- Risiko: overfit meta-model.
- Metrik sukses: multi-fold improvement melewati threshold dan stable.
- Stop jika: meta-model hanya menang pada fold terakhir.

### 13.11 Distribution constrained ERM

- Hipotesis: ERM row-wise perlu constraint global agar tidak collapse ke skor umum.
- Perubahan minimal: tambah penalty atau postprocess untuk top-score share, draw share, tail share.
- Risiko: local optimum sulit.
- Metrik sukses: AW-MAE membaik atau sama, distribusi lebih dekat validation/test prior.
- Stop jika: constraint memperburuk exact/outcome.

### 13.12 Hyperparameter search dengan nested validation atau OOF

- Hipotesis: tuning saat ini terlalu dekat ke validation.
- Perubahan minimal: outer folds untuk evaluasi, inner folds untuk tuning.
- Risiko: runtime tinggi.
- Metrik sukses: delta lebih terpercaya dan variance turun.
- Stop jika: biaya runtime tidak sebanding dengan insight.

## 14. Kesimpulan Praktis

### Komponen V5 yang harus dipertahankan

- Ensemble Poisson multi-model.
- Split team/opp blending.
- ERM score selection berbasis loss tensor.
- Draw dan low-score calibration.
- Outcome classifier sebagai sumber sinyal tambahan.
- Guard terhadap stacking.
- Config/report JSON lengkap.

### Komponen V5 yang harus diperbaiki

- Latest-window tunggal harus diganti atau dilengkapi multi-fold.
- Static consensus harus dipisah dari pure model.
- Static interactions yang dropped harus dirapikan.
- Dependency optional harus dikunci.
- Segment diagnostics harus ditambahkan.
- High-score tail warning harus menjadi hard diagnostic.

### Komponen V8 yang harus dipertahankan

- Anchor-safe philosophy.
- Multi-fold modern validation.
- Safety thresholds.
- Anti-score-collapse diagnostics.
- Segment diagnostics.
- Fallback sebagai final guard, dengan label yang jelas.

### Komponen V8 yang harus dibuang atau ditulis ulang

- Naming V6/V8 yang tidak konsisten.
- Cache version v6 pada pipeline v8.
- Gating yang bisa berakhir selalu aktif.
- Anchor berbasis skor diskret saja.
- Correction yang terlalu konservatif tanpa segment learning.
- Default model diversity yang lebih sempit dari V5.

### Desain pipeline baru yang paling masuk akal

Pipeline baru sebaiknya bukan "V9 lebih kompleks", tetapi "V5 pure yang divalidasi ulang dengan guardrail V8". Rancangannya:

1. Bangun V5 pure tanpa consensus sebagai baseline resmi.
2. Generate OOF predictions multi-fold untuk base models.
3. Tuning blend/calibration di OOF, bukan satu latest-window.
4. Simpan lambda/probability, bukan hanya skor diskret.
5. Tambahkan outcome-aware reranking.
6. Tambahkan segment-aware calibration.
7. Tambahkan optional V5/V3/V4 consensus sebagai layer terpisah.
8. Final decision memakai safety guard V8 dan bootstrap delta.

Rekomendasi final:

- Gunakan V5 sebagai baseline kompetitif.
- Ambil validasi, safety, dan diagnostics dari V8.
- Jangan menganggap V8 sebagai improvement final karena outputnya identik dengan V5.
- Fokus eksperimen berikutnya pada metric consistency, pure-vs-consensus ablation, dan probabilistic correction yang tidak kehilangan lambda V5.


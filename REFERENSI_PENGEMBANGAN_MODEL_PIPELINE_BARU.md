# Referensi Pengembangan Model Pipeline Baru

Audit ini membandingkan `src/model_pipeline_v5.py` dan `src/model_pipeline_v8_anchor_safe.py` berdasarkan kode, config, validation report, submission, evaluator lokal, dan feature engineering yang tersedia di repo. Fokus audit adalah kelebihan, kekurangan detail, risiko metodologis, risiko validasi, risiko overfitting, risiko leakage, serta arah desain untuk pipeline baru.

## 1. Executive Summary

1. V5 adalah pipeline yang lebih produktif secara submission karena menghasilkan file final yang dipakai sebagai baseline kuat. V8 Anchor Safe lebih defensif dan lebih rapi dari sisi safety, tetapi pada run ini tidak menghasilkan prediksi baru karena fallback ke V5.
2. V5 lebih stabil secara praktis karena memakai static feature surface dan ensemble yang kaya, tetapi validasinya lebih rapuh karena hanya memakai `latest_window`.
3. V8 lebih inovatif dari sisi metodologi: multi-fold modern validation, anchor-based bounded correction, anti-score-collapse guard, dan segment diagnostics. Namun inovasi ini tidak cukup kuat untuk melewati safety threshold.
4. Bukti final: `submission_v5.csv` dan `submission_v8_anchor_safe.csv` memiliki SHA256 yang sama (`B8A7CD5495597E74C879BD2E3C32F38959B43A4E6CAFC753F93A46A7661E8724`) dan sama-sama berisi 42.422 baris. Artinya final V8 identik dengan V5.
5. Alasan V8 tidak memberi peningkatan final adalah `fallback_to_v5=True` dengan alasan `weighted_awmae_improvement_too_small`. Hybrid hanya membaik sekitar 0.00024 dari anchor pada power 1.5, lebih kecil dari `MIN_AWMAE_IMPROVEMENT=0.001`.
6. V5 local improvement terhadap V4 sangat tipis: V4 local AW-MAE 2.54271, V5 local AW-MAE 2.54124, delta -0.00147. Ini menandakan gain praktis kecil dan perlu validasi lebih kuat sebelum diklaim robust.
7. Ada mismatch metrik penting: V5 memakai legacy AW-MAE power 1.3, sedangkan V8 tuning utama memakai power 1.5. Perbandingan langsung tanpa menyamakan power dapat menyesatkan.
8. Risiko leakage terbesar bukan pada rolling/Elo core features, karena fitur dihitung sebelum match lalu history diupdate setelah skor tersedia. Risiko yang lebih perlu dicek adalah context target encoding yang memakai KFold acak, bukan time-aware fold, sehingga train validation temporal bisa menerima statistik kategori dari masa depan.
9. Untuk pipeline baru, desain terbaik adalah menggabungkan model diversity dan calibration V5 dengan multi-fold validation, safety guard, dan diagnostics V8. Namun pipeline baru harus memisahkan mode "pure model" dari mode "submission consensus".
10. Komponen yang harus dipertahankan: AW-MAE-aware ERM, tournament weighting, split team/opp modeling, distribution diagnostics, segment report, dan safety checklist. Komponen yang perlu ditulis ulang: gating V8, traceability versi/cache, dan target encoding agar time-aware.

## 2. Ringkasan Arsitektur V5

V5 adalah static one-shot pipeline untuk prediksi skor. Ia membaca `train_final.csv` dan `test_final.csv`, menggabungkan metadata `date` dan `tournament` dari raw train/test, membuat `tournament_weight`, menambahkan static interactions, memilih fitur numerik, melatih beberapa model Poisson/regression untuk target `team_goals` dan `opp_goals`, lalu melakukan tuning ensemble dan ERM score selection.

### Flow data loading

Fungsi `load_data` membaca:

- `dataset/train_final.csv`
- `dataset/test_final.csv`
- `dataset/train.csv` untuk `Id`, `date`, `tournament`
- `dataset/test.csv` untuk `Id`, `date`, `tournament`

Setelah merge, `date` dikonversi menjadi datetime. `tournament_weight` dipetakan dari nama turnamen. Kolom `train_weight` dan `metric_weight` sama-sama diisi dari `tournament_weight`.

### Feature selection

V5 punya `FEATURE_PROFILE = "v4_stable_surface"`. Dalam mode ini, fitur yang dipakai hanya kolom numerik yang berakhiran `_feat` atau `_ctx`. Ini membuat fitur static interactions yang baru dibuat dengan suffix `_static` tidak masuk model. Dari config V5, feature count adalah 44. Kolom seperti `tournament_weight`, `abs_elo_diff_static`, `is_balanced_match_static`, `elo_diff_x_tournament_weight_static`, `year_static`, dan `is_major_tournament_static` masuk daftar dropped.

Implikasinya: V5 tampak membuat static interactions, tetapi profile stabilnya secara sengaja membatasi surface ke fitur core/context lama. Ini membantu stabilitas, tetapi juga menyembunyikan potensi fitur baru.

### Static interactions

`add_static_interactions` membuat fitur berbasis:

- Elo difference dan absolute Elo difference.
- Favorite/underdog indicator.
- Elo sum/ratio.
- Rank difference jika ada.
- Neutral x Elo.
- Year/month.
- Friendly/major tournament indicator.

Namun karena `FEATURE_PROFILE = "v4_stable_surface"`, sebagian besar fitur ini tidak dipakai dalam V5 run yang tercatat.

### Tournament weighting

V5 memakai peta bobot turnamen. Contoh:

- FIFA World Cup: 2.00
- AFC Asian Cup, AFCON, Copa America, UEFA Euro: 1.80
- World Cup qualification: 1.50
- Friendly: 0.96
- Default: 1.20

Bobot ini dipakai sebagai training weight dan metric weight. Ini masuk akal karena kompetisi besar dan qualifier cenderung lebih relevan untuk objektif.

### Validation strategy

V5 memakai `VALIDATION_MODE = "latest_window"` dengan `VALIDATION_FRACTION = 0.12`. Dari config:

- Full train rows: 78.772
- Train window: 69.319
- Validation rows: 9.453

Strategi ini cepat dan sederhana, tetapi hanya menguji satu jendela waktu akhir. Jika karakter test berbeda dari latest validation, estimasi performa bisa bias.

### Model families

V5 memakai base model:

- `xgb_poisson`
- `sk_hgb_poisson`
- `cat_poisson` jika CatBoost tersedia
- `lgb_poisson` jika LightGBM tersedia
- `xgb_pseudohuber` opsional, tetapi disabled pada run ini

Pada config V5, XGBoost, HGB, LightGBM, CatBoost, dan Optuna tersedia. Model diversity ini salah satu kekuatan V5.

### Ensemble dan blending

V5 membuat prediksi lambda untuk team dan opponent secara terpisah. Tuning Optuna memilih:

- team weights
- opponent weights
- team/opp scale
- team/opp bias
- draw boost
- low-score boost
- outcome blend alpha
- max goals

Run V5 memilih `ensemble_mode = weighted_blend`, bukan stacking. Bobot final:

- Team weights: XGB 0.2966, HGB 0.4495, CatBoost 0.1205, LGB 0.1333
- Opp weights: XGB 0.0429, HGB 0.6573, CatBoost 0.1979, LGB 0.1019

Bobot team/opp yang berbeda adalah keputusan baik karena distribusi gol team dan opponent tidak harus simetris.

### Outcome classifier

V5 melatih classifier outcome W/D/L (`cat_outcome` jika tersedia, fallback XGB). Pada validation:

- Outcome classifier accuracy: 0.60605
- Log loss: 0.86196
- Final selected score outcome accuracy: 0.59230
- Outcome blend alpha: 0.07379

Classifier memberi sinyal outcome yang lebih baik daripada score selection akhir, tetapi alpha kecil membuat efeknya terbatas.

### ERM score selection

V5 memakai Poisson score probability matrix, loss tensor AW-MAE, dan ERM untuk memilih pasangan skor integer yang meminimalkan expected loss. Ini penting karena objektif kompetisi bukan MSE lambda, melainkan skor diskret dengan penalty exact, outcome, dan goal-diff.

### Draw dan low-score calibration

Optuna memilih:

- `draw_boost = 1.03924`
- `low_score_boost = 0.95571`
- `max_goals = 10`

Draw sedikit dinaikkan, sedangkan low-score cells sedikit diturunkan. Ini menarik karena prediksi populer V5 sangat terkonsentrasi pada 1-2, 2-1, dan 1-1.

### Static submission consensus V3/V4

Fungsi `apply_static_submission_consensus` mengganti output model V5 dengan rule:

- Pakai V4 jika V3 dan V4 setuju outcome W/D/L.
- Jika tidak setuju, pakai V3.

Pada run V5:

- Consensus applied: true
- Rows from V4: 40.577
- Rows from V3: 1.845
- Agreement rate: 95.65%

Ini memberi prior stabilitas, tetapi membuat final V5 bukan lagi output murni model V5.

### Reporting dan config output

V5 menulis:

- `dataset/submission_v5.csv`
- `dataset/submission_v5_config.json`
- `dataset/submission_v5_validation_report.txt`

Config menyimpan fitur, bobot ensemble, metrik validation, prediction distribution, availability dependency, dan local reporting.

Fungsi penting:

- `load_data`: membaca final features, raw metadata, tournament weights, dan memilih feature columns.
- `validation_predictions`: membuat latest-window split dan prediksi validasi dari base models.
- `choose_best_config`: memilih antara weighted blend dan stacking dengan Optuna/fallback grid.
- `final_predictions`: melatih final models full train dan menghasilkan lambda final.
- `apply_static_submission_consensus`: mengganti final prediction dengan consensus V3/V4.
- `write_outputs`: menulis config dan report.

## 3. Kelebihan V5

### Static feature surface relatif stabil

Dengan `v4_stable_surface`, V5 membatasi fitur ke `_feat` dan `_ctx`. Ini mengurangi blast radius dari eksperimen static interaction baru. Dalam kompetisi, stabilitas sering lebih berharga daripada fitur tambahan yang belum tervalidasi.

### Ensemble model cukup kaya

V5 menggabungkan XGBoost Poisson, sklearn HistGradientBoosting Poisson, CatBoost Poisson, dan LightGBM Poisson. Kombinasi ini bagus karena tiap model punya bias berbeda. HGB terlihat dominan terutama untuk opponent prediction, sedangkan XGB tetap memberi kontribusi kuat untuk team prediction.

### Split team/opp weights membantu asimetri

Team dan opponent weights dituning terpisah. Ini keputusan tepat karena fitur team/opponent meski mirrored tetap bisa punya distribusi noise berbeda. Opponent prediction pada V5 sangat mengandalkan HGB, sedangkan team prediction lebih menyebar.

### Optuna mengoptimasi objektif yang dekat dengan target

Tuning tidak hanya mengejar lambda error, tetapi langsung mengejar AW-MAE lewat ERM result. Ini lebih selaras dengan leaderboard dibanding melatih regressor lalu rounding sederhana.

### Guard terhadap stacking mengurangi overfit

Stacking hanya dipilih jika menang minimal 0.04 AW-MAE dan outcome tidak turun lebih dari 0.005. Karena validation meta sample kecil, guard ini masuk akal. Pada run ini weighted blend dipilih, yang menunjukkan pipeline tidak otomatis memilih opsi paling kompleks.

### Outcome classifier sebagai soft calibration

Classifier outcome memberi informasi W/D/L yang berbeda dari model skor. Alpha kecil menjaga agar sinyal classifier tidak merusak score distribution, tetapi tetap memberi dorongan probabilistik.

### Static consensus memberi prior stabilitas

Consensus V3/V4 memanfaatkan dua submission sebelumnya. Rule ini sangat defensif: jika dua versi sepakat outcome, pakai V4; jika tidak, pakai V3 sebagai outcome-stability prior. Hasilnya final distribution cukup stabil.

### Reporting cukup lengkap

V5 report mencakup validation metrics, final distribution, lambda percentiles, common scores, local metrics, dependency availability, feature columns, dan dropped columns. Ini membuat eksperimen relatif mudah diaudit.

## 4. Kekurangan V5

### Latest-window validation terlalu sempit

Satu holdout terakhir 12% tidak cukup untuk menguji robustitas lintas era. Jika validation window kebetulan cocok dengan Optuna/calibration, hasil bisa terlihat baik tetapi tidak tahan terhadap distribusi test yang berbeda.

### Static consensus mengaburkan kontribusi model

Final V5 bukan prediksi murni model V5 karena output akhir diganti oleh V3/V4 consensus. Ini membuat sulit menjawab apakah improvement berasal dari training model, calibration, atau prior submission lama. Untuk pengembangan pipeline baru, harus ada dua mode:

- `pure_model_submission`
- `consensus_submission`

### Static interactions dibuat tetapi banyak tidak dipakai

Config V5 menunjukkan static interactions masuk dropped columns karena tidak berakhiran `_feat` atau `_ctx`. Ini menimbulkan dua masalah:

- Pembaca kode bisa mengira fitur tersebut dipakai padahal tidak.
- Potensi fitur seperti tournament_weight, year, month, major tournament, dan Elo interaction tidak diuji secara adil.

### Potensi overfitting pada validation window

V5 melakukan banyak tuning di validation yang sama:

- Model blend weights.
- Scale/bias.
- Draw boost.
- Low-score boost.
- Outcome blend alpha.
- Max goals.
- Stacking comparison.

Tanpa nested validation atau multi-fold OOF, improvement kecil berpotensi noise.

### Outcome classifier belum dimanfaatkan optimal

Outcome classifier accuracy 60.61% lebih tinggi daripada final score outcome accuracy 59.23%. Tetapi alpha hanya 0.07379. Ada peluang untuk outcome-aware reranking yang lebih eksplisit, misalnya memilih skor terbaik dalam outcome class yang dipercaya classifier.

### Tail score calibration perlu kontrol

V5 report memberi warning: final predictions dengan salah satu sisi >=5 adalah 3.02%, sedikit di atas threshold warning 3%. Ini bukan bencana, tetapi menunjukkan high-score tail perlu guard yang lebih eksplisit.

### Local gain terhadap V4 sangat tipis

Local V5 AW-MAE 2.54124 vs V4 2.54271, delta -0.00147. Gain sekecil ini mudah hilang jika validation split berubah. Pipeline baru perlu membuktikan improvement di beberapa fold, bukan hanya final local report.

### Stacking ada tetapi tidak memberikan nilai

Stacking implementation menambah kompleksitas, tetapi tidak dipilih. Ini bukan kesalahan, tetapi tanda bahwa meta-model belum cukup stabil atau meta-features belum cukup kaya.

### Reproducibility bergantung optional dependency

CatBoost, LightGBM, dan Optuna optional. Jika environment berbeda, model set dan tuner bisa berubah. Pipeline baru perlu mencatat dependency lock dan fallback path dengan jelas.

### Segment diagnostics kurang kuat

V5 tidak memberi segment summary yang setara dengan V8 untuk major tournaments, friendlies, qualifiers, neutral, non-neutral. Ini membuat blind spot pada jenis pertandingan yang berbeda.

## 5. Ringkasan Arsitektur V8 Anchor Safe

V8 Anchor Safe adalah pipeline `v5_centered_bounded_selective_correction`. Tujuan desainnya bukan mengganti V5 secara total, melainkan membuat koreksi kecil di sekitar anchor V5/static. Jika koreksi tidak jelas lebih baik pada train-only time validation, final submission fallback ke V5.

### Anchor V5 dan recent correction

V8 membangun anchor prediction pada fold validasi menggunakan model Poisson XGB dan HGB saja, bukan full V5 consensus. Lalu recent-era models dilatih pada cutoff seperti 1990 dan 2000. Recent predictions digabung dengan preset weights, lalu dipakai untuk mengoreksi anchor.

### Correction dibatasi

Fungsi `apply_bounded_correction` melakukan:

- Convert anchor skor diskret menjadi lambda dengan `anchor_offset`.
- Hitung delta recent lambda minus anchor lambda.
- Clip delta dengan `cap_team` dan `cap_opp`.
- Kalikan delta dengan `beta_team` dan `beta_opp`.
- Terapkan gate.

Config terpilih:

- recent weights: 1990 = 0.8, 2000 = 0.2, 2002 = 0.0
- beta team/opp = 0.35
- cap team/opp = 0.25
- anchor_offset = 0.3
- gate_threshold = 0.0
- max_goals = 10
- draw_boost = 1.0
- low_score_boost = 1.0

### Multi-fold modern validation

V8 memakai tiga fold:

- `fold_2003_2005`: train_end 2002-12-31, valid 2003-01-01 sampai 2005-12-31, weight 0.20
- `fold_2006_2008`: train_end 2005-12-31, valid 2006-01-01 sampai 2008-12-31, weight 0.30
- `fold_2009_2011`: train_end 2008-12-31, valid 2009-01-01 sampai 2011-08-04, weight 0.50

Ini lebih kuat daripada latest-window tunggal karena menguji beberapa periode modern.

### Safety constraints

V8 hanya boleh mengganti anchor jika lolos guard:

- Improvement AW-MAE minimal `MIN_AWMAE_IMPROVEMENT = 0.001`.
- Outcome drop maksimal 0.003.
- Exact drop maksimal 0.005.
- Common low score share tidak naik lebih dari 0.03.
- Top 3 score share tidak naik lebih dari 0.03.
- Draw share shift maksimal 0.03.
- Avg total goals drop maksimal 0.10.

Pada run ini safety gagal karena `weighted_awmae_improvement_too_small`.

### Optional models dimatikan

V8 config:

- `ENABLE_LGB = False`
- `ENABLE_CATBOOST = False`
- Active base models: XGB Poisson dan HGB Poisson

Ini mempercepat dan menstabilkan pipeline, tetapi mengurangi diversity dibanding V5.

### Cache

V8 memakai `dataset/cache_v6` dan `CACHE_VERSION = "v6_v5_centered_bounded_selective_correction_2026_04_30_a"`. Ini mempercepat eksperimen, tetapi naming `v6` pada pipeline V8 berisiko membingungkan traceability.

Fungsi penting:

- `load_data`: membaca final features, raw metadata tambahan, tournament weights, dan static interactions.
- `build_anchor_for_fold`: membuat anchor prediction untuk fold.
- `correction_gate`: menentukan baris yang boleh dikoreksi berdasarkan confidence/outcome/Elo/major signal.
- `apply_bounded_correction`: menerapkan correction cap/beta/offset/gate.
- `build_validation_artifacts`: membangun anchor dan recent predictions per fold.
- `evaluate_candidate`: mengevaluasi candidate config, metrics, distribution, penalties, dan safety.
- `tune_candidates`: mencari candidate terbaik dan candidate safe terbaik.
- `make_final_submission`: fallback ke V5 jika safety gagal, atau buat corrected submission jika lolos.
- `write_outputs`: menulis config dan report.

## 6. Kelebihan V8 Anchor Safe

### Lebih aman karena anchor-based

V8 tidak mengganti V5 secara agresif. Ia hanya mencoba koreksi kecil yang dibatasi. Ini desain yang baik jika baseline V5 sudah kuat dan improvement sulit.

### Multi-fold validation lebih robust

V8 mengevaluasi beberapa periode modern. Ini lebih dapat dipercaya daripada satu latest holdout karena memberi sinyal variance antar fold.

### Safety guard mencegah regresi

Fallback logic berhasil mencegah V8 menghasilkan submission yang belum terbukti lebih baik. Dalam konteks competition, ini keputusan konservatif yang baik.

### Anti-score-collapse diagnostics

V8 mengukur:

- common low score share
- top 1/top 3/top 5 score concentration
- draw share
- average total goals
- score >=5 share

Ini penting karena ERM AW-MAE bisa mendorong model terlalu sering memilih skor populer.

### Segment diagnostics lebih informatif

V8 report memiliki segment:

- major tournaments
- friendlies
- qualifiers
- neutral
- non-neutral

Ini membuka blind spot yang tidak terlihat di V5.

### Recent-era expert mencoba menangkap drift

Menggunakan cutoff 1990/2000/2002 adalah ide bagus untuk menangkap perubahan pola skor modern, aturan kompetisi, dan kualitas data yang berubah sepanjang waktu.

### Cap dan beta mengurangi prediksi liar

Correction tidak boleh melompat terlalu jauh dari anchor. Ini mengurangi risiko high variance dari recent-only models.

## 7. Kekurangan V8 Anchor Safe

### Final fallback berarti tidak ada improvement submission

Karena `fallback_to_v5=True`, final V8 adalah V5. Ini bukan pipeline baru yang menang; ini pipeline diagnosis/safety wrapper yang pada run ini memilih baseline.

### Improvement hybrid terlalu kecil

Validation power 1.5:

- Anchor weighted AW-MAE: 2.819948
- Recent-only weighted AW-MAE: 2.818830
- Hybrid weighted AW-MAE: 2.819705

Recent-only sebenarnya sedikit lebih baik dari anchor, tetapi hybrid hanya membaik 0.000244 dari anchor. Threshold minimum 0.001 tidak terpenuhi.

### Gate tidak selektif pada config terpilih

Config terpilih memakai `gate_threshold = 0.0`, sehingga `correction_gate` mengembalikan semua True. Fold summary menunjukkan gate share 1.000 di semua fold. Artinya mekanisme selective correction tidak benar-benar selektif pada run ini.

### Hybrid gagal mengambil manfaat recent-only

Recent-only weighted AW-MAE lebih baik dari anchor sekitar 0.001118, tetapi hybrid hanya mengambil sebagian kecil manfaat itu. Kemungkinan penyebab:

- Anchor score diskret + offset terlalu membatasi lambda.
- Cap 0.25 dan beta 0.35 terlalu konservatif.
- ERM setelah correction kembali memilih skor yang hampir sama dengan anchor.
- Correction tidak segment-aware.

### Anchor berbasis skor diskret membatasi probabilistik

V8 mengubah skor anchor diskret menjadi lambda dengan `anchor_offset`. Skor 1-2 menjadi lambda sekitar 1.3-2.3 sebelum correction. Ini kehilangan uncertainty asli dari model V5. Lebih baik anchor menggunakan probability matrix atau lambda V5 murni, bukan skor final consensus.

### AW-MAE power mismatch

V8 memakai `AWMAE_POWER = 1.5`, sedangkan evaluator lokal `evaluate_local.py` memakai default `nls_power = 1.3`. V8 memang melaporkan legacy power 1.3, tetapi tuning utama dilakukan pada 1.5. Ini membuat interpretasi "lebih baik" harus hati-hati.

### Naming dan traceability bermasalah

File bernama `model_pipeline_v8_anchor_safe.py`, tetapi report tertulis "V6 Validation Report" dan config memakai key seperti `local_v6_metrics_power_1_5`. Cache directory dan cache version juga memakai `v6`. Ini berisiko saat membandingkan eksperimen dan audit history.

### Model diversity lebih sempit daripada V5

V8 default hanya XGB dan HGB. LightGBM dan CatBoost disabled. Jika V5 mendapat stabilitas dari diversity model, V8 mungkin kehilangan sinyal penting.

### Fallback bisa menyembunyikan kegagalan eksperimen

Fallback bagus untuk submission safety, tetapi untuk riset ia bisa membuat output terlihat "tidak memburuk" padahal eksperimen correction gagal memberi nilai. Report harus selalu menonjolkan status fallback.

## 8. Perbandingan V5 vs V8

| Aspek | V5 | V8 Anchor Safe | Dampak | Pelajaran untuk pipeline baru |
|---|---|---|---|---|
| Validation strategy | Latest-window 12% | Multi-fold modern 2003-2011 | V5 cepat tetapi rapuh; V8 lebih robust | Pakai rolling/multi-fold plus latest holdout |
| Feature set | 44 fitur `_feat`/`_ctx` stabil | 58 fitur termasuk neutral, tournament, Elo interactions, year/month | V8 lebih kaya, V5 lebih terkendali | Buat ablation fitur, jangan campur semua tanpa bukti |
| Model diversity | XGB, HGB, CatBoost, LGB | XGB, HGB default | V5 lebih diverse | Pertahankan diversity, tetapi catat dependency |
| Ensemble strategy | Weighted blend team/opp + optional stacking | Anchor + bounded recent correction | V5 lebih langsung; V8 lebih defensif | Gabungkan OOF blend dengan correction yang selektif |
| Calibration | Draw boost, low-score boost, outcome alpha | Draw/low-score dalam candidate grid | Keduanya punya calibration | Pisahkan calibration layer dan validasi per segment |
| Safety guard | Guard stacking margin | Full fallback safety + distribution guard | V8 lebih aman | Safety guard wajib dipertahankan |
| Distribution diagnostics | Basic distribution dan warning score >=5 | Anti-collapse lengkap | V8 lebih informatif | Buat distribution dashboard standar |
| Segment diagnostics | Terbatas | Major/friendly/qualifier/neutral/non-neutral | V8 lebih baik | Segment diagnostics wajib |
| Reproducibility | Optional dependencies bisa mengubah hasil | Cache mempercepat, tetapi naming v6/v8 membingungkan | Keduanya punya risiko | Tambah experiment registry dan dependency lock |
| Runtime | Sekitar 2.74 menit pada report V5 | Sekitar 170.66 menit pada report V8 | V8 jauh lebih mahal | Cache boleh, tetapi naming dan invalidation harus ketat |
| Submission originality | Final dicampur V3/V4 consensus | Final identik V5 karena fallback | Keduanya tidak murni model baru | Pisahkan pure output dan consensus output |
| Risk of overfit | Tinggi pada single validation + Optuna | Lebih rendah, tetapi grid tetap tuned ke folds | V8 lebih defensif | Gunakan OOF/nested validation untuk tuning |
| Interpretability | Config detail baik, tetapi consensus mengaburkan | Diagnostics bagus, tetapi naming kacau | Keduanya perlu registry | Setiap submission harus punya lineage jelas |
| Final local performance | V5 legacy AW-MAE 2.54124 unweighted; weighted legacy 2.52887 di V8 config | Sama persis dengan V5 | V8 tidak menambah performa final | Jangan klaim improvement tanpa output berbeda dan validasi kuat |

## 9. Analisis Metrik dan Validasi

### AW-MAE formula

Evaluator lokal memakai:

- `mae = (abs(pred_team - true_team) + abs(pred_opp - true_opp)) / 2`
- penalty exact: `0.30 * (1 - exact)`
- penalty outcome: `0.25 * (1 - outcome_ok)`
- penalty goal difference: `0.15 * (1 - gd_ok)`
- multiplier 1.5 jika outcome salah
- hasil akhir dipangkatkan `nls_power`

V5 memakai formula yang sama dengan `NLS_POWER = 1.3`. V8 memakai `AWMAE_POWER = 1.5` untuk tuning utama dan `LEGACY_LOCAL_POWER = 1.3` untuk perbandingan.

### Dampak power 1.3 vs 1.5

Power lebih tinggi memperbesar hukuman untuk error besar. Dengan power 1.5, model cenderung lebih takut pada prediksi yang jauh meleset. Dengan power 1.3, penalty masih nonlinear tetapi lebih ringan. Jika pipeline A dituning pada 1.3 dan pipeline B pada 1.5, perbandingan bisa tidak fair.

Rekomendasi: pipeline baru harus menetapkan satu `METRIC_POWER` utama dan semua validation, tuning, reporting, local evaluator, dan loss tensor harus memakai nilai yang sama. Jika ingin melaporkan power alternatif, tandai sebagai secondary diagnostic.

### Weighted vs unweighted

Weighted AW-MAE memakai tournament weights. Ini penting karena kompetisi besar dan qualifier diberi bobot lebih tinggi. Namun evaluator lokal `evaluate_local.py` hanya menghitung unweighted score. V5 report menampilkan local AW-MAE unweighted, sedangkan V8 config menampilkan weighted dan unweighted. Ini harus diseragamkan.

Rekomendasi: selalu laporkan:

- validation weighted AW-MAE
- validation unweighted AW-MAE
- local weighted AW-MAE jika tournament metadata tersedia
- local unweighted AW-MAE
- delta terhadap baseline untuk semua metrik

### Local ground truth tidak boleh dipakai tuning

`test_ground_truth.csv` tersedia dan digunakan untuk local reporting. Ini hanya aman jika dipakai setelah model dipilih. Jika local ground truth ikut menentukan config, maka pipeline menjadi tidak valid untuk skenario kompetisi blind. Pada kode yang dibaca, V5 dan V8 memakai `test_ground_truth` untuk reporting output, bukan untuk tuning langsung. Namun keberadaan local reporting dekat dengan pipeline training tetap perlu disiplin eksperimen.

### Representativitas validation split

V5 latest-window mungkin cocok jika test period tepat setelah train dan distribusinya mirip. Tetapi ia tidak mengukur stabilitas antar periode. V8 multi-fold modern lebih baik untuk robustitas, meski validation end 2011-08-04 perlu dipastikan relevan dengan test period.

### Apakah improvement kecil meaningful?

Tidak selalu. Delta V5 terhadap V4 adalah -0.00147 legacy AW-MAE. Delta hybrid V8 terhadap anchor adalah sekitar -0.00024 power 1.5. Nilai sekecil ini bisa noise akibat split, dependency, random seed, atau tuning. Improvement baru layak dianggap meaningful jika:

- Konsisten di beberapa fold.
- Tidak menurunkan outcome/exact.
- Tidak membuat distribusi collapse.
- Tetap ada pada pure model output, bukan hanya consensus.
- Lebih besar dari threshold praktis, misalnya 0.003-0.005 AW-MAE tergantung variance fold.

### Metrik tambahan yang wajib masuk pipeline baru

- Calibration by total goals: 0-1, 2, 3, 4, 5+.
- Outcome confusion matrix: loss/draw/win.
- Exact score top-k hit rate dari probability matrix.
- Goal difference MAE.
- Segment by tournament type.
- Segment by neutral/non-neutral.
- Segment by Elo gap/rank gap.
- Distribution drift train/validation/test.
- Tail score share: side >=5 dan total goals >=6.
- Score concentration: top 1/top 3/top 5 share.

## 10. Analisis Risiko Leakage

| Area | Status | Analisis |
|---|---|---|
| Core Elo features | Aman dengan catatan | `feature_engineering_v2.py` menyimpan Elo sebelum pertandingan, lalu update hanya jika skor tersedia. Test rows punya target NaN sehingga tidak mengupdate Elo memakai test result. |
| Rolling form stats | Aman dengan catatan | Rolling stats dihitung dari `history` sebelum match, lalu history diupdate setelah skor row saat ini tersedia. Ini time-safe jika sort date/match_id benar. |
| H2H stats | Aman dengan catatan | H2H juga membaca history sebelum match dan update setelah skor tersedia. |
| Opponent mirrored features | Perlu dicek | Mirroring mengambil fitur opponent pada row match yang sama. Jika fitur opponent juga pre-match, aman. Perlu validasi untuk match group abnormal yang tidak tepat 2 rows. |
| Context target encoding | Berisiko | `fe_context.py` memakai KFold shuffle target encoding pada seluruh train. Untuk temporal validation, nilai TE pada validation-era train rows bisa mengandung statistik kategori dari masa depan relatif terhadap row tersebut. Test mapping dari seluruh train aman untuk blind test, tetapi validation score bisa optimistic. |
| Venue/confederation TE | Berisiko | Target adalah goal difference. KFold acak menghindari direct row leakage, tetapi bukan time-aware leakage. |
| V3/V4 static consensus | Perlu dicek | Bukan leakage jika V3/V4 hanya submission dari data train yang sama. Namun ia adalah external prior yang mengaburkan klaim model murni dan bisa memindahkan kesalahan lama. |
| `test_ground_truth.csv` | Perlu dicek | Kode memakai untuk local reporting. Tidak terlihat dipakai dalam tuning, tetapi proximity reporting dengan pipeline bisa memicu eksperimen manual yang overfit ke local test. |
| Cache V8 | Perlu dicek | Cache key memasukkan version, model, tag, rounds, params sebagian. Naming `cache_v6` pada V8 berisiko traceability. Pastikan feature set/hash data juga masuk cache key agar prediksi lama tidak salah pakai. |
| Validation artifacts V8 | Aman dengan catatan | Fold recent mask memakai `date <= train_end`; validation period dipisahkan. Anchor fold memakai `fold_train`. Ini time-safe secara split. |
| Raw date/tournament merge | Aman | Merge `Id`, `date`, `tournament` tidak memakai target. |
| Feature final train/test | Perlu dicek | Core features relatif time-safe. Context features perlu audit lebih detail terutama target encoding dan external socio/geography data timestamp. |

## 11. Analisis Feature Engineering

### Elo

Elo dihitung kronologis, disimpan sebelum match, dan diupdate setelah skor tersedia. Ada home advantage virtual +100 untuk expected score, tetapi tidak masuk Elo permanen. K-factor disesuaikan berdasarkan tournament dan confederation. Ini fitur kuat dan relatif aman.

Risiko:

- Sort order untuk match tanggal sama harus benar.
- Match group yang tidak tepat 2 row dilewati.
- Elo awal 1500 untuk semua team/gender bisa kasar untuk tim baru.

### Form

Rolling form memakai EWMA dengan half-life 90 hari dan window 10 match. Fitur mencakup points, goal difference, average goals for/against, win rate, dan days since last match. Ini relevan untuk performa terkini.

Risiko:

- Half-life 90 hari mungkin terlalu pendek untuk tim nasional yang jarang main.
- Missing early history perlu imputation yang konsisten.
- Form simple points dapat bias oleh kekuatan lawan.

### H2H

H2H memakai history pasangan team-opponent dengan window 5. Ini bisa berguna, tetapi untuk tim nasional banyak pasangan jarang bertemu sehingga coverage rendah dan noisy.

Rekomendasi:

- Laporkan missing rate H2H.
- Uji ablation H2H.
- Jangan terlalu percaya H2H untuk pair langka.

### Recent points dan goal difference EWMA

Fitur recent points dan GD EWMA bagus karena menangkap momentum dan kekuatan relatif. Derived diff team minus opponent membantu model tree menangkap gap performa.

Risiko:

- Tidak opponent-adjusted.
- Bisa overreact pada friendly atau pertandingan berat sebelah.

### Context features

Context mencakup:

- travel stress
- altitude shock
- temperature stress
- log GDP/population difference
- venue country target encoding
- confederation team target encoding
- venue country frequency

Fitur context dapat menangkap faktor eksternal yang tidak ada di Elo/form. Namun target encoding harus time-aware untuk validasi temporal.

### Tournament weight

V5 memakai tournament weight sebagai training/metric weight tetapi tidak sebagai fitur karena profile V5 membuangnya. V8 memasukkan `tournament_weight` sebagai feature. Ini perlu diuji:

- Sebagai weight saja.
- Sebagai feature saja.
- Sebagai keduanya.

### Neutral/rank/year/month

V8 memasukkan `neutral`, `year`, `month`, `is_friendly`, dan `is_major_tournament`. V5 tidak memakai fitur static tersebut karena profile. Ini kandidat ablation penting.

### Fitur yang dibuat tetapi tidak masuk model

Pada V5, fitur static interactions seperti `abs_elo_diff_static`, `is_balanced_match_static`, `elo_diff_x_tournament_weight_static`, `year_static`, dan `is_major_tournament_static` tidak masuk model. Ini harus dibersihkan atau dimasukkan secara eksplisit pada eksperimen.

### Fitur yang perlu distabilkan

- Target encoding context: ubah ke time-aware expanding encoding atau fold berdasarkan waktu.
- H2H: tambahkan fallback/smoothing kuat.
- Tournament/year/month: pastikan tidak menjadi proxy leakage period test.
- Rank features: pastikan ranking tersedia sebelum match date, bukan post-hoc.

## 12. Rekomendasi Pipeline Baru

### Prinsip desain

Pipeline baru sebaiknya menjadi gabungan konservatif dari V5 dan V8:

- V5 sebagai baseline dan source model diversity.
- V8 sebagai source validation discipline, safety guard, dan diagnostics.
- Jangan menjadikan fallback ke V5 sebagai bukti improvement.
- Pisahkan output pure model dan output consensus.
- Satu definisi metrik utama untuk semua tahap.

### Arsitektur yang disarankan

1. Feature layer:
   - Gunakan frozen chronological core features.
   - Rebuild context target encoding secara time-aware.
   - Buat feature manifest: nama fitur, asal, apakah time-safe, missing rate.

2. Validation layer:
   - Rolling folds modern.
   - Latest holdout.
   - Optional nested/OOF tuning untuk calibration.
   - Semua fold memakai metric power yang sama.

3. Base model layer:
   - XGB Poisson.
   - HGB Poisson.
   - LightGBM Poisson jika dependency tersedia.
   - CatBoost Poisson jika dependency tersedia.
   - Optional negative-binomial/quantile model sebagai eksperimen.

4. Probabilistic layer:
   - Simpan lambda dan score probability matrix.
   - Jangan hanya menyimpan skor diskret anchor.
   - Buat probability ensemble sebelum ERM.

5. Calibration layer:
   - Draw boost.
   - Low-score boost.
   - High-score tail dampener.
   - Outcome classifier/reranker.
   - Segment-aware calibration untuk friendly/qualifier/major/neutral.

6. Safety layer:
   - Distribution guard.
   - Segment regression guard.
   - Outcome/exact guard.
   - Minimum improvement threshold.
   - Fallback hanya untuk production submission, bukan untuk analisis eksperimen.

7. Reporting layer:
   - Experiment ID.
   - Version name konsisten.
   - Git hash jika tersedia.
   - Metric formula.
   - Dependency availability.
   - Feature list and dropped features.
   - Fold metrics.
   - Segment metrics.
   - Distribution diagnostics.
   - Leakage checklist.

### Gating baru

Gating V8 harus ditulis ulang agar benar-benar selektif. Kandidat gate:

- Correction hanya jika recent model dan anchor setuju outcome, tetapi recent confidence tinggi.
- Correction hanya jika anchor probability entropy tinggi.
- Correction hanya untuk segment yang recent expert terbukti unggul di validation.
- Correction magnitude proportional terhadap fold-level reliability.
- Gate threshold tidak boleh 0 kecuali mode diagnostic.

### Pisahkan consensus

Buat dua final output:

- `submission_new_pure.csv`: murni model baru.
- `submission_new_consensus.csv`: model baru + consensus/prior.

Report harus membandingkan keduanya. Jika consensus menang, tetap jelas bahwa gain berasal dari ensemble prior.

## 13. Eksperimen Lanjutan yang Disarankan

### 1. V5 pure tanpa static consensus

- Hipotesis: V5 model murni mungkin berbeda dari final consensus; perlu baseline bersih.
- Perubahan minimal: disable `USE_STATIC_SUBMISSION_CONSENSUS`.
- Risiko: local score bisa turun karena kehilangan prior V3/V4.
- Metrik sukses: pure V5 mendekati atau mengalahkan consensus di multi-fold dan local.
- Stop jika pure model konsisten lebih buruk >0.005 AW-MAE dan outcome turun >0.3%.

### 2. V5 dengan multi-fold validation

- Hipotesis: konfigurasi V5 yang menang latest-window belum tentu menang lintas fold.
- Perubahan minimal: aktifkan/implement multi-fold validation penuh untuk V5.
- Risiko: runtime naik dan Optuna tuning perlu disesuaikan.
- Metrik sukses: ranking config stabil lintas fold.
- Stop jika variance fold terlalu besar dan tidak ada config yang konsisten.

### 3. V5 dengan static interactions benar-benar dimasukkan

- Hipotesis: fitur static interactions bisa memperbaiki segment major/neutral/Elo gap.
- Perubahan minimal: ubah feature profile agar `_static` dan tournament features masuk.
- Risiko: overfit pada validation latest-window.
- Metrik sukses: improvement multi-fold tanpa distribution collapse.
- Stop jika fitur baru menaikkan concentration top3 atau menurunkan exact.

### 4. V8 tanpa fallback untuk diagnosis

- Hipotesis: corrected output punya pola error yang berguna meski belum safe.
- Perubahan minimal: generate diagnostic output tanpa fallback, dengan label jelas non-production.
- Risiko: bisa disalahanggap sebagai submission final.
- Metrik sukses: identifikasi segment tempat correction unggul.
- Stop jika semua segment kalah dari anchor.

### 5. V8 dengan selective gate threshold nyata

- Hipotesis: gate selektif bisa mengambil gain recent-only tanpa mengubah semua row.
- Perubahan minimal: hilangkan threshold 0 dari candidate safe atau beri penalty untuk gate_share terlalu tinggi.
- Risiko: terlalu sedikit row dikoreksi sehingga gain kecil.
- Metrik sukses: gate_share 10-50%, AW-MAE membaik >=0.001, outcome/exact tidak turun.
- Stop jika best config selalu gate 0 atau 1.

### 6. Recent expert per segment

- Hipotesis: recent expert lebih berguna di segment tertentu, bukan global.
- Perubahan minimal: train/evaluate recent correction untuk major, friendly, qualifier, neutral.
- Risiko: data segment kecil dan noisy.
- Metrik sukses: minimal satu segment besar membaik konsisten.
- Stop jika segment gains tidak stabil antar fold.

### 7. Calibration-only layer di atas V5

- Hipotesis: sebagian besar gain bisa datang dari score calibration, bukan model baru.
- Perubahan minimal: freeze V5 lambdas/probabilities, tune draw/low/high-tail/outcome rerank.
- Risiko: overfit calibration.
- Metrik sukses: AW-MAE membaik dengan distribution guard lolos.
- Stop jika calibration hanya mengubah score concentration tanpa gain.

### 8. Outcome-first reranker

- Hipotesis: classifier outcome yang lebih kuat bisa memperbaiki score choice.
- Perubahan minimal: pilih top score candidates dari probability matrix lalu rerank berdasarkan outcome probability.
- Risiko: exact score turun karena terlalu memaksa outcome.
- Metrik sukses: outcome naik tanpa exact turun lebih dari 0.2%.
- Stop jika AW-MAE memburuk meski outcome naik.

### 9. Poisson vs negative-binomial comparison

- Hipotesis: football goals overdispersed terhadap Poisson.
- Perubahan minimal: tambahkan negative-binomial atau dispersion calibration.
- Risiko: implementasi lebih kompleks dan tuning tidak stabil.
- Metrik sukses: tail calibration membaik dan AW-MAE turun.
- Stop jika high-score share makin liar.

### 10. Ensemble V5 pure + recent lambda dengan learned OOF meta-model

- Hipotesis: learned meta-model bisa menggabungkan anchor dan recent lebih baik daripada cap/beta manual.
- Perubahan minimal: buat OOF predictions untuk V5 pure dan recent experts, train ridge/XGB shallow meta.
- Risiko: meta-overfit.
- Metrik sukses: fold improvement konsisten dan selected features interpretable.
- Stop jika meta hanya menang pada satu fold.

### 11. Distribution constrained ERM

- Hipotesis: ERM perlu constraint agar tidak terlalu banyak memilih skor populer.
- Perubahan minimal: tambahkan penalty pada expected loss untuk score concentration/tail.
- Risiko: objective jadi tidak lagi row-independent.
- Metrik sukses: AW-MAE membaik atau sama dengan distribusi lebih sehat.
- Stop jika exact turun signifikan.

### 12. Hyperparameter search dengan nested validation atau OOF

- Hipotesis: sebagian gain saat ini adalah overfit validation.
- Perubahan minimal: outer folds untuk evaluasi, inner folds untuk tuning.
- Risiko: runtime tinggi.
- Metrik sukses: delta outer fold tetap positif.
- Stop jika runtime tidak sebanding dan hasil ranking config tidak berubah.

## 14. Kesimpulan Praktis

### Komponen V5 yang harus dipertahankan

- AW-MAE-aware ERM score selection.
- Split team/opp modeling dan weighting.
- Model diversity XGB/HGB/CatBoost/LGB.
- Tournament weighting.
- Guard terhadap stacking.
- Reporting config lengkap.

### Komponen V5 yang harus diperbaiki

- Latest-window validation harus diganti/ditambah multi-fold.
- Static interactions harus jelas: dipakai atau dihapus.
- Consensus V3/V4 harus dipisah dari pure model.
- Outcome classifier perlu reranking yang lebih eksplisit.
- Tail score warning perlu guard, bukan sekadar report.

### Komponen V8 yang harus dipertahankan

- Multi-fold modern validation.
- Safety guard dan fallback untuk production.
- Anti-score-collapse diagnostics.
- Segment diagnostics.
- Recent-era expert sebagai ide eksperimen.
- Stability penalty antar fold.

### Komponen V8 yang harus dibuang atau ditulis ulang

- Naming V6/V8 yang tidak konsisten.
- Cache version dan directory yang tidak sesuai versi pipeline.
- Gate threshold 0 sebagai kandidat production.
- Anchor dari skor diskret sebagai representasi utama.
- Fallback yang membuat final output identik tanpa menonjolkan status eksperimen gagal.

### Desain pipeline baru paling masuk akal

Pipeline baru sebaiknya bernama eksplisit, misalnya `model_pipeline_v9_oof_calibrated.py`, dengan prinsip:

- OOF/multi-fold sebagai pusat desain.
- Pure probabilistic ensemble, bukan skor diskret anchor.
- Context target encoding time-aware.
- Calibration layer terpisah.
- Segment-aware diagnostics wajib.
- Production safety guard ala V8.
- Submission consensus optional dan dilaporkan terpisah.

## 15. Final Recommendation

Jangan membangun pipeline baru dengan cara langsung memodifikasi V8 Anchor Safe dan berharap fallback menghasilkan improvement. V8 berguna sebagai kerangka audit dan safety, bukan sebagai source model yang terbukti menang. Fondasi model yang lebih kuat masih ada di V5 karena diversity dan calibration-nya.

Langkah paling sehat:

1. Jadikan V5 pure tanpa consensus sebagai baseline bersih.
2. Re-evaluate V5 dengan multi-fold modern.
3. Perbaiki target encoding agar time-aware.
4. Bangun OOF ensemble yang menyimpan lambda/probability, bukan hanya skor.
5. Tambahkan calibration dan safety guard dari V8.
6. Baru setelah itu buat consensus submission sebagai layer terpisah.

Dengan cara ini, pipeline baru tidak hanya "aman karena fallback", tetapi benar-benar punya bukti improvement yang bisa diaudit.

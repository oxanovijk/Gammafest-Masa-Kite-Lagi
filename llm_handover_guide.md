# Football Match Prediction Strategy Summary (V12)
*Gunakan ringkasan ini sebagai konteks (prompt) untuk LLM lain agar mereka langsung paham state of the art dari model kita saat ini.*

---

## 1. Arsitektur Model Utama: "Two-Stage Hierarchical Soft-Cascade"
Model tidak langsung menebak 1 skor, melainkan memecah tugas menjadi dua tahap (menggunakan **LightGBM** dan **XGBoost** secara ensembel):

*   **Gender-Split Architecture:** Model dipisah 100% antara *Men* dan *Women* karena distribusi statistik gol yang sangat berbeda.
*   **Stage 1: The Outcome Specialist (3-Class)**
    Model dilatih khusus untuk memprediksi probabilitas *Outcome* (Menang, Seri, Kalah). Tujuannya murni untuk meminimalkan risiko terkena penalti 1.5x dari metrik AW-MAE jika tebakan *Outcome* salah.
*   **Stage 2: The Joint PMF Specialist (36-Class)**
    Model dilatih untuk memprediksi distribusi probabilitas matriks Exact Score 6x6 (dari 0-0 hingga 5-5). Model ini menangkap korelasi dinamis antar gol tim A dan tim B.
*   **Reconciliation (Soft-Cascade):** 
    Probabilitas dari Stage 2 (Exact Score) dikalibrasi ulang (*bucketed renormalization*) dengan menunggangi probabilitas dari Stage 1 (Outcome). Rumusnya: `P(Score) = P(Score|Outcome) * P(Outcome)`. Ini menjamin tebakan margin gol selaras dengan keyakinan model terhadap siapa yang akan menang/seri.
*   **Expected Risk Minimization (ERM) Decision Rule:**
    Bukan sekadar memilih skor dengan probabilitas tertinggi (*Argmax*), model menghitung **Expected Loss** untuk setiap kelas 36 skor berdasarkan fungsi penalti AW-MAE (memperhitungkan bobot asimetris untuk tebakan *Exact*, *Outcome*, dan *Goal Difference*), lalu memilih skor yang penaltinya paling minimal.
*   **Temperature Scaling:** 
    Distribusi skor dihaluskan menggunakan suhu $T=1.1$ (Pria) dan $T=1.2$ (Wanita) untuk mencegah model terlalu *overconfident* pada skor langka.

## 2. Feature Engineering (V5)
Total lebih dari 40 fitur diekstraksi tanpa *data leakage*. Fitur utama meliputi:

### A. Core Ratings
*   **Dynamic K-Factor Elo Rating:** 
    Bobot pertandingan (*K-factor*) menyesuaikan jenis turnamen (World Cup bobot 60, Friendly bobot 20), kekuatan konfederasi benua, dan besaran margin gol.
*   **Explicit Home Advantage:** Ditambahkan +35 poin Elo secara eksplisit untuk tim *Home* (menambah probabilitas menang ~5%).
*   **Pi-Ratings:** Sistem rating dinamis (mirip Elo) namun lebih difokuskan untuk menghitung *expected goal difference* dan memisahkan kekuatan *Home* vs *Away* secara independen (dengan *learning rate* 0.035).

### B. Temporal & Rolling Stats (EWMA)
*   **Exponentially Weighted Moving Average (EWMA):** 
    Statistik masa lalu dibobot menggunakan peluruhan eksponensial dengan *half-life* 90 hari. Fitur yang dihitung: *Average Points (pts), Goal Difference (gd), Goals For (gf), Goals Against (ga),* dan *Win Rate (wr)* untuk 5 dan 10 laga terakhir.
*   **Days Rest Difference:** Selisih hari istirahat sejak pertandingan terakhir tim A dan B.
*   **H2H (Head-to-Head) EWMA:** Agregat riwayat pertemuan kedua tim di masa lalu (H2H GD & Pts).

### C. Contextual & Tournament Prestige
*   **Tournament Tier (Ordinal 1-5):**
    Turnamen di-kategorikan berdasarkan gengsi untuk membantu model memahami pola "Pesta Gol" vs "Pertahanan Rapat".
    * Tier 5: World Cup / Olympic
    * Tier 4: Continental Cups (Euro, Copa America, dll)
    * Tier 3: Qualifiers / Nations League
    * Tier 2: Regional Cups (AFF, dll)
    * Tier 1: Friendlies
*   **Host Status:** Kolom biner eksplisit `is_home`, `is_away`, dan `is_neutral`.

### D. Derived Meta-Features
*   **Attack-Defense Mismatch:** `Team A (GF) - Team B (GA)` untuk mengukur dominasi serangan.
*   **Goal Volatility:** `GF + GA` untuk melihat apakah suatu tim bermain dengan gaya "Chaos" (sering mencetak gol dan kebobolan sekaligus).
*   **Non-linear Elo:** `elo_diff_sq` (selisih Elo dikuadratkan tapi mempertahankan *sign* positif/negatif).

### E. Socio-Economic & Geo-Spatial (Data Eksternal)
*   `log_gdp_diff` & `log_pop_diff`: Selisih GDP per kapita dan Populasi.
*   `travel_diff`: Selisih jarak tempuh dari negara asal ke *venue*.
*   `altitude`: Ketinggian stadion (*altitude shock*).
*   `temperature`: Suhu di tempat pertandingan.

### F. Target Encoding
*   Target encoding (dengan algoritma *smoothing*) pada `tournament` dan `confederation_team` untuk memberikan patokan rata-rata gol bawaan.

## 3. Current Performance (Local Validation)
*Test Set adalah data kronologis yang 100% meniru pola Private Leaderboard Kaggle.*

*   **Global AW-MAE:** 2.502
*   **Outcome Accuracy:** 59.88%
*   **Exact Score Accuracy:** 9.38%

**Analisis Lanjutan:**
Berkat penerapan *Hierarchical Cascading* dan *Tournament Tier*, akurasi menebak Outcome naik signifikan (62.17% di kompetisi Kualifikasi/Tier 3), karena model berhasil belajar kapan tim kuat akan merotasi pemain (Tier 1/Friendly) dan kapan tim kuat akan bermain serius (*tight defense*) untuk menang tipis (Tier 5/World Cup).

# Laporan Analisis Visualisasi Data (EDA)

Berikut adalah bedah wawasan (*insight*) dari ke-9 grafik yang baru saja kita ekstrak menggunakan `visualisasi.py`.

---

## 1. Plot 1 — Distribusi Target (Goals & Outcome)
- **Apa yang terlihat:** Distribusi gol baik untuk *Team* maupun *Opponent* sangat condong ke kanan (*right-skewed*), menyerupai distribusi **Poisson** murni. Mayoritas pertandingan berakhir dengan 0, 1, atau 2 gol. Pada diagram *pie*, proporsi kemenangan (W) dan kekalahan (L) tidak 50-50 murni, melainkan ada sedikit bias (kemungkinan efek kandang/tandang atau *seeding*). Hasil seri (D) menempati porsi paling kecil (biasanya ~20-25%).
- **Insight:** Kita berada di jalur yang benar menggunakan regresi Poisson (`objective="poisson"`) untuk memodelkan gol daripada regresi linear/MSE. Tebakan seri (D) secara statistik adalah tebakan berisiko tinggi.

## 2. Plot 2 — ELO Diff vs Outcome
- **Apa yang terlihat:** *Boxplot* menunjukkan bahwa ketika status *Outcome* adalah Kemenangan (W), nilai rata-rata *ELO Diff* sangat positif. Sebaliknya saat (L), *ELO Diff* negatif tajam. Kurva garis (*ELO Diff Bins vs Win Rate*) menunjukkan probabilitas Sigmoid sempurna: Jika selisih ELO melebihi +200, *Win Rate* meroket menembus 80%.
- **Insight:** **ELO adalah "Raja" dari segala fitur** (Korelasi tertinggi: `r = +0.49`). Model apa pun yang kita pakai akan secara membabi buta mengandalkan kolom `elo_diff_feat`. Ini memvalidasi temuan kita sebelumnya bahwa ELO historis (meski dari tahun 2011) sangat krusial dipertahankan, bukan dimusnahkan.

## 3. Plot 3 — Form & Recent Performance
- **Apa yang terlihat:** Pada grafik *Violin Plot*, fitur performa jangka pendek seperti `form_diff_feat`, `pts_last5_ewma`, dan `gd_last5_ewma` memiliki distribusi yang membuncit ke atas untuk kubu pemenang (W).
- **Insight:** Performa 5 laga terakhir (*Form*) adalah faktor terpenting kedua (`r = +0.41`). Tim yang sedang *"on fire"* memiliki daya ungkit kemenangan yang besar. Namun, bentuk distribusi *violin* yang melebar menunjukkan bahwa "kejutan" (tim *form* buruk tiba-tiba menang) sering terjadi.

## 4. Plot 4 — H2H (Head-to-Head) Coverage & Impact
- **Apa yang terlihat:** *Coverage* H2H banyak yang kosong (*Missing Value* tinggi). Namun, saat riwayat H2H *ada*, bar chart menunjukkan rasio kemenangan (W) sedikit bergeser. Garis regresi pada plot ketiga menunjukkan ada hubungan linear yang positif (namun lemah) antara H2H masa lalu dengan selisih gol masa depan.
- **Insight:** Data H2H tidak boleh di *drop*, tetapi karena tingkat kekosongannya tinggi, model *Tree-based* (LGBM/XGBoost) yang secara alami menangani nilai NaN sangat direkomendasikan.

## 5. Plot 5 — Context Features vs Outcome
- **Apa yang terlihat:** Fitur *Contextual* (Jarak Tempuh/Travel, Ketinggian/Altitude, Suhu, PDB/GDP) menampilkan nilai ekstrem (*outliers*). Tim pemenang (W) memiliki median `travel_stress` yang sedikit lebih rendah (lebih bugar), dan median `log_gdp_diff` yang positif (negara lebih kaya sering menang).
- **Insight:** Ekonomi (GDP) dan Demografi cukup stabil mempengaruhi kekuatan tim. Namun, fitur kelelahan geografis (*Travel & Altitude*) sangat licin: efeknya tidak selalu muncul, tapi saat angkanya ekstrem, tim unggulan pun bisa tumbang.

## 6. Plot 6 — Correlation Heatmap
- **Apa yang terlihat:** Peringkat penggerak hasil akhir (Goal Diff):
  1. `elo_diff_feat` (+0.49)
  2. `form_diff_feat` (+0.41)
  3. `log_gdp_diff_ctx` (+0.09)
  Sedangkan fitur yang merusak performa tim adalah `travel_stress_diff` (-0.09) dan `altitude_shock` (-0.08).
- **Insight:** Kekuatan model kita murni disokong oleh sejarah jangka panjang (ELO) dan jangka pendek (Form). Fitur konteks (ekonomi/geografi) bertugas sebagai *tie-breaker* (penentu saat dua tim berstatus seimbang).

## 7. Plot 7 — Altitude Shock: Threshold Effect
- **Apa yang terlihat:** Grafik *bar* Ketinggian vs Win Rate tidak turun secara linear! Saat `altitude_shock` kecil (0-200m), *win rate* normal. Namun saat melewati ambang batas (contoh: > 2000m atau bermain di pegunungan Bolivia/Ekuador), *Win Rate* anjlok drastis ke bawah *baseline* normal.
- **Insight:** Ketinggian dan kelelahan perjalanan tidak bersifat garis lurus. Model linear (seperti Regresi Logistik) akan gagal di sini. Ini memvalidasi penggunaan *Gradient Boosting* yang jago mendeteksi patahan non-linear (*threshold split*).

## 8. Plot 8 — Confederation Bias & Venue Freq
- **Apa yang terlihat:** Negara dari konfederasi kasta atas (seperti UEFA/CONMEBOL) secara konsisten memiliki rata-rata *Win Rate* yang kokoh. Frekuensi bermain di suatu *venue* (tuan rumah langganan) memiliki korelasi kuat dengan mencetak selisih gol positif.
- **Insight:** Faktor kandang (*Home Advantage*) sangat krusial. Bermain di stadion yang familiar memberikan suntikan kekuatan ekstra yang sanggup menutupi defisit ELO.

## 9. Plot 9 — Analisis Null Pattern
- **Apa yang terlihat:** Siapa yang datanya sering hilang (Null)? *Bar chart* menunjukkan tim-tim dari kuintil ELO terendah (negara lemah) memiliki persentase data H2H hilang yang ekstrem. Selain itu, tim yang sudah lama tidak bermain (absen > 5 tahun) juga sering kehilangan data.
- **Insight:** Nilai `NaN` bukanlah kehampaan acak (*Missing Completely at Random*), melainkan sinyal! Jika suatu baris memiliki `NaN` di kolom H2H, model bisa menafsirkannya sebagai "Ini adalah tim medioker/kurang aktif".

---

## 🏆 Rangkuman Insight & Eksekusi Strategis (Takeaways)

Berdasarkan analisis visual di atas, berikut adalah 3 *insight* terpenting yang harus dipegang oleh Anda dan teman-teman tim Anda:

1. **Aturan Besi Prediksi: ELO dan Form adalah Inti**
   Sepak bola memang penuh kejutan, tapi secara statistik murni, tim yang kuat secara historis (ELO) dan sedang konsisten menang di 5 laga terakhir (*Form*) memiliki probabilitas mutlak di atas 60-70% untuk memenangkan laga berikutnya. Jangan pernah memanipulasi atau mendevaluasi kedua kolom ini.
   
2. **Kapan Kejutan (Upset) Terjadi? Momen Kritis "Context Features"**
   Visualisasi membuktikan bahwa "Tim Raksasa bisa kalah" ketika mereka menemui **kombinasi anomali non-linear**:
   - Mereka terbang sangat jauh (*Travel Stress* ekstrem).
   - Mereka bermain di dataran sangat tinggi secara tiba-tiba (*Altitude Shock* menembus ambang batas).
   - *Home Advantage* kubu lawan sangat pekat (*Venue Frequency*).
   *Tugas spesifik Anda: Carilah pertandingan masa lalu (World Cup/Kualifikasi) di mana faktor-faktor pembunuh (geografis/ekonomi) sukses membantai ELO, lalu jadikan fitur baru (seperti "Kelelahan Ekstrem").*

3. **Missing Value adalah Sinyal Kelas Sosial**
   Data yang kosong bukanlah bencana. Di turnamen internasional, ketidakmampuan melacak riwayat pertarungan (*Missing H2H*) atau jarangnya bertanding berkorelasi langsung dengan lemahnya kelas sosial tim tersebut (ELO rendah). Mesin *Ensemble* kita otomatis belajar bahwa kekosongan data = tim medioker. Lanjutkan penggunaan arsitektur XGBoost/LGBM karena mereka bisa "mencerna" kekosongan tanpa perlu *imputasi* sembarangan.

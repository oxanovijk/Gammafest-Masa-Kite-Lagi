Dataset Overview
Tujuan dari kompetisi ini adalah untuk memprediksi hasil skor pertandingan sepak bola internasional. Dataset ini mencakup rentang waktu yang sangat luas, mulai dari pertandingan pertama di tahun 1872 hingga proyeksi pertandingan di masa depan hingga tahun 2026.

Tantangan utama adalah memprediksi dua variabel target: team_goals: Jumlah gol yang dicetak oleh tim utama. opp_goals: Jumlah gol yang dicetak oleh tim lawan.

Dataset ini menggabungkan statistik performa tim, peringkat Elo, peringkat FIFA, hingga data pendukung seperti kondisi geografis (ketinggian), cuaca, dan profil sosio-ekonomi negara (GDP dan populasi).

Files
train.csv: Dataset pelatihan yang berisi data historis pertandingan dari tahun 1872 hingga Agustus 2011. File ini mencakup kolom target (team_goals, opp_goals) dan fitur performa mendalam (Elo, Rank, statistik laga terakhir).
test.csv: Dataset pengujian yang mencakup pertandingan dari Agustus 2011 hingga Maret 2026. Penting: Beberapa fitur performa historis (seperti Elo, Rank, dan statistik last 5/10 matches) tidak disediakan di file ini untuk mensimulasikan tantangan prediksi masa depan.
sample submission.csv: File contoh format submisi. Berisi kolom Id, team_goals, dan opp_goals.
metadata.txt: File berisikan deskripsi setiap feature pada dataset.

 Augmented Weighted Mean Absolute Error (AW-MAE)
Kompetisi ini menggunakan Augmented Weighted Mean Absolute Error (AW-MAE) sebagai metrik evaluasi.

Berbeda dari MAE biasa, metrik ini tidak hanya melihat seberapa dekat prediksi skor dengan hasil asli, tetapi juga memperhatikan:

ketepatan skor akhir,
ketepatan hasil pertandingan (menang / seri / kalah),
ketepatan selisih gol,
dan tingkat kepentingan turnamen.
Semakin kecil nilai AW-MAE, semakin baik performa model

1. Base Error (MAE)
Untuk setiap pertandingan, pertama dihitung error dasar menggunakan rata-rata selisih absolut antara skor prediksi dan skor asli:


Semakin dekat prediksi dengan skor asli, semakin kecil nilai MAE.

2. Penalty Components
Setelah MAE dihitung, sistem akan menambahkan penalti jika prediksi meleset pada aspek-aspek penting berikut:

Exact Score Penalty (0.30)
Dikenakan jika skor prediksi tidak sama persis dengan skor asli.

Outcome Penalty (0.25)
Dikenakan jika hasil pertandingan yang diprediksi (Menang / Seri / Kalah) tidak sesuai dengan hasil asli.

Goal Difference Penalty (0.15)
Dikenakan jika selisih gol prediksi tidak sama dengan selisih gol hasil asli.

Secara ringkas:


dengan:

Exact = 1 jika skor prediksi tepat, selain itu 0
Outcome = 1 jika hasil menang/seri/kalah benar, selain itu 0
GD = 1 jika selisih gol benar, selain itu 0
Artinya, jika suatu aspek diprediksi benar, maka penalti untuk aspek tersebut adalah 0.

3. Outcome Multiplier
Dalam kompetisi ini, ketepatan menebak hasil pertandingan menjadi faktor penting.

Jika prediksi salah pada outcome, total error akan diperbesar dengan pengali:

1.0 jika outcome benar
1.5 jika outcome salah
Dengan demikian, prediksi yang salah pada hasil inti pertandingan akan mendapat konsekuensi lebih besar dibanding kesalahan biasa.

4. Non-Linear Scaling
Setelah MAE dan penalti digabung, nilai tersebut masih akan dipangkatkan agar prediksi yang sangat meleset menerima hukuman lebih besar daripada prediksi yang hampir tepat.

Langkahnya:



Pendekatan ini membuat gap antar peserta menjadi lebih jelas, terutama antara prediksi yang benar-benar kuat dan prediksi yang hanya mendekati.

5. Tournament Weighting
Setiap pertandingan memiliki bobot berbeda sesuai tingkat kepentingan turnamennya.

Turnamen yang lebih prestisius akan memberikan pengaruh lebih besar terhadap skor akhir dibanding pertandingan dengan bobot lebih rendah.

Sebagai contoh:

FIFA World Cup memiliki bobot tinggi, hingga 2.00
AFC Championship memiliki bobot 1.80
Friendly Match memiliki bobot lebih rendah, yaitu 0.96
Jika turnamen tidak termasuk dalam daftar khusus, maka digunakan default weight = 1.20
Dengan kata lain, kesalahan pada pertandingan yang lebih penting akan berdampak lebih besar terhadap nilai akhir.

Metrik ini dirancang agar model yang:

dekat dengan skor asli,
benar menebak hasil pertandingan,
dan tepat pada selisih gol
akan memperoleh nilai yang lebih baik dibanding model yang hanya mendekati skor secara umum.

baca overview dan aturan lomba ini. Kemudian berikan saya strategi (jangan masuk ke kode dulu) untuk mengerjakan lomba ini.
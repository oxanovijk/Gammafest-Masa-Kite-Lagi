# Analisa Hasil V26: Mengapa ERM Menghindari 0-0 dan Strategi Baru

## 1. Pertanyaan Anda: "Mengapa Model Masih Susah Menebak 0-0 dan 1-0?"

### Jawaban Singkat

**Draw Boost dan Tier-Adaptive parameter yang saya pasang di V26 TIDAK BISA memperbaiki masalah ini**, karena akar masalahnya bukan di probabilitas outcome, melainkan di **algoritma ERM (Expected Risk Minimization) itu sendiri**. ERM memiliki bias struktural yang secara matematis akan selalu menghindari 0-0 dan meminimalkan prediksi skor seri/rendah.

### Bukti Numerik

Saya melakukan simulasi: diberikan distribusi probabilitas yang realistis (P(Win)=38%, P(Draw)=25%, P(Loss)=37%), ERM menghitung expected loss untuk setiap kandidat skor. Hasilnya:

```
Expected Loss per Kandidat Prediksi:
  2-1:  3.083  ← ERM pilih ini (terendah)
  1-2:  3.083  ← atau ini
  1-1:  3.092  ← hampir sama, tapi kalah tipis
  1-0:  3.422  ← jauh lebih buruk
  0-0:  4.262  ← TERBURUK dari semua skor rendah
```

**0-0 memiliki expected loss 38% lebih tinggi dari 2-1.** Tidak ada parameter tuning yang bisa mengatasi gap sebesar ini.

### Mengapa 2-1 Selalu Menang dari 0-0?

Karena penalti **1.5x untuk salah outcome** di AW-MAE.

Jika kita prediksi **0-0** (draw):
- Saat kebenaran = Win (38%): SALAH outcome → penalti **1.5x**
- Saat kebenaran = Draw (25%): benar outcome
- Saat kebenaran = Loss (37%): SALAH outcome → penalti **1.5x**
- **75% kasus kena penalti 1.5x**

Jika kita prediksi **2-1** (win):
- Saat kebenaran = Win (38%): benar outcome
- Saat kebenaran = Draw (25%): SALAH outcome → penalti 1.5x
- Saat kebenaran = Loss (37%): SALAH outcome → penalti 1.5x
- **62% kasus kena penalti 1.5x** (lebih sedikit!)

2-1 menang karena Win adalah outcome paling sering. ERM berpikir: "Lebih aman menebak skor yang outcome-nya sama dengan outcome paling sering."

### Tipping Point: P(Draw) Berapa Agar ERM Mau Pilih 1-1?

```
P(Draw)=0.10: ERM pilih 1-2 (E[2-1]=2.897, E[1-1]=3.196)
P(Draw)=0.15: ERM pilih 1-2 (E[2-1]=2.914, E[1-1]=3.070)
P(Draw)=0.20: ERM pilih 1-2 (E[2-1]=2.930, E[1-1]=2.945) ← hampir sama!
P(Draw)=0.25: ERM pilih 1-1 (E[2-1]=2.946, E[1-1]=2.819) ← SWITCH!
P(Draw)=0.30: ERM pilih 1-1 (E[2-1]=2.963, E[1-1]=2.694)
```

**ERM baru mau menebak 1-1 jika P(Draw) >= 25%.** Tapi rata-rata draw rate di data hanya 20.5%! Ini artinya ERM akan **hampir tidak pernah** menebak skor seri.

Dan untuk 0-0? Bahkan di P(Draw)=50% sekalipun, E[0-0]=3.046 masih kalah dari E[1-1]=2.192. **ERM tidak pernah memilih 0-0 dalam kondisi apapun** karena 1-1 selalu lebih dekat ke skor rata-rata.

### Bukti dari Data Aktual

| Metrik | V12 | V26c | Ground Truth |
|:---|:---|:---|:---|
| Total prediksi Draw | 5.163 (12.2%) | 5.721 (13.5%) | **8.688 (20.5%)** |
| Prediksi 0-0 | **0** (0.0%) | **0** (0.0%) | 3.248 (7.7%) |
| Prediksi 1-1 | 5.163 (12.2%) | 5.721 (13.5%) | 3.828 (9.0%) |
| Prediksi 2-1 | 10.935 (25.8%) | 10.577 (24.9%) | 2.621 (6.2%) |

Model menebak **0 kali** 0-0 dan **terlalu banyak** 2-1/1-2 (~50% vs ~12% di GT). Draw Boost saya di V26 hanya berhasil menaikkan prediksi 1-1 dari 12.2% ke 13.5% — **jauh dari cukup**.

---

## 2. Mengapa Parameter V26 Tidak Bisa Memperbaiki Ini?

Draw Boost bekerja pada **outcome probability** (menaikkan P(Draw) sebelum cascade). Tapi ERM menghitung expected loss **di level skor (36-class)**, bukan di level outcome. Walaupun P(Draw) dinaikkan dari 20% ke 25%, ERM masih harus memilih antara 36 skor, dan 2-1/1-2 tetap memiliki expected loss terendah karena:

1. **2-1 punya MAE rendah terhadap SEMUA skor Win** (1-0, 2-0, 2-1, 3-0, 3-1, dst)
2. **0-0 punya MAE tinggi terhadap SEMUA skor non-zero** (1-0, 0-1, 1-1, 2-1, dst)
3. ERM otomatis memilih skor yang "tengah-tengah" (hedging), bukan skor yang paling sering muncul

**Ini adalah cacat desain fundamental dari ERM ketika digunakan untuk metrik AW-MAE.** ERM bagus untuk meminimalkan expected loss secara teori, tapi dalam praktik ia menghasilkan distribusi prediksi yang sangat terkonsentrasi dan tidak natural.

---

## 3. Strategi Baru: Hard Cascade ERM

### Konsep Inti

Alih-alih ERM memilih dari **semua 36 skor** (di mana 2-1 selalu menang), kita pisahkan keputusan menjadi 2 tahap:

**Tahap 1: PUTUSKAN outcome dulu (Win/Draw/Loss)**
- Pilih outcome dengan probabilitas tertinggi dari Stage 1

**Tahap 2: KUNCI ke outcome tersebut, lalu ERM HANYA dalam bucket itu**
- Jika Draw: ERM pilih antara {0-0, 1-1, 2-2, 3-3, 4-4, 5-5} saja
- Jika Win: ERM pilih antara {1-0, 2-0, 2-1, 3-0, 3-1, ...} saja
- Jika Loss: ERM pilih antara {0-1, 0-2, 1-2, 0-3, 1-3, ...} saja

### Mengapa Ini Bisa Membantu?

Saya sudah menghitung: **jika ERM di-lock ke bucket Draw saja**, hasilnya:

```
ERM dalam Draw bucket:
  Predict 1-1: E[loss] = 1.139  ← pilihan terbaik (benar!)
  Predict 0-0: E[loss] = 1.839
  Predict 2-2: E[loss] = 1.656
```

Dan **jika ERM di-lock ke bucket Win saja**:

```
ERM dalam Win bucket:
  Predict 2-0: E[loss] = 1.094  ← pilihan terbaik
  Predict 2-1: E[loss] = 1.172
  Predict 1-0: E[loss] = 1.402
```

Ini berarti Hard Cascade akan menghasilkan:
- **Lebih banyak 1-1** (setiap kali outcome=Draw → otomatis 1-1)
- **Lebih banyak 2-0** (bukan selalu 2-1)
- Total prediksi draw ~20% (mendekati GT 20.5%)

### Risiko

- Jika outcome Stage 1 SALAH, kita terjebak di bucket yang salah tanpa "jaring pengaman" dari soft cascade
- Outcome accuracy saat ini ~59%, artinya ~41% prediksi terkunci di bucket salah
- Ini bisa membuat AW-MAE **lebih buruk** jika outcome model tidak cukup akurat

### Mitigasi Risiko: Hybrid Cascade

**Konsep**: Gunakan Hard Cascade hanya untuk prediksi dengan **confidence tinggi**, dan fallback ke Soft Cascade (V12 style) untuk prediksi yang tidak pasti.

```python
threshold = 0.45  # tunable

for each match:
    if max(P(Win), P(Draw), P(Loss)) > threshold:
        # Confident → Hard Cascade
        outcome = argmax(P)
        score = ERM within outcome bucket only
    else:
        # Not confident → Soft Cascade (V12 style)
        score = ERM across all 36 scores
```

### Estimasi Dampak

- Matches dengan confidence > 45%: sekitar 60-70% dari total
- Dari matches confident: outcome accuracy ~65-70% (lebih tinggi dari rata-rata)
- Prediksi draw naik dari 12% ke ~18-20%
- Expected AW-MAE improvement: **-0.02 to -0.05**

---

## 4. Strategi Alternatif: Custom Loss Tensor

### Konsep

Alih-alih mengubah alur cascade, kita modifikasi **loss tensor** agar ERM tidak terlalu bias ke 2-1:

```python
# Tambahkan "diversity penalty" ke loss tensor
# Skor yang terlalu "safe/hedge" (2-1, 1-2) diberi penalty tambahan
diversity_penalty = {
    (2,1): 0.15,  # reduce ERM's love for 2-1
    (1,2): 0.15,
}

for (a,b), pen in diversity_penalty.items():
    for gt in range(M):
        for go in range(M):
            loss_tensor[a,b,gt,go] += pen
```

Ini akan membuat 2-1 dan 1-2 sedikit kurang menarik bagi ERM, memaksa model untuk lebih sering memilih 1-0, 0-1, atau 1-1.

### Pro & Kontra

- Pro: Mudah diimplementasi, tidak mengubah arsitektur
- Kontra: Angka penalty perlu di-tune dengan hati-hati, bisa merusak jika terlalu besar

---

## 5. Rekomendasi Implementasi

| Prioritas | Strategi | Effort | Expected Impact |
|:---|:---|:---|:---|
| 1 | **Hybrid Cascade (Hard + Soft)** | Sedang | -0.02 to -0.05 |
| 2 | **Custom Loss Tensor (diversity penalty)** | Rendah | -0.01 to -0.03 |
| 3 | **Gabungan keduanya** | Sedang | -0.03 to -0.06 |

Implementasikan **Hybrid Cascade** terlebih dahulu (satu perubahan = satu eksperimen, sesuai prinsip ablation).

---

## 6. Ringkasan

**Pertanyaan**: Mengapa 0-0 tidak pernah diprediksi?

**Jawaban**: Karena AW-MAE memberikan penalti 1.5x untuk salah outcome, dan ERM secara rasional memilih skor yang outcome-nya paling sering benar (Win → 2-1). Untuk ERM mau pilih draw, P(Draw) harus >= 25%, tapi rata-rata draw rate hanya 20.5%. Untuk 0-0 secara spesifik, bahkan di dalam bucket Draw, ERM selalu memilih 1-1 karena jaraknya lebih dekat ke skor-skor lain. **Ini bukan masalah parameter tuning, tapi cacat desain structural ERM terhadap AW-MAE.**

**Solusi**: Gunakan **Hybrid Hard Cascade** yang memisahkan keputusan outcome dari keputusan skor, agar skor draw (1-1, 0-0) mendapat kesempatan muncul proporsional.

# Plan 03: Compact Draw Specialist

## Tujuan

Membangun specialist untuk segmen draw-heavy dan low-score. Plan ini menjadi lawan natural dari Women Tail Specialist, sehingga tidak boleh aktif pada segmen blowout.

## Prioritas Insight

1. P04 Men African tournament compact draw.
2. P05 Men COSAFA draw pocket.
3. P17 Women African compact-draw pocket.
4. P11 Confederation pair x gender, terutama M CAF-CAF dan W CAF-CAF.
5. P12 Scoreline modes differ by archetype.
6. P14 Draw calibration conflict.
7. P10 Neutral side-bias, khusus outcome direction.
8. P01 Men compact global shape.
9. P03 Men 2020 anomaly, low-weight only.
10. P15 Pair symmetry repair.

## Segmen Aktif

Aktif kuat:

- M African Cup of Nations
- M COSAFA Cup
- M CAF-CAF
- M African Cup of Nations qualification, moderate
- W African Cup of Nations
- W CAF Olympic Qualifying Tournament
- W CAF-CAF, moderate

Aktif lemah:

- M Friendly jika tournament/context tidak high-tail
- M UEFA Nations League jika no mismatch signal
- M Gulf Cup
- M Copa America, only with caution

Tidak aktif:

- W AFC/CONCACAF/AFF high-tail qualifier
- M CONCACAF Nations League high-tail cells
- Regional volatile seperti CONIFA/Island Games/Pacific Games
- W FIFA WC qualification high-tail cells

## Arsitektur

```text
candidate pool
  -> compact draw gate
  -> low-total cap
  -> draw candidate injection
  -> one-goal-margin rerank
  -> tail penalty
  -> neutral side-bias correction
  -> mirror repair
```

## Candidate Set Utama

Draw candidates:

- `0-0`
- `1-1`
- `2-2`, hanya jika total prior tidak terlalu rendah

One-goal / low-score candidates:

- `1-0`
- `0-1`
- `2-1`
- `1-2`
- `2-0`
- `0-2`

Tail candidates:

- `3-0` atau `0-3` hanya jika base candidate kuat.
- `4-0+` dimatikan kecuali ada multiple-expert agreement.

## Draw Calibration

Target draw behavior:

| Segment | Draw Action |
|---|---|
| M AFCON | strong draw boost |
| M COSAFA | strong draw boost, especially 0-0 |
| M CAF-CAF | draw boost and total cap |
| W AFCON | medium draw boost |
| W CAF Olympic | strong draw boost but sample guarded |
| M Friendly | mild draw boost |

Draw boost harus dibatasi oleh total prior:

- Jika candidate total >3, draw boost kecil.
- Jika candidate adalah `0-0` atau `1-1`, draw boost penuh.
- Jika base submissions sepakat non-draw dengan margin >=2, draw boost dikurangi.

## Low-Total Cap

Cap rule:

```text
if compact_draw_level == strong:
    penalize total >=4
    strongly penalize total >=5
elif compact_draw_level == medium:
    penalize total >=5
```

Exception:

- Jangan cap jika tournament x era atau confederation pair masuk regional volatile.
- Jangan cap jika base model punya high-confidence blowout dari beberapa expert.

## Konflik Yang Sengaja Dihindari

| Konflik | Guard |
|---|---|
| Draw boost merusak women qualifier blowout | explicit off-list untuk W high-tail |
| Low-score cap merusak regional volatile | volatile archetype override |
| Metric-aware terlalu draw-heavy global | hanya pakai metric-aware draw di compact gate |
| Men 2020 terlalu mempengaruhi semua men | 2020 hanya weak modifier |
| Neutral salah dianggap home | neutral hanya mengurangi side advantage |

## Kelebihan

- Menangkap scoreline mode yang sangat khas: 0-0/1-1/1-0.
- Cocok sebagai specialist override pada Plan 01.
- Mengurangi over-tail pada women Africa compact.

## Risiko

- Bisa underpredict total di men qualifier mismatch.
- Bisa salah bila tournament compact punya match mismatch ekstrim.
- Draw exact score sulit; 0-0 vs 1-1 perlu tuning.

## Audit Wajib

- Predicted draw naik hanya di compact segments.
- Predicted total >=5 turun di M AFCON/COSAFA/W CAF compact.
- W high-tail qualifier tidak terkena compact cap.
- Pair inconsistency 0%.


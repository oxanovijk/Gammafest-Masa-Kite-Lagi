# Plan 02: Women Tail Specialist

## Tujuan

Membangun specialist untuk segmen wanita yang ground truth-nya menunjukkan tail ekstrem dan old submissions underpredict total goals. Plan ini sengaja agresif, tetapi hanya boleh aktif pada segmen yang lolos guard.

## Prioritas Insight

1. P07 Women continental qualifier blowout tail.
2. P13 Old submissions underpredict women high-tail.
3. P12 Scoreline modes differ by archetype.
4. P02 Women tail shrink in recent era.
5. P08 Women UEFA qualifier era reversal.
6. P09 Women elite and invitational compactness sebagai guard negatif.
7. P17 Women African compact-draw pocket sebagai guard negatif.
8. P11 Confederation pair x gender, khusus W AFC-AFC dan W CONCACAF-CONCACAF.
9. P15 Pair symmetry repair.
10. P14 Draw suppression pada high-tail, bukan global draw rule.

## Segmen Aktif

Aktif kuat:

- W AFC Asian Cup qualification
- W CONCACAF Championship qualification
- W AFF Championship
- W FIFA World Cup qualification, tetapi lebih moderat
- W AFC Olympic Qualifying Tournament, moderat
- W CONCACAF Gold Cup qualification, moderat
- W AFC-AFC dan W CONCACAF-CONCACAF jika tournament tidak compact

Aktif lemah:

- W Friendly jika base model sudah memilih total tinggi
- W UEFA Nations League jika mismatch kuat
- Regional women volatile seperti Island Games

Tidak aktif:

- W FIFA World Cup
- W UEFA Euro
- Olympic Games
- Cyprus Cup
- Algarve Cup
- W African Cup of Nations
- W CAF Olympic Qualifying Tournament
- W CAF-CAF compact cells

## Arsitektur

```text
base prediction / candidate pool
  -> women segment gate
  -> tail intensity estimator
       tournament archetype
       era shrink
       confederation pair
       old-submission underprediction profile
  -> high-score candidate expansion
  -> direction selector
       keep base outcome direction unless weak
       use margin tendency, not row lookup
  -> draw suppression
  -> exact score rerank
  -> pair mirror
```

## Tail Intensity

Tail intensity levels:

| Level | Trigger | Candidate Tail |
|---|---|---|
| 3 Extreme | W AFC qual, W CONCACAF qual, W AFF | 4-0, 5-0, 6-0, 0-4, 0-5, 0-6, plus 3-0 |
| 2 Strong | W FIFA WCQ, W AFC Olympic, W CONCACAF Gold Cup qual | 3-0, 4-0, 5-0, mirrored |
| 1 Soft | W Friendly, W UEFA NL, regional volatile | 3-0, 4-0 only when base total already high |
| 0 Off | elite compact / women Africa compact | no tail expansion |

Era adjustment:

- 2011-2014: no shrink.
- 2015-2018: small shrink.
- 2019-2022: allow strong tail if tournament supports it.
- 2023-2026: shrink one level unless tournament x era still has clear high-tail.

Specific W UEFA Euro qualification:

- Pre-2023: can be Level 2.
- 2023-2026: Level 0 or 1 only.

## Candidate Expansion

For Level 3:

```text
Base outcome W -> add 3-0, 4-0, 5-0, 6-0
Base outcome L -> add 0-3, 0-4, 0-5, 0-6
Base draw/weak outcome -> add both directions but downweight direction flips
```

For Level 2:

```text
Add 3-0, 4-0, 5-0 or mirrored.
Keep 2-0/2-1 as fallback if base confidence is not tail enough.
```

For Level 1:

```text
Only add 3-0/0-3 and 4-0/0-4.
Require base model total >=3 or margin >=2.
```

## Draw Suppression

In Level 3:

- Penalize `0-0`, `1-1`, `2-2`.
- Allow draw only if multiple base submissions agree on draw and tournament-specific draw rate is not below 10%.

In Level 2:

- Penalize draw softly.
- Keep `1-1` if base outcome is uncertain and era is 2023-2026.

## Konflik Yang Sengaja Dihindari

| Konflik | Guard |
|---|---|
| Women global tail merusak W elite | explicit off-list untuk elite compact |
| W 2023-2026 tail turun | era shrink satu level |
| Women Africa lebih compact | W CAF compact override mematikan tail |
| High-tail mengubah pemenang terlalu sering | direction follows base outcome unless weak |
| Total naik tapi exact score ngawur | rerank memakai shape 4-0/5-0/6-0 dan margin bins |

## Kelebihan

- Langsung menarget kegagalan terbesar old submissions: underpredict women high-tail.
- Cocok sebagai post-processing layer di atas Plan 01.
- Bisa menghasilkan scoreline yang old submissions jarang pilih.

## Risiko

- Overpredict total goals bila gate bocor ke women elite/friendly.
- Salah arah blowout jika direction selector lemah.
- Sample beberapa regional women kecil dan perlu shrink.

## Audit Wajib

- Cek predicted total >=5 khusus W AFC/CONCACAF/AFF naik, bukan semua W.
- Cek predicted draw di W high-tail turun, tetapi W African compact tidak ikut turun.
- Cek W UEFA Euro qualification 2023-2026 tidak mendapat tail boost besar.
- Cek pair inconsistency 0%.


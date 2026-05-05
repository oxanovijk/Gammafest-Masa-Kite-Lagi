# 🔍 Analisis 100 Match Paling Anomali di Data Training

> **Definisi Anomali**: Tim yang berdasarkan semua indikator fitur (Elo, Form, Win Rate, H2H, Goal Difference) seharusnya menang, tetapi justru **kalah**.

> **Total anomali terdeteksi**: 18292 dari 78772 baris data

---

## 📊 Ringkasan Statistik

| Kategori | Jumlah Match | Deskripsi |
|---|---|---|
| **FORM_CRASH** | 79 | Tim dengan form sangat bagus (diff >4) tiba-tiba collapse |
| **ALTITUDE_SHOCK** | 57 | Tim terpengaruh oleh perbedaan ketinggian ekstrem |
| **HUMILIATING_DEFEAT** | 48 | Kekalahan dengan margin ≥4 gol |
| **TRAVEL_FATIGUE** | 43 | Stres perjalanan jauh mempengaruhi performa |
| **AWAY_DISADVANTAGE** | 42 | Tim unggulan bermain tandang (tanpa keuntungan tuan rumah) |
| **FRIENDLY_LOW_MOTIVATION** | 35 | Kekalahan terjadi di pertandingan persahabatan (motivasi rendah) |
| **GIANT_KILLING** | 30 | Tim dengan Elo >300 poin lebih tinggi kalah (kejutan besar) |
| **SIGNIFICANT_UPSET** | 23 | Tim dengan Elo 150-300 poin lebih tinggi kalah |
| **H2H_DOMINANT_BUT_LOST** | 19 | Tim dominan dalam rekor H2H tetap kalah |
| **GDP_ADVANTAGE_BUT_LOST** | 19 | Tim dari negara lebih kaya kalah (tidak selalu relevan) |
| **HEAVY_DEFEAT** | 17 | Kekalahan dengan margin 3 gol |
| **TEAM_RUSTY_LONG_BREAK** | 10 | Tim tidak bermain dalam waktu lama (>300 hari) |
| **TEAM_FATIGUE_SHORT_REST** | 5 | Tim bermain dengan jeda istirahat sangat singkat (<5 hari) |

---

## 📋 Daftar 100 Match Paling Anomali (Urut Skor Anomali)

| # | Match ID | Tim Favored | vs | Skor | Elo Diff | Form Diff | Anomaly Score | Kategori |
|---|---|---|---|---|---|---|---|---|
| 1 | `W001350` | **Fiji** | Australia | 0-17 | -113 | +6.1 | 58.26 | ALTITUDE_SHOCK, FORM_CRASH, TEAM_FATIGUE_SHORT_REST, HUMILIATING_DEFEAT |
| 2 | `W002463` | **Nigeria** | Senegal | 4-11 | +307 | +7.1 | 54.17 | TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING, HUMILIATING_DEFEAT, H2H_DOMINANT_BUT_LOST |
| 3 | `M030968` | **Guam** | Taiwan | 0-10 | -78 | +7.1 | 46.09 | GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, HUMILIATING_DEFEAT |
| 4 | `W001512` | **Kazakhstan** | South Korea | 0-6 | -47 | +8.6 | 40.91 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, HUMILIATING_DEFEAT |
| 5 | `M004110` | **Costa Rica** | Guatemala | 1-3 | +189 | +11.0 | 40.21 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, SIGNIFICANT_UPSET, H2H_DOMINANT_BUT_LOST |
| 6 | `M026282` | **New Caledonia** | Papua New Guinea | 1-4 | +142 | +9.7 | 39.54 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, TEAM_FATIGUE_SHORT_REST, HEAVY_DEFEAT, H2H_DOMINANT_BUT_LOST |
| 7 | `M034260` | **British Virgin Islands** | Dominican Republic | 0-17 | -87 | +4.2 | 39.51 | FORM_CRASH, HUMILIATING_DEFEAT |
| 8 | `W001674` | **Romania** | Iceland | 0-8 | +189 | +4.9 | 38.24 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 9 | `M004237` | **Cambodia** | Malaysia | 2-9 | +22 | +6.5 | 36.15 | ALTITUDE_SHOCK, FORM_CRASH, TEAM_RUSTY_LONG_BREAK, HUMILIATING_DEFEAT |
| 10 | `M013625` | **Algeria** | Benin | 2-6 | +372 | +5.1 | 34.72 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, GIANT_KILLING, HUMILIATING_DEFEAT, H2H_DOMINANT_BUT_LOST |
| 11 | `M002767` | **Argentina** | Paraguay | 1-5 | +362 | +5.1 | 34.63 | AWAY_DISADVANTAGE, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING, HUMILIATING_DEFEAT |
| 12 | `W002735` | **Mozambique** | South Africa | 2-6 | +46 | +8.3 | 34.25 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, HUMILIATING_DEFEAT |
| 13 | `W001198` | **Chile** | Peru | 0-1 | +60 | +13.0 | 34.16 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH |
| 14 | `M032900` | **Argentina** | Bolivia | 1-6 | +401 | +4.3 | 34.16 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, GIANT_KILLING, HUMILIATING_DEFEAT |
| 15 | `M000534` | **Denmark** | Norway | 1-3 | +400 | +6.1 | 32.68 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING, H2H_DOMINANT_BUT_LOST |
| 16 | `W001898` | **Canada** | Finland | 0-3 | +294 | +6.5 | 32.45 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, SIGNIFICANT_UPSET, HEAVY_DEFEAT |
| 17 | `M001702` | **Kenya** | Uganda | 1-13 | -60 | +3.8 | 31.95 | FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, TEAM_RUSTY_LONG_BREAK, HUMILIATING_DEFEAT |
| 18 | `M000151` | **Scotland** | Northern Ireland | 0-2 | +502 | +5.0 | 30.99 | FORM_CRASH, GIANT_KILLING, H2H_DOMINANT_BUT_LOST |
| 19 | `W000547` | **Brazil** | Venezuela | 6-11 | +91 | +6.0 | 30.83 | TRAVEL_FATIGUE, FORM_CRASH, HUMILIATING_DEFEAT |
| 20 | `W002737` | **Togo** | Congo | 0-9 | +47 | +4.2 | 30.47 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, FORM_CRASH, HUMILIATING_DEFEAT |
| 21 | `M006392` | **Norway** | Finland | 0-4 | +240 | +5.5 | 30.04 | AWAY_DISADVANTAGE, TRAVEL_FATIGUE, FORM_CRASH, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 22 | `M021168` | **Tajikistan** | Uzbekistan | 0-5 | -91 | +6.3 | 29.79 | ALTITUDE_SHOCK, FORM_CRASH, HUMILIATING_DEFEAT, H2H_DOMINANT_BUT_LOST |
| 23 | `W003735` | **Estonia** | Iceland | 0-12 | -509 | +3.8 | 29.50 | TRAVEL_FATIGUE, HUMILIATING_DEFEAT |
| 24 | `M004456` | **Martinique** | Guadeloupe | 2-5 | +123 | +6.9 | 29.41 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, FORM_CRASH, HEAVY_DEFEAT, H2H_DOMINANT_BUT_LOST |
| 25 | `W000892` | **Uzbekistan** | South Korea | 0-6 | +167 | +4.2 | 29.26 | ALTITUDE_SHOCK, FORM_CRASH, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 26 | `W003444` | **Jamaica** | Mexico | 1-8 | -58 | +6.3 | 29.03 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, HUMILIATING_DEFEAT |
| 27 | `M007044` | **Romania** | Switzerland | 1-7 | +203 | +4.0 | 29.00 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 28 | `M004253` | **Malaysia** | Cambodia | 2-3 | +64 | +9.8 | 28.54 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, H2H_DOMINANT_BUT_LOST |
| 29 | `W002802` | **Netherlands Antilles** | Suriname | 1-7 | +123 | +4.5 | 28.48 | FORM_CRASH, HUMILIATING_DEFEAT |
| 30 | `M011774` | **Mozambique** | Zimbabwe | 0-6 | +66 | +4.7 | 28.16 | FRIENDLY_LOW_MOTIVATION, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, HUMILIATING_DEFEAT |
| 31 | `W003681` | **Palestine** | Kyrgyzstan | 1-4 | -70 | +8.0 | 28.06 | ALTITUDE_SHOCK, FORM_CRASH, HEAVY_DEFEAT |
| 32 | `M012124` | **New Caledonia** | Australia | 0-8 | -2 | +4.8 | 27.95 | FORM_CRASH, HUMILIATING_DEFEAT |
| 33 | `M003446` | **Turkey** | Israel | 1-5 | +137 | +5.8 | 27.75 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, HUMILIATING_DEFEAT |
| 34 | `M001719` | **Scotland** | Wales | 2-5 | +342 | +4.6 | 27.59 | FORM_CRASH, GIANT_KILLING, HEAVY_DEFEAT |
| 35 | `M000178` | **France** | Belgium | 0-7 | +38 | +4.6 | 27.52 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, FORM_CRASH, HUMILIATING_DEFEAT |
| 36 | `M002050` | **Sweden** | Belgium | 1-5 | +355 | +3.2 | 27.34 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, TRAVEL_FATIGUE, GIANT_KILLING, HUMILIATING_DEFEAT |
| 37 | `M024395` | **New Caledonia** | Vanuatu | 0-6 | +176 | +4.0 | 27.28 | GDP_ADVANTAGE_BUT_LOST, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 38 | `M027944` | **Taiwan** | Palestine | 0-8 | -137 | +4.6 | 27.26 | ALTITUDE_SHOCK, FORM_CRASH, HUMILIATING_DEFEAT |
| 39 | `W003027` | **Jordan** | Japan | 0-13 | -246 | +3.1 | 27.05 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, TEAM_RUSTY_LONG_BREAK, HUMILIATING_DEFEAT |
| 40 | `M003996` | **Sweden** | Russia | 0-7 | +226 | +3.1 | 27.03 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 41 | `W004256` | **Luxembourg** | Macedonia | 1-5 | +197 | +4.5 | 26.97 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 42 | `M032381` | **Switzerland** | Luxembourg | 1-2 | +777 | +2.9 | 26.91 | GIANT_KILLING, H2H_DOMINANT_BUT_LOST |
| 43 | `M003146` | **Israel** | United States | 1-3 | -33 | +8.8 | 26.89 | FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, TEAM_RUSTY_LONG_BREAK |
| 44 | `M011333` | **Martinique** | Puerto Rico | 1-4 | +358 | +4.1 | 26.84 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, FORM_CRASH, GIANT_KILLING, HEAVY_DEFEAT |
| 45 | `M009173` | **Tahiti** | Chile | 1-10 | -1 | +4.6 | 26.80 | FRIENDLY_LOW_MOTIVATION, FORM_CRASH, TEAM_RUSTY_LONG_BREAK, HUMILIATING_DEFEAT |
| 46 | `M000602` | **Italy** | Switzerland | 0-3 | +145 | +6.0 | 26.46 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, HEAVY_DEFEAT |
| 47 | `M000156` | **Argentina** | Uruguay | 2-3 | +54 | +9.0 | 26.23 | FRIENDLY_LOW_MOTIVATION, FORM_CRASH, TEAM_RUSTY_LONG_BREAK, H2H_DOMINANT_BUT_LOST |
| 48 | `M013793` | **New Zealand** | Fiji | 0-2 | +269 | +5.8 | 26.18 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, SIGNIFICANT_UPSET, H2H_DOMINANT_BUT_LOST |
| 49 | `M012211` | **South Korea** | Luxembourg | 2-3 | +764 | +3.0 | 26.14 | FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GIANT_KILLING |
| 50 | `M026901` | **Bhutan** | Maldives | 0-6 | +13 | +4.9 | 26.11 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, HUMILIATING_DEFEAT |
| 51 | `M008829` | **Kuwait** | Cambodia | 0-4 | +232 | +5.0 | 26.01 | TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 52 | `M006627` | **Libya** | Lithuania | 0-6 | +296 | +4.4 | 25.55 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, TRAVEL_FATIGUE, FORM_CRASH, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 53 | `M000281` | **Italy** | Hungary | 1-6 | +30 | +5.3 | 25.52 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, HUMILIATING_DEFEAT |
| 54 | `M008491` | **Myanmar** | Singapore | 0-1 | +537 | +4.9 | 25.44 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING, H2H_DOMINANT_BUT_LOST |
| 55 | `M032205` | **Egypt** | Sudan | 0-4 | +395 | +2.1 | 25.35 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, GDP_ADVANTAGE_BUT_LOST, GIANT_KILLING, HUMILIATING_DEFEAT, H2H_DOMINANT_BUT_LOST |
| 56 | `M003285` | **Denmark** | Finland | 0-2 | +406 | +4.0 | 25.21 | TRAVEL_FATIGUE, GIANT_KILLING, H2H_DOMINANT_BUT_LOST |
| 57 | `M007710` | **Spain** | Finland | 0-2 | +482 | +4.1 | 25.16 | AWAY_DISADVANTAGE, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING |
| 58 | `M002692` | **Costa Rica** | Guatemala | 2-3 | +372 | +5.5 | 24.95 | ALTITUDE_SHOCK, FORM_CRASH, GIANT_KILLING |
| 59 | `M033376` | **Brazil** | Bolivia | 1-2 | +564 | +4.1 | 24.90 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, GIANT_KILLING |
| 60 | `M005207` | **Saint Lucia** | Grenada | 0-4 | +9 | +6.0 | 24.88 | FORM_CRASH, HUMILIATING_DEFEAT |
| 61 | `W002646` | **Lithuania** | Estonia | 1-4 | +209 | +5.0 | 24.76 | FORM_CRASH, SIGNIFICANT_UPSET, HEAVY_DEFEAT |
| 62 | `W003216` | **Iceland** | Slovenia | 1-2 | +356 | +6.8 | 24.75 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, GIANT_KILLING |
| 63 | `M002484` | **China PR** | Japan | 0-3 | +333 | +3.8 | 24.75 | FRIENDLY_LOW_MOTIVATION, GIANT_KILLING, TEAM_RUSTY_LONG_BREAK, HEAVY_DEFEAT |
| 64 | `M023018` | **Tonga** | Tahiti | 0-5 | -297 | +5.3 | 24.70 | ALTITUDE_SHOCK, FORM_CRASH, TEAM_FATIGUE_SHORT_REST, HUMILIATING_DEFEAT |
| 65 | `M001310` | **Uruguay** | Paraguay | 1-3 | +388 | +4.0 | 24.52 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, GIANT_KILLING, H2H_DOMINANT_BUT_LOST |
| 66 | `W000700` | **Hong Kong** | Taiwan | 0-2 | -304 | +8.0 | 24.43 | ALTITUDE_SHOCK, FORM_CRASH |
| 67 | `M027172` | **Saudi Arabia** | Liechtenstein | 0-1 | +560 | +4.6 | 24.34 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, GIANT_KILLING |
| 68 | `W002974` | **Colombia** | Chile | 1-3 | +169 | +5.8 | 24.30 | FORM_CRASH, SIGNIFICANT_UPSET, H2H_DOMINANT_BUT_LOST |
| 69 | `M007250` | **Tunisia** | Saudi Arabia | 0-4 | +100 | +5.8 | 24.23 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, HUMILIATING_DEFEAT |
| 70 | `M008656` | **Saint Vincent and the Grenadines** | Dominica | 0-3 | +236 | +4.3 | 24.20 | AWAY_DISADVANTAGE, FORM_CRASH, SIGNIFICANT_UPSET, HEAVY_DEFEAT |
| 71 | `M007721` | **Eswatini** | Malawi | 0-3 | +173 | +4.8 | 24.15 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, SIGNIFICANT_UPSET, TEAM_RUSTY_LONG_BREAK, HEAVY_DEFEAT |
| 72 | `W004147` | **United States** | Mexico | 1-2 | +589 | +3.6 | 24.15 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, GIANT_KILLING |
| 73 | `M004415` | **Uruguay** | Colombia | 0-1 | +432 | +5.7 | 24.10 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING |
| 74 | `M002848` | **Suriname** | Aruba | 1-8 | +67 | +3.7 | 24.07 | FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, HUMILIATING_DEFEAT |
| 75 | `W003815` | **China PR** | Chile | 0-1 | +500 | +4.6 | 24.06 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING |
| 76 | `M007906` | **Indonesia** | Singapore | 2-3 | +369 | +5.0 | 24.05 | TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING, H2H_DOMINANT_BUT_LOST |
| 77 | `M008995` | **New Caledonia** | New Zealand | 1-4 | +191 | +4.9 | 23.91 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, SIGNIFICANT_UPSET, TEAM_RUSTY_LONG_BREAK, HEAVY_DEFEAT |
| 78 | `M011338` | **Malaysia** | Singapore | 0-1 | +389 | +5.6 | 23.75 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING |
| 79 | `W003300` | **Antigua and Barbuda** | Jamaica | 0-12 | -154 | +3.1 | 23.64 | GDP_ADVANTAGE_BUT_LOST, HUMILIATING_DEFEAT |
| 80 | `M019114` | **Laos** | Myanmar | 1-7 | -148 | +4.5 | 23.62 | GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, HUMILIATING_DEFEAT |
| 81 | `M001085` | **Guernsey** | Jersey | 1-5 | +127 | +4.4 | 23.45 | FORM_CRASH, HUMILIATING_DEFEAT |
| 82 | `M003537` | **New Caledonia** | New Zealand | 4-6 | +235 | +5.0 | 23.42 | FRIENDLY_LOW_MOTIVATION, FORM_CRASH, SIGNIFICANT_UPSET |
| 83 | `W001156` | **Bulgaria** | Turkey | 1-2 | +148 | +7.2 | 23.41 | FORM_CRASH, H2H_DOMINANT_BUT_LOST |
| 84 | `M022949` | **Guam** | Northern Mariana Islands | 0-3 | -238 | +6.5 | 23.35 | FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, HEAVY_DEFEAT |
| 85 | `M022236` | **Denmark** | Bosnia and Herzegovina | 0-3 | +409 | +3.1 | 23.35 | AWAY_DISADVANTAGE, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, GIANT_KILLING, HEAVY_DEFEAT |
| 86 | `M002765` | **Latvia** | Lithuania | 3-7 | +237 | +3.7 | 23.31 | FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, SIGNIFICANT_UPSET, TEAM_RUSTY_LONG_BREAK, HUMILIATING_DEFEAT |
| 87 | `M001066` | **Czechoslovakia** | Catalonia | 1-2 | +238 | +6.6 | 23.22 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, FORM_CRASH, SIGNIFICANT_UPSET |
| 88 | `W001675` | **Hungary** | Netherlands | 0-3 | +49 | +5.9 | 23.21 | TRAVEL_FATIGUE, FORM_CRASH, HEAVY_DEFEAT |
| 89 | `M031163` | **Italy** | Hungary | 1-3 | +463 | +3.4 | 23.07 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, GIANT_KILLING |
| 90 | `M012907` | **Nigeria** | Iceland | 0-3 | +390 | +3.5 | 23.01 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, TRAVEL_FATIGUE, GIANT_KILLING, HEAVY_DEFEAT |
| 91 | `M031518` | **Saint Kitts and Nevis** | Antigua and Barbuda | 0-2 | +175 | +5.7 | 23.01 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, FORM_CRASH, SIGNIFICANT_UPSET |
| 92 | `M025418` | **Tonga** | Australia | 0-22 | -467 | +2.0 | 22.99 | TRAVEL_FATIGUE, TEAM_FATIGUE_SHORT_REST, HUMILIATING_DEFEAT |
| 93 | `M003398` | **England** | United States | 0-1 | +432 | +4.6 | 22.97 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, GIANT_KILLING |
| 94 | `W002803` | **United States Virgin Islands** | Dominican Republic | 1-3 | +75 | +6.6 | 22.87 | AWAY_DISADVANTAGE, FORM_CRASH |
| 95 | `M002816` | **Luxembourg** | Belgium | 0-7 | -77 | +3.7 | 22.82 | FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, HUMILIATING_DEFEAT |
| 96 | `W003889` | **Suriname** | Guyana | 0-2 | +36 | +7.0 | 22.82 | AWAY_DISADVANTAGE, FORM_CRASH, TEAM_FATIGUE_SHORT_REST |
| 97 | `M003015` | **Italy** | Austria | 1-5 | +264 | +3.1 | 22.78 | AWAY_DISADVANTAGE, FRIENDLY_LOW_MOTIVATION, ALTITUDE_SHOCK, SIGNIFICANT_UPSET, HUMILIATING_DEFEAT |
| 98 | `W002854` | **Suriname** | Haiti | 0-3 | +5 | +6.4 | 22.77 | TRAVEL_FATIGUE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH, HEAVY_DEFEAT |
| 99 | `W001000` | **Poland** | Belarus | 0-2 | +12 | +7.0 | 22.75 | AWAY_DISADVANTAGE, GDP_ADVANTAGE_BUT_LOST, FORM_CRASH |
| 100 | `M008873` | **Iran** | Republic of Ireland | 1-2 | +294 | +5.9 | 22.74 | ALTITUDE_SHOCK, TRAVEL_FATIGUE, FORM_CRASH, SIGNIFICANT_UPSET |

---

## 🧠 Analisis Per Kategori

### ⚡ Giant Killing (Elo Gap > 300 poin)

Ini adalah kasus paling ekstrem: tim yang secara peringkat **jauh lebih kuat** (selisih Elo >300 poin) justru kalah. Dalam sepak bola, ini sering terjadi di:
- **Kualifikasi turnamen**: Tim besar underestimate lawan kecil
- **Friendly**: Menggunakan pemain cadangan/eksperimen
- **Faktor non-teknis**: Cuaca, perjalanan, ketinggian

| Match ID | Favored | vs | Skor | Elo Gap | Tournament |
|---|---|---|---|---|---|
| `M032381` | Switzerland (1807) | Luxembourg (1030) | 1-2 | +777 | FIFA World Cup qualification |
| `M012211` | South Korea (1905) | Luxembourg (1141) | 2-3 | +764 | Friendly |
| `W004147` | United States (2339) | Mexico (1750) | 1-2 | +589 | CONCACAF Gold Cup |
| `M033376` | Brazil (2167) | Bolivia (1602) | 1-2 | +564 | FIFA World Cup qualification |
| `M027172` | Saudi Arabia (1694) | Liechtenstein (1134) | 0-1 | +560 | Friendly |
| `M008491` | Myanmar (1766) | Singapore (1228) | 0-1 | +537 | Merdeka Tournament |
| `M000151` | Scotland (1739) | Northern Ireland (1236) | 0-2 | +502 | British Home Championship |
| `W003815` | China PR (1904) | Chile (1403) | 0-1 | +500 | International Tournament |
| `M007710` | Spain (1705) | Finland (1223) | 0-2 | +482 | FIFA World Cup qualification |
| `M031163` | Italy (2039) | Hungary (1575) | 1-3 | +463 | Friendly |
| `M004415` | Uruguay (1755) | Colombia (1323) | 0-1 | +432 | Copa América |
| `M003398` | England (1853) | United States (1421) | 0-1 | +432 | FIFA World Cup |
| `M022236` | Denmark (1920) | Bosnia and Herzegovina (1512) | 0-3 | +409 | FIFA World Cup qualification |
| `M003285` | Denmark (1597) | Finland (1191) | 0-2 | +406 | Nordic Championship |
| `M032900` | Argentina (2005) | Bolivia (1605) | 1-6 | +401 | FIFA World Cup qualification |
| `M000534` | Denmark (1714) | Norway (1315) | 1-3 | +400 | Friendly |
| `M032205` | Egypt (1862) | Sudan (1468) | 0-4 | +395 | Friendly |
| `M012907` | Nigeria (1743) | Iceland (1354) | 0-3 | +390 | Friendly |
| `M011338` | Malaysia (1636) | Singapore (1247) | 0-1 | +389 | King's Cup |
| `M001310` | Uruguay (1807) | Paraguay (1419) | 1-3 | +388 | Friendly |

### 🤝 Friendly (Motivasi Rendah)

Pertandingan persahabatan adalah sumber anomali terbesar karena:
- Tim besar sering **merotasi skuad** dan mencoba formasi baru
- Tidak ada tekanan kompetitif → intensitas lebih rendah
- Sering digunakan untuk **eksperimen taktik**, bukan menang

**Total match anomali di Friendly: 35**

| Match ID | Favored | vs | Skor | Elo Gap |
|---|---|---|---|---|
| `M012211` | South Korea | Luxembourg | 2-3 | +764 |
| `M027172` | Saudi Arabia | Liechtenstein | 0-1 | +560 |
| `M031163` | Italy | Hungary | 1-3 | +463 |
| `M000534` | Denmark | Norway | 1-3 | +400 |
| `M032205` | Egypt | Sudan | 0-4 | +395 |
| `M012907` | Nigeria | Iceland | 0-3 | +390 |
| `M001310` | Uruguay | Paraguay | 1-3 | +388 |
| `M011333` | Martinique | Puerto Rico | 1-4 | +358 |
| `M002050` | Sweden | Belgium | 1-5 | +355 |
| `M002484` | China PR | Japan | 0-3 | +333 |
| `M006627` | Libya | Lithuania | 0-6 | +296 |
| `M013793` | New Zealand | Fiji | 0-2 | +269 |
| `M003015` | Italy | Austria | 1-5 | +264 |
| `M001066` | Czechoslovakia | Catalonia | 1-2 | +238 |
| `M002765` | Latvia | Lithuania | 3-7 | +237 |

### 📉 Form Crash (Kolaps Mendadak)

Tim yang sedang dalam performa puncak (form diff > 4.0) tiba-tiba kalah. Ini bisa disebabkan oleh:
- **Overconfidence** setelah winning streak
- **Kelelahan akumulatif** dari jadwal padat
- Lawan yang **bermain ultra-defensif** (parking the bus)

| Match ID | Favored | vs | Skor | Form Diff | Win Rate Team vs Opp |
|---|---|---|---|---|---|
| `W001198` | Chile | Peru | 0-1 | +13.0 | 1.00 vs 0.00 |
| `M004110` | Costa Rica | Guatemala | 1-3 | +11.0 | 1.00 vs 0.00 |
| `M004253` | Malaysia | Cambodia | 2-3 | +9.8 | 1.00 vs 0.02 |
| `M026282` | New Caledonia | Papua New Guinea | 1-4 | +9.7 | 0.99 vs 0.00 |
| `M000156` | Argentina | Uruguay | 2-3 | +9.0 | 1.00 vs 0.00 |
| `M003146` | Israel | United States | 1-3 | +8.8 | 1.00 vs 0.00 |
| `W001512` | Kazakhstan | South Korea | 0-6 | +8.6 | 0.99 vs 0.02 |
| `W002735` | Mozambique | South Africa | 2-6 | +8.3 | 1.00 vs 0.01 |
| `W003681` | Palestine | Kyrgyzstan | 1-4 | +8.0 | 1.00 vs 0.00 |
| `W000700` | Hong Kong | Taiwan | 0-2 | +8.0 | 1.00 vs 0.00 |
| `W001156` | Bulgaria | Turkey | 1-2 | +7.2 | 0.98 vs 0.00 |
| `M030968` | Guam | Taiwan | 0-10 | +7.1 | 0.91 vs 0.13 |
| `W002463` | Nigeria | Senegal | 4-11 | +7.1 | 0.76 vs 0.00 |
| `W003889` | Suriname | Guyana | 0-2 | +7.0 | 1.00 vs 0.00 |
| `W001000` | Poland | Belarus | 0-2 | +7.0 | 0.95 vs 0.00 |

### 🏟️ Away Disadvantage

Tim unggulan yang bermain tandang dan kalah. Home advantage di sepak bola internasional diperkirakan bernilai **50-100 poin Elo**. Faktor:
- Dukungan penonton tuan rumah
- Familiaritas lapangan dan iklim
- Bias wasit (terbukti secara statistik)

**Total: 42 match**

### 🔄 H2H Dominan Tapi Kalah

Tim yang mendominasi rekor head-to-head (rata-rata GD > 3 dalam 5 laga terakhir vs lawan yang sama) tapi tetap kalah. Ini menunjukkan bahwa **rekor historis tidak menjamin hasil masa depan**.

| Match ID | Favored | vs | Skor | H2H GD Avg |
|---|---|---|---|---|
| `M013793` | New Zealand | Fiji | 0-2 | +9.9 |
| `M000534` | Denmark | Norway | 1-3 | +8.6 |
| `M004110` | Costa Rica | Guatemala | 1-3 | +8.0 |
| `M004253` | Malaysia | Cambodia | 2-3 | +7.0 |
| `M007906` | Indonesia | Singapore | 2-3 | +6.7 |
| `W002463` | Nigeria | Senegal | 4-11 | +6.0 |
| `M000156` | Argentina | Uruguay | 2-3 | +6.0 |
| `M008491` | Myanmar | Singapore | 0-1 | +6.0 |
| `M026282` | New Caledonia | Papua New Guinea | 1-4 | +4.9 |
| `M004456` | Martinique | Guadeloupe | 2-5 | +4.5 |

### 💀 Kekalahan Telak (Margin ≥ 3 Gol)

Tim favored yang tidak hanya kalah, tapi **dipermalukan** dengan margin besar. Ini adalah anomali ganda: bukan hanya hasil yang kebalik, tapi intensitasnya juga ekstrem.

| Match ID | Favored | vs | Skor | Elo Gap | Tournament |
|---|---|---|---|---|---|
| `M025418` | Tonga | Australia | 0-22 | -467 | FIFA World Cup qualification |
| `W001350` | Fiji | Australia | 0-17 | -113 | OFC Championship |
| `M034260` | British Virgin Islands | Dominican Republic | 0-17 | -87 | CFU Caribbean Cup qualification |
| `W003027` | Jordan | Japan | 0-13 | -246 | Asian Games |
| `M001702` | Kenya | Uganda | 1-13 | -60 | Friendly |
| `W003735` | Estonia | Iceland | 0-12 | -509 | FIFA World Cup qualification |
| `W003300` | Antigua and Barbuda | Jamaica | 0-12 | -154 | CONCACAF Olympic Qualifying Tournament qualification |
| `M030968` | Guam | Taiwan | 0-10 | -78 | EAFF Championship |
| `W002737` | Togo | Congo | 0-9 | +47 | African Championship qualification |
| `M009173` | Tahiti | Chile | 1-10 | -1 | Friendly |
| `W001674` | Romania | Iceland | 0-8 | +189 | UEFA Euro qualification |
| `M012124` | New Caledonia | Australia | 0-8 | -2 | Oceania Nations Cup |
| `M027944` | Taiwan | Palestine | 0-8 | -137 | FIFA World Cup qualification |
| `W002463` | Nigeria | Senegal | 4-11 | +307 | African Championship qualification |
| `M004237` | Cambodia | Malaysia | 2-9 | +22 | AFC Asian Cup qualification |

---

## 💡 Insight untuk Peningkatan Model

Dari analisis 100 match paling anomali, beberapa insight kunci:

1. **Friendly matches** adalah sumber noise terbesar. Model mungkin perlu **menurunkan bobot** friendly atau menambah fitur boolean `is_friendly`.
2. **Giant killing** konsisten terjadi di semua era. Ini menunjukkan bahwa Elo saja tidak cukup — model perlu fitur **uncertainty/variance** dari performa tim.
3. **Away disadvantage** signifikan. Fitur `neutral` venue sudah ada, tapi mungkin perlu fitur eksplisit `is_home` atau interaksi `elo_diff * is_home`.
4. **Form crash** menunjukkan bahwa form tinggi bisa menjadi **sinyal berbahaya** (regression to mean). Fitur non-linear dari form mungkin membantu.
5. **Pure anomaly** yang tidak bisa dijelaskan (~faktor acak) menunjukkan **batas bawah error** model. Ini adalah noise irreducible dalam sepak bola.

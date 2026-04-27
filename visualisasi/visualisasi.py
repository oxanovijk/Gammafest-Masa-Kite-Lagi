import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train_final.csv")
TEST_PATH  = os.path.join(BASE_DIR, "dataset", "test_final.csv")
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# ─── LOAD ─────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

# Derive outcome labels dari train
train["team_goals"]  = train["team_goals"].astype(int)
train["opp_goals"]   = train["opp_goals"].astype(int)
train["goal_diff"]   = train["team_goals"] - train["opp_goals"]
train["outcome"]     = train["goal_diff"].apply(
    lambda x: "W" if x > 0 else ("L" if x < 0 else "D")
)

FEAT_COLS = [c for c in test.columns if c != "Id"]
CTX_COLS  = [c for c in test.columns if "_ctx" in c]
ELO_COLS  = [c for c in test.columns if "elo" in c]
FORM_COLS = [c for c in test.columns if "form" in c or "pts_last" in c or "gd_last" in c]
H2H_COLS  = [c for c in test.columns if "h2h" in c]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Distribusi Target: Goals & Outcome
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Plot 1 — Distribusi Target (train)", fontweight="bold")

# Goals distribution
for ax, col, color, label in zip(
    axes[:2],
    ["team_goals", "opp_goals"],
    ["steelblue", "tomato"],
    ["Team Goals", "Opp Goals"]
):
    vals = train[col].clip(0, 10).value_counts().sort_index()
    ax.bar(vals.index, vals.values, color=color, alpha=0.8)
    ax.set_xlabel(label); ax.set_ylabel("Frekuensi")
    ax.set_title(f"Distribusi {label}")
    mean_val = train[col].mean()
    ax.axvline(mean_val, color="black", linestyle="--", label=f"Mean={mean_val:.2f}")
    ax.legend()

# Outcome pie
outcome_counts = train["outcome"].value_counts()
axes[2].pie(outcome_counts, labels=outcome_counts.index, autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c", "#3498db"], startangle=90)
axes[2].set_title("Distribusi Outcome (W/D/L)")

plt.tight_layout()
plt.savefig("plot1_target_distribution.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — ELO Diff vs Outcome
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Plot 2 — ELO Diff vs Outcome", fontweight="bold")

# Box plot ELO diff per outcome
order = ["W", "D", "L"]
colors = {"W": "#2ecc71", "D": "#3498db", "L": "#e74c3c"}
sns.boxplot(data=train, x="outcome", y="elo_diff_feat", order=order,
            palette=colors, ax=axes[0])
axes[0].set_title("ELO Diff per Outcome")
axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)

# ELO diff bins vs win rate
bins = pd.cut(train["elo_diff_feat"], bins=20)
win_rate = train.groupby(bins, observed=True)["outcome"].apply(
    lambda x: (x == "W").mean()
).reset_index()
win_rate["mid"] = win_rate["elo_diff_feat"].apply(lambda x: x.mid)
axes[1].plot(win_rate["mid"], win_rate["outcome"], marker="o", color="steelblue")
axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.6, label="50% baseline")
axes[1].set_xlabel("ELO Diff (bins)"); axes[1].set_ylabel("Win Rate")
axes[1].set_title("ELO Diff Bins vs Win Rate"); axes[1].legend()

# ELO team vs opp scatter (color by outcome)
sample = train.sample(min(5000, len(train)), random_state=42)
for o, c in colors.items():
    sub = sample[sample["outcome"] == o]
    axes[2].scatter(sub["elo_team_feat"], sub["elo_opponent_feat"],
                    alpha=0.3, s=5, color=c, label=o)
axes[2].set_xlabel("ELO Team"); axes[2].set_ylabel("ELO Opp")
axes[2].set_title("ELO Team vs Opp (warna = outcome)")
axes[2].legend(markerscale=3)

plt.tight_layout()
plt.savefig("plot2_elo_vs_outcome.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Form & Recent Performance
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Plot 3 — Form & Recent Performance vs Outcome", fontweight="bold")

key_form_feats = [
    ("form_diff_feat",       "Form Diff"),
    ("pts_last5_ewma_diff_feat", "Pts Last5 EWMA Diff"),
    ("gd_last5_ewma_diff_feat",  "GD Last5 EWMA Diff"),
    ("team_avg_gf_last5_ewma_feat", "Team Avg GF Last5"),
    ("team_avg_ga_last5_ewma_feat", "Team Avg GA Last5"),
    ("opp_avg_gf_last5_ewma_feat",  "Opp Avg GF Last5"),
]

for ax, (feat, label) in zip(axes.flatten(), key_form_feats):
    data_clean = train.dropna(subset=[feat])
    sns.violinplot(data=data_clean, x="outcome", y=feat, order=["W", "D", "L"],
                   palette=colors, ax=ax, inner="box", cut=0)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_title(label); ax.set_xlabel(""); ax.set_ylabel("")

plt.tight_layout()
plt.savefig("plot3_form_vs_outcome.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — H2H Coverage & Impact
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Plot 4 — H2H: Coverage & Dampak", fontweight="bold")

# H2H null coverage
h2h_null_pct = train[H2H_COLS].isnull().mean() * 100
axes[0].barh(h2h_null_pct.index, h2h_null_pct.values, color="salmon")
axes[0].set_xlabel("% Missing"); axes[0].set_title("H2H Null Rate (%)")
axes[0].axvline(50, color="black", linestyle="--")

# H2H ada vs tidak → win rate
train["has_h2h"] = train[H2H_COLS[0]].notna()
h2h_wr = train.groupby("has_h2h")["outcome"].apply(
    lambda x: x.value_counts(normalize=True)
).unstack().fillna(0)
h2h_wr.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#3498db", "#e74c3c"],
             rot=0)
axes[1].set_xticklabels(["No H2H", "Has H2H"])
axes[1].set_title("Win Rate: Punya H2H vs Tidak")
axes[1].set_ylabel("Proporsi"); axes[1].legend(["D", "L", "W"])

# H2H GD vs goal diff aktual
valid = train.dropna(subset=["h2h_gd_last5_simple_feat"])
axes[2].scatter(valid["h2h_gd_last5_simple_feat"], valid["goal_diff"],
                alpha=0.1, s=5, color="purple")
# Add regression line
m, b, r, p, _ = stats.linregress(
    valid["h2h_gd_last5_simple_feat"], valid["goal_diff"]
)
x_range = np.linspace(valid["h2h_gd_last5_simple_feat"].min(),
                       valid["h2h_gd_last5_simple_feat"].max(), 100)
axes[2].plot(x_range, m * x_range + b, color="red", linewidth=2,
             label=f"r={r:.3f}, p={p:.4f}")
axes[2].set_xlabel("H2H GD Last5 Simple"); axes[2].set_ylabel("Actual Goal Diff")
axes[2].set_title("H2H GD vs Actual GD"); axes[2].legend()

plt.tight_layout()
plt.savefig("plot4_h2h.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Context Features: Travel, Altitude, Temperature
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Plot 5 — Context Features vs Outcome", fontweight="bold")

ctx_plot_feats = [
    ("travel_stress_diff_ctx", "Travel Stress Diff (km)"),
    ("altitude_shock_team_ctx", "Altitude Shock Team (m)"),
    ("altitude_shock_opp_ctx", "Altitude Shock Opp (m)"),
    ("temp_stress_ctx", "Temperature Stress (°C)"),
    ("log_gdp_diff_ctx", "Log GDP Diff"),
    ("log_pop_diff_ctx", "Log Pop Diff"),
]

for ax, (feat, label) in zip(axes.flatten(), ctx_plot_feats):
    data_clean = train.dropna(subset=[feat])
    sns.boxplot(data=data_clean, x="outcome", y=feat, order=["W", "D", "L"],
                palette=colors, ax=ax, showfliers=False)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_title(label); ax.set_xlabel(""); ax.set_ylabel("")

plt.tight_layout()
plt.savefig("plot5_context_vs_outcome.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Correlation Heatmap (Top fitur terhadap goal_diff)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Plot 6 — Correlation Analysis", fontweight="bold")

# Corr with goal_diff → bar chart
numeric_train = train[FEAT_COLS + ["goal_diff"]].select_dtypes(include=[np.number])
corr_with_gd = numeric_train.corr()["goal_diff"].drop("goal_diff").sort_values()
top_bottom = pd.concat([corr_with_gd.head(15), corr_with_gd.tail(15)])
bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in top_bottom.values]
axes[0].barh(range(len(top_bottom)), top_bottom.values, color=bar_colors)
axes[0].set_yticks(range(len(top_bottom)))
axes[0].set_yticklabels(top_bottom.index, fontsize=8)
axes[0].axvline(0, color="black")
axes[0].set_title("Top 15 & Bottom 15 Korelasi dengan Goal Diff")
axes[0].set_xlabel("Pearson r")

# Heatmap dari fitur kunci saja
key_feats = [
    "elo_diff_feat", "form_diff_feat", "pts_last5_ewma_diff_feat",
    "gd_last5_ewma_diff_feat", "h2h_gd_last5_ewma_feat",
    "travel_stress_diff_ctx", "altitude_shock_team_ctx",
    "log_gdp_diff_ctx", "log_pop_diff_ctx", "temp_stress_ctx",
    "goal_diff"
]
key_feats = [f for f in key_feats if f in numeric_train.columns]
corr_matrix = numeric_train[key_feats].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=axes[1], linewidths=0.5, annot_kws={"size": 8})
axes[1].set_title("Heatmap Korelasi Fitur Kunci")

plt.tight_layout()
plt.savefig("plot6_correlation.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Altitude Shock: Non-linear Threshold Effect
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 7 — Altitude Shock: Threshold Effect", fontweight="bold")

# team altitude shock vs win rate
alt_col = "altitude_shock_team_ctx"
train_alt = train.dropna(subset=[alt_col])
bins_alt = pd.cut(train_alt[alt_col], bins=[-1, 0, 200, 500, 1000, 2000, 5000])
wr_alt = train_alt.groupby(bins_alt, observed=True)["outcome"].apply(
    lambda x: (x == "W").mean()
).reset_index()
wr_alt["label"] = wr_alt[alt_col].astype(str)
axes[0].bar(range(len(wr_alt)), wr_alt["outcome"], color="steelblue", alpha=0.8)
axes[0].axhline(train["outcome"].eq("W").mean(), color="red", linestyle="--",
                label=f"Baseline={train['outcome'].eq('W').mean():.2f}")
axes[0].set_xticks(range(len(wr_alt)))
axes[0].set_xticklabels(wr_alt["label"], rotation=30, ha="right", fontsize=8)
axes[0].set_title("Altitude Shock Tim (m) vs Win Rate")
axes[0].set_ylabel("Win Rate"); axes[0].legend()

# Scatter travel stress vs goal diff
travel_col = "travel_stress_diff_ctx"
train_tr = train.dropna(subset=[travel_col])
# Clip extreme outliers untuk visualisasi
clip_val = train_tr[travel_col].quantile(0.99)
train_tr_clip = train_tr[train_tr[travel_col].abs() <= clip_val]
axes[1].scatter(train_tr_clip[travel_col], train_tr_clip["goal_diff"],
                alpha=0.05, s=4, color="teal")
bins_tr = pd.cut(train_tr_clip[travel_col], bins=10)
mean_gd = train_tr_clip.groupby(bins_tr, observed=True)["goal_diff"].mean().reset_index()
mean_gd["mid"] = mean_gd[travel_col].apply(lambda x: x.mid)
axes[1].plot(mean_gd["mid"], mean_gd["goal_diff"], color="red", linewidth=2,
             marker="o", markersize=4, label="Mean per bin")
axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
axes[1].axvline(0, color="black", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Travel Stress Diff (km, positif = lawan yang lebih jauh travel)")
axes[1].set_ylabel("Goal Diff"); axes[1].set_title("Travel Stress vs Goal Diff")
axes[1].legend()

plt.tight_layout()
plt.savefig("plot7_altitude_travel.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — Confederation Bias & Venue Freq
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 8 — Confederation & Venue Effects", fontweight="bold")

# Confederation TE vs outcome
conf_col = "confederation_team_te_ctx"
bins_conf = pd.qcut(train[conf_col], q=5, duplicates="drop")
wr_conf = train.groupby(bins_conf, observed=True)["outcome"].apply(
    lambda x: (x == "W").mean()
).reset_index()
wr_conf["label"] = wr_conf[conf_col].astype(str)
axes[0].bar(range(len(wr_conf)), wr_conf["outcome"], color="mediumpurple", alpha=0.8)
axes[0].axhline(train["outcome"].eq("W").mean(), color="red", linestyle="--",
                label="Baseline")
axes[0].set_xticks(range(len(wr_conf)))
axes[0].set_xticklabels(wr_conf["label"], rotation=30, ha="right", fontsize=8)
axes[0].set_title("Confederation TE (quintiles) vs Win Rate")
axes[0].set_ylabel("Win Rate"); axes[0].legend()

# Venue frequency vs goal diff
venue_col = "venue_country_freq_ctx"
bins_venue = pd.qcut(train[venue_col], q=5, duplicates="drop")
mean_gd_venue = train.groupby(bins_venue, observed=True)["goal_diff"].mean().reset_index()
mean_gd_venue["label"] = mean_gd_venue[venue_col].astype(str)
axes[1].bar(range(len(mean_gd_venue)), mean_gd_venue["goal_diff"],
            color="darkorange", alpha=0.8)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_xticks(range(len(mean_gd_venue)))
axes[1].set_xticklabels(mean_gd_venue["label"], rotation=30, ha="right", fontsize=8)
axes[1].set_title("Venue Country Freq (quintiles) vs Avg Goal Diff")
axes[1].set_ylabel("Avg Goal Diff")

plt.tight_layout()
plt.savefig("plot8_confederation_venue.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9 — Null Pattern: Siapa yang sering hilang H2H?
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 9 — Analisis Null Pattern H2H", fontweight="bold")

# H2H missing rate berdasarkan ELO quintile
train["elo_quintile"] = pd.qcut(train["elo_team_feat"], q=5,
                                  labels=["Q1\n(Lemah)", "Q2", "Q3", "Q4", "Q5\n(Kuat)"])
h2h_missing = train.groupby("elo_quintile", observed=True)[H2H_COLS[0]].apply(
    lambda x: x.isna().mean() * 100
).reset_index()
axes[0].bar(h2h_missing["elo_quintile"].astype(str), h2h_missing[H2H_COLS[0]],
            color="coral", alpha=0.8)
axes[0].set_xlabel("ELO Quintile Tim")
axes[0].set_ylabel("% H2H Missing")
axes[0].set_title("Tim Lemah vs Kuat: Seberapa Sering H2H Hilang?")

# Days since last match vs H2H availability
train["has_h2h_num"] = train[H2H_COLS[0]].notna().astype(int)
days_col = "days_since_last_team_feat"
train_d = train.dropna(subset=[days_col])
bins_days = pd.cut(train_d[days_col].clip(0, 100), bins=10)
h2h_avail = train_d.groupby(bins_days, observed=True)["has_h2h_num"].mean().reset_index()
h2h_avail["label"] = h2h_avail[days_col].astype(str)
axes[1].bar(range(len(h2h_avail)), h2h_avail["has_h2h_num"],
            color="steelblue", alpha=0.8)
axes[1].set_xticks(range(len(h2h_avail)))
axes[1].set_xticklabels(h2h_avail["label"], rotation=30, ha="right", fontsize=8)
axes[1].set_title("Days Since Last Match vs H2H Availability Rate")
axes[1].set_ylabel("Proporsi Punya H2H")

plt.tight_layout()
plt.savefig("plot9_null_pattern.png")
plt.show()

print("\nSemua 9 plot tersimpan sebagai PNG.")
print("\nSummary statistik korelasi utama:")
for feat in ["elo_diff_feat", "form_diff_feat", "travel_stress_diff_ctx",
             "altitude_shock_team_ctx", "log_gdp_diff_ctx"]:
    if feat in numeric_train.columns:
        r = numeric_train[feat].corr(numeric_train["goal_diff"])
        print(f"  {feat:40s}: r={r:+.4f}")
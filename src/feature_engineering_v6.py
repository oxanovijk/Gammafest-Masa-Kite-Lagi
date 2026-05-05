"""
Feature Engineering V6 — Gender-Aware + Friendly + 6 New Features
=================================================================
Delta dari V5:
  1. Gender interaction features (elo_diff_x_women, elo_team_x_women)
  2. Friendly interaction (elo_diff_x_friendly)
  3. Fatigue score
  4. Altitude x home interaction
  5. Form acceleration
  6. Scoring streak & clean sheet streak
  7. Tournament tier TE per gender
  8. Confederation strength mapping
"""
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"
OUT_TRAIN = DATA_DIR / "train_final.csv"
OUT_TEST  = DATA_DIR / "test_final.csv"

ELO_INIT = 1500
ELO_K    = 32
ELO_HOME_ADVANTAGE = 35

TOURNAMENT_K_WEIGHT = {
    "FIFA World Cup": 60, "FIFA World Cup qualification": 40,
    "Confederations Cup": 50, "Copa America": 50, "UEFA Euro": 50,
    "UEFA Euro qualification": 40, "African Cup of Nations": 50,
    "African Cup of Nations qualification": 40, "AFC Asian Cup": 50,
    "AFC Asian Cup qualification": 40, "Gold Cup": 45,
    "CONCACAF Championship": 45, "CONCACAF Gold Cup qualification": 40,
    "CONCACAF Nations League": 40, "UEFA Nations League": 45,
    "Oceania Nations Cup": 40, "Friendly": 20, "Olympic Games": 40,
    "Finalissima": 50, "CONMEBOL Nations League": 40,
}

CONFEDERATION_K_MULTIPLIER = {
    "UEFA": 1.00, "CONMEBOL": 1.00, "CAF": 0.95,
    "AFC": 0.90, "CONCACAF": 0.90, "OFC": 0.80, "Unknown": 0.85,
}

CONFEDERATION_STRENGTH = {
    "UEFA": 1.00, "CONMEBOL": 0.95, "CONCACAF": 0.75,
    "AFC": 0.70, "CAF": 0.65, "OFC": 0.40, "Unknown": 0.55,
}

EWMA_HALF_LIFE = 90
EWMA_ALPHA = math.log(2) / EWMA_HALF_LIFE

# ===========================================================================
# 1. LOAD & MERGE
# ===========================================================================
def load_and_merge():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    train["is_test"] = False
    test["is_test"]  = True
    for col in ["team_goals", "opp_goals"]:
        if col not in test.columns:
            test[col] = np.nan
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    return df

# ===========================================================================
# 2. ELO RATING
# ===========================================================================
def get_k_factor(tournament, conf_a, conf_b):
    base_k = TOURNAMENT_K_WEIGHT.get(tournament, ELO_K)
    mult_a = CONFEDERATION_K_MULTIPLIER.get(conf_a, 0.85)
    mult_b = CONFEDERATION_K_MULTIPLIER.get(conf_b, 0.85)
    return base_k * (mult_a + mult_b) / 2.0

def calc_elo_change(elo_a, elo_b, score_a, k, home_adv_a=0.0):
    effective_elo_a = elo_a + home_adv_a
    expected = 1.0 / (1.0 + 10 ** ((elo_b - effective_elo_a) / 400.0))
    return k * (score_a - expected)

def compute_elo(df):
    elo = defaultdict(lambda: ELO_INIT)
    n = len(df)
    elo_team_vals = np.full(n, np.nan)
    elo_opp_vals  = np.full(n, np.nan)
    for match_id, group in df.groupby("match_id", sort=False):
        if len(group) != 2: continue
        idx = group.index.tolist()
        row_a, row_b = df.loc[idx[0]], df.loc[idx[1]]
        gender = row_a["gender"]
        team_a, team_b = row_a["team"], row_b["team"]
        conf_a = str(row_a.get("confederation_team", "Unknown"))
        conf_b = str(row_b.get("confederation_team", "Unknown"))
        key_a, key_b = (team_a, gender), (team_b, gender)
        cur_a, cur_b = elo[key_a], elo[key_b]
        elo_team_vals[idx[0]], elo_opp_vals[idx[0]] = cur_a, cur_b
        elo_team_vals[idx[1]], elo_opp_vals[idx[1]] = cur_b, cur_a
        goals_a, goals_b = row_a["team_goals"], row_b["team_goals"]
        if pd.notna(goals_a) and pd.notna(goals_b):
            score_a = 1.0 if goals_a > goals_b else (0.0 if goals_a < goals_b else 0.5)
            k = get_k_factor(row_a["tournament"], conf_a, conf_b)
            gd = abs(goals_a - goals_b)
            if gd == 2: k *= 1.5
            elif gd == 3: k *= 1.75
            elif gd > 3: k *= (1.75 + (gd - 3) / 8.0)
            is_home_a = row_a.get("is_home", 0)
            is_home_b = row_b.get("is_home", 0)
            neutral = row_a.get("neutral", 0)
            home_adv_a = 0.0
            if neutral != 1:
                if is_home_a == 1: home_adv_a = ELO_HOME_ADVANTAGE
                elif is_home_b == 1: home_adv_a = -ELO_HOME_ADVANTAGE
            delta = calc_elo_change(cur_a, cur_b, score_a, k, home_adv_a)
            elo[key_a] = cur_a + delta
            elo[key_b] = cur_b - delta
    df["elo_team"] = elo_team_vals
    df["elo_opp"]  = elo_opp_vals
    return df

# ===========================================================================
# 3. PI-RATINGS
# ===========================================================================
def compute_pi_ratings(df):
    pi_home = defaultdict(float)
    pi_away = defaultdict(float)
    n = len(df)
    pi_team_vals = np.full(n, np.nan)
    pi_opp_vals  = np.full(n, np.nan)
    LR_HOME = 0.035
    LR_AWAY = 0.035
    for match_id, group in df.groupby("match_id", sort=False):
        if len(group) != 2: continue
        idx = group.index.tolist()
        row_a, row_b = df.loc[idx[0]], df.loc[idx[1]]
        gender = row_a["gender"]
        team_a, team_b = row_a["team"], row_b["team"]
        key_a, key_b = (team_a, gender), (team_b, gender)
        is_home_a = row_a.get("is_home", 0)
        is_home_b = row_b.get("is_home", 0)
        neutral = row_a.get("neutral", 0)
        ra_h, ra_a = pi_home[key_a], pi_away[key_a]
        rb_h, rb_a = pi_home[key_b], pi_away[key_b]
        if neutral == 1:
            eff_ra, eff_rb = (ra_h+ra_a)/2.0, (rb_h+rb_a)/2.0
        elif is_home_a == 1:
            eff_ra, eff_rb = ra_h, rb_a
        elif is_home_b == 1:
            eff_ra, eff_rb = ra_a, rb_h
        else:
            eff_ra, eff_rb = (ra_h+ra_a)/2.0, (rb_h+rb_a)/2.0
        pi_team_vals[idx[0]], pi_opp_vals[idx[0]] = eff_ra, eff_rb
        pi_team_vals[idx[1]], pi_opp_vals[idx[1]] = eff_rb, eff_ra
        goals_a, goals_b = row_a["team_goals"], row_b["team_goals"]
        if pd.notna(goals_a) and pd.notna(goals_b):
            gd, e_gd = goals_a-goals_b, eff_ra-eff_rb
            error = gd - e_gd
            k = get_k_factor(row_a["tournament"],
                             str(row_a.get("confederation_team","Unknown")),
                             str(row_b.get("confederation_team","Unknown")))
            k_mult = k / 32.0
            lr_h_eff, lr_a_eff = LR_HOME*k_mult, LR_AWAY*k_mult
            if neutral == 1:
                pi_home[key_a] += lr_h_eff*error*0.5; pi_away[key_a] += lr_a_eff*error*0.5
                pi_home[key_b] -= lr_h_eff*error*0.5; pi_away[key_b] -= lr_a_eff*error*0.5
            elif is_home_a == 1:
                pi_home[key_a] += lr_h_eff*error; pi_away[key_b] -= lr_a_eff*error
            elif is_home_b == 1:
                pi_away[key_a] += lr_a_eff*error; pi_home[key_b] -= lr_h_eff*error
            else:
                pi_home[key_a] += lr_h_eff*error*0.5; pi_away[key_a] += lr_a_eff*error*0.5
                pi_home[key_b] -= lr_h_eff*error*0.5; pi_away[key_b] -= lr_a_eff*error*0.5
    df["pi_team"] = pi_team_vals
    df["pi_opp"]  = pi_opp_vals
    return df

# ===========================================================================
# 4. EWMA ROLLING
# ===========================================================================
def _ewma_agg(history, current_date, last_n=None):
    items = list(history)
    if last_n: items = items[-last_n:]
    if not items: return None
    weights = [math.exp(-EWMA_ALPHA * max(0, (current_date - d).days)) for (_, _, _, d) in items]
    tw = sum(weights) or 1e-9
    return {
        "pts": sum(w*x[0] for w,x in zip(weights,items))/tw,
        "gd":  sum(w*(x[1]-x[2]) for w,x in zip(weights,items))/tw,
        "gf":  sum(w*x[1] for w,x in zip(weights,items))/tw,
        "ga":  sum(w*x[2] for w,x in zip(weights,items))/tw,
        "wr":  sum(w*(1.0 if x[0]==3 else 0.0) for w,x in zip(weights,items))/tw,
    }

def compute_rolling(df):
    n = len(df)
    cols = {k: np.full(n, np.nan) for k in [
        "pts5","pts10","gd5","gf5","ga5","wr10","days_since"
    ]}
    history = defaultdict(lambda: deque(maxlen=10))
    last_date = {}
    # V6: Track streaks
    scoring_streak = defaultdict(int)  # consecutive matches scoring >=1 goal
    clean_sheet_streak = defaultdict(int)  # consecutive clean sheets
    streak_scoring = np.full(n, np.nan)
    streak_clean   = np.full(n, np.nan)
    streak_scoring_allowed = np.full(n, np.nan)  # consecutive conceding
    streak_conceding_allowed = defaultdict(int)

    for i in range(n):
        row = df.iloc[i]
        key = (row["team"], row["gender"])
        date_i = row["date"]
        hist = history[key]

        # Rolling stats
        if len(hist) > 0:
            a5 = _ewma_agg(hist, date_i, 5)
            a10 = _ewma_agg(hist, date_i, 10)
            if a5:
                cols["pts5"][i] = a5["pts"]
                cols["gd5"][i]  = a5["gd"]
                cols["gf5"][i]  = a5["gf"]
                cols["ga5"][i]  = a5["ga"]
            if a10:
                cols["pts10"][i] = a10["pts"]
                cols["wr10"][i]  = a10["wr"]

        if key in last_date:
            cols["days_since"][i] = (date_i - last_date[key]).days

        # Streaks before this match
        streak_scoring[i] = scoring_streak[key]
        streak_clean[i] = clean_sheet_streak[key]
        streak_conceding_allowed[i] = streak_conceding_allowed[key]

        gf, ga = row["team_goals"], row["opp_goals"]
        if pd.notna(gf) and pd.notna(ga):
            pts = 3 if gf > ga else (1 if gf == ga else 0)
            hist.append((pts, gf, ga, date_i))
            # Update streaks
            if gf >= 1: scoring_streak[key] += 1
            else: scoring_streak[key] = 0
            if ga == 0: clean_sheet_streak[key] += 1
            else: clean_sheet_streak[key] = 0
            if ga >= 1: streak_conceding_allowed[key] += 1
            else: streak_conceding_allowed[key] = 0
        last_date[key] = date_i

    for k, arr in cols.items():
        df[f"team_{k}"] = arr
    df["team_scoring_streak"] = streak_scoring
    df["team_clean_sheet_streak"] = streak_clean
    df["team_conceding_streak"] = streak_conceding_allowed
    return df

# ===========================================================================
# 5. H2H
# ===========================================================================
def compute_h2h(df):
    n = len(df)
    h2h_gd = np.full(n, np.nan)
    h2h_pts = np.full(n, np.nan)
    h2h_hist = defaultdict(lambda: deque(maxlen=5))
    for i in range(n):
        row = df.iloc[i]
        key = (row["team"], row["opponent"], row["gender"])
        date_i = row["date"]
        hist = h2h_hist[key]
        if len(hist) > 0:
            agg = _ewma_agg(hist, date_i)
            if agg:
                h2h_gd[i]  = agg["gd"]
                h2h_pts[i] = agg["pts"]
        gf, ga = row["team_goals"], row["opp_goals"]
        if pd.notna(gf) and pd.notna(ga):
            pts = 3 if gf > ga else (1 if gf == ga else 0)
            hist.append((pts, gf, ga, date_i))
    df["h2h_gd"] = h2h_gd
    df["h2h_pts"] = h2h_pts
    return df

# ===========================================================================
# 6. MIRROR OPPONENT
# ===========================================================================
def mirror_opponent(df):
    team_cols = ["team_pts5","team_pts10","team_gd5","team_gf5","team_ga5",
                 "team_wr10","team_days_since",
                 "team_scoring_streak","team_clean_sheet_streak","team_conceding_streak"]
    opp_cols  = ["opp_pts5","opp_pts10","opp_gd5","opp_gf5","opp_ga5",
                 "opp_wr10","opp_days_since",
                 "opp_scoring_streak","opp_clean_sheet_streak","opp_conceding_streak"]
    for oc in opp_cols: df[oc] = np.nan
    for _, group in df.groupby("match_id", sort=False):
        if len(group) != 2: continue
        idx = group.index.tolist()
        for tc, oc in zip(team_cols, opp_cols):
            df.at[idx[0], oc] = df.at[idx[1], tc]
            df.at[idx[1], oc] = df.at[idx[0], tc]
    return df

# ===========================================================================
# 7. DERIVED + V6 NEW FEATURES
# ===========================================================================
def compute_derived(df):
    # Core diffs
    df["elo_diff"] = df["elo_team"] - df["elo_opp"]
    df["pi_diff"] = df["pi_team"] - df["pi_opp"]
    df["pts5_diff"] = df["team_pts5"] - df["opp_pts5"]
    df["pts10_diff"] = df["team_pts10"] - df["opp_pts10"]
    df["gd5_diff"] = df["team_gd5"] - df["opp_gd5"]

    # Form composite
    df["form_team"] = df["team_pts5"].fillna(0) + df["team_gd5"].fillna(0)*0.5
    df["form_opp"]  = df["opp_pts5"].fillna(0) + df["opp_gd5"].fillna(0)*0.5
    df["form_diff"] = df["form_team"] - df["form_opp"]

    # Binary context
    df["is_home"]    = df["is_home"].fillna(0).astype(int)
    df["is_neutral"] = df["neutral"].fillna(0).astype(int)
    df["is_away"]    = ((df["is_home"]==0) & (df["is_neutral"]==0)).astype(int)
    df["is_women"]   = (df["Id"].str.startswith("W")).astype(int)
    df["is_friendly"]= (df["tournament"].str.contains("Friendly", case=False, na=False)).astype(int)

    # Tournament tier
    def get_tournament_tier(t_name):
        t = str(t_name).lower()
        if "friendly" in t: return 1
        if "world cup" in t and "qualification" not in t: return 5
        if "olympic" in t and "qualification" not in t: return 5
        if "finalissima" in t: return 5
        for t2 in ["euro","copa america","african cup of nations","afc asian cup",
                    "gold cup","confederations cup","concacaf championship"]:
            if t2 in t and "qualification" not in t: return 4
        if "qualification" in t or "nations league" in t: return 3
        return 2
    df["tournament_tier"] = df["tournament"].apply(get_tournament_tier)

    # Non-linear Elo
    df["elo_diff_sq"] = df["elo_diff"]**2 * np.sign(df["elo_diff"])

    # Days rest difference
    df["days_rest_diff"] = df["team_days_since"] - df["opp_days_since"]

    # Attack vs defense mismatch
    df["atk_def_mismatch"] = (df["team_gf5"].fillna(0) - df["opp_ga5"].fillna(0))

    # Goal-scoring volatility
    df["goal_volatility_team"] = df["team_gf5"].fillna(0) + df["team_ga5"].fillna(0)
    df["goal_volatility_opp"]  = df["opp_gf5"].fillna(0) + df["opp_ga5"].fillna(0)

    # ============================
    # V6 NEW FEATURES
    # ============================

    # 1. Gender interaction features (P0)
    df["elo_diff_x_women"] = df["elo_diff"] * df["is_women"].astype(float)
    df["elo_team_x_women"] = df["elo_team"] * df["is_women"].astype(float)

    # 2. Friendly interaction (P3)
    df["elo_diff_x_friendly"] = df["elo_diff"] * df["is_friendly"].astype(float)

    # 3. Fatigue score (P4)
    dt = df.get("distance_travel_team", pd.Series(0, index=df.index)).fillna(0)
    ds = df["team_days_since"].fillna(999).clip(lower=0)
    df["fatigue_score"] = (1.0 / (ds + 1.0)) * dt / 1000.0

    # 4. Altitude x home interaction (P4)
    alt = df.get("altitude_venue", pd.Series(0, index=df.index)).fillna(0)
    df["altitude_x_home"] = alt * df["is_home"].astype(float)

    # 5. Form acceleration (P4) — compute lagged form diff per team
    df["form_diff_prev"] = np.nan
    team_form_history = {}
    for i in range(len(df)):
        row = df.iloc[i]
        key = (row["team"], row["gender"])
        if key in team_form_history:
            df.at[df.index[i], "form_diff_prev"] = team_form_history[key]
        team_form_history[key] = row["form_diff"]
    df["form_accel"] = df["form_diff"] - df["form_diff_prev"].fillna(0)

    # 6. Confederation strength index (P4)
    df["conf_strength_team"] = df["confederation_team"].map(CONFEDERATION_STRENGTH).fillna(0.55)
    df["conf_strength_opp"]  = df["confederation_opp"].map(CONFEDERATION_STRENGTH).fillna(0.55)
    df["conf_strength_diff"] = df["conf_strength_team"] - df["conf_strength_opp"]

    # 7. Goal ratio (scoring / conceding) as overall team quality proxy
    df["goal_ratio_team"] = df["team_gf5"].fillna(0) / (df["team_ga5"].fillna(0) + 0.5)
    df["goal_ratio_opp"]  = df["opp_gf5"].fillna(0) / (df["opp_ga5"].fillna(0) + 0.5)
    df["goal_ratio_diff"] = df["goal_ratio_team"] - df["goal_ratio_opp"]

    return df

# ===========================================================================
# 8. CONTEXT FEATURES
# ===========================================================================
def compute_context(df):
    gdp_t = df.get("gdp_per_capita_team", pd.Series(np.nan, index=df.index))
    gdp_o = df.get("gdp_per_capita_opp", pd.Series(np.nan, index=df.index))
    df["log_gdp_diff"] = np.log1p(gdp_t.fillna(0)) - np.log1p(gdp_o.fillna(0))

    pop_t = df.get("population_team", pd.Series(np.nan, index=df.index))
    pop_o = df.get("population_opp", pd.Series(np.nan, index=df.index))
    df["log_pop_diff"] = np.log1p(pop_t.fillna(0)) - np.log1p(pop_o.fillna(0))

    alt = df.get("altitude_venue", pd.Series(0, index=df.index)).fillna(0)
    df["altitude"] = alt

    dt = df.get("distance_travel_team", pd.Series(0, index=df.index)).fillna(0)
    do = df.get("distance_travel_opp", pd.Series(0, index=df.index)).fillna(0)
    df["travel_diff"] = dt - do

    df["temperature"] = df.get("temperature_venue", pd.Series(np.nan, index=df.index)).fillna(15)

    if "rank_team" in df.columns and "rank_opponent" in df.columns:
        df["rank_diff"] = df["rank_team"].fillna(100) - df["rank_opponent"].fillna(100)
    else:
        df["rank_diff"] = 0

    return df

# ===========================================================================
# 9. TARGET ENCODING (with gender separation)
# ===========================================================================
def target_encode(df, col, target="team_goals", smoothing=50):
    train_mask = df["is_test"] == False
    train_df = df[train_mask]
    global_mean = train_df[target].mean()
    agg = train_df.groupby(col)[target].agg(["mean","count"])
    agg["te"] = (agg["count"]*agg["mean"] + smoothing*global_mean)/(agg["count"]+smoothing)
    te_map = agg["te"].to_dict()
    df[f"{col}_te"] = df[col].map(te_map).fillna(global_mean)
    return df

def target_encode_per_gender(df, col, target="team_goals", smoothing=50):
    """Target encoding SEPARATELY for men and women (P0)."""
    train_mask = df["is_test"] == False
    train_df_full = df[train_mask]
    global_mean_m = train_df_full[train_df_full["Id"].str.startswith("M")][target].mean()
    global_mean_w = train_df_full[train_df_full["Id"].str.startswith("W")][target].mean()
    name = f"{col}_te_gender"
    df[name] = np.nan
    for gender_prefix, gmean in [("M", global_mean_m), ("W", global_mean_w)]:
        gender_mask = df["Id"].str.startswith(gender_prefix)
        train_gender = df[train_mask & gender_mask]
        agg = train_gender.groupby(col)[target].agg(["mean","count"])
        agg["te"] = (agg["count"]*agg["mean"] + smoothing*gmean)/(agg["count"]+smoothing)
        te_map = agg["te"].to_dict()
        df.loc[gender_mask, name] = df.loc[gender_mask, col].map(te_map).fillna(gmean)
    return df

# ===========================================================================
# 10. FINALIZE
# ===========================================================================
def finalize(df):
    # Target encode (standard + per-gender)
    df = target_encode(df, "tournament")
    df = target_encode_per_gender(df, "tournament")  # P0: per-gender TE

    if "confederation_team" in df.columns:
        df = target_encode(df, "confederation_team")
        df = target_encode_per_gender(df, "confederation_team")
    if "confederation_opp" in df.columns:
        df = target_encode(df, "confederation_opp")
        df = target_encode_per_gender(df, "confederation_opp")

    FEATURES = [
        # Elo
        "elo_team","elo_opp","elo_diff","elo_diff_sq",
        "pi_team","pi_opp","pi_diff",
        # Form
        "form_team","form_opp","form_diff",
        "pts5_diff","pts10_diff","gd5_diff",
        # Team rolling
        "team_gf5","team_ga5","team_wr10",
        # Opponent rolling
        "opp_gf5","opp_ga5","opp_wr10",
        # H2H
        "h2h_gd","h2h_pts",
        # Days
        "team_days_since","opp_days_since","days_rest_diff",
        # Context binary
        "is_home","is_away","is_neutral","is_women","is_friendly",
        "tournament_tier",
        # Derived
        "atk_def_mismatch","goal_volatility_team","goal_volatility_opp",
        # Geo/socio
        "log_gdp_diff","log_pop_diff","altitude","travel_diff",
        "temperature","rank_diff",
        # Target encoding
        "tournament_te","tournament_te_gender",
        # V6 NEW
        "elo_diff_x_women","elo_team_x_women",
        "elo_diff_x_friendly",
        "fatigue_score",
        "altitude_x_home",
        "form_accel",
        "team_scoring_streak","team_clean_sheet_streak","team_conceding_streak",
        "opp_scoring_streak","opp_clean_sheet_streak","opp_conceding_streak",
        "conf_strength_team","conf_strength_opp","conf_strength_diff",
        "goal_ratio_team","goal_ratio_opp","goal_ratio_diff",
    ]

    if "confederation_team_te" in df.columns:
        FEATURES.append("confederation_team_te")
        FEATURES.append("confederation_team_te_gender")
    if "confederation_opp_te" in df.columns:
        FEATURES.append("confederation_opp_te")
        FEATURES.append("confederation_opp_te_gender")

    FEATURES = [f for f in FEATURES if f in df.columns]

    train_df = df[df["is_test"]==False].copy()
    test_df  = df[df["is_test"]==True].copy()

    train_out = train_df[["Id"]+FEATURES+["team_goals","opp_goals"]].copy()
    test_out  = test_df[["Id"]+FEATURES].copy()

    train_raw_info = train_df[["Id","date","tournament"]].copy()
    test_raw_info  = test_df[["Id","date","tournament"]].copy()
    train_raw_info.to_csv(DATA_DIR/"train_meta.csv", index=False)
    test_raw_info.to_csv(DATA_DIR/"test_meta.csv", index=False)

    train_out.to_csv(OUT_TRAIN, index=False)
    test_out.to_csv(OUT_TEST, index=False)

    print(f"[OK] train_final.csv: {train_out.shape}")
    print(f"[OK] test_final.csv:  {test_out.shape}")
    print(f"\nFeatures ({len(FEATURES)}):")
    for f in FEATURES:
        na_pct = train_out[f].isna().mean()*100
        print(f"  {f:40s}  NA: {na_pct:5.1f}%")

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("="*60)
    print("FEATURE ENGINEERING V6 (GENDER-AWARE + FRIENDLY + STREAKS)")
    print("="*60)
    print("\n[1/8] Loading data...")
    df = load_and_merge()
    print(f"      Total: {len(df)} rows")
    print("\n[2/8] Computing Elo...")
    df = compute_elo(df)
    print("\n[3/8] Computing Pi-Ratings...")
    df = compute_pi_ratings(df)
    print("\n[4/8] Computing EWMA rolling + streaks...")
    df = compute_rolling(df)
    print("\n[5/8] Computing H2H...")
    df = compute_h2h(df)
    print("\n[6/8] Mirroring opponent features...")
    df = mirror_opponent(df)
    print("\n[7/8] Computing derived features (V6)...")
    df = compute_derived(df)
    print("\n[8/8] Computing context & finalizing...")
    df = compute_context(df)
    finalize(df)
    print("\n"+"="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()
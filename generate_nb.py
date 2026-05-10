import nbformat as nbf
import os
from pathlib import Path

nb = nbf.v4.new_notebook()

md1 = nbf.v4.new_markdown_cell('# Gammafest Masa Kite Lagi - Data Preprocessing\nPipeline untuk feature engineering dari `train.csv` & `test.csv` menjadi `train_final.csv` & `test_final.csv`.')
code1 = nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import math
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('dataset')
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH  = DATA_DIR / 'test.csv'

OUT_TRAIN_FINAL = DATA_DIR / 'train_final.csv'
OUT_TEST_FINAL  = DATA_DIR / 'test_final.csv'
''')

md2 = nbf.v4.new_markdown_cell('## 1. Core Historical Features (Anggota 1)\nMenghitung Elo rating, EWMA rolling stats, dan Head-to-Head.')
code2 = nbf.v4.new_code_cell('''
# ===========================================================================
# 1. LOAD DATA & GABUNGKAN SECARA KRONOLOGIS
# ===========================================================================
def load_and_merge():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    train['is_test'] = False
    test['is_test']  = True

    for col in ['team_goals', 'opp_goals']:
        if col not in test.columns:
            test[col] = np.nan
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan

    df = pd.concat([train, test], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'match_id', 'Id']).reset_index(drop=True)
    return df

df = load_and_merge()
print(f'Total data gabungan: {len(df)} baris')
''')

code3 = nbf.v4.new_code_cell('''
# ===========================================================================
# 2. ELO RATING V2
# ===========================================================================
ELO_INIT = 1500
ELO_K    = 32
ELO_HOME_ADVANTAGE = 100

TOURNAMENT_K_WEIGHT = {
    'FIFA World Cup': 60,
    'FIFA World Cup qualification': 40,
    'Confederations Cup': 50,
    'Copa America': 50,
    'UEFA Euro': 50,
    'UEFA Euro qualification': 40,
    'African Cup of Nations': 50,
    'African Cup of Nations qualification': 40,
    'AFC Asian Cup': 50,
    'AFC Asian Cup qualification': 40,
    'Gold Cup': 45,
    'CONCACAF Championship': 45,
    'CONCACAF Gold Cup qualification': 40,
    'CONCACAF Nations League': 40,
    'UEFA Nations League': 45,
    'Oceania Nations Cup': 40,
    'Friendly': 20,
    'Olympic Games': 40,
    'Confederations Cup': 50,
    'Finalissima': 50,
    'CONMEBOL Nations League': 40,
}

CONFEDERATION_K_MULTIPLIER = {
    'UEFA':     1.00,
    'CONMEBOL': 1.00,
    'CAF':      0.95,
    'AFC':      0.90,
    'CONCACAF': 0.90,
    'OFC':      0.80,
    'Unknown':  0.85,
}

def get_k_factor(tournament: str, conf_a: str, conf_b: str) -> float:
    base_k = TOURNAMENT_K_WEIGHT.get(tournament, ELO_K)
    mult_a = CONFEDERATION_K_MULTIPLIER.get(conf_a, 0.85)
    mult_b = CONFEDERATION_K_MULTIPLIER.get(conf_b, 0.85)
    conf_mult = (mult_a + mult_b) / 2.0
    return base_k * conf_mult

def calc_elo_change(elo_a: float, elo_b: float, score_a: float, k: float, home_adv_a: float = 0.0) -> float:
    effective_elo_a = elo_a + home_adv_a
    expected = 1.0 / (1.0 + 10 ** ((elo_b - effective_elo_a) / 400.0))
    return k * (score_a - expected)

def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    elo = defaultdict(lambda: ELO_INIT)
    n = len(df)
    elo_team_vals = np.full(n, np.nan)
    elo_opp_vals  = np.full(n, np.nan)

    match_groups = df.groupby('match_id', sort=False)

    for match_id, group in match_groups:
        if len(group) != 2: continue
        idx = group.index.tolist()
        row_a = df.loc[idx[0]]
        row_b = df.loc[idx[1]]

        gender = row_a['gender']
        team_a, team_b = row_a['team'], row_b['team']
        tournament = row_a['tournament']
        conf_a = str(row_a.get('confederation_team', 'Unknown'))
        conf_b = str(row_b.get('confederation_team', 'Unknown'))

        key_a, key_b = (team_a, gender), (team_b, gender)
        cur_elo_a, cur_elo_b = elo[key_a], elo[key_b]

        elo_team_vals[idx[0]] = cur_elo_a
        elo_opp_vals[idx[0]]  = cur_elo_b
        elo_team_vals[idx[1]] = cur_elo_b
        elo_opp_vals[idx[1]]  = cur_elo_a

        goals_a, goals_b = row_a['team_goals'], row_b['team_goals']
        if pd.notna(goals_a) and pd.notna(goals_b):
            if goals_a > goals_b: score_a = 1.0
            elif goals_a < goals_b: score_a = 0.0
            else: score_a = 0.5

            k = get_k_factor(tournament, conf_a, conf_b)
            gd = abs(goals_a - goals_b)
            if gd == 2: k *= 1.5
            elif gd == 3: k *= 1.75
            elif gd > 3: k *= (1.75 + (gd - 3) / 8.0)

            is_home_a, is_home_b = row_a.get('is_home', 0), row_b.get('is_home', 0)
            neutral = row_a.get('neutral', 0)

            if neutral == 1: home_adv_a = 0.0
            elif is_home_a == 1: home_adv_a = ELO_HOME_ADVANTAGE
            elif is_home_b == 1: home_adv_a = -ELO_HOME_ADVANTAGE
            else: home_adv_a = 0.0

            delta = calc_elo_change(cur_elo_a, cur_elo_b, score_a, k, home_adv_a)
            elo[key_a] = cur_elo_a + delta
            elo[key_b] = cur_elo_b - delta

    df['elo_team_calc']     = elo_team_vals
    df['elo_opponent_calc'] = elo_opp_vals
    return df

df = compute_elo(df)
print('Elo computed.')
''')

code4 = nbf.v4.new_code_cell('''
# ===========================================================================
# 3. EWMA ROLLING STATS
# ===========================================================================
EWMA_HALF_LIFE_DAYS = 90
EWMA_ALPHA = math.log(2) / EWMA_HALF_LIFE_DAYS
ROLLING_WINDOW = 10

def _ewma_aggregate(history_list, current_date, last_n=None):
    items = list(history_list)
    if last_n is not None: items = items[-last_n:]
    if len(items) == 0: return None

    weights = []
    for (pts, gf, ga, d) in items:
        days_gap = max(0, (current_date - d).days)
        weights.append(math.exp(-EWMA_ALPHA * days_gap))

    total_w = sum(weights) or 1e-9
    pts_ewma   = sum(w * x[0] for w, x in zip(weights, items)) / total_w
    gf_ewma    = sum(w * x[1] for w, x in zip(weights, items)) / total_w
    ga_ewma    = sum(w * x[2] for w, x in zip(weights, items)) / total_w
    gd_ewma    = sum(w * (x[1] - x[2]) for w, x in zip(weights, items)) / total_w
    wins_ewma  = sum(w * (1.0 if x[0] == 3 else 0.0) for w, x in zip(weights, items)) / total_w
    pts_simple = sum(x[0] for x in items)

    return {
        'pts_ewma': pts_ewma, 'gd_ewma': gd_ewma, 'avg_gf_ewma': gf_ewma,
        'avg_ga_ewma': ga_ewma, 'win_rate_ewma': wins_ewma, 'pts_simple': pts_simple,
    }

def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    col_names = [
        'pts_last5_ewma', 'pts_last10_ewma', 'gd_last5_ewma', 'gd_last10_ewma',
        'avg_gf_last5_ewma', 'avg_ga_last5_ewma', 'win_rate_last10_ewma',
        'days_since_last_calc', 'pts_last5_simple', 'pts_last10_simple',
    ]
    cols_out = {c: np.full(n, np.nan) for c in col_names}
    history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    last_date = {}

    for i in range(n):
        row = df.iloc[i]
        key = (row['team'], row['gender'])
        date_i = row['date']
        hist = history[key]

        if len(hist) > 0:
            agg5  = _ewma_aggregate(hist, date_i, last_n=5)
            agg10 = _ewma_aggregate(hist, date_i, last_n=10)
            if agg5:
                cols_out['pts_last5_ewma'][i]    = agg5['pts_ewma']
                cols_out['gd_last5_ewma'][i]     = agg5['gd_ewma']
                cols_out['avg_gf_last5_ewma'][i] = agg5['avg_gf_ewma']
                cols_out['avg_ga_last5_ewma'][i] = agg5['avg_ga_ewma']
                cols_out['pts_last5_simple'][i]  = agg5['pts_simple']
            if agg10:
                cols_out['pts_last10_ewma'][i]      = agg10['pts_ewma']
                cols_out['gd_last10_ewma'][i]       = agg10['gd_ewma']
                cols_out['win_rate_last10_ewma'][i] = agg10['win_rate_ewma']
                cols_out['pts_last10_simple'][i]    = agg10['pts_simple']

        if key in last_date:
            cols_out['days_since_last_calc'][i] = (date_i - last_date[key]).days

        gf, ga = row['team_goals'], row['opp_goals']
        if pd.notna(gf) and pd.notna(ga):
            pts = 3 if gf > ga else 1 if gf == ga else 0
            hist.append((pts, gf, ga, date_i))

        last_date[key] = date_i

    for c, arr in cols_out.items(): df[c] = arr
    return df

df = compute_rolling_stats(df)
print('Rolling stats computed.')
''')

code5 = nbf.v4.new_code_cell('''
# ===========================================================================
# 4. HEAD-TO-HEAD EWMA & MIRROR OPPONENT
# ===========================================================================
def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    h2h_pts, h2h_gd = np.full(n, np.nan), np.full(n, np.nan)
    h2h_pts_ewma, h2h_gd_ewma = np.full(n, np.nan), np.full(n, np.nan)
    h2h_history = defaultdict(lambda: deque(maxlen=5))

    for i in range(n):
        row = df.iloc[i]
        key = (row['team'], row['opponent'], row['gender'])
        date_i = row['date']
        hist = h2h_history[key]

        if len(hist) > 0:
            h2h_pts[i] = sum(x[0] for x in hist)
            h2h_gd[i]  = sum(x[1] - x[2] for x in hist)
            agg = _ewma_aggregate(hist, date_i)
            if agg:
                h2h_pts_ewma[i], h2h_gd_ewma[i] = agg['pts_ewma'], agg['gd_ewma']

        gf, ga = row['team_goals'], row['opp_goals']
        if pd.notna(gf) and pd.notna(ga):
            p = 3 if gf > ga else 1 if gf == ga else 0
            hist.append((p, gf, ga, date_i))

    df['h2h_pts_last5_simple'] = h2h_pts
    df['h2h_gd_last5_simple']  = h2h_gd
    df['h2h_pts_last5_ewma']   = h2h_pts_ewma
    df['h2h_gd_last5_ewma']    = h2h_gd_ewma
    return df

def mirror_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    team_rolling_cols = [
        'pts_last5_ewma', 'pts_last10_ewma', 'gd_last5_ewma', 'gd_last10_ewma',
        'avg_gf_last5_ewma', 'avg_ga_last5_ewma', 'win_rate_last10_ewma',
        'days_since_last_calc', 'pts_last5_simple', 'pts_last10_simple',
    ]
    opp_col_names = ['opp_' + c for c in team_rolling_cols]
    for oc in opp_col_names: df[oc] = np.nan

    match_groups = df.groupby('match_id', sort=False)
    for match_id, group in match_groups:
        if len(group) != 2: continue
        idx = group.index.tolist()
        for col, opp_col in zip(team_rolling_cols, opp_col_names):
            df.at[idx[0], opp_col] = df.at[idx[1], col]
            df.at[idx[1], opp_col] = df.at[idx[0], col]
    return df

df = compute_h2h(df)
df = mirror_opponent_features(df)
print('H2H and Mirror Opponent computed.')
''')

code6 = nbf.v4.new_code_cell('''
# ===========================================================================
# 5. DERIVED FEATURES & FINALIZE CORE
# ===========================================================================
def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df['elo_diff_calc'] = df['elo_team_calc'] - df['elo_opponent_calc']
    df['pts_last5_ewma_diff']  = df['pts_last5_ewma']  - df['opp_pts_last5_ewma']
    df['pts_last10_ewma_diff'] = df['pts_last10_ewma'] - df['opp_pts_last10_ewma']
    df['gd_last5_ewma_diff']   = df['gd_last5_ewma']   - df['opp_gd_last5_ewma']
    df['form_index_team'] = df['pts_last5_ewma'].fillna(0) + df['gd_last5_ewma'].fillna(0) * 0.5
    df['form_index_opp'] = df['opp_pts_last5_ewma'].fillna(0) + df['opp_gd_last5_ewma'].fillna(0) * 0.5
    df['form_index_diff'] = df['form_index_team'] - df['form_index_opp']
    df['pts_last5_simple_diff']  = df['pts_last5_simple'] - df['opp_pts_last5_simple']
    df['pts_last10_simple_diff'] = df['pts_last10_simple'] - df['opp_pts_last10_simple']
    return df

df = compute_derived(df)

rename_map = {
    'elo_team_calc': 'elo_team_feat', 'elo_opponent_calc': 'elo_opponent_feat', 'elo_diff_calc': 'elo_diff_feat',
    'pts_last5_ewma': 'team_pts_last5_ewma_feat', 'pts_last10_ewma': 'team_pts_last10_ewma_feat',
    'gd_last5_ewma': 'team_gd_last5_ewma_feat', 'gd_last10_ewma': 'team_gd_last10_ewma_feat',
    'avg_gf_last5_ewma': 'team_avg_gf_last5_ewma_feat', 'avg_ga_last5_ewma': 'team_avg_ga_last5_ewma_feat',
    'win_rate_last10_ewma': 'team_win_rate_last10_ewma_feat', 'days_since_last_calc': 'days_since_last_team_feat',
    'pts_last5_simple': 'team_pts_last5_simple_feat', 'pts_last10_simple': 'team_pts_last10_simple_feat',
    'opp_pts_last5_ewma': 'opp_pts_last5_ewma_feat', 'opp_pts_last10_ewma': 'opp_pts_last10_ewma_feat',
    'opp_gd_last5_ewma': 'opp_gd_last5_ewma_feat', 'opp_gd_last10_ewma': 'opp_gd_last10_ewma_feat',
    'opp_avg_gf_last5_ewma': 'opp_avg_gf_last5_ewma_feat', 'opp_avg_ga_last5_ewma': 'opp_avg_ga_last5_ewma_feat',
    'opp_win_rate_last10_ewma': 'opp_win_rate_last10_ewma_feat', 'opp_days_since_last_calc': 'days_since_last_opp_feat',
    'opp_pts_last5_simple': 'opp_pts_last5_simple_feat', 'opp_pts_last10_simple': 'opp_pts_last10_simple_feat',
    'h2h_pts_last5_simple': 'h2h_pts_last5_simple_feat', 'h2h_gd_last5_simple': 'h2h_gd_last5_simple_feat',
    'h2h_pts_last5_ewma': 'h2h_pts_last5_ewma_feat', 'h2h_gd_last5_ewma': 'h2h_gd_last5_ewma_feat',
    'pts_last5_ewma_diff': 'pts_last5_ewma_diff_feat', 'pts_last10_ewma_diff': 'pts_last10_ewma_diff_feat',
    'gd_last5_ewma_diff': 'gd_last5_ewma_diff_feat', 'form_index_team': 'form_team_feat',
    'form_index_opp': 'form_opp_feat', 'form_index_diff': 'form_diff_feat',
    'pts_last5_simple_diff': 'pts_last5_simple_diff_feat', 'pts_last10_simple_diff': 'pts_last10_simple_diff_feat',
}
df = df.rename(columns=rename_map)

# Kita simpan core features untuk proses merge di akhir
feat_cols = sorted([c for c in df.columns if c.endswith('_feat')])
print(f'Total core features: {len(feat_cols)}')
''')

md3 = nbf.v4.new_markdown_cell('## 2. Contextual & Socio-Economic Features (Anggota 2)\nMenambahkan Geo-Spatial Stress, Socio-Economic Asymmetry, dan Target Encoding.')
code7 = nbf.v4.new_code_cell('''
# ===========================================================================
# 1. GEO-SPATIAL & PHYSICAL STRESS
# ===========================================================================
MISSING_SENTINEL, MISSING_STR, IDEAL_TEMP = -9999, "Unknown", 22.0

def replace_sentinel_with_nan(d, columns):
    for col in columns:
        if col not in d.columns: continue
        d[col] = d[col].replace(MISSING_STR, np.nan)
        d[col] = pd.to_numeric(d[col], errors="coerce").replace(MISSING_SENTINEL, np.nan)
    return d

def impute_with_median(series, median_val): return series.fillna(median_val)

def build_geo_features(df_full):
    geo_cols = ["distance_travel_team", "distance_travel_opp", "temperature_venue", "altitude_venue"]
    df_full = replace_sentinel_with_nan(df_full, geo_cols)
    
    medians = {col: df_full[col].median() if col in df_full.columns else 0.0 for col in geo_cols}
    for col in geo_cols:
        df_full[col] = impute_with_median(df_full[col], medians[col])

    df_full["travel_stress_diff_ctx"] = df_full["distance_travel_team"] - df_full["distance_travel_opp"]

    home_rows = df_full[df_full["is_home"] == 1]
    team_alt_proxy = home_rows.groupby("team")["altitude_venue"].median().rename("team_home_alt_proxy").reset_index()
    opp_alt_proxy = home_rows.groupby("team")["altitude_venue"].median().rename("opp_home_alt_proxy").reset_index().rename(columns={"team": "opponent"})

    global_alt_median = medians["altitude_venue"]
    df_full = df_full.merge(team_alt_proxy, on="team", how="left")
    df_full["team_home_alt_proxy"] = df_full["team_home_alt_proxy"].fillna(global_alt_median)
    df_full = df_full.merge(opp_alt_proxy, on="opponent", how="left")
    df_full["opp_home_alt_proxy"] = df_full["opp_home_alt_proxy"].fillna(global_alt_median)

    df_full["altitude_shock_team_ctx"] = abs(df_full["altitude_venue"] - df_full["team_home_alt_proxy"])
    df_full["altitude_shock_opp_ctx"]  = abs(df_full["altitude_venue"] - df_full["opp_home_alt_proxy"])
    df_full.drop(columns=["team_home_alt_proxy", "opp_home_alt_proxy"], inplace=True)
    df_full["temp_stress_ctx"] = abs(df_full["temperature_venue"] - IDEAL_TEMP)
    
    return df_full

df = build_geo_features(df)
print('Geo features computed.')
''')

code8 = nbf.v4.new_code_cell('''
# ===========================================================================
# 2. SOCIO-ECONOMIC ASYMMETRY
# ===========================================================================
def build_socio_features(df_full):
    socio_cols = ["gdp_per_capita_team", "gdp_per_capita_opp", "population_team", "population_opp"]
    df_full = replace_sentinel_with_nan(df_full, socio_cols)

    medians = {col: df_full[col].median() if col in df_full.columns else 1.0 for col in socio_cols}
    for col in socio_cols:
        df_full[col] = impute_with_median(df_full[col], medians.get(col, 1.0))

    df_full["log_gdp_diff_ctx"] = np.log1p(df_full["gdp_per_capita_team"]) - np.log1p(df_full["gdp_per_capita_opp"])
    df_full["log_pop_diff_ctx"] = np.log1p(df_full["population_team"]) - np.log1p(df_full["population_opp"])
    return df_full

df = build_socio_features(df)
print('Socio-Economic computed.')
''')

code9 = nbf.v4.new_code_cell('''
# ===========================================================================
# 3. CATEGORICAL & TARGET ENCODING
# ===========================================================================
N_FOLDS, SMOOTH_ALPHA = 5, 10.0

def smooth_target_encode(series, target, global_mean, alpha=SMOOTH_ALPHA):
    stats = pd.DataFrame({"cat": series, "target": target})
    agg = stats.groupby("cat")["target"].agg(["mean", "count"])
    agg["smoothed"] = ((agg["count"] * agg["mean"] + alpha * global_mean) / (agg["count"] + alpha))
    return series.map(agg["smoothed"])

def build_encoding_features(df_full):
    train = df_full[df_full['is_test'] == False].copy()
    test  = df_full[df_full['is_test'] == True].copy()

    cat_cols = ["venue_country", "confederation_team", "confederation_opp"]
    for col in cat_cols:
        if col in train.columns: train[col] = train[col].replace(MISSING_STR, "UNKNOWN")
        if col in test.columns:  test[col]  = test[col].replace(MISSING_STR, "UNKNOWN")

    train["_goal_diff_temp"] = train["team_goals"] - train["opp_goals"]
    global_mean = train["_goal_diff_temp"].mean()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # venue_country
    train["venue_country_te_ctx"] = np.nan
    if "venue_country" in train.columns:
        for trn_idx, val_idx in kf.split(train):
            trn_fold, val_fold = train.iloc[trn_idx], train.iloc[val_idx]
            encoded = smooth_target_encode(trn_fold["venue_country"], trn_fold["_goal_diff_temp"], global_mean)
            mapping = dict(zip(trn_fold["venue_country"], encoded))
            train.loc[train.index[val_idx], "venue_country_te_ctx"] = val_fold["venue_country"].map(mapping).values
        train["venue_country_te_ctx"] = train["venue_country_te_ctx"].fillna(global_mean)
        vc_global_map = dict(zip(train["venue_country"], smooth_target_encode(train["venue_country"], train["_goal_diff_temp"], global_mean)))
        test["venue_country_te_ctx"] = test["venue_country"].map(vc_global_map).fillna(global_mean) if "venue_country" in test.columns else global_mean
    else:
        train["venue_country_te_ctx"] = test["venue_country_te_ctx"] = global_mean

    # confederation_team
    train["confederation_team_te_ctx"] = np.nan
    if "confederation_team" in train.columns:
        for trn_idx, val_idx in kf.split(train):
            trn_fold, val_fold = train.iloc[trn_idx], train.iloc[val_idx]
            encoded = smooth_target_encode(trn_fold["confederation_team"], trn_fold["_goal_diff_temp"], global_mean)
            mapping = dict(zip(trn_fold["confederation_team"], encoded))
            train.loc[train.index[val_idx], "confederation_team_te_ctx"] = val_fold["confederation_team"].map(mapping).values
        train["confederation_team_te_ctx"] = train["confederation_team_te_ctx"].fillna(global_mean)
        conf_global_map = dict(zip(train["confederation_team"], smooth_target_encode(train["confederation_team"], train["_goal_diff_temp"], global_mean)))
        test["confederation_team_te_ctx"] = test["confederation_team"].map(conf_global_map).fillna(global_mean) if "confederation_team" in test.columns else global_mean
    else:
        train["confederation_team_te_ctx"] = test["confederation_team_te_ctx"] = global_mean

    # Frequency Encoding
    if "venue_country" in train.columns:
        vc_freq = train["venue_country"].value_counts(normalize=True)
        train["venue_country_freq_ctx"] = train["venue_country"].map(vc_freq).fillna(0.0)
        test["venue_country_freq_ctx"] = test["venue_country"].map(vc_freq).fillna(0.0) if "venue_country" in test.columns else 0.0
    else:
        train["venue_country_freq_ctx"] = test["venue_country_freq_ctx"] = 0.0

    train.drop(columns=["_goal_diff_temp"], inplace=True)
    
    # Gabungkan kembali
    df_combined = pd.concat([train, test]).sort_values(['date', 'match_id', 'Id']).reset_index(drop=True)
    return df_combined

df = build_encoding_features(df)
ctx_cols = [c for c in df.columns if c.endswith('_ctx')]
print(f'Total context features: {len(ctx_cols)}')
''')

md4 = nbf.v4.new_markdown_cell('## 3. Merging & Export\nMenyimpan dataset akhir (`train_final.csv` dan `test_final.csv`) yang hanya berisi Id, target, dan fitur (_feat dan _ctx).')
code10 = nbf.v4.new_code_cell('''
# ===========================================================================
# 5. FINALISASI & SIMPAN OUTPUT
# ===========================================================================
# Menggabungkan fitur core (_feat) dan fitur context (_ctx) ke bentuk akhir
all_features = sorted(feat_cols + ctx_cols)

keep_cols_train = ['Id'] + all_features + ['team_goals', 'opp_goals']
keep_cols_test  = ['Id'] + all_features

train_final = df[df['is_test'] == False][keep_cols_train].copy()
test_final  = df[df['is_test'] == True][keep_cols_test].copy()

import os
os.makedirs(DATA_DIR, exist_ok=True)

train_final.to_csv(OUT_TRAIN_FINAL, index=False)
test_final.to_csv(OUT_TEST_FINAL, index=False)

print(f"Dataset Train Final: {train_final.shape}")
print(f"Dataset Test Final : {test_final.shape}")
print("Selesai! File output telah di-generate di folder dataset/.")
''')

nb['cells'] = [md1, code1, md2, code2, code3, code4, code5, code6, md3, code7, code8, code9, md4, code10]

with open('data_preprocessing.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook generated.")

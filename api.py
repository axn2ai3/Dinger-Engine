from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from pybaseball import statcast
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import requests
import warnings
import json
import traceback
from datetime import date, datetime, timedelta

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Fix numpy/pandas types not being JSON serializable
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

model_cache = {}

# ==========================================
# PARK FACTORS
# ==========================================

HR_PARK_FACTORS = {
    'CIN': 150, 'COL': 131, 'CWS': 125, 'LAA': 120, 'LAD': 118,
    'MIL': 114, 'BAL': 113, 'NYY': 112, 'ATL': 111, 'PHI': 111,
    'BOS': 110, 'TEX': 108, 'HOU': 105, 'TOR': 103, 'CHW': 102,
    'TB':  100, 'MIN': 99,  'MIA': 98,  'SEA': 97,  'WSH': 96,
    'CHC': 95,  'SF':  92,  'SD':  91,  'ARI': 90,  'NYM': 89,
    'STL': 87,  'OAK': 85,  'PIT': 84,  'CLE': 82,  'DET': 80,
    'KC':  78,
    # Full name mappings for statsapi
    'Cincinnati Reds': 150, 'Colorado Rockies': 131, 'Chicago White Sox': 125,
    'Los Angeles Angels': 120, 'Los Angeles Dodgers': 118, 'Milwaukee Brewers': 114,
    'Baltimore Orioles': 113, 'New York Yankees': 112, 'Atlanta Braves': 111,
    'Philadelphia Phillies': 111, 'Boston Red Sox': 110, 'Texas Rangers': 108,
    'Houston Astros': 105, 'Toronto Blue Jays': 103,
    'Tampa Bay Rays': 100, 'Minnesota Twins': 99, 'Miami Marlins': 98,
    'Seattle Mariners': 97, 'Washington Nationals': 96, 'Chicago Cubs': 95,
    'San Francisco Giants': 92, 'San Diego Padres': 91, 'Arizona Diamondbacks': 90,
    'New York Mets': 89, 'St. Louis Cardinals': 87, 'Oakland Athletics': 85,
    'Pittsburgh Pirates': 84, 'Cleveland Guardians': 82, 'Detroit Tigers': 80,
    'Kansas City Royals': 78
}

VENUE_COORDS = {
    'Angel Stadium': (33.8003, -117.8827), 'Busch Stadium': (38.6226, -90.1928),
    'Chase Field': (33.4455, -112.0667), 'Citi Field': (40.7571, -73.8458),
    'Citizens Bank Park': (39.9061, -75.1665), 'Comerica Park': (42.3390, -83.0485),
    'Coors Field': (39.7559, -104.9942), 'Dodger Stadium': (34.0739, -118.2400),
    'Fenway Park': (42.3467, -71.0972), 'Globe Life Field': (32.7474, -97.0845),
    'Great American Ball Park': (39.0974, -84.5065), 'Guaranteed Rate Field': (41.8299, -87.6338),
    'Kauffman Stadium': (39.0517, -94.4803), 'loanDepot park': (25.7781, -80.2196),
    'Minute Maid Park': (29.7573, -95.3555), 'Nationals Park': (38.8730, -77.0074),
    'Oracle Park': (37.7786, -122.3893), 'Oriole Park at Camden Yards': (39.2838, -76.6218),
    'Petco Park': (32.7076, -117.1570), 'PNC Park': (40.4469, -80.0058),
    'Progressive Field': (41.4962, -81.6852), 'Rogers Centre': (43.6414, -79.3894),
    'T-Mobile Park': (47.5914, -122.3325), 'Target Field': (44.9817, -93.2776),
    'Tropicana Field': (27.7682, -82.6534), 'Truist Park': (33.8911, -84.4681),
    'Wrigley Field': (41.9484, -87.6553), 'Yankee Stadium': (40.8296, -73.9262),
    'American Family Field': (43.0280, -87.9712),
}

# ==========================================
# FEATURE ENGINEERING FUNCTIONS
# ==========================================

def fetch_statcast_data(start_date, end_date):
    print(f"Fetching Statcast data from {start_date} to {end_date}...")
    df = statcast(start_dt=start_date, end_dt=end_date)
    df = df.dropna(subset=['events'])
    df['is_hr'] = np.where(df['events'] == 'home_run', 1, 0)
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    return df


def add_park_factors(df):
    pf_df = pd.DataFrame(list(HR_PARK_FACTORS.items()), columns=['home_team', 'hr_park_factor'])
    df = pd.merge(df, pf_df, on='home_team', how='left')
    df['hr_park_factor'] = df['hr_park_factor'].fillna(100)
    return df


def calculate_pitcher_tendencies(df):
    pitcher_totals = df.groupby('pitcher').size().reset_index(name='total_pitches')
    pitch_mix = df.groupby(['pitcher', 'pitch_type']).size().reset_index(name='pitch_count')
    mix_df = pd.merge(pitch_mix, pitcher_totals, on='pitcher')
    mix_df['pitch_pct'] = mix_df['pitch_count'] / mix_df['total_pitches']
    return mix_df[['pitcher', 'pitch_type', 'pitch_pct']]


def calculate_batter_strengths(df):
    batter_stats = df.groupby(['batter', 'pitch_type']).agg(
        avg_exit_velo=('launch_speed', 'mean'),
        avg_launch_angle=('launch_angle', 'mean'),
        total_batted_balls=('batter', 'count'),
        home_runs=('is_hr', 'sum')
    ).reset_index()
    batter_stats['hr_rate_vs_pitch'] = batter_stats['home_runs'] / batter_stats['total_batted_balls']
    return batter_stats


# --- NEW FEATURE 1: Platoon Splits ---
def calculate_platoon_features(df):
    """Compute batter HR rates by handedness matchup"""
    if 'stand' not in df.columns or 'p_throws' not in df.columns:
        return pd.DataFrame()

    df['platoon_adv'] = ((df['stand'] == 'L') & (df['p_throws'] == 'R') |
                         (df['stand'] == 'R') & (df['p_throws'] == 'L')).astype(int)

    platoon_stats = df.groupby(['batter', 'platoon_adv']).agg(
        platoon_hr_rate=('is_hr', 'mean'),
        platoon_avg_ev=('launch_speed', 'mean'),
        platoon_count=('batter', 'count')
    ).reset_index()

    # Also get each batter's handedness
    batter_hand = df.groupby('batter')['stand'].first().reset_index()
    batter_hand.columns = ['batter', 'batter_hand']

    return platoon_stats, batter_hand


# --- NEW FEATURE 2: Recent Form (rolling windows) ---
def calculate_recent_form(df, windows=[7, 14, 30]):
    """Compute rolling window stats for batters"""
    if 'game_date' not in df.columns:
        return pd.DataFrame()

    max_date = df['game_date'].max()
    results = []

    for window in windows:
        cutoff = max_date - timedelta(days=window)
        recent = df[df['game_date'] >= cutoff]

        form = recent.groupby('batter').agg(**{
            f'ev_{window}d': ('launch_speed', 'mean'),
            f'la_{window}d': ('launch_angle', 'mean'),
            f'hr_rate_{window}d': ('is_hr', 'mean'),
            f'pa_{window}d': ('batter', 'count'),
        }).reset_index()
        results.append(form)

    merged = results[0]
    for r in results[1:]:
        merged = pd.merge(merged, r, on='batter', how='outer')

    return merged.fillna(0)


# --- NEW FEATURE 3: Pitch Location Zones ---
def calculate_zone_features(df):
    """Compute batter HR rates by pitch zone"""
    if 'plate_x' not in df.columns or 'plate_z' not in df.columns:
        return pd.DataFrame()

    # Define zones: up/down x in/away
    df = df.copy()
    df['vert_zone'] = pd.cut(df['plate_z'], bins=[-np.inf, 2.0, 3.0, np.inf],
                              labels=['low', 'mid', 'high'])
    df['horiz_zone'] = pd.cut(df['plate_x'], bins=[-np.inf, -0.5, 0.5, np.inf],
                               labels=['away', 'middle', 'in'])
    df['pitch_zone'] = df['vert_zone'].astype(str) + '_' + df['horiz_zone'].astype(str)

    zone_stats = df.groupby(['batter', 'pitch_zone']).agg(
        zone_hr_rate=('is_hr', 'mean'),
        zone_ev=('launch_speed', 'mean'),
        zone_count=('batter', 'count')
    ).reset_index()

    # Pivot to wide format for each batter
    zone_pivot = zone_stats.pivot_table(
        index='batter', columns='pitch_zone',
        values=['zone_hr_rate', 'zone_ev'],
        fill_value=0
    )
    zone_pivot.columns = ['_'.join(col).strip() for col in zone_pivot.columns]
    zone_pivot = zone_pivot.reset_index()
    return zone_pivot


# --- NEW FEATURE 4: Pitcher Fatigue ---
def calculate_pitcher_fatigue(df):
    """Compute pitcher workload metrics"""
    if 'game_date' not in df.columns:
        return pd.DataFrame()

    max_date = df['game_date'].max()

    # Pitches in last 7/14/30 days
    fatigue = []
    for window in [7, 14, 30]:
        cutoff = max_date - timedelta(days=window)
        recent = df[df['game_date'] >= cutoff]
        pitch_counts = recent.groupby('pitcher').agg(**{
            f'pitches_{window}d': ('pitcher', 'count'),
            f'avg_velo_{window}d': ('release_speed', 'mean') if 'release_speed' in df.columns else ('pitcher', 'count'),
        }).reset_index()
        fatigue.append(pitch_counts)

    merged = fatigue[0]
    for f in fatigue[1:]:
        merged = pd.merge(merged, f, on='pitcher', how='outer')

    # Days since last outing
    last_outing = df.groupby('pitcher')['game_date'].max().reset_index()
    last_outing['days_rest'] = (max_date - last_outing['game_date']).dt.days
    last_outing = last_outing[['pitcher', 'days_rest']]

    merged = pd.merge(merged, last_outing, on='pitcher', how='outer')
    return merged.fillna(0)


# ==========================================
# ENHANCED TRAINING PIPELINE
# ==========================================

def engineer_training_data(df):
    print("Engineering enhanced features...")
    df = add_park_factors(df)
    pitcher_mix = calculate_pitcher_tendencies(df)
    batter_strength = calculate_batter_strengths(df)
    recent_form = calculate_recent_form(df)
    pitcher_fatigue = calculate_pitcher_fatigue(df)

    # Platoon
    platoon_data = None
    batter_hand = None
    try:
        platoon_data, batter_hand = calculate_platoon_features(df)
    except:
        pass

    # Base matchups
    matchups = df[['game_pk', 'game_date', 'batter', 'pitcher', 'pitch_type',
                    'hr_park_factor', 'is_hr']].copy()

    # Add handedness info
    if 'stand' in df.columns and 'p_throws' in df.columns:
        hand_info = df[['batter', 'pitcher', 'stand', 'p_throws']].drop_duplicates()
        hand_info['platoon_adv'] = ((hand_info['stand'] == 'L') & (hand_info['p_throws'] == 'R') |
                                    (hand_info['stand'] == 'R') & (hand_info['p_throws'] == 'L')).astype(int)
        matchups = pd.merge(matchups, hand_info[['batter', 'pitcher', 'platoon_adv']],
                            on=['batter', 'pitcher'], how='left')

    matchups = pd.merge(matchups, pitcher_mix, on=['pitcher', 'pitch_type'], how='left')
    matchups = pd.merge(matchups, batter_strength, on=['batter', 'pitch_type'], how='left')
    matchups = pd.merge(matchups, recent_form, on='batter', how='left')
    matchups = pd.merge(matchups, pitcher_fatigue, on='pitcher', how='left')

    # Interaction features
    matchups['ev_x_park'] = matchups.get('avg_exit_velo', 0) * matchups.get('hr_park_factor', 100) / 100
    if 'platoon_adv' in matchups.columns:
        matchups['platoon_x_hr_rate'] = matchups.get('platoon_adv', 0) * matchups.get('hr_rate_vs_pitch', 0)

    matchups = matchups.fillna(0)

    return matchups, pitcher_mix, batter_strength, recent_form, pitcher_fatigue


def get_feature_list(df):
    """Dynamically get all numeric feature columns"""
    base_features = ['pitch_pct', 'avg_exit_velo', 'avg_launch_angle', 'hr_rate_vs_pitch',
                     'hr_park_factor', 'ev_x_park']

    optional = ['platoon_adv', 'platoon_x_hr_rate',
                'ev_7d', 'la_7d', 'hr_rate_7d', 'pa_7d',
                'ev_14d', 'la_14d', 'hr_rate_14d', 'pa_14d',
                'ev_30d', 'la_30d', 'hr_rate_30d', 'pa_30d',
                'pitches_7d', 'pitches_14d', 'pitches_30d', 'days_rest']

    features = base_features.copy()
    for f in optional:
        if f in df.columns:
            features.append(f)

    return features


def train_hr_model(features_df):
    print("Training XGBoost Model with calibration...")
    features = get_feature_list(features_df)
    X = features_df[features]
    y = features_df['is_hr']

    # Temporal split: use last 20% of dates as test
    if 'game_date' in features_df.columns:
        dates_sorted = features_df['game_date'].sort_values()
        cutoff_idx = int(len(dates_sorted) * 0.8)
        cutoff_date = dates_sorted.iloc[cutoff_idx]
        train_mask = features_df['game_date'] <= cutoff_date
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        print(f"Temporal split: train up to {cutoff_date.date()}, test after")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Class imbalance: scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / max(pos_count, 1)
    print(f"Class ratio: {neg_count}:{pos_count}, scale_pos_weight={scale_weight:.1f}")

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=6,
        learning_rate=0.08,
        n_estimators=200,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5
    )

    # Calibrated model for better probability estimates
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

    predictions = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, predictions)
    print(f"Calibrated Model AUC Score: {auc:.4f}")

    # Feature importance from base estimators
    importance = {}
    try:
        for est in calibrated_model.calibrated_classifiers_:
            base = est.estimator
            for fname, imp in zip(features, base.feature_importances_):
                importance[fname] = importance.get(fname, 0) + imp
        for k in importance:
            importance[k] = float(importance[k] / len(calibrated_model.calibrated_classifiers_))
    except:
        importance = {f: 0.0 for f in features}

    return calibrated_model, features, float(auc), importance


# ==========================================
# LINEUP SCRAPING
# ==========================================

def scrape_todays_lineups():
    print("Fetching today's lineups from MLB Stats API...")
    import statsapi

    today = date.today().strftime('%Y-%m-%d')
    todays_matchups = []
    game_info = []

    try:
        schedule = statsapi.schedule(date=today)
        print(f"Found {len(schedule)} games today")

        for game in schedule:
            try:
                game_id = game['game_id']
                away_team = game.get('away_name', 'UNK')
                home_team = game.get('home_name', 'UNK')
                venue = game.get('venue_name', 'UNK')

                home_pitcher_name = game.get('home_probable_pitcher', '')
                away_pitcher_name = game.get('away_probable_pitcher', '')

                if not home_pitcher_name or not away_pitcher_name:
                    print(f"  Skipping {away_team} @ {home_team} — no probable pitchers")
                    continue

                home_pitcher_search = statsapi.lookup_player(home_pitcher_name)
                away_pitcher_search = statsapi.lookup_player(away_pitcher_name)

                if not home_pitcher_search or not away_pitcher_search:
                    continue

                home_pitcher_id = home_pitcher_search[0]['id']
                away_pitcher_id = away_pitcher_search[0]['id']

                live = statsapi.get('game', {'gamePk': game_id})
                lineup_data = live['liveData']['boxscore']['teams']

                home_batters = []
                away_batters = []

                for pid, pdata in lineup_data['home']['players'].items():
                    if pdata.get('battingOrder') and str(pdata['battingOrder']).endswith('00'):
                        home_batters.append({
                            'id': pdata['person']['id'],
                            'name': pdata['person']['fullName'],
                            'position': pdata.get('position', {}).get('abbreviation', 'UNK')
                        })

                for pid, pdata in lineup_data['away']['players'].items():
                    if pdata.get('battingOrder') and str(pdata['battingOrder']).endswith('00'):
                        away_batters.append({
                            'id': pdata['person']['id'],
                            'name': pdata['person']['fullName'],
                            'position': pdata.get('position', {}).get('abbreviation', 'UNK')
                        })

                if not home_batters and not away_batters:
                    print(f"  Lineups not posted yet for {away_team} @ {home_team}")
                    continue

                game_meta = {
                    'game_id': game_id,
                    'away_team': away_team,
                    'home_team': home_team,
                    'venue': venue,
                    'home_pitcher': home_pitcher_name,
                    'away_pitcher': away_pitcher_name,
                    'home_pitcher_id': home_pitcher_id,
                    'away_pitcher_id': away_pitcher_id,
                    'game_time': game.get('game_datetime', ''),
                    'status': game.get('status', '')
                }
                game_info.append(game_meta)

                for b in away_batters:
                    todays_matchups.append({
                        'batter': b['id'],
                        'batter_name': b['name'],
                        'pitcher': home_pitcher_id,
                        'pitcher_name': home_pitcher_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'batting_team': away_team,
                        'venue': venue,
                        'game_id': game_id
                    })
                for b in home_batters:
                    todays_matchups.append({
                        'batter': b['id'],
                        'batter_name': b['name'],
                        'pitcher': away_pitcher_id,
                        'pitcher_name': away_pitcher_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'batting_team': home_team,
                        'venue': venue,
                        'game_id': game_id
                    })

                print(f"  {away_team} @ {home_team}: {len(away_batters)} away, {len(home_batters)} home batters")

            except Exception as e:
                print(f"  Error parsing game {game.get('game_id')}: {e}")
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"Error fetching schedule: {e}")
        traceback.print_exc()
        return pd.DataFrame(), []

    print(f"Total matchups found: {len(todays_matchups)}")
    return pd.DataFrame(todays_matchups), game_info


# ==========================================
# WEATHER
# ==========================================

def fetch_weather_for_venues(game_info):
    """Fetch weather data for each venue"""
    weather_data = {}
    for game in game_info:
        venue = game.get('venue', '')
        coords = VENUE_COORDS.get(venue)
        if not coords:
            weather_data[venue] = {'temp_f': 72, 'wind_mph': 5, 'wind_dir': 'N/A', 'humidity': 50}
            continue
        try:
            lat, lon = coords
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            cw = data.get('current_weather', {})
            temp_c = cw.get('temperature', 22)
            temp_f = temp_c * 9/5 + 32
            wind_kmh = cw.get('windspeed', 10)
            wind_mph = wind_kmh * 0.621371
            weather_data[venue] = {
                'temp_f': round(temp_f, 1),
                'wind_mph': round(wind_mph, 1),
                'wind_dir': cw.get('winddirection', 0),
                'humidity': 50
            }
        except:
            weather_data[venue] = {'temp_f': 72, 'wind_mph': 5, 'wind_dir': 'N/A', 'humidity': 50}
    return weather_data


# ==========================================
# FANDUEL ODDS SCRAPING
# ==========================================

def scrape_fanduel_hr_odds():
    """Scrape HR prop odds — uses The Odds API (free tier) or fallback"""
    odds_data = {}

    # Try The Odds API (free tier, 500 requests/month)
    try:
        import os
        api_key = os.environ.get("ODDS_API_KEY", "demo")
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'batter_home_runs',
            'oddsFormat': 'american',
            'bookmakers': 'fanduel'
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            events = resp.json()
            for event in events:
                for bookmaker in event.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'batter_home_runs':
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('description', outcome.get('name', ''))
                                odds_data[player_name.lower()] = {
                                    'american_odds': outcome.get('price', 0),
                                    'point': outcome.get('point', 0.5)
                                }
    except Exception as e:
        print(f"Odds API error: {e}")

    # If no data, generate synthetic odds based on historical HR rates
    # This ensures the EV calc still works for demo/development
    if not odds_data:
        print("No live odds available — will use implied odds from model")

    return odds_data


def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def calculate_ev(model_prob, american_odds):
    """Calculate expected value of a HR bet"""
    if american_odds > 0:
        profit = american_odds / 100
    else:
        profit = 100 / abs(american_odds)

    ev = (model_prob * profit) - ((1 - model_prob) * 1)
    return round(ev, 4)


# ==========================================
# PREDICTION
# ==========================================

def predict_game_matchups(daily_df, model, pitcher_mix, batter_strength,
                          recent_form, pitcher_fatigue, features_list, weather_data, odds_data):
    print("Generating predictions for today's slate...")
    daily_df = add_park_factors(daily_df)

    results = []
    for _, row in daily_df.iterrows():
        b_id, p_id = row['batter'], row['pitcher']

        p_arsenal = pitcher_mix[pitcher_mix['pitcher'] == p_id]
        if p_arsenal.empty:
            continue

        matchup_pitches = pd.merge(
            p_arsenal,
            batter_strength[batter_strength['batter'] == b_id],
            on='pitch_type', how='left'
        )
        matchup_pitches['hr_park_factor'] = row['hr_park_factor']

        # Add recent form
        batter_form = recent_form[recent_form['batter'] == b_id]
        if not batter_form.empty:
            for col in batter_form.columns:
                if col != 'batter':
                    matchup_pitches[col] = batter_form[col].values[0]

        # Add pitcher fatigue
        p_fatigue = pitcher_fatigue[pitcher_fatigue['pitcher'] == p_id]
        if not p_fatigue.empty:
            for col in p_fatigue.columns:
                if col != 'pitcher':
                    matchup_pitches[col] = p_fatigue[col].values[0]

        # Interaction features
        matchup_pitches['ev_x_park'] = matchup_pitches.get('avg_exit_velo', pd.Series([0])).fillna(0) * \
                                        matchup_pitches.get('hr_park_factor', pd.Series([100])).fillna(100) / 100

        # Platoon (default 0 if unknown)
        matchup_pitches['platoon_adv'] = 0
        matchup_pitches['platoon_x_hr_rate'] = 0

        matchup_pitches = matchup_pitches.fillna(0)

        # Ensure all features exist
        for f in features_list:
            if f not in matchup_pitches.columns:
                matchup_pitches[f] = 0

        X_inference = matchup_pitches[features_list]

        try:
            pitch_probs = model.predict_proba(X_inference)[:, 1]
            weights = matchup_pitches['pitch_pct'].values
            if weights.sum() > 0:
                weighted_hr_prob = float(np.average(pitch_probs, weights=weights))
            else:
                weighted_hr_prob = float(pitch_probs.mean())
        except Exception as e:
            print(f"  Prediction error for batter {b_id}: {e}")
            continue

        # Weather adjustment
        venue = row.get('venue', '')
        weather = weather_data.get(venue, {})
        temp_adj = (weather.get('temp_f', 72) - 72) * 0.002  # ~0.2% per degree F
        wind_adj = weather.get('wind_mph', 5) * 0.001
        adjusted_prob = weighted_hr_prob * (1 + temp_adj + wind_adj)
        adjusted_prob = max(0, min(adjusted_prob, 1))

        # Odds & EV
        batter_name = row.get('batter_name', '').lower()
        player_odds = odds_data.get(batter_name, {})
        american_odds = player_odds.get('american_odds', None)

        ev = None
        implied_prob = None
        if american_odds:
            implied_prob = american_to_implied_prob(american_odds)
            ev = calculate_ev(adjusted_prob, american_odds)
        else:
            # Generate synthetic odds from model prob for display
            if adjusted_prob > 0:
                synthetic_odds = int(-100 * (1 - adjusted_prob) / adjusted_prob) if adjusted_prob > 0.5 \
                    else int(100 * (1 - adjusted_prob) / adjusted_prob)
                american_odds = synthetic_odds
                implied_prob = adjusted_prob  # No edge if synthetic
                ev = 0.0

        results.append({
            'batter': str(b_id),
            'batter_name': row.get('batter_name', str(b_id)),
            'pitcher': str(p_id),
            'pitcher_name': row.get('pitcher_name', str(p_id)),
            'home_team': row.get('home_team', ''),
            'away_team': row.get('away_team', ''),
            'batting_team': row.get('batting_team', ''),
            'venue': venue,
            'base_hr_prob': round(weighted_hr_prob, 6),
            'adj_hr_prob': round(adjusted_prob, 6),
            'weather_temp': weather.get('temp_f', 72),
            'weather_wind': weather.get('wind_mph', 5),
            'american_odds': american_odds,
            'implied_prob': round(implied_prob, 4) if implied_prob else None,
            'ev': ev,
            'ev_plus': ev is not None and ev > 0,
            'game_id': row.get('game_id', '')
        })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('adj_hr_prob', ascending=False)
    return results_df


# ==========================================
# FLASK ROUTES
# ==========================================

@app.route('/train', methods=['POST'])
def train():
    try:
        # Rolling 18-month window with recency weighting
        end_date = date.today().strftime('%Y-%m-%d')
        start_date = (date.today() - timedelta(days=540)).strftime('%Y-%m-%d')

        hist_data = fetch_statcast_data(start_date, end_date)
        train_data, p_mix, b_strength, r_form, p_fatigue = engineer_training_data(hist_data)
        model, features, auc, importance = train_hr_model(train_data)

        model_cache['model'] = model
        model_cache['p_mix'] = p_mix
        model_cache['b_strength'] = b_strength
        model_cache['r_form'] = r_form
        model_cache['p_fatigue'] = p_fatigue
        model_cache['features'] = features
        model_cache['auc'] = auc
        model_cache['importance'] = importance
        model_cache['train_rows'] = len(train_data)
        model_cache['train_date'] = datetime.now().isoformat()

        return jsonify({
            "status": "ok",
            "auc": auc,
            "features": features,
            "importance": importance,
            "train_rows": len(train_data),
            "date_range": f"{start_date} to {end_date}"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['GET'])
def predict():
    if 'model' not in model_cache:
        return jsonify({"error": "Model not trained yet. Train first."}), 400
    try:
        lineups, game_info = scrape_todays_lineups()
        if lineups.empty:
            return jsonify({"error": "No lineups found yet — check back closer to game time."}), 404

        weather_data = fetch_weather_for_venues(game_info)
        odds_data = scrape_fanduel_hr_odds()

        preds = predict_game_matchups(
            lineups,
            model_cache['model'],
            model_cache['p_mix'],
            model_cache['b_strength'],
            model_cache['r_form'],
            model_cache['p_fatigue'],
            model_cache['features'],
            weather_data,
            odds_data
        )

        return jsonify({
            "predictions": preds.to_dict(orient='records'),
            "games": game_info,
            "weather": weather_data,
            "odds_source": "the-odds-api" if odds_data else "synthetic",
            "model_auc": model_cache.get('auc'),
            "total_matchups": len(preds)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "model_trained": 'model' in model_cache,
        "auc": model_cache.get('auc'),
        "features": model_cache.get('features', []),
        "importance": model_cache.get('importance', {}),
        "train_rows": model_cache.get('train_rows', 0),
        "train_date": model_cache.get('train_date')
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    # debug=False for cloud deployment; set DEBUG=1 env var to enable locally
    debug = os.environ.get("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

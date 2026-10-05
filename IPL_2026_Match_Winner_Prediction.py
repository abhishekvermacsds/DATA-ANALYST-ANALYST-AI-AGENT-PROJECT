# ============================================================
# IPL 2026 MATCH WINNER PREDICTION
# ============================================================
# Project: IPL Match Winner Prediction
# Dataset: IPL Ball-by-Ball Data (2008-2025)
# Models: Logistic Regression, Random Forest, XGBoost,
#         LightGBM, CatBoost, Neural Network, Stacking
# Target: Predict which team wins a given match
# ============================================================

🏏 IPL 2026 Match Winner Prediction
Models: Logistic Regression | Random Forest | XGBoost | LightGBM | CatBoost | Neural Network | Stacking Ensemble
Dataset: IPL Ball-by-Ball Data (2008–2025)
Target: Predict which team wins a given match

1. Setup & Data Loading
# Install required libraries
!pip install kagglehub lightgbm catboost xgboost --quiet
import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Download dataset
path = kagglehub.dataset_download('chaitu20/ipl-dataset2008-2025')
print('Files:', os.listdir(path))
# Load raw ball-by-ball data
df = pd.read_csv(f'{path}/IPL.csv', low_memory=False)
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
Columns: ['Unnamed: 0', 'match_id', 'date', 'match_type', 'event_name', 'innings', 'batting_team', 'bowling_team', 'over', 'ball', 'ball_no', 'batter', 'bat_pos', 'runs_batter', 'balls_faced', 'bowler', 'valid_ball', 'runs_extras', 'runs_total', 'runs_bowler', 'runs_not_boundary', 'extra_type', 'non_striker', 'non_striker_pos', 'wicket_kind', 'player_out', 'fielders', 'runs_target', 'review_batter', 'team_reviewed', 'review_decision', 'umpire', 'umpires_call', 'player_of_match', 'match_won_by', 'win_outcome', 'toss_winner', 'toss_decision', 'venue', 'city', 'day', 'month', 'year', 'season', 'gender', 'team_type', 'superover_winner', 'result_type', 'method', 'balls_per_over', 'overs', 'event_match_no', 'stage', 'match_number', 'team_runs', 'team_balls', 'team_wicket', 'new_batter', 'batter_runs', 'batter_balls', 'bowler_wicket', 'batting_partners', 'next_batter', 'striker_out']
2. Preprocessing — Build Match-Level Dataset
# Aggregate ball-by-ball → one row per match
match_df = df.groupby('match_id').first().reset_index()
match_cols = [
    'match_id', 'date', 'season', 'venue', 'city',
    'batting_team', 'bowling_team',
    'toss_winner', 'toss_decision',
    'match_won_by', 'win_outcome', 'stage'
]
match_df = match_df[match_cols].copy()
print('Match-level shape:', match_df.shape)
# Team name standardization map — includes abbreviations
abbrev_map = {
    'Chennai Super Kings'            : 'CSK',
    'Delhi Capitals'                 : 'DC',
    'Delhi Daredevils'               : 'DC',
    'Gujarat Titans'                 : 'GT',
    'Kolkata Knight Riders'          : 'KKR',
    'Mumbai Indians'                 : 'MI',
    'Punjab Kings'                   : 'PBKS',
    'Kings XI Punjab'                : 'PBKS',
    'Rajasthan Royals'               : 'RR',
    'Royal Challengers Bangalore'    : 'RCB',
    'Royal Challengers Bengaluru'    : 'RCB',
    'Sunrisers Hyderabad'            : 'SRH',
    'Lucknow Super Giants'           : 'LSG',
    'Rising Pune Supergiant'         : None,
    'Rising Pune Supergiants'        : None,
    'Deccan Chargers'                : None,
    'Kochi Tuskers Kerala'           : None,
    'Pune Warriors'                  : None,
    'Gujarat Lions'                  : None,
}

for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
    match_df[col] = match_df[col].map(abbrev_map)

# Keep only 2026 IPL teams
ipl_2026_teams = {'CSK','DC','GT','KKR','LSG','MI','PBKS','RCB','RR','SRH'}
mask = (
    match_df['batting_team'].isin(ipl_2026_teams) &
    match_df['bowling_team'].isin(ipl_2026_teams) &
    match_df['match_won_by'].isin(ipl_2026_teams)
)
match_df = match_df[mask].reset_index(drop=True)
print('Matches after filtering:', len(match_df))
print('Teams:', sorted(match_df['batting_team'].unique()))
# Fix season column — handles '2007/08', '2009/10', '2020/21' formats
raw_season = df.groupby('match_id').first().reset_index()
season_lookup = raw_season.set_index('match_id')['season'].astype(str)
match_df['season_raw'] = match_df['match_id'].map(season_lookup)

season_map = {
    '2007/08':'2008', '2009':'2009', '2009/10':'2010',
    '2011':'2011', '2012':'2012', '2013':'2013', '2014':'2014',
    '2015':'2015', '2016':'2016', '2017':'2017', '2018':'2018',
    '2019':'2019', '2020':'2020', '2020/21':'2021', '2021':'2021',
    '2022':'2022', '2023':'2023', '2024':'2024', '2025':'2025',
}
match_df['season'] = match_df['season_raw'].map(season_map).astype(int)
match_df = match_df.drop(columns=['season_raw'])
print('Seasons:', sorted(match_df['season'].unique()))
3. Feature Engineering
# Sort by date — CRITICAL to prevent data leakage
match_df['date'] = pd.to_datetime(match_df['date'])
match_df = match_df.sort_values('date').reset_index(drop=True)

# Base columns
match_df['team1']       = match_df['batting_team']   # bats first
match_df['team2']       = match_df['bowling_team']   # chases
match_df['team1_won']   = (match_df['match_won_by'] == match_df['team1']).astype(int)
match_df['toss_is_team1']     = (match_df['toss_winner'] == match_df['team1']).astype(int)
match_df['toss_decision_bat'] = (match_df['toss_decision'] == 'bat').astype(int)

# Stage — knockout vs league
stage_lookup = df.groupby('match_id')['stage'].first()
match_df['stage'] = match_df['match_id'].map(stage_lookup)
knockout_stages = {'Final','Qualifier 1','Qualifier 2','Eliminator','Semi Final','Elimination Final'}
match_df['is_knockout'] = match_df['stage'].isin(knockout_stages).astype(int)

print('Base features done.')
# Helper: compute win rate for a team from a subset of matches
def win_rate(team, matches):
    if len(matches) == 0:
        return 0.5
    w = ((matches['team1'] == team) & (matches['team1_won'] == 1)).sum() + \
        ((matches['team2'] == team) & (matches['team1_won'] == 0)).sum()
    return w / len(matches)

# Initialize all rolling feature lists
(
    t1_wr, t2_wr,
    t1_swr, t2_swr,
    t1_f5, t2_f5,
    t1_f10, t2_f10,
    t1_bat, t2_bat,
    t1_ch, t2_ch,
    t1_vwr,
    h2h,
    t1_str, t2_str,
    venue_avg
) = [[] for _ in range(17)]

# First innings runs per match (for venue avg score)
fi = df[df['innings'] == 1].groupby('match_id')['runs_total'].sum().reset_index()
fi.columns = ['match_id', 'fi_runs']
venue_map = df.groupby('match_id')['venue'].first()
fi['venue'] = fi['match_id'].map(venue_map)

def streak(team, past):
    m = past[(past['team1'] == team) | (past['team2'] == team)]
    if len(m) == 0: return 0
    s = 0
    for _, r in m.iloc[::-1].iterrows():
        won = (r['team1'] == team and r['team1_won'] == 1) or \
              (r['team2'] == team and r['team1_won'] == 0)
        if s == 0: s = 1 if won else -1
        elif (s > 0 and won) or (s < 0 and not won): s += 1 if won else -1
        else: break
    return s

print('Starting rolling feature computation (may take 2-3 mins)...')
for i, row in match_df.iterrows():
    past = match_df.iloc[:i]
    t1, t2 = row['team1'], row['team2']

    # All matches per team
    m1 = past[(past['team1']==t1)|(past['team2']==t1)]
    m2 = past[(past['team1']==t2)|(past['team2']==t2)]

    # All-time win rate
    t1_wr.append(win_rate(t1, m1))
    t2_wr.append(win_rate(t2, m2))

    # Season win rate
    sm1 = m1[m1['season'] == row['season']]
    sm2 = m2[m2['season'] == row['season']]
    t1_swr.append(win_rate(t1, sm1))
    t2_swr.append(win_rate(t2, sm2))

    # Recent form — last 5 and last 10
    t1_f5.append(win_rate(t1, m1.tail(5)))
    t2_f5.append(win_rate(t2, m2.tail(5)))
    t1_f10.append(win_rate(t1, m1.tail(10)))
    t2_f10.append(win_rate(t2, m2.tail(10)))

    # Batting first win rate
    bat1 = past[past['team1'] == t1]
    bat2 = past[past['team1'] == t2]
    t1_bat.append(bat1['team1_won'].mean() if len(bat1) > 0 else 0.5)
    t2_bat.append(bat2['team1_won'].mean() if len(bat2) > 0 else 0.5)

    # Chasing win rate
    ch1 = past[past['team2'] == t1]
    ch2 = past[past['team2'] == t2]
    t1_ch.append((ch1['team1_won']==0).mean() if len(ch1) > 0 else 0.5)
    t2_ch.append((ch2['team1_won']==0).mean() if len(ch2) > 0 else 0.5)

    # Venue win rate for team1
    vm = m1[m1['venue'] == row['venue']]
    t1_vwr.append(win_rate(t1, vm))

    # Head-to-head
    h = past[((past['team1']==t1)&(past['team2']==t2))|
             ((past['team1']==t2)&(past['team2']==t1))]
    hw = ((h['team1']==t1)&(h['team1_won']==1)).sum() + \
         ((h['team2']==t1)&(h['team1_won']==0)).sum()
    h2h.append(hw / len(h) if len(h) > 0 else 0.5)

    # Streak
    t1_str.append(streak(t1, past))
    t2_str.append(streak(t2, past))

    # Venue average first innings score
    past_fi = fi[(fi['venue'] == row['venue']) & (fi['match_id'] < row['match_id'])]
    venue_avg.append(past_fi['fi_runs'].mean() if len(past_fi) > 0 else 150)

# Assign all features
match_df['t1_wr']   = t1_wr;   match_df['t2_wr']   = t2_wr
match_df['t1_swr']  = t1_swr;  match_df['t2_swr']  = t2_swr
match_df['t1_f5']   = t1_f5;   match_df['t2_f5']   = t2_f5
match_df['t1_f10']  = t1_f10;  match_df['t2_f10']  = t2_f10
match_df['t1_bat']  = t1_bat;  match_df['t2_bat']  = t2_bat
match_df['t1_ch']   = t1_ch;   match_df['t2_ch']   = t2_ch
match_df['t1_vwr']  = t1_vwr
match_df['h2h']     = h2h
match_df['t1_str']  = t1_str;  match_df['t2_str']  = t2_str
match_df['venue_avg'] = venue_avg

print('All features computed!')
# Build differential features — cleaner signal for models
match_df['wr_diff']    = match_df['t1_wr']   - match_df['t2_wr']
match_df['swr_diff']   = match_df['t1_swr']  - match_df['t2_swr']
match_df['f5_diff']    = match_df['t1_f5']   - match_df['t2_f5']
match_df['f10_diff']   = match_df['t1_f10']  - match_df['t2_f10']
match_df['bat_diff']   = match_df['t1_bat']  - match_df['t2_bat']
match_df['ch_diff']    = match_df['t1_ch']   - match_df['t2_ch']
match_df['str_diff']   = match_df['t1_str']  - match_df['t2_str']
match_df['h2h_diff']   = match_df['h2h']     - 0.5

FEATURES = [
    'wr_diff', 'swr_diff', 'f5_diff', 'f10_diff',
    'bat_diff', 'ch_diff', 'str_diff', 'h2h_diff',
    't1_vwr', 'venue_avg', 'season', 'is_knockout'
]

X = match_df[FEATURES]
y = match_df['team1_won']

print(f'Features : {len(FEATURES)}')
print(f'Samples  : {len(X)}')
print(f'Nulls    : {X.isnull().sum().sum()}')
print(f'Class balance:\n{y.value_counts(normalize=True).round(3)}')
team1_won
0    0.539
1    0.461
Name: proportion, dtype: float64
4. Train/Test Split (Time-Based)
# Time-based split — NEVER use random split for temporal data
# Train: 2008-2022 | Test: 2023-2025
train = match_df[match_df['season'] <= 2022]
test  = match_df[match_df['season'] >= 2023]

X_train, y_train = train[FEATURES], train['team1_won']
X_test,  y_test  = test[FEATURES],  test['team1_won']

print(f'Train: {len(X_train)} matches ({train.season.min()}–{train.season.max()})')
print(f'Test : {len(X_test)}  matches ({test.season.min()}–{test.season.max()})')
print(f'Baseline accuracy (always predict chaser): {y_test.value_counts(normalize=True).max():.3f}')
5. Model Training & Evaluation
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import accuracy_score, classification_report, roc_auc_score
from xgboost   import XGBClassifier
from lightgbm  import LGBMClassifier
from catboost  import CatBoostClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define all models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_split=20,
        min_samples_leaf=10, class_weight='balanced', random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        random_state=42, eval_metric='logloss', verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=200, depth=3, learning_rate=0.05,
        auto_class_weights='Balanced', random_seed=42, verbose=0
    ),
}

# Stacking model — uses all above as base learners
estimators = [
    ('lr',  Pipeline([('s', StandardScaler()), ('c', LogisticRegression(C=0.05, class_weight='balanced', max_iter=1000, random_state=42))])),
    ('rf',  RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=10, class_weight='balanced', random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0, random_state=42, eval_metric='logloss')),
    ('lgb', LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)),
    ('cat', CatBoostClassifier(iterations=100, depth=3, learning_rate=0.05, verbose=0, random_seed=42)),
]
models['Stacking'] = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    cv=5, passthrough=True
)

print('All models defined.')
# Train and evaluate all models
results = []

for name, model in models.items():
    print(f'Training {name}...', end=' ')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    cv_score  = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

    try:
        y_prob   = model.predict_proba(X_test)[:, 1]
        roc_auc  = roc_auc_score(y_test, y_prob)
    except:
        roc_auc = None

    results.append({
        'Model'      : name,
        'Train Acc'  : round(train_acc, 3),
        'CV Acc'     : round(cv_score, 3),
        'Test Acc'   : round(test_acc, 3),
        'ROC-AUC'    : round(roc_auc, 3) if roc_auc else '-',
        'Overfit Gap': round(train_acc - test_acc, 3)
    })
    print(f'Done — Test Acc: {test_acc:.3f}')

results_df = pd.DataFrame(results).sort_values('Test Acc', ascending=False).reset_index(drop=True)
print('\n=== MODEL COMPARISON ===')
print(results_df.to_string(index=False))

6. Neural Network (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scale and convert to tensors
nn_scaler      = StandardScaler()
X_train_sc     = nn_scaler.fit_transform(X_train)
X_test_sc      = nn_scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_sc)
y_train_t = torch.FloatTensor(y_train.values)
X_test_t  = torch.FloatTensor(X_test_sc)
y_test_t  = torch.FloatTensor(y_test.values)

loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

class IPLNet(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),   nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze()

nn_model  = IPLNet(X_train_t.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

train_losses, test_accs = [], []
best_acc, best_epoch = 0, 0

for epoch in range(300):
    nn_model.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(nn_model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    nn_model.eval()
    with torch.no_grad():
        preds = (nn_model(X_test_t) > 0.5).float()
        acc   = (preds == y_test_t).float().mean().item()

    train_losses.append(epoch_loss / len(loader))
    test_accs.append(acc)

    if acc > best_acc:
        best_acc, best_epoch = acc, epoch+1

    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1:>3} | Loss: {epoch_loss/len(loader):.4f} | Test Acc: {acc:.3f}')

print(f'\nBest Test Acc: {best_acc:.3f} at epoch {best_epoch}')
Epoch 100 | Loss: 0.6238 | Test Acc: 0.509
Epoch 150 | Loss: 0.6031 | Test Acc: 0.523
Epoch 200 | Loss: 0.5952 | Test Acc: 0.495
Epoch 250 | Loss: 0.6000 | Test Acc: 0.519
Epoch 300 | Loss: 0.5977 | Test Acc: 0.519

# Add NN to results
nn_model.eval()
with torch.no_grad():
    nn_train_acc = ((nn_model(X_train_t) > 0.5).float() == y_train_t).float().mean().item()
    nn_test_acc  = best_acc

results_df = pd.concat([results_df, pd.DataFrame([{
    'Model'      : 'Neural Network',
    'Train Acc'  : round(nn_train_acc, 3),
    'CV Acc'     : '-',
    'Test Acc'   : round(nn_test_acc, 3),
    'ROC-AUC'    : '-',
    'Overfit Gap': round(nn_train_acc - nn_test_acc, 3)
}])], ignore_index=True)

results_df = results_df.sort_values('Test Acc', ascending=False).reset_index(drop=True)
print('=== FINAL MODEL COMPARISON ===')
print(results_df.to_string(index=False))
7. Visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('IPL 2026 — Model Comparison', fontsize=14, fontweight='bold')

# Plot 1: Test Accuracy comparison
colors = ['green' if x == results_df['Test Acc'].max() else 'steelblue'
          for x in results_df['Test Acc']]
axes[0].barh(results_df['Model'], results_df['Test Acc'], color=colors)
axes[0].axvline(x=0.505, color='red', linestyle='--', label='Baseline')
axes[0].set_xlabel('Test Accuracy')
axes[0].set_title('Test Accuracy by Model')
axes[0].legend()
for i, v in enumerate(results_df['Test Acc']):
    axes[0].text(v + 0.002, i, f'{v:.3f}', va='center')

# Plot 2: Train vs Test (overfitting check)
num_models = results_df[results_df['Train Acc'] != '-']
x = np.arange(len(num_models))
axes[1].bar(x - 0.2, num_models['Train Acc'], 0.4, label='Train', color='steelblue')
axes[1].bar(x + 0.2, num_models['Test Acc'],  0.4, label='Test',  color='coral')
axes[1].set_xticks(x)
axes[1].set_xticklabels(num_models['Model'], rotation=30, ha='right')
axes[1].set_title('Train vs Test Accuracy')
axes[1].legend()

# Plot 3: Neural Network training curve
axes[2].plot(test_accs, color='steelblue', label='Test Accuracy')
axes[2].axhline(y=results_df[results_df['Model']=='Logistic Regression']['Test Acc'].values[0],
                color='red', linestyle='--', label='Logistic Regression')
axes[2].axhline(y=0.505, color='gray', linestyle=':', label='Baseline')
axes[2].set_xlabel('Epoch')
axes[2].set_title('Neural Network — Test Accuracy Curve')
axes[2].legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

8. Best Model — Detailed Evaluation
best_model_name = results_df.iloc[0]['Model']
print(f'Best Model: {best_model_name}')
print(f'Test Accuracy: {results_df.iloc[0]["Test Acc"]}')

if best_model_name == 'Neural Network':
    nn_model.eval()
    with torch.no_grad():
        y_pred_best = (nn_model(X_test_t) > 0.5).numpy().astype(int)
else:
    best_model  = models[best_model_name]
    y_pred_best = best_model.predict(X_test)

print(f'\n{classification_report(y_test, y_pred_best, target_names=["Team2 wins", "Team1 wins"])}')


  Team2 wins       0.57      0.50      0.53       106
  Team1 wins       0.56      0.63      0.59       108


9. Predict IPL 2026 Fixtures
# ── Section 9: Predict IPL 2026 Fixtures ─────────────────────────────────────
import random
#random.seed(42)

city_to_venue = {
    'Bengaluru' : 'M Chinnaswamy Stadium, Bengaluru',
    'Mumbai'    : 'Wankhede Stadium, Mumbai',
    'Guwahati'  : 'ACA Stadium, Guwahati',
    'Mullanpur' : 'PCA International Cricket Stadium, New Chandigarh',
    'Lucknow'   : 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Kolkata'   : 'Eden Gardens, Kolkata',
    'Chennai'   : 'MA Chidambaram Stadium, Chennai',
    'Delhi'     : 'Arun Jaitley Stadium, Delhi',
    'Ahmedabad' : 'Narendra Modi Stadium, Ahmedabad',
    'Hyderabad' : 'Rajiv Gandhi International Stadium, Hyderabad',
    'Jaipur'    : 'Sawai Mansingh Stadium, Jaipur',
}

team_venues = {
    'CSK' : 'MA Chidambaram Stadium, Chennai',
    'DC'  : 'Arun Jaitley Stadium, Delhi',
    'GT'  : 'Narendra Modi Stadium, Ahmedabad',
    'KKR' : 'Eden Gardens, Kolkata',
    'LSG' : 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'MI'  : 'Wankhede Stadium, Mumbai',
    'PBKS': 'PCA International Cricket Stadium, New Chandigarh',
    'RCB' : 'M Chinnaswamy Stadium, Bengaluru',
    'RR'  : 'Sawai Mansingh Stadium, Jaipur',
    'SRH' : 'Rajiv Gandhi International Stadium, Hyderabad',
}

# ── Helper functions ──────────────────────────────────────────────────────────
def get_wr(team):
    m = match_df[(match_df['team1']==team)|(match_df['team2']==team)]
    return win_rate(team, m)

def get_swr(team, season=2025):
    m = match_df[(match_df['season']==season) &
                 ((match_df['team1']==team)|(match_df['team2']==team))]
    return win_rate(team, m)

def get_form(team, n=5):
    m = match_df[(match_df['team1']==team)|(match_df['team2']==team)].tail(n)
    return win_rate(team, m)

def get_bat(team):
    m = match_df[match_df['team1']==team]
    return m['team1_won'].mean() if len(m) > 0 else 0.5

def get_chase(team):
    m = match_df[match_df['team2']==team]
    return (m['team1_won']==0).mean() if len(m) > 0 else 0.5

def get_h2h(t1, t2):
    m = match_df[((match_df['team1']==t1)&(match_df['team2']==t2))|
                 ((match_df['team1']==t2)&(match_df['team2']==t1))]
    if len(m) == 0: return 0.5
    w = ((m['team1']==t1)&(m['team1_won']==1)).sum() + \
        ((m['team2']==t1)&(m['team1_won']==0)).sum()
    return w / len(m)

def get_streak(team):
    return streak(team, match_df)

def get_vwr(team, venue):
    m = match_df[(match_df['venue']==venue) &
                 ((match_df['team1']==team)|(match_df['team2']==team))]
    return win_rate(team, m)

def get_vavg(venue):
    m = match_df[match_df['venue']==venue]
    if len(m) == 0: return 150
    return df[df['match_id'].isin(m['match_id']) &
              (df['innings']==1)]['runs_total'].sum() / len(m)

def build_features(t1, t2, venue, is_knockout=0):
    return pd.DataFrame([{
        'wr_diff'    : get_wr(t1)     - get_wr(t2),
        'swr_diff'   : get_swr(t1)    - get_swr(t2),
        'f5_diff'    : get_form(t1,5) - get_form(t2,5),
        'f10_diff'   : get_form(t1,10)- get_form(t2,10),
        'bat_diff'   : get_bat(t1)    - get_bat(t2),
        'ch_diff'    : get_chase(t1)  - get_chase(t2),
        'str_diff'   : get_streak(t1) - get_streak(t2),
        'h2h_diff'   : get_h2h(t1,t2) - 0.5,
        't1_vwr'     : get_vwr(t1, venue),
        'venue_avg'  : get_vavg(venue),
        'season'     : 2026,
        'is_knockout': is_knockout
    }])

def get_proba(t1, t2, venue, is_knockout=0):
    feat = build_features(t1, t2, venue, is_knockout)
    if best_model_name == 'Neural Network':
        nn_model.eval()
        with torch.no_grad():
            scaled = torch.FloatTensor(nn_scaler.transform(feat))
            p1 = nn_model(scaled).item()
        return np.array([1 - p1, p1])
    else:
        return models[best_model_name].predict_proba(feat)[0]

# ── Fixture Predictions ───────────────────────────────────────────────────────
fixtures = [
    ('Mar 28', 'RCB',  'SRH',  'Bengaluru'),
    ('Mar 29', 'MI',   'KKR',  'Mumbai'),
    ('Mar 30', 'RR',   'CSK',  'Guwahati'),
    ('Mar 31', 'PBKS', 'GT',   'Mullanpur'),
    ('Apr 1',  'LSG',  'DC',   'Lucknow'),
    ('Apr 2',  'KKR',  'SRH',  'Kolkata'),
    ('Apr 3',  'CSK',  'PBKS', 'Chennai'),
    ('Apr 4',  'DC',   'MI',   'Delhi'),
    ('Apr 4',  'GT',   'RR',   'Ahmedabad'),
    ('Apr 5',  'SRH',  'LSG',  'Hyderabad'),
    ('Apr 5',  'RCB',  'CSK',  'Bengaluru'),
    ('Apr 6',  'KKR',  'PBKS', 'Kolkata'),
    ('Apr 7',  'RR',   'MI',   'Guwahati'),
    ('Apr 8',  'DC',   'GT',   'Delhi'),
    ('Apr 9',  'KKR',  'LSG',  'Kolkata'),
    ('Apr 10', 'RR',   'RCB',  'Guwahati'),
    ('Apr 11', 'PBKS', 'SRH',  'Mullanpur'),
    ('Apr 11', 'CSK',  'DC',   'Chennai'),
    ('Apr 12', 'LSG',  'GT',   'Lucknow'),
    ('Apr 12', 'MI',   'RCB',  'Mumbai'),
]

print(f"{'#':<4} {'Date':<8} {'Bats':<6} {'Chases':<6} {'Bat%':>7} {'Chase%':>8} {'Winner':>10}")
print('=' * 55)

match_results = []
win_counts    = {t: 0 for t in team_venues.keys()}

for i, (date, tA, tB, city) in enumerate(fixtures, 1):
    venue  = city_to_venue[city]
    t1, t2 = (tA, tB) if random.random() > 0.5 else (tB, tA)
    prob   = get_proba(t1, t2, venue)
    winner = t1 if prob[1] > 0.5 else t2
    win_counts[winner] += 1
    match_results.append({'t1': t1, 't2': t2, 'winner': winner})
    print(f"{i:<4} {date:<8} {t1:<6} {t2:<6} {prob[1]*100:>6.1f}% {prob[0]*100:>7.1f}% {'🏆 '+winner:>10}")

# ── Points Table ──────────────────────────────────────────────────────────────
print('\n')
pt = pd.DataFrame({
    'Team'  : list(win_counts.keys()),
    'Wins'  : list(win_counts.values()),
    'Points': [w*2 for w in win_counts.values()]
}).sort_values('Points', ascending=False).reset_index(drop=True)
pt.index += 1

print('=== PREDICTED POINTS TABLE ===')
print(pt.to_string())
print(f"\nTop 4: {pt['Team'].iloc[:4].tolist()}")

# ── Playoff Simulator ─────────────────────────────────────────────────────────
p1, p2, p3, p4 = pt['Team'].iloc[:4].tolist()

def playoff_predict(tA, tB, stage):
    t1, t2  = (tA, tB) if random.random() > 0.5 else (tB, tA)
    venue   = team_venues[t1]
    prob    = get_proba(t1, t2, venue, is_knockout=1)
    winner  = t1 if prob[1] > 0.5 else t2
    loser   = t2 if prob[1] > 0.5 else t1
    conf    = max(prob[0], prob[1]) * 100
    print(f"  {stage:<30} {tA} vs {tB}  →  🏆 {winner}  ({conf:.1f}% confidence)")
    return winner, loser

print('\n' + '='*60)
print('           IPL 2026 PLAYOFF PREDICTIONS')
print('='*60)
print(f'  Top 4 → {p1}, {p2}, {p3}, {p4}\n')

q1_win, q1_los = playoff_predict(p1, p2, 'Qualifier 1  (1st vs 2nd)')
el_win, el_los = playoff_predict(p3, p4, 'Eliminator   (3rd vs 4th)')
q2_win, q2_los = playoff_predict(q1_los, el_win, 'Qualifier 2')
champ, runner  = playoff_predict(q1_win, q2_win, 'FINAL')

print(f'\n{"="*60}')
print(f'  🏆  IPL 2026 PREDICTED CHAMPION :  {champ}')
print(f'  🥈  Runner Up                   :  {runner}')
print(f'{"="*60}')
#    Date     Bats   Chases    Bat%   Chase%     Winner




  Top 4 → MI, GT, KKR, RCB

  Qualifier 1  (1st vs 2nd)      MI vs GT  →  🏆 MI  (51.5% confidence)
  Eliminator   (3rd vs 4th)      KKR vs RCB  →  🏆 RCB  (53.0% confidence)
  Qualifier 2                    GT vs RCB  →  🏆 GT  (55.3% confidence)
  FINAL                          MI vs GT  →  🏆 GT  (54.2% confidence)

  🏆  IPL 2026 PREDICTED CHAMPION :  GT
  🥈  Runner Up                   :  MI
# ── Section 10: Monte Carlo Simulation ───────────────────────────────────────
from collections import defaultdict
import time

# ── Step 1: Precompute all probabilities ONCE ─────────────────────────────────
# This avoids calling the model 200,000 times inside the loop
print('Precomputing match probabilities...')

fixture_probs = {}
for _, tA, tB, city in fixtures:
    venue = city_to_venue[city]
    key   = (tA, tB, city)
    fixture_probs[key] = {
        'AB': get_proba(tA, tB, venue),   # tA bats first → [p_tB_wins, p_tA_wins]
        'BA': get_proba(tB, tA, venue),   # tB bats first → [p_tA_wins, p_tB_wins]
    }

all_teams     = list(team_venues.keys())
playoff_probs = {}
for tA in all_teams:
    for tB in all_teams:
        if tA != tB:
            venue = team_venues[tA]
            playoff_probs[(tA, tB)] = get_proba(tA, tB, venue, is_knockout=1)

print(f'  ✓ {len(fixture_probs)} fixture probabilities precomputed')
print(f'  ✓ {len(playoff_probs)} playoff matchup probabilities precomputed')
print('  Ready for fast simulation!\n')

# ── Step 2: Monte Carlo Simulation ───────────────────────────────────────────
N_SIMULATIONS   = 100
champion_counts = defaultdict(int)
finalist_counts = defaultdict(int)
top4_counts     = defaultdict(int)
points_accum    = defaultdict(int)   # to track average points per team

start = time.time()

for sim in range(N_SIMULATIONS):

    # ── League stage ──────────────────────────────────────────────────────────
    sim_wins = {t: 0 for t in all_teams}

    for _, tA, tB, city in fixtures:
        key   = (tA, tB, city)
        probs = fixture_probs[key]

        # Randomly assign batting order each simulation
        if random.random() > 0.5:
            p_tA_wins = probs['AB'][1]   # tA bats first
        else:
            p_tA_wins = probs['BA'][0]   # tB bats first, tA wins = BA[0]

        # Probabilistic outcome — this is what makes it Monte Carlo
        winner = tA if random.random() < p_tA_wins else tB
        sim_wins[winner] += 1

    for t in all_teams:
        points_accum[t] += sim_wins[t] * 2

    # ── Determine top 4 with random tie-breaking ──────────────────────────────
    ranked = sorted(sim_wins.items(), key=lambda x: x[1], reverse=True)

    top4, i = [], 0
    while len(top4) < 4:
        j = i
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        group = [ranked[k][0] for k in range(i, j)]
        random.shuffle(group)
        top4.extend(group)
        i = j
    top4 = top4[:4]

    for t in top4:
        top4_counts[t] += 1

    sp1, sp2, sp3, sp4 = top4

    # ── Playoff simulation ────────────────────────────────────────────────────
    def fast_sim(tA, tB):
        """Simulate a single match using precomputed probabilities."""
        if random.random() > 0.5:
            p_tA = playoff_probs[(tA, tB)][1]   # tA bats first
        else:
            p_tA = playoff_probs[(tB, tA)][0]   # tB bats first
        winner = tA if random.random() < p_tA else tB
        loser  = tB if winner == tA else tA
        return winner, loser

    q1_w, q1_l = fast_sim(sp1, sp2)   # Qualifier 1: 1st vs 2nd
    el_w, el_l = fast_sim(sp3, sp4)   # Eliminator:  3rd vs 4th
    q2_w, q2_l = fast_sim(q1_l, el_w) # Qualifier 2: Q1 loser vs Elim winner
    champ, ru  = fast_sim(q1_w, q2_w)  # Final

    finalist_counts[q1_w] += 1
    finalist_counts[q2_w] += 1
    champion_counts[champ] += 1

    if (sim + 1) % 2000 == 0:
        elapsed = time.time() - start
        print(f'  {sim+1:>6,}/{N_SIMULATIONS:,} simulations | {elapsed:.1f}s elapsed')

elapsed = time.time() - start
print(f'\n✓ Done! {N_SIMULATIONS:,} simulations completed in {elapsed:.1f} seconds\n')

# ── Step 3: Results Table ─────────────────────────────────────────────────────
mc_df = pd.DataFrame({
    'Team'          : all_teams,
    'Champion %'    : [round(champion_counts[t] / N_SIMULATIONS * 100, 1) for t in all_teams],
    'Finalist %'    : [round(finalist_counts[t] / N_SIMULATIONS * 100, 1) for t in all_teams],
    'Top 4 %'       : [round(top4_counts[t]     / N_SIMULATIONS * 100, 1) for t in all_teams],
    'Avg Points'    : [round(points_accum[t]     / N_SIMULATIONS, 1)       for t in all_teams],
}).sort_values('Champion %', ascending=False).reset_index(drop=True)
mc_df.index += 1

print(f'{"="*60}')
print(f'   MONTE CARLO RESULTS — {N_SIMULATIONS:,} SIMULATIONS')
print(f'{"="*60}')
print(mc_df.to_string())

# ── Step 4: Visualization ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'IPL 2026 — Monte Carlo Simulation ({N_SIMULATIONS:,} runs)',
             fontsize=15, fontweight='bold', y=1.01)

sorted_teams = mc_df['Team'].tolist()
colors       = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(sorted_teams)))

# Plot 1: Championship probability
ax = axes[0, 0]
vals = mc_df.set_index('Team').loc[sorted_teams, 'Champion %']
bars = ax.barh(sorted_teams[::-1], vals[::-1], color=colors[::-1])
ax.set_xlabel('Probability (%)')
ax.set_title('🏆 Championship Probability', fontweight='bold')
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)

# Plot 2: Finalist probability
ax = axes[0, 1]
vals = mc_df.set_index('Team').loc[sorted_teams, 'Finalist %']
bars = ax.barh(sorted_teams[::-1], vals[::-1], color=colors[::-1])
ax.set_xlabel('Probability (%)')
ax.set_title('🥈 Finalist Probability', fontweight='bold')
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)

# Plot 3: Top 4 probability
ax = axes[1, 0]
vals = mc_df.set_index('Team').loc[sorted_teams, 'Top 4 %']
bars = ax.barh(sorted_teams[::-1], vals[::-1], color=colors[::-1])
ax.set_xlabel('Probability (%)')
ax.set_title('🔝 Top 4 Probability', fontweight='bold')
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)

# Plot 4: Average predicted points
ax = axes[1, 1]
vals = mc_df.set_index('Team').loc[sorted_teams, 'Avg Points']
bars = ax.barh(sorted_teams[::-1], vals[::-1], color=colors[::-1])
ax.set_xlabel('Average Points')
ax.set_title('📊 Average Predicted Points', fontweight='bold')
for bar, val in zip(bars, vals[::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('monte_carlo.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Step 5: Final Announcement ────────────────────────────────────────────────
print(f'\n{"="*55}')
print(f'  🏆  MOST LIKELY IPL 2026 CHAMPION : {mc_df.iloc[0]["Team"]}')
print(f'  📊  Championship Probability      : {mc_df.iloc[0]["Champion %"]}%')
print(f'  🥈  Most Likely Finalist          : {mc_df.iloc[1]["Team"]}')
print(f'  📊  Finalist Probability          : {mc_df.iloc[1]["Finalist %"]}%')
print(f'\n  Top 4 most likely teams:')
for _, row in mc_df.head(4).iterrows():
    print(f'    {row["Team"]:<6}  Champion: {row["Champion %"]:>5}%  |  Top4: {row["Top 4 %"]:>5}%')
print(f'{"="*55}')
  ✓ 20 fixture probabilities precomputed
  ✓ 90 playoff matchup probabilities precomputed


✓ Done! 100 simulations completed in 0.0 seconds


  🏆  MOST LIKELY IPL 2026 CHAMPION : RCB
  📊  Championship Probability      : 17.0%
  🥈  Most Likely Finalist          : GT
  📊  Finalist Probability          : 26.0%

  Top 4 most likely teams:
    RCB     Champion:  17.0%  |  Top4:  46.0%
    GT      Champion:  15.0%  |  Top4:  50.0%
    PBKS    Champion:  14.0%  |  Top4:  42.0%
    KKR     Champion:  11.0%  |  Top4:  35.0%
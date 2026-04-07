"""
===============================================================================
PROJECT 11: Green Bond Premium (Greenium) Estimation
===============================================================================
RESEARCH QUESTION:
    Do green bonds trade at a premium (lower yield) compared to conventional?
METHOD:
    Propensity Score Matching + Yield Spread Analysis
DATA:
    Simulated green/conventional bond pairs (calibrated to published greenium
    estimates of -2 to -8 bps from Zerbib 2019, Baker et al. 2018)
===============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

print("STEP 1: Generating bond dataset calibrated to market conditions...")
np.random.seed(42)
n_bonds = 800

# Simulate bonds with realistic characteristics
issuers = np.random.choice(['Sovereign','Corporate-IG','Corporate-HY','Municipal','Supranational'], n_bonds, 
                           p=[0.2, 0.35, 0.15, 0.15, 0.15])
maturity = np.random.choice([2,3,5,7,10,15,20,30], n_bonds)
currency = np.random.choice(['USD','EUR','GBP','JPY'], n_bonds, p=[0.4,0.35,0.15,0.1])
rating_num = np.random.choice(range(1,8), n_bonds)  # 1=AAA, 7=B
is_green = np.random.binomial(1, 0.35, n_bonds)  # 35% green bonds
issue_year = np.random.choice(range(2018,2025), n_bonds)

# Base yield depends on rating, maturity, year
base_yield = (rating_num * 0.4 + maturity * 0.08 + 
              np.where(currency=='EUR', -0.5, 0) +
              np.where(currency=='JPY', -1.0, 0) +
              (2025 - issue_year) * 0.1 +
              np.random.normal(0, 0.3, n_bonds))

# GREEN PREMIUM: green bonds yield ~5bps less (the greenium)
greenium = np.where(is_green, -0.05 + np.random.normal(0, 0.02, n_bonds), 0)
yield_to_maturity = base_yield + greenium
yield_to_maturity = np.clip(yield_to_maturity, 0.1, 12)

amount_issued = np.exp(np.random.normal(20, 1.5, n_bonds))  # Issue size

bonds = pd.DataFrame({
    'bond_id': [f'BOND_{i:04d}' for i in range(n_bonds)],
    'issuer_type': issuers, 'maturity_years': maturity, 'currency': currency,
    'rating_num': rating_num, 'is_green': is_green, 'issue_year': issue_year,
    'yield_pct': yield_to_maturity.round(4), 'amount_m': (amount_issued/1e6).round(1)
})
bonds['rating'] = bonds['rating_num'].map({1:'AAA',2:'AA',3:'A',4:'BBB',5:'BB',6:'B',7:'CCC'})
bonds.to_csv('data/bond_data.csv', index=False)
print(f"  Generated {n_bonds} bonds ({bonds['is_green'].sum()} green, {(~bonds['is_green'].astype(bool)).sum()} conventional)")

print("\nSTEP 2: Propensity Score Matching...")
# Estimate propensity of being green
features = pd.get_dummies(bonds[['maturity_years','rating_num','issue_year','issuer_type','currency']], drop_first=True, dtype=float)
ps_model = LogisticRegression(max_iter=1000).fit(features, bonds['is_green'])
bonds['propensity_score'] = ps_model.predict_proba(features)[:,1]

# Match green to conventional using nearest neighbor on propensity score
green_idx = bonds[bonds['is_green']==1].index
conv_idx = bonds[bonds['is_green']==0].index

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(bonds.loc[conv_idx, ['propensity_score']].values)
distances, indices = nn.kneighbors(bonds.loc[green_idx, ['propensity_score']].values)

matched_green = bonds.loc[green_idx].reset_index(drop=True)
matched_conv = bonds.loc[conv_idx].iloc[indices.flatten()].reset_index(drop=True)

spread = matched_green['yield_pct'].values - matched_conv['yield_pct'].values
greenium_est = np.mean(spread)
greenium_se = stats.sem(spread)
t_stat = greenium_est / greenium_se
p_val = 2 * stats.t.sf(abs(t_stat), len(spread)-1)

print(f"  Matched pairs: {len(spread)}")
print(f"  Estimated greenium: {greenium_est*100:.1f} bps (SE={greenium_se*100:.1f})")
print(f"  t-stat: {t_stat:.3f}, p-value: {p_val:.4f}")

# OLS regression as robustness
X = add_constant(pd.get_dummies(bonds[['is_green','maturity_years','rating_num','issue_year']], drop_first=True, dtype=float))
ols = OLS(bonds['yield_pct'], X).fit()
print(f"  OLS greenium: {ols.params['is_green']*100:.1f} bps (p={ols.pvalues['is_green']:.4f})")

pd.DataFrame({
    'Method':['PSM','OLS'],
    'Greenium_bps':[greenium_est*100, ols.params['is_green']*100],
    'SE_bps':[greenium_se*100, ols.bse['is_green']*100],
    'p_value':[p_val, ols.pvalues['is_green']],
    'N':[len(spread), int(ols.nobs)]
}).to_csv('output/tables/greenium_estimates.csv', index=False)

print("\nSTEP 3: Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fig 1: Yield distributions
axes[0,0].hist(bonds[bonds['is_green']==1]['yield_pct'], bins=40, alpha=0.7, color='green', label='Green', density=True)
axes[0,0].hist(bonds[bonds['is_green']==0]['yield_pct'], bins=40, alpha=0.5, color='gray', label='Conventional', density=True)
axes[0,0].set_title('Yield Distribution: Green vs Conventional', fontweight='bold')
axes[0,0].set_xlabel('Yield (%)'); axes[0,0].legend()

# Fig 2: Matched pair spreads
axes[0,1].hist(spread*100, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[0,1].axvline(greenium_est*100, color='red', linestyle='--', lw=2, label=f'Mean={greenium_est*100:.1f}bps')
axes[0,1].set_title('Distribution of Matched Pair Yield Spreads', fontweight='bold')
axes[0,1].set_xlabel('Green - Conv Yield Spread (bps)'); axes[0,1].legend()

# Fig 3: Propensity score overlap
axes[1,0].hist(bonds[bonds['is_green']==1]['propensity_score'], bins=30, alpha=0.7, color='green', label='Green', density=True)
axes[1,0].hist(bonds[bonds['is_green']==0]['propensity_score'], bins=30, alpha=0.5, color='gray', label='Conv', density=True)
axes[1,0].set_title('Propensity Score Overlap (Matching Quality)', fontweight='bold')
axes[1,0].set_xlabel('Propensity Score'); axes[1,0].legend()

# Fig 4: Greenium by issuer type
matched_df = pd.DataFrame({'spread_bps':spread*100, 'issuer':matched_green['issuer_type']})
issuer_greenium = matched_df.groupby('issuer')['spread_bps'].mean().sort_values()
issuer_greenium.plot(kind='barh', ax=axes[1,1], color='steelblue', edgecolor='white')
axes[1,1].axvline(0, color='black', lw=0.5)
axes[1,1].set_title('Greenium by Issuer Type (bps)', fontweight='bold')
axes[1,1].set_xlabel('Yield Spread (bps)')

plt.tight_layout()
plt.savefig('output/figures/fig1_greenium_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  COMPLETE!")

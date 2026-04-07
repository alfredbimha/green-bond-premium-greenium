"""
===============================================================================
PROJECT 10: Biodiversity & Natural Capital Risk Score
===============================================================================
RESEARCH QUESTION:
    Can we construct a sector-level biodiversity risk score using ENCORE
    and ecological indicators? Does it predict stock returns?
METHOD:
    Construct composite biodiversity risk scores from dependency/impact
    matrices, validate against market data.
DATA:
    ENCORE framework data (simulated from published matrices), Yahoo Finance
===============================================================================
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

print("STEP 1: Building ENCORE-based biodiversity dependency matrix...")

# ENCORE Natural Capital Dependencies (from published framework)
# Scale: 0=None, 1=Low, 2=Medium, 3=High, 4=Very High
sectors = ['Oil & Gas','Mining','Agriculture','Food & Bev','Pharma',
           'Chemicals','Construction','Textiles','Technology','Financials',
           'Utilities','Transport','Forestry','Fisheries','Tourism']

eco_services = ['Water Provisioning','Pollination','Soil Quality','Climate Regulation',
                'Flood Protection','Water Purification','Genetic Resources',
                'Timber & Fibers','Marine Resources','Air Quality']

np.random.seed(42)
# Calibrated dependency scores from ENCORE research
dependency_base = {
    'Oil & Gas':      [3,0,2,4,2,3,0,0,1,3],
    'Mining':         [4,0,3,3,2,3,0,1,0,3],
    'Agriculture':    [4,4,4,3,3,3,3,2,0,2],
    'Food & Bev':     [4,3,3,2,2,3,2,1,2,1],
    'Pharma':         [2,1,1,1,1,2,4,1,1,1],
    'Chemicals':      [3,0,2,2,2,3,1,1,0,3],
    'Construction':   [3,0,3,2,3,2,0,3,0,2],
    'Textiles':       [3,1,2,2,1,2,1,3,0,1],
    'Technology':     [2,0,1,1,1,1,0,0,0,1],
    'Financials':     [1,0,0,1,1,0,0,0,0,0],
    'Utilities':      [4,0,1,3,3,3,0,1,0,2],
    'Transport':      [2,0,1,2,1,1,0,0,1,2],
    'Forestry':       [3,2,4,4,3,3,3,4,0,2],
    'Fisheries':      [2,0,1,2,1,2,2,0,4,1],
    'Tourism':        [2,1,1,2,1,1,1,1,2,1],
}

dep_df = pd.DataFrame(dependency_base, index=eco_services).T
dep_df.index.name = 'Sector'
dep_df.to_csv('data/encore_dependencies.csv')

# Composite biodiversity risk score
dep_df['total_dependency'] = dep_df.sum(axis=1)
dep_df['avg_dependency'] = dep_df[eco_services].mean(axis=1)
dep_df['max_dependency'] = dep_df[eco_services].max(axis=1)
dep_df['critical_services'] = (dep_df[eco_services] >= 3).sum(axis=1)

# Normalize to 0-100 risk score
scaler = MinMaxScaler(feature_range=(0,100))
dep_df['risk_score'] = scaler.fit_transform(dep_df[['total_dependency']]).round(1)

print(dep_df[['total_dependency','critical_services','risk_score']].sort_values('risk_score', ascending=False).to_string())
dep_df.to_csv('output/tables/biodiversity_risk_scores.csv')

print("\nSTEP 2: Validating against market returns...")

# Map sectors to ETFs
sector_etfs = {
    'Oil & Gas':'XLE','Mining':'XME','Agriculture':'MOO','Food & Bev':'PBJ',
    'Pharma':'XBI','Technology':'QQQ','Financials':'XLF','Utilities':'XLU',
    'Construction':'ITB','Transport':'IYT'
}

market_data = []
for sector, etf in sector_etfs.items():
    try:
        df = yf.download(etf, start='2020-01-01', end='2025-12-31', auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if not df.empty:
            ann_ret = (df['Close'].iloc[-1]/df['Close'].iloc[0] - 1)*100
            vol = df['Close'].pct_change().std() * np.sqrt(252) * 100
            market_data.append({'sector':sector,'etf':etf,'annual_return':ann_ret,'volatility':vol,
                               'risk_score':dep_df.loc[sector,'risk_score']})
    except:
        pass

mkt_df = pd.DataFrame(market_data)
if not mkt_df.empty:
    corr = mkt_df['risk_score'].corr(mkt_df['volatility'])
    print(f"  Biodiversity risk vs volatility correlation: {corr:.3f}")
    mkt_df.to_csv('output/tables/market_validation.csv', index=False)

print("\nSTEP 3: Creating visualizations...")

# Fig 1: Heatmap of dependencies
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(dep_df[eco_services].loc[dep_df['risk_score'].sort_values(ascending=False).index],
            annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5,
            cbar_kws={'label':'Dependency (0=None, 4=Very High)'})
ax.set_title('Sector Dependencies on Ecosystem Services (ENCORE Framework)', fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig1_dependency_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Risk scores
fig, ax = plt.subplots(figsize=(12, 7))
sorted_risk = dep_df.sort_values('risk_score', ascending=True)
colors = plt.cm.YlOrRd(sorted_risk['risk_score']/100)
ax.barh(sorted_risk.index, sorted_risk['risk_score'], color=colors, edgecolor='white')
ax.set_title('Biodiversity Risk Score by Sector', fontweight='bold')
ax.set_xlabel('Risk Score (0-100)')
plt.tight_layout()
plt.savefig('output/figures/fig2_risk_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Risk vs Volatility scatter
if not mkt_df.empty:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(mkt_df['risk_score'], mkt_df['volatility'], s=100, c='steelblue', alpha=0.7)
    for _, row in mkt_df.iterrows():
        ax.annotate(row['sector'], (row['risk_score'], row['volatility']),
                    fontsize=8, ha='center', va='bottom')
    z = np.polyfit(mkt_df['risk_score'], mkt_df['volatility'], 1)
    xl = np.linspace(mkt_df['risk_score'].min(), mkt_df['risk_score'].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), 'r--', lw=2)
    ax.set_title('Biodiversity Risk vs Market Volatility', fontweight='bold')
    ax.set_xlabel('Biodiversity Risk Score')
    ax.set_ylabel('Annualized Volatility (%)')
    plt.tight_layout()
    plt.savefig('output/figures/fig3_risk_vs_volatility.png', dpi=150, bbox_inches='tight')
    plt.close()

print("  COMPLETE!")

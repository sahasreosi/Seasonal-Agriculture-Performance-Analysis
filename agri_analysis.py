import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path('/mnt/data/seasonal_agriculture_performance_dataset(1).xlsx')
OUTDIR = Path('/mnt/data/agri_analysis_outputs')
OUTDIR.mkdir(exist_ok=True)

# Load dataset
raw = pd.read_excel(INPUT)
print('Shape:', raw.shape)
print('\nMissing values:')
print(raw.isna().sum()[raw.isna().sum() > 0])

# Basic cleaning
_df = raw.copy()
_df.columns = _df.columns.str.strip()
num_cols = _df.select_dtypes(include=np.number).columns
_df[num_cols] = _df[num_cols].apply(pd.to_numeric, errors='coerce')
_df = _df.drop_duplicates()
_df = _df.dropna(subset=['Season', 'Crop', 'Yield_Tonnes_Ha', 'Revenue_INR', 'Profit_INR'])

# Analysis tables
season = _df.groupby('Season').agg(
    farms=('Farm_ID','count'),
    avg_yield_t_ha=('Yield_Tonnes_Ha','mean'),
    avg_production_t=('Production_Tonnes','mean'),
    avg_revenue_inr=('Revenue_INR','mean'),
    avg_profit_inr=('Profit_INR','mean'),
    avg_water_eff=('Water_Efficiency_t_per_1000m3','mean'),
    avg_pest_risk_pct=('Disease_Pest_Risk_pct','mean'),
    avg_rainfall_mm=('Rainfall_mm','mean'),
    avg_cost_inr=('Total_Cost_INR','mean')
).sort_values('avg_profit_inr', ascending=False)

crop = _df.groupby('Crop').agg(
    farms=('Farm_ID','count'),
    avg_yield_t_ha=('Yield_Tonnes_Ha','mean'),
    avg_profit_inr=('Profit_INR','mean'),
    avg_revenue_inr=('Revenue_INR','mean'),
    avg_water_eff=('Water_Efficiency_t_per_1000m3','mean')
).sort_values('avg_yield_t_ha', ascending=False)

irrig = _df.groupby('Irrigation_Method').agg(
    farms=('Farm_ID','count'),
    avg_yield_t_ha=('Yield_Tonnes_Ha','mean'),
    avg_profit_inr=('Profit_INR','mean'),
    avg_water_eff=('Water_Efficiency_t_per_1000m3','mean'),
    avg_water_used_m3=('Water_Used_m3','mean')
).sort_values('avg_water_eff', ascending=False)

# Season x crop for consistency
season_crop = _df.pivot_table(index='Season', columns='Crop', values='Yield_Tonnes_Ha', aggfunc='mean')

# Correlations with yield and profit
corr_cols = ['Rainfall_mm','Avg_Temperature_C','Humidity_pct','Sunlight_Hours_Day','Soil_Moisture_pct',
             'Nitrogen_kg_ha','Phosphorus_kg_ha','Potassium_kg_ha','Fertilizer_kg_ha',
             'Pesticide_Litre_ha','Seed_Quality_Score','Water_Efficiency_t_per_1000m3','Disease_Pest_Risk_pct',
             'Yield_Tonnes_Ha','Profit_INR']
corr = _df[corr_cols].corr(numeric_only=True)
yield_corr = corr['Yield_Tonnes_Ha'].drop('Yield_Tonnes_Ha').sort_values(key=np.abs, ascending=False)
profit_corr = corr['Profit_INR'].drop('Profit_INR').sort_values(key=np.abs, ascending=False)

# Profitability rate by season
profit_rate = _df.assign(Profitable=_df['Profit_INR'] > 0).groupby('Season')['Profitable'].mean().sort_values(ascending=False) * 100

# Save results
season.to_csv(OUTDIR/'season_summary.csv')
crop.to_csv(OUTDIR/'crop_summary.csv')
irrig.to_csv(OUTDIR/'irrigation_summary.csv')
season_crop.to_csv(OUTDIR/'season_crop_yield.csv')
yield_corr.to_csv(OUTDIR/'yield_correlations.csv')
profit_corr.to_csv(OUTDIR/'profit_correlations.csv')
profit_rate.to_csv(OUTDIR/'profitability_rate_by_season.csv')

# Charts used in presentation
plt.figure(figsize=(9,5))
season['avg_yield_t_ha'].plot(kind='bar')
plt.title('Average Yield by Season')
plt.ylabel('Tonnes per Hectare')
plt.xlabel('Season')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTDIR/'yield_by_season.png', dpi=180)
plt.close()

plt.figure(figsize=(9,5))
season['avg_profit_inr'].plot(kind='bar')
plt.title('Average Profit by Season')
plt.ylabel('Average Profit (INR)')
plt.xlabel('Season')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTDIR/'profit_by_season.png', dpi=180)
plt.close()

plt.figure(figsize=(9,5))
irrig['avg_water_eff'].plot(kind='bar')
plt.title('Average Water Efficiency by Irrigation Method')
plt.ylabel('Tonnes per 1,000 m³')
plt.xlabel('Irrigation Method')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTDIR/'water_efficiency_irrigation.png', dpi=180)
plt.close()

# Console-friendly answers to project questions
print('\n=== PROJECT ANSWERS ===')
print('1. How does performance vary across seasons?')
print(season[['avg_yield_t_ha','avg_profit_inr','avg_water_eff','avg_pest_risk_pct']].round(2))
print('\n2. Highest average yield season:', season['avg_yield_t_ha'].idxmax())
print('3. Highest average profit season:', season['avg_profit_inr'].idxmax())
print('4. Highest water-efficiency season:', season['avg_water_eff'].idxmax())
print('5. Lowest disease/pest-risk season:', season['avg_pest_risk_pct'].idxmin())
print('\n6. Highest-yield crop:', crop['avg_yield_t_ha'].idxmax(), round(crop.iloc[0]['avg_yield_t_ha'],2), 't/ha')
print('7. Highest-profit crop:', crop['avg_profit_inr'].idxmax(), round(crop.sort_values('avg_profit_inr', ascending=False).iloc[0]['avg_profit_inr'],2), 'INR')
print('\n8. Irrigation comparison:')
print(irrig.round(2))
print('\n9. Strongest relationships with yield:')
print(yield_corr.head(5).round(3))
print('\n10. Strongest relationships with profit:')
print(profit_corr.head(5).round(3))
print('\n11. Profitability rate by season (% farms with profit > 0):')
print(profit_rate.round(1))

print('\nOutputs saved to', OUTDIR)

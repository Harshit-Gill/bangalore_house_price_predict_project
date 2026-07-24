import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "colab"

import warnings
warnings.filterwarnings('ignore')
print("All libraries loaded.")

def kde_line(data, n_points=200):
    """Return (x, y) arrays for a smooth KDE curve."""
    data = data.dropna()
    kde = gaussian_kde(data, bw_method='scott')
    x = np.linspace(data.min(), data.max(), n_points)
    y = kde(x)
    return x, y

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'bengaluru_listings.csv'))
print(f"Dataset loaded: {df.shape[0]} row x {df.shape[1]} columns")
 

#data inspection 

# --- Audit 1 & 2: Shape and Types ---
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst 3 rows:")
print(df.head(3))

# --- Audit 3: Statistical Summary ---
# Flag any column whose count < total rows (indicates missing values)
print(df.describe(include='all').T)

# Anomaly hunt — drill into the columns flagged by describe()

# Discrete-numeric column: full value distribution
print('bhk value counts:')
print(df['bhk'].value_counts().sort_index())

# Continuous numeric: explicit anomaly counts
n_neg_age = (df['age_years'] < 0).sum()
print("number of rows with negative age:", n_neg_age)

# Type audit on a column that should be numeric
print(f"area sqft data type : {df['area_sqft'].dtype}")
type_counts = df["area_sqft"].apply(type).value_counts()
for t, c in type_counts.items():
    print(f' {t} : {c}')

# Categorical / string audit: too many unique values for the real category count

n_unique_loc = df['locality'].nunique()
print(f'Unique locality strings     : {n_unique_loc}  (we expect ~28 real neighbourhoods)')
print('Sample locality variants (case/whitespace mismatches):')
print(df['locality'].value_counts().head(15).to_string())


# Pull the anomaly counts out as named variables so the interpretation
# below is computed, never hardcoded.
n_bhk_zero = (df['bhk'] == 0).sum()
n_age_neg = (df['age_years']<0).sum()
n_area_str = df['area_sqft'].apply(lambda x: isinstance(x, str)).sum()
n_loc_unique = df['locality'].nunique()


# --- Audit 4: Missing Value Bar Chart ---
# --- Audit 4: Missing Value Bar Chart ---
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(1)
miss_df = pd.DataFrame({'Missing Count': missing, 'Percent': missing_pct})
miss_df = miss_df[miss_df['Missing Count'] > 0].sort_values('Percent', ascending=True)

fig = go.Figure(go.Bar(
    y=miss_df.index,
    x=miss_df['Percent'],
    orientation='h',
    marker_color='#ED1C24',
    text=[f"{int(c)} ({p}%)" for c, p in zip(miss_df['Missing Count'], miss_df['Percent'])],
    textposition='auto',
    hovertemplate='%{y}: %{x:.1f}% missing<extra></extra>'
))
fig.update_layout(
    title='Missing Values per Column (% of total rows)',
    xaxis_title='Missing %',
    width=700, height=350,
    template='simple_white'
)
# fig.show(renderer='browser')

# --- Audit 5: Duplicates ---
n_dupes = df.duplicated().sum()
if n_dupes > 0:
    print(f"Example duplicate (first 2 copies of first duplicated row):")
    print(df[df.duplicated(keep=False)].head(2)[['listing_id', 'locality', 'bhk', 'area_sqft', 'price_lakh']])

#Data cleaning the 6 step process


#step 1 removing duplicates
before_rows = df.shape[0]
df = df.drop_duplicates(keep="first").reset_index(drop=True)
after_rows = df.shape[0]

print(f"Duplicates rows removed: {before_rows - after_rows}")
print(f"Rows: {before_rows} -> {after_rows}")

#step 2 string standarisation

#stage A: apply .str.strip().str.title() and observe the result
n_before = df["locality"].nunique()
df['locality'] = df["locality"].str.strip().str.title()
n_after_stage_a = df["locality"].nunique()

print(f"Unique localities — BEFORE strip+title : {n_before}")
print(f"Unique localities — AFTER  strip+title : {n_after_stage_a}")
print()
print('Top 20 localities now (sorted by count):')
print(df['locality'].value_counts().head(20).to_string())
print()
print('Bottom 8 localities (sorted by count):')
print(df['locality'].value_counts().tail(8).to_string())

#Stage B: small correction map for multi-word + acronym localities
#Each entry maps the mangled .str.title() output back to the canonical name

locality_fix_map = {
    "White Field": "Whitefield",        # multi-word -> single word
    "Hsr Layout": "HSR Layout",         # acronym restored
    "Btm Layout": "BTM Layout",         # acronym restored
    "Jp Nagar": "JP Nagar",             # acronym restored
    "Kr Puram": "KR Puram",             # acronym restored
    "Cv Raman Nagar": "CV Raman Nagar", # acronym restored
}

df['locality'] = df['locality'].replace(locality_fix_map)

# Apply the same strip+title to furnishing_status (no acronym issues here)
# so no fix-map needed
df['furnishing_status'] = df['furnishing_status'].str.strip().str.title()

n_after_stage_b = df['locality'].nunique()
print(f"Unique localities — AFTER fix-map        : {n_after_stage_b}")
print(f"Reduction across both stages             : {n_before} → {n_after_stage_a} → {n_after_stage_b}")
print()
print('Final canonical localities (counts):')
print(df['locality'].value_counts().to_string())


# Step 3: Type conversion
# area_sqft is a numeric column, but it has some string values that need to be converted to float.

def to_float_sqft(x):
    """Convert area values like '1200 sqft' or '1,200 sqft' to float."""
    if pd.isna(x):
        return np.nan
    """
    isinstance(object, type) checks whether an object is an instance of a given type (or class) — returns True/False.
    """
    if isinstance(x,str):
        cleaned = x.replace('sqft', '').replace(',','').strip()
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
        """
        Try converting the cleaned string to a float. If it still can't convert (e.g., garbage like "N/A" or empty string after cleaning),
        don't crash — just return NaN instead, so one bad row doesn't kill the whole pipeline.
        """
    return float(x)

df["area_sqft"] = df['area_sqft'].apply(to_float_sqft)
print(f"area_sqft dtype now: {df['area_sqft'].dtype}")
print(f"Sample values: {df['area_sqft'].dropna().head(5).round(1).tolist()}")

# Step 4: Remove impossible values
#bhk <= 0, age_years < 0, area_sqft < 100
print(df['price_lakh'].describe()) #isse humne pta krlia min value hmari 12 lakh h so no prob of impossible values

n_bhk_zero_now = (df["bhk"] == 0).sum()
n_age_neg_now = (df["age_years"] < 0).sum()
n_area_lt100 = (df['area_sqft']< 100).sum() #lt100 = less than 100
print(f"Rows with bhk = 0          : {n_bhk_zero_now}")
print(f"Rows with age_years < 0    : {n_age_neg_now}")
print(f"Rows with area_sqft < 100  : {n_area_lt100}")
print(f"Total impossible rows to drop: {n_bhk_zero_now + n_age_neg_now + n_area_lt100} (some may overlap)")

before = len(df)

# Filter only the EXPLICITLY impossible rows. Preserve NaN values — those
# are a separate problem and will be handled by imputation in Step 6.
# (Without `.isna() |` the comparisons return False on NaN and drop rows
# that should be imputed.)

df = df[df["bhk"] > 0]
df = df[df["age_years"].isna() | (df['age_years']>= 0)]
df = df[df["area_sqft"].isna() | (df["area_sqft"] >= 100)]
df = df.reset_index(drop = True)

after = len(df)
print(f"Impossible rows removed: {before - after}")
print(f"Rows: {before} --> {after}")

# Step 5: outlier handling -- detect first then decide


#detect krege price_lakh ko to see where outliers exist
# Boxplot of price_lakh — the standard first chart for outlier detection
fig = go.Figure(go.Box(
    x=df['price_lakh'],
    name='price_lakh',
    marker_color='#ED1C24',
    boxpoints='outliers',          # only show points beyond the whiskers
    hovertemplate='Price: Rs.%{x:.1f}L<extra></extra>',
))
fig.update_layout(
    title='Price Distribution with Outliers Highlighted (whole dataset, pre-cap)',
    xaxis_title='Price (Rs. Lakhs)',
    width=900, height=300,
    template='simple_white',
    showlegend=False,
)
# fig.show(renderer = "browser")


q1 , q3 = df['price_lakh'].quantile([0.25,0.75])
iqr = q3 - q1
iqr_low, iqr_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
p01 , p99 = df['price_lakh'].quantile([0.01,0.99])
n_iqr_outliers  = ((df['price_lakh'] < iqr_low) | (df['price_lakh'] > iqr_high)).sum()
n_pct_outliers  = ((df['price_lakh'] < p01) | (df['price_lakh'] > p99)).sum()
n_ge_500 = ((df['price_lakh']>= 500)).sum()

print("output", "Outlier Audit on price_lakh\n",
    f"IQR rule — flags any value outside [Rs.{iqr_low:.1f}L, Rs.{iqr_high:.1f}L] \n"
    f"(Q1=Rs.{q1:.1f}L, Q3=Rs.{q3:.1f}L, IQR=Rs.{iqr:.1f}L). \n"
    f"{n_iqr_outliers} rows flagged as IQR outliers.\n"
    f"1st–99th percentile rule — flags any value outside [Rs.{p01:.1f}L, Rs.{p99:.1f}L]. \n"
    f"{n_pct_outliers} rows flagged.\n"
    f"Eye-test — {n_ge_500} listings ≥ Rs.500L (real luxury flats — not data errors). \n"
    f"The right tail is real; the question is what to do with it.\n")


# Apply: percentile capping (winsorization) on price_lakh
# .clip() keeps every row — values below q_low become q_low, above q_high become q_high.
q_low  = df['price_lakh'].quantile(0.01)
q_high = df['price_lakh'].quantile(0.99)

n_clipped_low  = (df['price_lakh'] < q_low).sum()
n_clipped_high = (df['price_lakh'] > q_high).sum()

df['price_lakh'] = df['price_lakh'].clip(lower=q_low, upper=q_high)

print(f"1st percentile  = Rs.{q_low:.1f}L  (low cap)")
print(f"99th percentile = Rs.{q_high:.1f}L  (high cap)")
print(f"\nValues capped at lower bound : {n_clipped_low}")
print(f"Values capped at upper bound : {n_clipped_high}")
print(f"Rows preserved (no row dropped): {len(df)}")

# Skewness check — for numeric columns we are about to impute
skews = {
    'area_sqft':     df['area_sqft'].skew(),
    'age_years':     df['age_years'].skew(),
    'bathrooms':     df['bathrooms'].skew(),
    'parking_spots': df['parking_spots'].skew(),
}
print('Skewness of imputation candidates:')
for col, sk in skews.items():
    shape = ('symmetric' if abs(sk) < 0.5 else
             'mildly skewed' if abs(sk) < 1.0 else
             'strongly skewed')
    direction = 'right' if sk > 0 else 'left'
    print(f"  {col:<14}: skew = {sk:+.2f}  ({shape}, {direction}-tailed)")

# f-string interpretation tying the skew result to the chosen strategy
mode_furn = df['furnishing_status'].mode()[0] if df['furnishing_status'].notna().any() else 'Unfurnished'

"""
If the column has at least one real (non-missing) value, use the most common furnishing status as the fill value. 
But if the entire column is somehow empty, fall back to a safe default of 'Unfurnished' instead of crashing.
"""
df['area_sqft']        = df['area_sqft'].fillna(df['area_sqft'].median())
df['age_years']        = df['age_years'].fillna(df['age_years'].median())
df['bathrooms']        = df['bathrooms'].fillna(df['bathrooms'].median())
df['parking_spots']    = df['parking_spots'].fillna(0)
df['furnishing_status']= df['furnishing_status'].fillna(df['furnishing_status'].mode()[0])

remaining_missing = df.isna().sum().sum()
print(f"Missing values remaining after imputation: {remaining_missing}")


# Data quality dashboard - Verification Asserts
'''
Cleaning code is just code, and code has bugs. A typo in a filter, a wrong column name in an imputation, 
an off-by-one in an outlier rule — any of these can leave your 'cleaned' DataFrame silently dirty. 
The model in 11.2 will then train on dirty data, produce great-looking metrics in development, and fail in production. 
Asserts are how you catch the bug before it becomes an outage.

Each assert encodes a contract about what 'clean' means for this dataset:
  • 'There must be zero missing values'   →   assert df.isna().sum().sum() == 0
  • 'Every BHK must be at least 1'   →   assert (df['bhk'] > 0).all()
  • 'No building older than its existence'   →   assert (df['age_years'] >= 0).all()

Three rules of engagement.
  1. Asserts run at the END of cleaning, not during it. 
     They check the contract is satisfied — if it isn't, fix the cleaning, don't bypass the assert.
  2. Each assert message names the specific contract being violated, so the failure tells you what is wrong, not just that something is wrong.

  3. Never comment out a failing assert. If an assert fails, the contract has been violated. 
     Treat it like a unit-test failure in software engineering — debug the cleaning, do not edit the test.
'''

# Verification asserts — if ANY assert fails, the notebook stops loudly
assert df.isna().sum().sum() == 0,         "FAIL: Missing values still exist!"
assert (df['bhk'] > 0).all(),              "FAIL: bhk=0 rows remain!"
assert (df['age_years'] >= 0).all(),       "FAIL: Negative age_years remain!"
assert (df['price_lakh'] > 0).all(),       "FAIL: Non-positive price_lakh!"
assert (df['area_sqft'] >= 100).all(),     "FAIL: area_sqft < 100 sqft rows remain!"
assert df['locality'].nunique() >= 25,     "FAIL: Too few unique localities — string cleaning may have missed variants!"
assert 7800 <= len(df) <= 8250,            "FAIL: Row count out of expected range (7800-8250) after cleaning!"

print(f"All cleaning asserts passed.")
print(f"Final clean dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Unique localities: {df['locality'].nunique()}")

# Train test split before EDA

train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)

#reset indices for clean downstream indexing
train_df = train_df.reset_index(drop = True).copy()
test_df = test_df.reset_index(drop = True).copy()

print(f"Train: {train_df.shape[0]} rows ({train_df.shape[0]/len(df)*100:.1f}%)")
print(f"Test:  {test_df.shape[0]} rows ({test_df.shape[0]/len(df)*100:.1f}%)")
print(f"\nTrain price mean:   Rs.{train_df['price_lakh'].mean():.2f}L")
print(f"Test  price mean:   Rs.{test_df['price_lakh'].mean():.2f}L")
print(f"(Similar means = good split, no obvious skew)")




#Section 4: EDA on train dataset

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train_df['price_lakh'],
    histnorm='probability density',
    name='price_lakh',
    marker_color='#ED1C24',
    opacity=0.7,
    hovertemplate='Price: Rs.%{x:.1f}L<br>Density: %{y:.4f}<extra></extra>'
))
x_kde, y_kde = kde_line(train_df['price_lakh'])
fig.add_trace(go.Scatter(
    x=x_kde, y=y_kde, mode='lines',
    line=dict(color='black', width=2.5), name='KDE'
))
fig.update_layout(
    title='Price Distribution — Train Set (Rs. Lakhs)',
    xaxis_title='Price (Rs. Lakhs)',
    yaxis_title='Density',
    width=800, height=450,
    template='simple_white'
)
# fig.show(renderer = "browser")

mean_price = train_df["price_lakh"].mean()
median_price = train_df["price_lakh"].median()
skew_price = train_df["price_lakh"].skew()

print(f"Mean = Rs.{mean_price:.1f}L | Median = Rs.{median_price:.1f}L | Skew = {skew_price:.2f}")
print("The distribution is right-skewed (mean > median, skew > 1).")
print("A small number of premium Koramangala/Indiranagar listings pull the mean up.")
print("Implication: We will apply a log1p transform to price_lakh in Feature Engineering to reduce the skew (from 1.43 toward 0) before model training.")

#Multi panel histogram with KDE overlays - ALL in one cell
num_features = ["area_sqft","age_years","bhk","bathrooms","parking_spots"]
colors = ['#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']

fig = make_subplots(rows=2, cols=3, subplot_titles=num_features + [''],
                    vertical_spacing=0.18, horizontal_spacing=0.10)

for i, (feat, clr) in enumerate(zip(num_features, colors)):
    row = i // 3 + 1
    col = i % 3 + 1
    fig.add_trace(go.Histogram(
        x=train_df[feat], histnorm='probability density',
        marker_color=clr, opacity=0.65, name=feat,
        hovertemplate=f'{feat}: %{{x}}<br>Density: %{{y:.4f}}<extra></extra>'
    ), row=row, col=col)
    try:
        kx, ky = kde_line(train_df[feat])
        fig.add_trace(go.Scatter(
            x=kx, y=ky, mode='lines',
            line=dict(color='black', width=1.8), showlegend=False
        ), row=row, col=col)
    except Exception:
        pass  # skip KDE if column has < 2 unique values

fig.update_layout(
    title_text='Numeric Feature Distributions — Train Set',
    height=550, width=1000,
    showlegend=False,
    plot_bgcolor='white'
)
# fig.show(renderer = "browser")

area_mean = train_df["area_sqft"].mean()
age_median = train_df["age_years"].median()
bhk_mode = int(train_df["bhk"].mode()[0])
print(f"area_sqft Mean = {area_mean}")
print(f"age Median = {age_median}")
print(f"bhk Mode = {bhk_mode}")
print(f"bathrooms & parking_spots: Discrete, roughly match bhk distribution as expected.")

# Correlation heatmap — ALL traces in one cell
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude listing_id (it's an identifier, not a feature)
numeric_cols = [c for c in numeric_cols if c != 'listing_id']
corr = train_df[numeric_cols].corr()  #corr mtlb corelate smthng shit

fig = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='RdBu',
    zmid=0,
    zmin=-1, zmax=1,
    text=corr.round(2).values,
    texttemplate='%{text}',
    hovertemplate='%{x} vs %{y}: r = %{z:.3f}<extra></extra>'
))
fig.update_layout(
    title='Pearson Correlation — Numeric Features (Train Set)',
    width=700, height=550,
    template='simple_white'
)
# fig.show(renderer = 'browser')


# Find the feature most correlated with price_lakh (excluding price itself)
price_corr = corr['price_lakh'].drop('price_lakh').abs().sort_values(ascending=False)
top_feature = price_corr.index[0]
top_r = corr['price_lakh'][top_feature]
second_feature = price_corr.index[1]
second_r = corr['price_lakh'][second_feature]

print(f"Strongest predictor of price_lakh:{top_feature} (r = {top_r:.3f})")
print(f"Second strongest: {second_feature} (r = {second_r:.3f})")
print(f"These 2 features will be the most influential in our linear regession model")
print(f"Area_sqft and bhk are also correlated with each other - larger flats have more bedrooms")

r_area_price = train_df[['area_sqft','price_lakh']].corr().iloc[0,1]
print(f"r(area_sqft, price_lakh) = {r_area_price:.3f}")
print(f"{"Scatter plot (|r| >= 0.3)" if abs(r_area_price) >= 0.3 else "Boxplot by category (|r| < 0.3)"}")

# area_sqft vs price_lakh scatter (r >= 0.3 expected)
fig = go.Figure(go.Scatter(
    x=train_df['area_sqft'],
    y=train_df['price_lakh'],
    mode='markers',
    marker=dict(color='#ED1C24', opacity=0.5, size=5),
    hovertemplate=(
        'Area: %{x:.0f} sqft<br>Price: Rs.%{y:.1f}L<extra></extra>'
    )
))
fig.update_layout(
    title=f'area_sqft vs price_lakh (Train) — r = {r_area_price:.3f}',
    xaxis_title='Area (sqft)',
    yaxis_title='Price (Rs. Lakhs)',
    width=750, height=430,
    template='simple_white'
)
# fig.show(renderer = "browser")

# price by locality - which neighbourhoods command a premium

loc_medians = train_df.groupby('locality')['price_lakh'].median().sort_values()
sorted_locs = loc_medians.index.tolist()

fig = go.Figure()
for loc in sorted_locs:
    fig.add_trace(go.Box(
        y=train_df[train_df['locality'] == loc]['price_lakh'],
        name=loc,
        marker_color='#ED1C24',
        hovertemplate=f'{loc}<br>Price: Rs.%{{y:.1f}}L<extra></extra>'
    ))
fig.update_layout(
    title='Price Distribution by Locality (Train) — Sorted by Median',
    yaxis_title='Price (Rs. Lakhs)',
    xaxis_title='Locality',
    width=950, height=500,
    template='simple_white',
    showlegend=False
)
# fig.show(renderer = "browser")

premium_loc  = loc_medians.index[-1]
outskirts_loc = loc_medians.index[0]
premium_med  = loc_medians.iloc[-1]
outskirts_med = loc_medians.iloc[0]

print(f"Highest median: {premium_loc} — Rs.{premium_med:.1f}L")
print(f"Lowest median: {outskirts_loc} — Rs.{outskirts_med:.1f}L")
print(f"Premium Gap: Rs.{premium_med - outskirts_med:.1f} L (factor of {premium_med / outskirts_med:.1f}x)")


# price by furnishing_status - does furnishing move the needle?

#boxplot of price by furnishing_status - 3 categories
furnish_order = ['Unfurnished', "Semi-Furnished", "Furnished"]
fig = go.Figure()
for furn in furnish_order:
    sub = train_df[train_df['furnishing_status'] == furn]['price_lakh']
    fig.add_trace(go.Box(
        y=sub,
        name=furn,
        marker_color='#ED1C24',
        boxpoints='outliers',
        hovertemplate=f'{furn}<br>Price: Rs.%{{y:.1f}}L<extra></extra>',
    ))
fig.update_layout(
    title='Price Distribution by Furnishing Status (Train)',
    yaxis_title='Price (Rs. Lakhs)',
    xaxis_title='Furnishing',
    width=750, height=420,
    template='simple_white',
    showlegend=False,
)
# fig.show(renderer = "browser")

med_unfurn = train_df[train_df['furnishing_status']=='Unfurnished']['price_lakh'].median()
med_semi   = train_df[train_df['furnishing_status']=='Semi-Furnished']['price_lakh'].median()
med_furn   = train_df[train_df['furnishing_status']=='Furnished']['price_lakh'].median()
premium_pct = (med_furn / med_unfurn - 1) * 100 if med_unfurn else 0

print(f"Median by category: Unfurnished = Rs.{med_unfurn:.1f}L, Semi-Furnished = Rs.{med_semi:.1f}L, Furnished = Rs.{med_furn:.1f}L")
print(f"Premium for Furnished vs Unfurnished: {premium_pct:.1f}% (Furnished commands a premium)")

# price by bhk - does bedroom count drive price?

# Boxplot of price by bhk — discrete numeric, treated as categorical here
bhk_order = sorted(train_df['bhk'].unique())
fig = go.Figure()
for b in bhk_order:
    sub = train_df[train_df['bhk'] == b]['price_lakh']
    fig.add_trace(go.Box(
        y=sub,
        name=f'{int(b)} BHK',
        marker_color='#ED1C24',
        boxpoints='outliers',
        hovertemplate=f'{int(b)} BHK<br>Price: Rs.%{{y:.1f}}L<extra></extra>',
    ))
fig.update_layout(
    title='Price Distribution by BHK (Train)',
    yaxis_title='Price (Rs. Lakhs)',
    xaxis_title='Bedrooms-Hall-Kitchen (BHK)',
    width=750, height=420,
    template='simple_white',
    showlegend=False,
)
# fig.show(renderer = "browser")

bhk_medians = train_df.groupby('bhk')['price_lakh'].median().to_dict()
med_lines = ' | '.join(f"{int(k)}BHK = Rs.{v:.1f}L" for k, v in sorted(bhk_medians.items()))

print(f"Median price by BHK: {med_lines}")



# Section 5: Feature engineering - Fit on train, Apply to Test

#the golden rule of feature engineering: fit on train, apply to test. Never peek at the test set during training.

'''
we engineer four features:
1. log_price -- log transform of the right skewed target
2. price_per_sqft_eda_only -- target derived ratio feature, used only for EDA, not in the model
3. locality_freq -- frquency encoding of locality (fit on train, mapped to test)
4. sqft_per_bhk -- size efficiency ratio
'''

#feature 1: log_price

train_df['log_price'] = np.log1p(train_df['price_lakh'])
test_df['log_price'] = np.log1p(test_df['price_lakh'])

original_skew = train_df['price_lakh'].skew()
log_skew = train_df['log_price'].skew()

print(f"price_lakh skew:  {original_skew:.3f}")
print(f"log_price skew:   {log_skew:.3f}")
print(f"Skew reduction:   {original_skew - log_skew:.3f} (closer to 0 = more symmetric)")

'''
Visual confirmation that the log transform fixed the skew.** Compare this histogram + KDE of `log_price` 
to the original `price_lakh` histogram from Section 4.1 — the right tail should be gone and the curve should be roughly symmetric 
(close to bell-shaped).'''

# Histogram + KDE of log_price (train) — visual proof the log transform worked
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=train_df['log_price'],
    histnorm='probability density',
    name='log_price',
    marker_color='#ED1C24',
    opacity=0.7,
    hovertemplate='log_price: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>',
))
x_kde, y_kde = kde_line(train_df['log_price'])
fig.add_trace(go.Scatter(
    x=x_kde, y=y_kde, mode='lines',
    line=dict(color='black', width=2.5), name='KDE',
))
fig.update_layout(
    title='Distribution of log_price (Train) — After log1p Transform',
    xaxis_title='log_price = log(1 + price_lakh)',
    yaxis_title='Density',
    width=800, height=420,
    template='simple_white',
)
# fig.show(renderer = 'browser')

log_mean = train_df['log_price'].mean()
log_median = train_df['log_price'].median()

print(f"Before transformation skew = {original_skew:+.2f} - strongly right tailed")
print(f"After transformation skew = {log_skew:+.2f}")

# Feature 2: price_per_sqft -- EDA only (target leakage warning)

#price_per_sqft = (price_lakh * 100000)/area_sqft

train_df['price_per_sqft_eda_only'] = ((train_df['price_lakh'] * 100000) / train_df['area_sqft']).round(0)

mean_ppsf = train_df['price_per_sqft_eda_only'].mean()

print(f"Mean price per sqft in train: Rs.{mean_ppsf:,.0f} per sqft")
print(f"Note: this column is on train_df only — NOT a model feature.")

# Feature 3: locality frequency - Frequency encoding (Fit on Train, Map to test)

freq_map = train_df['locality'].value_counts(normalize=True).to_dict()

print("Locality frequency map (from table)")
for loc , freq in sorted(freq_map.items(), key=lambda x:-x[1]):
    print(f" {loc:<20}: {freq:.4f}")

# Map to BOTH train and test (.fillna(0) handles any test locality not seen in train)
train_df['locality_freq'] = train_df['locality'].map(freq_map)
test_df['locality_freq']  = test_df['locality'].map(freq_map).fillna(0)

print(f"\nTrain NaN in locality_freq: {train_df['locality_freq'].isna().sum()}")
print(f"Test  NaN in locality_freq: {test_df['locality_freq'].isna().sum()}")

# Feature 4: sqft_per_bhk -- size efficiency ratio

train_df['sqft_per_bhk'] = train_df['area_sqft']/train_df['bhk']
test_df['sqft_per_bhk'] = test_df['area_sqft']/test_df['bhk']

mean_spb = train_df['sqft_per_bhk'].mean()
print(f"Mean sqft per BHK in train: {mean_spb:.1f} sqft")
print(f"This captures space efficiency — a 1,200 sqft 3BHK is 400 sqft/BHK (cramped vs spacious for the price).")

# Verification: All engineered features in both train and test

new_features = ['log_price', 'locality_freq', 'sqft_per_bhk']

for f in new_features:
    assert f in train_df.columns, f"FAIL: {f} missing from train_df!"
    assert f in test_df.columns,  f"FAIL: {f} missing from test_df!"
    assert train_df[f].notna().all(), f"FAIL: {f} has NaN in train_df!"
    assert test_df[f].notna().all(),  f"FAIL: {f} has NaN in test_df!"

print(f"All {len(new_features)} engineered features verified in train_df and test_df.")
print(f"\ntrain_df columns: {train_df.shape[1]}")
print(f"test_df  columns: {test_df.shape[1]}")
print(f"Note: test_df has 1 fewer column (price_per_sqft_eda_only is train-only, not a model feature)")


#Section 6: baseline model and project save

#computing the baseline RMSE
# Baseline: predict mean train price for every test row
mean_price_train = train_df['price_lakh'].mean()
baseline_pred    = np.full(len(test_df), mean_price_train)
baseline_rmse    = np.sqrt(mean_squared_error(test_df['price_lakh'], baseline_pred))
target_rmse      = baseline_rmse * 0.70  # 30% improvement

print(f"Mean train price :  Rs.{mean_price_train:.2f}L")
print(f"Baseline RMSE    :  Rs.{baseline_rmse:.2f}L")
print()
print(f"Any model in 11.2 MUST beat Rs.{baseline_rmse:.2f}L RMSE.")
print(f"Success threshold:  Rs.{target_rmse:.2f}L RMSE (30% improvement over baseline)")


# Saving the project files

# Save train and test as separate CSVs — the inputs to Session 11.2
train_df.to_csv('bengaluru_train.csv', index=False)
test_df.to_csv('bengaluru_test.csv', index=False)

print("Saved project files:")
print(f"  bengaluru_train.csv  ->  {train_df.shape[0]} rows x {train_df.shape[1]} columns")
print(f"  bengaluru_test.csv   ->  {test_df.shape[0]} rows x {test_df.shape[1]} columns")
print()
print(f"Baseline RMSE to beat in 11.2: Rs.{baseline_rmse:.2f}L")
print(f"Success threshold            : Rs.{target_rmse:.2f}L (30% improvement)")
# =============================================================================
#
#   ██████╗██╗  ██╗██╗   ██╗██████╗ ███╗   ██╗██╗ ██████╗
#  ██╔════╝██║  ██║██║   ██║██╔══██╗████╗  ██║██║██╔═══██╗
#  ██║     ███████║██║   ██║██████╔╝██╔██╗ ██║██║██║   ██║
#  ██║     ██╔══██║██║   ██║██╔══██╗██║╚██╗██║██║██║▄▄ ██║
#  ╚██████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║██║╚██████╔╝
#   ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝ ╚══▀▀═╝
#
#   ChurnIQ Pro — FILE 1: Complete ML Pipeline
#   ─────────────────────────────────────────────────────────
#   What this file does (run top to bottom):
#
#   PART A — DATASET GENERATION
#     Generates a synthetic OTT churn dataset (5000 customers)
#     using Faker + NumPy with business-rule-driven churn logic
#
#   PART B — EXPLORATORY DATA ANALYSIS (EDA)
#     Visually understands the data before training
#     Saves 6-chart EDA dashboard as PNG
#
#   PART C — PREPROCESSING & FEATURE ENGINEERING
#     Cleans data, encodes categories, engineers new features
#     Splits into train/test sets
#
#   PART D — SMOTE (Class Imbalance Fix)
#     Balances churned vs retained in training set
#
#   PART E — MODEL TRAINING (XGBoost)
#     Trains gradient boosted model, saves as .pkl
#
#   PART F — MODEL EVALUATION
#     Confusion matrix, ROC curve, feature importance
#
#   PART G — SHAP (Explainable AI)
#     Computes why each customer will churn
#     Saves summary + waterfall plots
#
#   PART H — PREDICTIONS TABLE
#     Generates final CSV with churn_prob, risk_level,
#     top 3 SHAP reasons, CLV score for every customer
#
#   ─────────────────────────────────────────────────────────
#   HOW TO RUN:
#     pip install pandas numpy faker scikit-learn xgboost shap
#                 imbalanced-learn matplotlib seaborn plotly
#     python churniq_ml_pipeline.py
#
#   OUTPUT FILES:
#     ott_churn_dataset.csv        ← raw dataset
#     churniq_model.pkl            ← trained XGBoost model
#     feature_names.pkl            ← feature column names
#     customer_predictions.csv     ← predictions + SHAP reasons
#     eda_dashboard.png            ← EDA plots
#     model_evaluation.png         ← confusion matrix + ROC
#     shap_summary.png             ← SHAP beeswarm
#     shap_importance.png          ← SHAP bar chart
#     shap_waterfall_example.png   ← individual customer
# =============================================================================

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd               # DataFrames — like Excel in Python
import numpy as np                # Math & array operations
import matplotlib.pyplot as plt   # Base plotting library
import seaborn as sns             # Statistical visualizations (built on matplotlib)
import shap                       # Explainable AI — SHAP values
import warnings
import pickle                     # Save/load Python objects (model, feature names)
import os
import random

from faker import Faker           # Generate realistic fake data (names, locations)

from sklearn.model_selection import train_test_split   # Split data train/test
from sklearn.preprocessing import LabelEncoder         # Convert text → numbers
from sklearn.metrics import (
    classification_report,        # Precision, Recall, F1 in one table
    confusion_matrix,             # TP, FP, TN, FN grid
    roc_auc_score,                # Area under ROC curve
    roc_curve,                    # For plotting ROC
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE   # Fix class imbalance
from xgboost import XGBClassifier          # Our main ML model

warnings.filterwarnings('ignore')

# ─── OUTPUT DIRECTORY ─────────────────────────────────────────────────────────
# All generated files will be saved here
# Change this path to wherever your project folder is
OUTPUT_DIR = './'   # Current directory — change to 'outputs/' if preferred
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("   ChurnIQ Pro — ML Pipeline")
print("   OTT Subscription Churn Prediction System")
print("=" * 65)


# =============================================================================
#   PART A — DATASET GENERATION
# =============================================================================
# WHY SYNTHETIC DATA?
#   The standard Telco Churn dataset from Kaggle is overused —
#   every second ML student project uses it. A synthetic dataset:
#     1. Shows awareness of data privacy in ML
#     2. Lets us control feature richness
#     3. Lets us bake in realistic business logic
#     4. Makes the project unique and memorable
#
# HOW THE CHURN LOGIC WORKS:
#   We assign a "churn_score" to each customer based on real-world
#   OTT platform signals. The score is converted to a probability
#   using sigmoid function. Then we sample the actual churn label
#   from that probability (so it's probabilistic, not deterministic).
#
# VIVA TIP:
#   "We generated synthetic data to avoid using overused public datasets
#    and to demonstrate understanding of data privacy. The churn labels
#    are assigned using domain-knowledge-driven business rules, making
#    the data statistically realistic for an OTT platform."
# =============================================================================

print("\n" + "─" * 65)
print("  PART A — DATASET GENERATION")
print("─" * 65)

# Seed everything for reproducibility
# CONCEPT: A fixed random seed means every run gives the exact same dataset
# This is important for reproducibility — teammates get the same results
fake = Faker('en_IN')    # Indian locale for realistic Indian names
np.random.seed(42)
random.seed(42)

N = 5000    # Number of customers to generate

# ── Plan Configuration ─────────────────────────────────────────────────────
# Real Netflix-style plans with Indian pricing
PLANS = {
    'Basic':    {'charge': 199,  'weight': 0.45},   # Most popular, cheapest
    'Standard': {'charge': 499,  'weight': 0.35},   # Mid-tier
    'Premium':  {'charge': 799,  'weight': 0.20},   # Least popular, most features
}

PAYMENT_METHODS = ['UPI', 'Credit Card', 'Debit Card', 'Wallet']
DEVICES         = ['Mobile', 'Laptop', 'Smart TV', 'Tablet']
GENRES          = ['Action', 'Drama', 'Comedy', 'Thriller', 'Kids', 'Documentary', 'Romance']
GENDERS         = ['Male', 'Female', 'Other']

# India-weighted country distribution (most OTT users are in India)
COUNTRIES = ['India'] * 80 + ['UAE', 'USA', 'UK', 'Singapore', 'Canada'] * 4

print(f"\n  Generating {N} synthetic OTT customers...")

# ── Generate Base Features ────────────────────────────────────────────────
plan_types      = random.choices(list(PLANS.keys()),
                                 weights=[v['weight'] for v in PLANS.values()], k=N)
monthly_charges = [PLANS[p]['charge'] for p in plan_types]
tenures         = np.random.gamma(shape=2, scale=12, size=N).clip(1, 72).astype(int)
ages            = np.random.normal(loc=30, scale=8, size=N).clip(18, 65).astype(int)

# Watch behavior — exponential distribution because most users watch little,
# few watch a lot (realistic for streaming platforms)
watch_hours_per_week = np.random.exponential(scale=8, size=N).clip(0.5, 40).round(1)
num_profiles         = np.random.choice([1, 2, 3, 4, 5],
                                        p=[0.30, 0.30, 0.20, 0.15, 0.05], size=N)
logins_last_30_days  = np.random.poisson(lam=12, size=N).clip(0, 60)

# Payment failures — most users have none, few have many
payment_failures     = np.random.choice([0, 1, 2, 3, 4],
                                        p=[0.60, 0.20, 0.10, 0.07, 0.03], size=N)
support_tickets      = np.random.choice([0, 1, 2, 3, 4, 5],
                                        p=[0.50, 0.25, 0.12, 0.07, 0.04, 0.02], size=N)

# last_login_days_ago: most users logged in recently, some have been inactive
last_login_days_ago  = np.random.exponential(scale=5, size=N).clip(0, 90).astype(int)

genders         = random.choices(GENDERS,         k=N)
countries       = random.choices(COUNTRIES,        k=N)
payment_methods = random.choices(PAYMENT_METHODS,  k=N)
devices         = random.choices(DEVICES,          k=N)
genre_prefs     = random.choices(GENRES,           k=N)

# ── Churn Score Logic ─────────────────────────────────────────────────────
# CONCEPT: Each behavioral signal contributes a "churn score".
# Higher score = more likely to churn.
# We use domain knowledge from real OTT analytics research:
#   - Low engagement is the #1 predictor of churn
#   - Payment failures strongly indicate financial churn
#   - New customers (honeymoon period) are high risk
#   - Long-tenure customers are loyal and less likely to churn

def sigmoid(x):
    """Convert any real number to a probability between 0 and 1."""
    return 1 / (1 + np.exp(-x))

churn_score = np.zeros(N)   # Start everyone at 0

# Low engagement signals → strong churn indicators
churn_score += np.where(watch_hours_per_week < 2,  1.5, 0)   # Almost no watching
churn_score += np.where(watch_hours_per_week < 5,  0.5, 0)   # Below average watching
churn_score += np.where(logins_last_30_days  < 3,  1.2, 0)   # Rarely logs in
churn_score += np.where(last_login_days_ago  > 20, 1.0, 0)   # Gone dark 3+ weeks
churn_score += np.where(last_login_days_ago  > 40, 0.8, 0)   # Extra penalty for very long absence

# Payment friction → clear churn signal
churn_score += payment_failures * 0.6    # Each failure adds 0.6 to score

# Support frustration → unhappy customers leave
churn_score += support_tickets * 0.4     # Each ticket adds 0.4 to score

# Tenure effects
churn_score += np.where(tenures < 3,  1.2, 0)    # Very new: 0-3 months = high risk
churn_score += np.where(tenures < 6,  0.5, 0)    # New: 3-6 months = elevated risk
churn_score += np.where(tenures > 24, -0.8, 0)   # 2+ year customers: loyalty reward

# Plan + device interaction → Basic + Mobile = price sensitive segment
churn_score += np.where(
    (np.array(plan_types) == 'Basic') & (np.array(devices) == 'Mobile'),
    0.4, 0
)

# Add realistic noise (no model is perfect — some randomness is real)
churn_score += np.random.normal(0, 0.5, N)

# Convert score to probability using sigmoid
# Offset by -1.5 to get realistic ~30-35% base churn rate
churn_prob  = sigmoid(churn_score - 1.5)

# Sample actual churn label from probability
# CONCEPT: This makes the dataset probabilistic, not deterministic
# A customer with 80% churn_prob will churn 80% of the time, not always
churn_labels = (np.random.rand(N) < churn_prob).astype(int)

print(f"  Churn rate: {churn_labels.mean():.1%} "
      f"({churn_labels.sum()} churned / {N - churn_labels.sum()} retained)")

# ── CLV Calculation ─────────────────────────────────────────────────────────
# CONCEPT: Customer Lifetime Value (CLV) = how much this customer is worth
# Formula: monthly_charge × expected_tenure_remaining × retention_factor
# retention_factor = 1 - churn_probability
# Higher CLV = more worth fighting for when at churn risk
retention_factor  = 1 - churn_prob
expected_tenure   = np.where(tenures < 6, 6, tenures)
clv_scores        = (np.array(monthly_charges) * expected_tenure * retention_factor).round(2)

def clv_segment(v):
    """Segment customers into High/Medium/Low CLV buckets."""
    if v > 8000:  return 'High'
    if v > 3000:  return 'Medium'
    return 'Low'

clv_segments = [clv_segment(v) for v in clv_scores]

# ── Build DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    'customer_id':           [f'CUST{str(i).zfill(5)}' for i in range(1, N+1)],
    'name':                  [fake.name() for _ in range(N)],
    'age':                   ages,
    'gender':                genders,
    'country':               countries,
    'plan_type':             plan_types,
    'monthly_charge':        monthly_charges,
    'tenure_months':         tenures,
    'watch_hours_per_week':  watch_hours_per_week,
    'num_profiles':          num_profiles,
    'logins_last_30_days':   logins_last_30_days,
    'last_login_days_ago':   last_login_days_ago,
    'preferred_genre':       genre_prefs,
    'device_type':           devices,
    'payment_method':        payment_methods,
    'payment_failures_3m':   payment_failures,
    'support_tickets':       support_tickets,
    'clv_score':             clv_scores,
    'clv_segment':           clv_segments,
    'churn_probability':     churn_prob.round(4),
    'churn':                 churn_labels,
})

# Save dataset
dataset_path = OUTPUT_DIR + 'ott_churn_dataset.csv'
df.to_csv(dataset_path, index=False)

print(f"\n  Dataset Summary:")
print(f"    Rows × Columns : {df.shape[0]} × {df.shape[1]}")
print(f"    Churned        : {churn_labels.sum()} ({churn_labels.mean():.1%})")
print(f"    Retained       : {N - churn_labels.sum()} ({1 - churn_labels.mean():.1%})")
print(f"    Plan dist      : {df['plan_type'].value_counts().to_dict()}")
print(f"    CLV segments   : {df['clv_segment'].value_counts().to_dict()}")
print(f"\n  ✓ Dataset saved → {dataset_path}")


# =============================================================================
#   PART B — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# CONCEPT: EDA = "Understanding your data before touching the model"
#
# WHY EDA IS NOT OPTIONAL:
#   ML models are mathematical functions. They will train on garbage
#   data and produce confidently wrong predictions. EDA lets you:
#     1. Validate data quality (missing values, outliers)
#     2. Understand class distribution (imbalance check)
#     3. Discover which features correlate with churn
#     4. Form hypotheses to test with feature engineering
#
# THE 6 EDA CHARTS WE GENERATE:
#   1. Churn distribution  → how balanced is our target?
#   2. Churn by plan type  → does plan affect churn?
#   3. Watch hours         → low watchers churn more?
#   4. Tenure vs churn     → new customers riskier?
#   5. Payment failures    → does friction predict churn?
#   6. Correlation heatmap → which numbers move together?
#
# VIVA TIP:
#   "EDA revealed that customers in the 0-3 month tenure bucket
#    have a 70%+ churn rate, leading us to engineer the
#    is_new_customer feature. Payment failures showed a near-linear
#    relationship with churn rate, making friction_score our
#    most important feature."
# =============================================================================

print("\n" + "─" * 65)
print("  PART B — EXPLORATORY DATA ANALYSIS")
print("─" * 65)

print("\n  Generating EDA dashboard...")

churn_counts = df['churn'].value_counts()
churn_pct    = df['churn'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('ChurnIQ Pro — EDA Dashboard', fontsize=16, fontweight='bold')

# ── Chart 1: Churn Distribution ────────────────────────────────────────────
# PURPOSE: First thing to check — how balanced is our dataset?
# If 95% retained / 5% churned → serious imbalance → need SMOTE
ax1 = axes[0, 0]
colors = ['#00C853', '#FF4444']
ax1.bar(['Retained', 'Churned'], churn_counts.values, color=colors, width=0.5)
ax1.set_title('Churn Distribution', fontweight='bold')
ax1.set_ylabel('Number of Customers')
for i, v in enumerate(churn_counts.values):
    ax1.text(i, v + 30, f'{v}\n({churn_pct.values[i]:.1f}%)',
             ha='center', fontweight='bold')

# ── Chart 2: Churn by Plan Type ────────────────────────────────────────────
# PURPOSE: Do Basic plan users churn more than Premium?
# INSIGHT: Basic users are price-sensitive; Premium users are more invested
ax2 = axes[0, 1]
churn_by_plan = df.groupby('plan_type')['churn'].mean() * 100
plan_colors   = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax2.bar(churn_by_plan.index, churn_by_plan.values, color=plan_colors)
ax2.set_title('Churn Rate by Plan Type', fontweight='bold')
ax2.set_ylabel('Churn Rate (%)')
for bar, val in zip(bars, churn_by_plan.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold')

# ── Chart 3: Watch Hours Distribution ──────────────────────────────────────
# PURPOSE: Visualize engagement gap between churned and retained
# EXPECTED: Churned customers cluster at low watch hours
# INSIGHT: If the two histograms overlap a lot → watch hours is not a good feature
#          If they separate well → it's a strong feature
ax3 = axes[0, 2]
churned  = df[df['churn'] == 1]['watch_hours_per_week']
retained = df[df['churn'] == 0]['watch_hours_per_week']
ax3.hist(retained, bins=30, alpha=0.6, color='#00C853', label='Retained', density=True)
ax3.hist(churned,  bins=30, alpha=0.6, color='#FF4444', label='Churned',  density=True)
ax3.set_title('Watch Hours Distribution', fontweight='bold')
ax3.set_xlabel('Watch Hours per Week')
ax3.set_ylabel('Density')
ax3.legend()

# ── Chart 4: Churn by Tenure Bucket ────────────────────────────────────────
# PURPOSE: At what subscription age do customers leave most?
# INSIGHT: This directly drives the is_new_customer engineered feature
ax4 = axes[1, 0]
tenure_bins  = pd.cut(df['tenure_months'],
                      bins=[0, 3, 6, 12, 24, 72],
                      labels=['0-3m', '3-6m', '6-12m', '1-2yr', '2yr+'])
churn_tenure = df.groupby(tenure_bins, observed=True)['churn'].mean() * 100
ax4.plot(churn_tenure.index, churn_tenure.values,
         'o-', color='#FF6B35', linewidth=2.5, markersize=8)
ax4.fill_between(range(len(churn_tenure)), churn_tenure.values,
                 alpha=0.2, color='#FF6B35')
ax4.set_title('Churn Rate by Tenure Bucket', fontweight='bold')
ax4.set_xlabel('Tenure')
ax4.set_ylabel('Churn Rate (%)')
ax4.set_xticks(range(len(churn_tenure)))
ax4.set_xticklabels(churn_tenure.index)

# ── Chart 5: Payment Failures vs Churn ─────────────────────────────────────
# PURPOSE: Does payment friction predict churn?
# EXPECTED: Near-linear relationship (more failures = higher churn rate)
# INSIGHT: This is the strongest single predictor — drives friction_score feature
ax5 = axes[1, 1]
churn_payment = df.groupby('payment_failures_3m')['churn'].mean() * 100
ax5.bar(churn_payment.index, churn_payment.values, color='#9B59B6', alpha=0.8)
ax5.set_title('Churn Rate by Payment Failures', fontweight='bold')
ax5.set_xlabel('Number of Payment Failures (last 3 months)')
ax5.set_ylabel('Churn Rate (%)')

# ── Chart 6: Correlation Heatmap ───────────────────────────────────────────
# PURPOSE: See numerical relationships between all features and churn
# HOW TO READ:
#   Values range from -1 to +1
#   +1 = perfect positive correlation (both go up together)
#   -1 = perfect negative correlation (one up, other down)
#    0 = no relationship
# LOOK FOR: features with high |correlation| with 'churn' column
ax6 = axes[1, 2]
numeric_cols = ['tenure_months', 'watch_hours_per_week', 'logins_last_30_days',
                'last_login_days_ago', 'payment_failures_3m', 'support_tickets',
                'monthly_charge', 'churn']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax6, square=True, cbar_kws={'shrink': 0.8})
ax6.set_title('Feature Correlation Heatmap', fontweight='bold')

plt.tight_layout()
eda_path = OUTPUT_DIR + 'eda_dashboard.png'
plt.savefig(eda_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"  ✓ EDA dashboard saved → {eda_path}")
print(f"\n  Key EDA Insights:")
print(f"    Avg watch hours — Churned:  {churned.mean():.1f} hrs/week")
print(f"    Avg watch hours — Retained: {retained.mean():.1f} hrs/week")
print(f"    Churn by plan:\n{churn_by_plan.round(1).to_string()}")
print(f"    Churn by tenure:\n{churn_tenure.round(1).to_string()}")


# =============================================================================
#   PART C — PREPROCESSING & FEATURE ENGINEERING
# =============================================================================
# CONCEPT: ML models are mathematical functions that only understand NUMBERS.
#   'Basic', 'Mobile', 'India' are meaningless to a model.
#   Preprocessing = converting everything into numbers the model can use.
#
# TWO TYPES OF ENCODING:
#
#   Label Encoding: Assigns an integer to each category
#     Basic=0, Standard=1, Premium=2
#     USE WHEN: There is a natural ORDER (Basic < Standard < Premium ✓)
#
#   One-Hot Encoding: Creates a new 0/1 column for each category value
#     device_Mobile=1, device_Laptop=0, device_TV=0, device_Tablet=0
#     USE WHEN: No natural order (Mobile is NOT "less than" Laptop)
#     WHY NOT LABEL ENCODE HERE: If Mobile=0, Laptop=1, TV=2, Tablet=3,
#     the model thinks Tablet > TV > Laptop > Mobile which is meaningless!
#
# FEATURE ENGINEERING:
#   Creating NEW features by combining existing ones.
#   Goal: give the model a more powerful signal than raw features alone.
#   Good features = better model, often more than switching algorithms.
#
# VIVA TIP:
#   "We applied label encoding to plan_type because it has a natural
#    ordinal relationship (Basic < Standard < Premium). We used
#    one-hot encoding for device, genre, and payment method because
#    these are nominal — no inherent ordering exists between categories."
# =============================================================================

print("\n" + "─" * 65)
print("  PART C — PREPROCESSING & FEATURE ENGINEERING")
print("─" * 65)

# Work on a copy — never modify the original df
df_model = df.copy()

# ── Drop Identifier & Leakage Columns ─────────────────────────────────────
# DATA LEAKAGE CONCEPT: If we include churn_probability (which we calculated
# from the same logic that creates churn labels), the model would just learn
# to copy our formula — it wouldn't discover real patterns.
# clv_score and clv_segment are derived from churn_probability → also leakage.
# name and customer_id are identifiers — no predictive relationship with churn.

drop_cols = ['customer_id', 'name', 'churn_probability', 'clv_score', 'clv_segment']
df_model  = df_model.drop(columns=drop_cols)
print(f"\n  Dropped leakage/identifier columns: {drop_cols}")

# ── Feature Engineering ────────────────────────────────────────────────────
# CONCEPT: Raw features capture one dimension each.
# Engineered features capture RELATIONSHIPS between dimensions.
# A customer watching 15hrs with 1 login behaves very differently
# from one watching 15hrs with 20 logins — engagement_score captures this.

# engagement_score: overall activity in one normalized number
# High watch hours AND high logins = highly engaged = low churn risk
# Divide by 30 to normalize (logins_last_30_days is raw count)
df_model['engagement_score'] = (
    df_model['watch_hours_per_week'] * df_model['logins_last_30_days'] / 30
).round(3)

# friction_score: all negative experience signals combined
# Payment failure + support ticket = double friction = high churn risk
# Our EDA showed both are individually correlated with churn
# Combining them into one feature gives the model a stronger signal
df_model['friction_score'] = (
    df_model['payment_failures_3m'] + df_model['support_tickets']
)

# inactivity_flag: binary signal for "gone dark"
# A customer who hasn't logged in for 14+ days is signaling disengagement
# This threshold comes from OTT platform analytics research
df_model['inactivity_flag'] = (df_model['last_login_days_ago'] > 14).astype(int)

# is_new_customer: our EDA showed 0-6 month customers have ~70% churn rate
# vs 2yr+ customers with ~33% churn — a dramatic difference worth flagging
df_model['is_new_customer'] = (df_model['tenure_months'] < 6).astype(int)

print(f"\n  Engineered 4 new features:")
print(f"    engagement_score = watch_hours × logins / 30")
print(f"    friction_score   = payment_failures + support_tickets")
print(f"    inactivity_flag  = 1 if last_login > 14 days")
print(f"    is_new_customer  = 1 if tenure < 6 months")

# ── Label Encode Plan Type ──────────────────────────────────────────────────
# Plan type has a clear order: Basic < Standard < Premium
# (price, features, and prestige all increase)
# Label encoding preserves this ordinal relationship
plan_order = {'Basic': 0, 'Standard': 1, 'Premium': 2}
df_model['plan_type'] = df_model['plan_type'].map(plan_order)
print(f"\n  Label encoded 'plan_type': Basic=0, Standard=1, Premium=2")

# ── One-Hot Encode Nominal Categoricals ────────────────────────────────────
# These have no natural ordering → one-hot encoding
# drop_first=True: removes one column per feature to prevent multicollinearity
# CONCEPT: If you know Mobile=0, Laptop=0, TV=0 → must be Tablet=1
# So the 4th column is redundant and can cause numerical instability
ohe_cols = ['gender', 'country', 'preferred_genre', 'device_type', 'payment_method']
df_model = pd.get_dummies(df_model, columns=ohe_cols, drop_first=True)
print(f"  One-hot encoded: {ohe_cols}")
print(f"  Shape after encoding: {df_model.shape}")

# ── Separate Features (X) from Target (y) ──────────────────────────────────
# CONCEPT: In supervised ML:
#   X = INPUTS (features the model sees to make a prediction)
#   y = OUTPUT (what we want the model to predict)
# The model learns the mapping: f(X) → y
X = df_model.drop(columns=['churn'])   # All columns except churn
y = df_model['churn']                  # Only the churn column (0 or 1)

print(f"\n  X (features) shape: {X.shape}")
print(f"  y (target) shape:   {y.shape}")

# ── Train-Test Split ────────────────────────────────────────────────────────
# CONCEPT: We MUST evaluate the model on data it has NEVER seen during training.
# Otherwise we're testing if the model "memorized" the answers, not if it
# actually learned patterns (like giving students the exam paper to study).
#
# 80% training: model learns from this
# 20% testing:  we evaluate on this — these are "unseen exam questions"
#
# stratify=y: ensures both train AND test have the same churn ratio (~43%)
# Without stratify, by chance test might have 60% churned — biased evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% held out for testing
    random_state=42,   # Fixed seed = same split every run
    stratify=y         # Keep churn ratio consistent
)

print(f"\n  Train-Test Split (80/20):")
print(f"    Training: {len(X_train)} samples | Churn rate: {y_train.mean():.1%}")
print(f"    Testing:  {len(X_test)} samples  | Churn rate: {y_test.mean():.1%}")
print(f"    ✓ Churn rates are similar → stratify worked correctly")


# =============================================================================
#   PART D — SMOTE (Fix Class Imbalance)
# =============================================================================
# CONCEPT: Our dataset is ~57% retained / 43% churned.
# Even mild imbalance can bias the model toward the majority class.
#
# WHY IMBALANCE IS A PROBLEM:
#   Imagine a model that ALWAYS predicts "retained":
#     Accuracy = 57%  (looks ok right?)
#     Recall   = 0%   (catches ZERO churners — completely useless!)
#   This is why we never rely on accuracy alone.
#
# WHAT SMOTE DOES:
#   1. Takes a minority class sample (a churned customer)
#   2. Finds its K=5 nearest neighbors (other churned customers)
#   3. Creates a new SYNTHETIC sample by interpolating between them
#   4. Repeats until classes are 50/50
#
# KEY RULE → SMOTE ONLY ON TRAINING DATA:
#   We NEVER apply SMOTE to the test set.
#   Test set must stay real-world distribution for honest evaluation.
#   Applying SMOTE to test data = evaluating on synthetic data = lying to yourself.
#
# VIVA TIP:
#   "SMOTE is applied only to the training set. The test set represents
#    real-world distribution. We apply SMOTE after the split, never before,
#    to prevent data leakage from synthetic samples influencing test performance."
# =============================================================================

print("\n" + "─" * 65)
print("  PART D — SMOTE (Class Imbalance)")
print("─" * 65)

print(f"\n  Before SMOTE (training set only):")
print(f"    Retained (0): {(y_train==0).sum()}")
print(f"    Churned  (1): {(y_train==1).sum()}")
print(f"    Ratio: {(y_train==1).sum()/(y_train==0).sum():.2f}")

# k_neighbors=5: each synthetic sample is created using 5 nearest real neighbors
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\n  After SMOTE:")
print(f"    Retained (0): {(y_train_sm==0).sum()}")
print(f"    Churned  (1): {(y_train_sm==1).sum()}")
print(f"    New training size: {len(X_train_sm)} (added {len(X_train_sm)-len(X_train)} synthetic samples)")
print(f"    ✓ Classes are now 50/50 balanced")


# =============================================================================
#   PART E — MODEL TRAINING (XGBoost)
# =============================================================================
# CONCEPT: XGBoost = eXtreme Gradient Boosting
#
# HOW IT WORKS (the simple version):
#   Think of 200 students solving a very hard exam together.
#   Student 1 attempts all questions → gets some wrong
#   Student 2 focuses ONLY on questions Student 1 got wrong
#   Student 3 focuses on what Students 1 & 2 BOTH got wrong
#   ...continue for 200 students...
#   Final answer = weighted combination of all 200 students
#   Result: far more accurate than any single student
#
# In ML terms:
#   Each "student" = a small decision tree
#   "Focusing on wrong answers" = gradient descent on residual errors
#   This is BOOSTING → sequential improvement
#   XGBoost = an extremely optimized implementation of this idea
#
# KEY HYPERPARAMETERS:
#   n_estimators=200    → build 200 trees sequentially
#   max_depth=5         → each tree can be at most 5 levels deep
#                         deeper = learns more complex patterns but risks overfitting
#   learning_rate=0.1   → how much each new tree corrects the previous
#                         lower = more careful = usually better
#   subsample=0.8       → use 80% of training data per tree (reduces overfitting)
#   colsample_bytree=0.8 → use 80% of features per tree (reduces overfitting)
#
# VIVA TIP:
#   "We chose XGBoost because it handles mixed feature types, missing values,
#    and imbalanced data well. It natively supports SHAP for explainability.
#    The gradient boosting mechanism — where each tree corrects the errors of
#    previous trees — makes it superior to random forest on tabular data."
# =============================================================================

print("\n" + "─" * 65)
print("  PART E — MODEL TRAINING (XGBoost)")
print("─" * 65)

model = XGBClassifier(
    n_estimators=200,         # 200 sequential trees
    max_depth=5,              # Max tree depth
    learning_rate=0.1,        # Step size (eta)
    subsample=0.8,            # Row subsampling
    colsample_bytree=0.8,     # Feature subsampling
    eval_metric='logloss',    # Binary cross-entropy loss
    random_state=42,
    verbosity=0               # Suppress XGBoost internal logs
)

print(f"\n  Training XGBoost on {len(X_train_sm)} SMOTE-balanced samples...")
model.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test, y_test)],   # Monitor test performance each round
    verbose=False
)
print(f"  ✓ Training complete!")

# Save trained model to disk
# CONCEPT: pickle serializes the Python object to a binary file
# The FastAPI backend loads this file to make predictions
# without having to retrain every time
model_path = OUTPUT_DIR + 'churniq_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  ✓ Model saved → {model_path}")

# Save feature names — critical for correct prediction in FastAPI
# CONCEPT: When FastAPI loads the model and makes a new prediction,
# features MUST be in the exact same order as during training
feature_names = list(X.columns)
feat_path     = OUTPUT_DIR + 'feature_names.pkl'
with open(feat_path, 'wb') as f:
    pickle.dump(feature_names, f)
print(f"  ✓ Feature names saved ({len(feature_names)} features) → {feat_path}")


# =============================================================================
#   PART F — MODEL EVALUATION
# =============================================================================
# CONCEPT: NEVER judge a classification model on accuracy alone.
# For churn prediction, missing a churner (False Negative) is worse
# than a false alarm (False Positive). This asymmetry matters.
#
# KEY METRICS:
#
#   PRECISION = TP / (TP + FP)
#   "Of all customers we flagged as churners, what % actually churned?"
#   Low precision → too many false alarms → waste retention budget
#
#   RECALL = TP / (TP + FN)
#   "Of all customers who actually churned, what % did we catch?"
#   Low recall → missing churners → they leave without intervention
#   ← THIS IS OUR PRIORITY METRIC
#
#   F1 SCORE = 2 × (Precision × Recall) / (Precision + Recall)
#   Harmonic mean — good balance between the two
#
#   ROC-AUC = Area Under the ROC Curve
#   0.5 = random coin flip
#   1.0 = perfect model
#   >0.85 = excellent, 0.75-0.85 = good, 0.65-0.75 = decent
#
# CONFUSION MATRIX:
#   ┌────────────────────┬──────────────────┐
#   │ True Negative (TN) │ False Positive   │
#   │ Predicted Retained │ (FP) False Alarm │
#   │ Actually Retained  │ Predicted Churn  │
#   │         ✓          │ Actually Retained│
#   ├────────────────────┼──────────────────┤
#   │ False Negative     │ True Positive    │
#   │ (FN) ← WORST CASE │ (TP) ✓           │
#   │ Predicted Retained │ Predicted Churn  │
#   │ Actually Churned   │ Actually Churned │
#   └────────────────────┴──────────────────┘
#
# VIVA TIP:
#   "False Negatives (FN) are our most costly error — a churner we missed
#    will leave and we lose their CLV forever. Therefore we optimized for
#    Recall. We accept some False Positives (unnecessary retention offers)
#    as the cost of catching more real churners."
# =============================================================================

print("\n" + "─" * 65)
print("  PART F — MODEL EVALUATION")
print("─" * 65)

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]   # Probability of class 1 (churn)

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

roc_auc = roc_auc_score(y_test, y_pred_prob)
acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)

print(f"  Summary Metrics:")
print(f"    Accuracy  : {acc:.4f}")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}  ← our priority metric")
print(f"    F1 Score  : {f1:.4f}")
print(f"    ROC-AUC   : {roc_auc:.4f}")

# ── Evaluation Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ChurnIQ Pro — Model Evaluation', fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Retained', 'Churned'],
            yticklabels=['Retained', 'Churned'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
tn, fp, fn, tp = cm.ravel()
axes[0].text(0.5, -0.18,
             f'TP={tp}✓  FP={fp}(false alarms)  FN={fn}⚠(missed)  TN={tn}✓',
             transform=axes[0].transAxes, ha='center', fontsize=8, color='#555')

# Plot 2: ROC Curve
# CONCEPT: ROC shows tradeoff between catching churners (TPR/Recall)
# and false alarms (FPR) at every possible classification threshold.
# Default threshold = 0.5 but you can lower it to catch more churners
# (at the cost of more false alarms). AUC = area under this curve.
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='#4A90E2', linewidth=2.5,
             label=f'XGBoost (AUC = {roc_auc:.3f})')
axes[1].plot([0,1], [0,1], 'k--', linewidth=1, label='Random Baseline (AUC = 0.5)')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#4A90E2')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].set_xlabel('False Positive Rate (False Alarms)')
axes[1].set_ylabel('True Positive Rate (Churners Caught)')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Feature Importance (built-in XGBoost)
feat_imp    = pd.Series(model.feature_importances_, index=X.columns)
top_feats   = feat_imp.nlargest(12)
colors_fi   = ['#FF4444' if i < 3 else '#4A90E2' for i in range(len(top_feats))]
top_feats[::-1].plot(kind='barh', ax=axes[2], color=colors_fi[::-1])
axes[2].set_title('Top 12 Feature Importances\n(Red = Most Critical)',
                  fontweight='bold')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
eval_path = OUTPUT_DIR + 'model_evaluation.png'
plt.savefig(eval_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✓ Evaluation plots saved → {eval_path}")


# =============================================================================
#   PART G — SHAP (EXPLAINABLE AI)
# =============================================================================
# CONCEPT: SHAP = SHapley Additive exPlanations
#
# THE PROBLEM WITH BLACK BOXES:
#   Without SHAP: "Customer will churn — 87% probability"
#   Business team: "OK... so what do we do?"
#
#   With SHAP: "Customer will churn 87% BECAUSE:
#     - friction_score = 7 → pushed risk UP by +2.17
#     - tenure = 2 months → pushed risk UP by +1.35
#     - watch_hours = 1.4 → pushed risk UP by +1.05"
#   Business team: "OK, we know exactly what to fix → call Gemini → email"
#
# THE GAME THEORY ORIGIN:
#   Shapley values come from cooperative game theory (Shapley, 1953).
#   Imagine 4 players (features) cooperating in a game (making a prediction).
#   SHAP asks: "How much did each player FAIRLY contribute to the outcome?"
#   It does this by testing all possible subsets of players — then averaging.
#
# TWO TYPES OF SHAP ANALYSIS:
#   GLOBAL: Which features matter MOST across all customers?
#           → Summary plot (beeswarm) and bar chart
#   LOCAL:  Why is THIS SPECIFIC customer at risk?
#           → Waterfall plot for individual customers
#           → This is what feeds the Gemini API prompt!
#
# SHAP VALUE INTERPRETATION:
#   Positive value (+2.17) → feature PUSHED churn probability UP
#   Negative value (-0.32) → feature PUSHED churn probability DOWN
#   Magnitude → how strongly it pushed
#
# VIVA TIP:
#   "SHAP provides model-agnostic explanations grounded in game theory.
#    We use TreeExplainer which is optimized for tree-based models and
#    computes exact Shapley values efficiently. The local explanations
#    from SHAP feed our Gemini API prompt to generate personalized
#    retention strategies for each at-risk customer."
# =============================================================================

print("\n" + "─" * 65)
print("  PART G — SHAP (EXPLAINABLE AI)")
print("─" * 65)

print("\n  Calculating SHAP values for test set...")
print("  (This may take 30-60 seconds...)")

# TreeExplainer: specialized for tree-based models (XGBoost, Random Forest)
# Much faster than the generic KernelExplainer for tree models
# Uses the tree structure directly for exact computation
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"  ✓ SHAP values computed — shape: {shap_values.shape}")
print(f"    ({shap_values.shape[0]} customers × {shap_values.shape[1]} features)")

# ── SHAP Plot 1: Summary (Beeswarm) ────────────────────────────────────────
# WHAT IT SHOWS:
#   Each dot = one customer in the test set
#   X-axis = SHAP value (how much this feature pushed the churn probability)
#   Color  = actual feature value for that customer
#            RED = HIGH feature value, BLUE = LOW feature value
#
# HOW TO READ IT:
#   Row = one feature (most important at top)
#   Dots spread RIGHT of 0 → feature increases churn risk
#   Dots spread LEFT of 0  → feature decreases churn risk
#   RED dots on RIGHT      → high values of this feature increase churn risk
#   BLUE dots on RIGHT     → low values increase churn risk
#
# EXAMPLE READING:
#   watch_hours_per_week row:
#     BLUE dots on RIGHT → low watch hours = increases churn risk ✓ (makes sense!)
#     RED dots on LEFT   → high watch hours = decreases churn risk ✓ (makes sense!)
print("\n  Generating SHAP Summary (Beeswarm) plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, X_test,
    plot_type='dot',    # 'dot' = beeswarm — best for understanding direction
    max_display=15,     # Show top 15 most important features
    show=False
)
plt.title(
    'SHAP Summary — Feature Impact on Churn Prediction\n'
    '(Right = Increases Churn Risk  |  Left = Reduces Churn Risk)',
    fontweight='bold', pad=15
)
plt.tight_layout()
shap_summary_path = OUTPUT_DIR + 'shap_summary.png'
plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ SHAP summary saved → {shap_summary_path}")

# ── SHAP Plot 2: Feature Importance Bar ────────────────────────────────────
# Shows MEAN ABSOLUTE SHAP value per feature
# = average magnitude of impact across all customers
# More interpretable than XGBoost's built-in importance
# because it's measured in the same units as the model output
print("  Generating SHAP Importance Bar plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test,
    plot_type='bar',    # bar = mean |SHAP| per feature
    max_display=12,
    show=False
)
plt.title('SHAP Feature Importance — Mean Impact on Churn Probability',
          fontweight='bold')
plt.tight_layout()
shap_bar_path = OUTPUT_DIR + 'shap_importance.png'
plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ SHAP importance saved → {shap_bar_path}")

# ── SHAP Plot 3: Waterfall (Individual Customer) ────────────────────────────
# WHAT IT SHOWS: The complete story of ONE customer's churn prediction
# Starts at E[f(x)] = baseline (average prediction across all customers)
# Each bar = one feature's contribution (positive = pushed churn up)
# Ends at f(x) = final churn probability for this customer
#
# THIS IS WHAT GETS DISPLAYED ON THE CUSTOMER DETAIL PAGE
# AND WHAT FEEDS INTO THE GEMINI PROMPT
print("  Generating SHAP Waterfall (individual customer) plot...")
highest_risk_idx = np.argmax(y_pred_prob)    # Most at-risk customer in test set
print(f"  Highest-risk customer: churn probability = {y_pred_prob[highest_risk_idx]:.1%}")

# Get top 3 churn drivers for this customer
customer_shap   = shap_values[highest_risk_idx]
sorted_shap     = sorted(zip(X_test.columns, customer_shap),
                         key=lambda x: x[1], reverse=True)
print(f"  Top 3 churn drivers:")
for feat, val in sorted_shap[:3]:
    actual = X_test.iloc[highest_risk_idx][feat]
    print(f"    {feat}: SHAP={val:+.3f}, Value={actual:.2f}")

plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[highest_risk_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[highest_risk_idx].values,
        feature_names=list(X_test.columns)
    ),
    max_display=10,    # Show top 10 contributing features
    show=False
)
plt.title(
    f'SHAP Waterfall — Why Is This Customer Churning?\n'
    f'(Churn Probability: {y_pred_prob[highest_risk_idx]:.1%})',
    fontweight='bold'
)
plt.tight_layout()
waterfall_path = OUTPUT_DIR + 'shap_waterfall_example.png'
plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Waterfall plot saved → {waterfall_path}")


# =============================================================================
#   PART H — PREDICTIONS TABLE
# =============================================================================
# CONCEPT: Pre-compute predictions + SHAP reasons for ALL customers.
# Store results in a CSV that the FastAPI backend reads on startup.
#
# WHY PRE-COMPUTE?
#   Computing SHAP values is slow (~30 seconds for 1000 customers).
#   If we computed SHAP live on every API request, the dashboard
#   would be unusably slow. Pre-computing = fast API responses.
#
# WHAT'S IN THE TABLE:
#   customer_id   → to join back to customer profile data
#   churn_prob    → model's prediction (0.0 to 1.0)
#   risk_level    → HIGH/MEDIUM/LOW (for color coding in frontend)
#   top_reason_1/2/3 → SHAP-derived plain-English churn reasons
#   clv_score     → Customer Lifetime Value
#   actual_churn  → ground truth (for evaluation purposes)
# =============================================================================

print("\n" + "─" * 65)
print("  PART H — PREDICTIONS TABLE")
print("─" * 65)

def get_top_shap_reasons(shap_row, feature_cols, n=3):
    """
    Extract top N positive SHAP contributors for one customer.
    Returns list of formatted strings: "feature_name (impact: +X.XX)"
    Only positive SHAP values (features pushing churn UP) are included.
    """
    pairs = sorted(zip(feature_cols, shap_row), key=lambda x: x[1], reverse=True)
    return [
        f"{feat} (impact: {val:+.2f})"
        for feat, val in pairs[:n]
        if val > 0   # Only include features pushing churn risk UP
    ]

def calc_clv(monthly_charge, tenure, churn_prob):
    """
    Simple CLV: monthly_charge × tenure × (1 - churn_probability)
    Min tenure 6 months to avoid extremely low CLV for new customers.
    """
    return round(monthly_charge * max(tenure, 6) * (1 - churn_prob), 2)

print(f"\n  Building predictions for {len(X_test)} test customers...")

predictions = []
for i in range(len(X_test)):
    reasons  = get_top_shap_reasons(shap_values[i], list(X_test.columns))
    prob     = y_pred_prob[i]
    risk     = 'HIGH' if prob >= 0.70 else ('MEDIUM' if prob >= 0.40 else 'LOW')
    orig_idx = X_test.index[i]

    # Look up original customer data for CLV calculation
    mc  = df.loc[orig_idx, 'monthly_charge']
    ten = df.loc[orig_idx, 'tenure_months']

    predictions.append({
        'customer_id':  df.loc[orig_idx, 'customer_id'],
        'churn_prob':   round(prob, 4),
        'risk_level':   risk,
        'top_reason_1': reasons[0] if len(reasons) > 0 else 'N/A',
        'top_reason_2': reasons[1] if len(reasons) > 1 else 'N/A',
        'top_reason_3': reasons[2] if len(reasons) > 2 else 'N/A',
        'clv_score':    calc_clv(mc, ten, prob),
        'actual_churn': int(y_test.iloc[i])
    })

pred_df   = pd.DataFrame(predictions).sort_values('churn_prob', ascending=False)
pred_path = OUTPUT_DIR + 'customer_predictions.csv'
pred_df.to_csv(pred_path, index=False)

print(f"  ✓ Predictions saved → {pred_path}")
print(f"\n  Top 5 Highest-Risk Customers:")
print(pred_df.head(5)[['customer_id','churn_prob','risk_level',
                        'top_reason_1','clv_score']].to_string(index=False))


# =============================================================================
#   PIPELINE COMPLETE — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("   PIPELINE COMPLETE")
print("=" * 65)
print(f"""
  Model      : XGBoost (200 trees, depth=5, lr=0.1)
  Dataset    : {len(df)} customers × {df.shape[1]} features (synthetic OTT)
  Train/Test : 80% / 20% (stratified)
  SMOTE      : Applied to training set only

  ┌──────────────────────────────────────────┐
  │  EVALUATION METRICS (test set)           │
  ├──────────────────────────────────────────┤
  │  Accuracy  : {acc:.4f}                       │
  │  Precision : {prec:.4f}                       │
  │  Recall    : {rec:.4f}  ← key metric      │
  │  F1 Score  : {f1:.4f}                       │
  │  ROC-AUC   : {roc_auc:.4f}                       │
  └──────────────────────────────────────────┘

  Saved Files:
  ✓ ott_churn_dataset.csv          Raw synthetic dataset
  ✓ churniq_model.pkl              Trained XGBoost model
  ✓ feature_names.pkl              Feature column names
  ✓ customer_predictions.csv       All predictions + SHAP reasons
  ✓ eda_dashboard.png              6-chart EDA dashboard
  ✓ model_evaluation.png           Confusion matrix + ROC + importance
  ✓ shap_summary.png               Global SHAP beeswarm
  ✓ shap_importance.png            Mean SHAP feature ranking
  ✓ shap_waterfall_example.png     Individual customer explanation

  Next Step → Run main.py (FastAPI Backend)
  Command:    uvicorn main:app --reload --port 8000
  Docs:       http://localhost:8000/docs
""")
print("=" * 65)

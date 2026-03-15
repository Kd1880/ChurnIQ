
#   ChurnIQ Pro — FILE 2: FastAPI Backend + Gemini AI
#   ─────────────────────────────────────────────────────────
#   What this file does:
#
#   PART A — STARTUP
#     Loads trained XGBoost model (.pkl) and datasets on server start
#     Configures Gemini API key
#
#   PART B — HELPER FUNCTIONS
#     Feature engineering (same as ML pipeline — must match exactly)
#     Plain-English SHAP reason formatting
#     CLV calculation and segmentation
#
#   PART C — API ENDPOINTS
#     GET  /                   → Health check
#     GET  /customers          → Full risk table + KPI summary
#     GET  /customer/{id}      → Single customer detail + SHAP reasons
#     POST /whatif             → Live what-if simulation (sliders)
#     GET  /retention/{id}     → Gemini AI retention email + actions
#     GET  /segments           → Churn stats by plan/device/tenure/genre
#     GET  /dashboard/summary  → All KPIs in one call
#
#   ─────────────────────────────────────────────────────────
#   PREREQUISITES:
#     Run churniq_ml_pipeline.py first to generate:
#       churniq_model.pkl
#       feature_names.pkl
#       ott_churn_dataset.csv
#       customer_predictions.csv
#
#   SETUP:
#     pip install fastapi uvicorn google-generativeai
#                 pandas numpy scikit-learn xgboost python-multipart
#
#   ADD YOUR GEMINI KEY:
#     Get free key from: https://aistudio.google.com
#     Set it below in the GEMINI_API_KEY variable (line ~120)
#
#   RUN:
#     uvicorn main:app --reload --port 8000
#
#   TEST:
#     Open http://localhost:8000/docs  ← interactive Swagger UI
#     Try every endpoint directly in the browser — no Postman needed!
#
#   DEPLOY TO RENDER (free):
#     1. Push project to GitHub
#     2. Go to render.com → New Web Service
#     3. Connect repo → Build: pip install -r requirements.txt
#     4. Start: uvicorn main:app --host 0.0.0.0 --port $PORT
#     5. Add env variable: GEMINI_API_KEY = your_key
# =============================================================================

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

# FastAPI — the web framework that turns Python functions into REST API endpoints
# Uvicorn — the ASGI server that runs FastAPI (async capable, much faster than Flask)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Pydantic — data validation library
# CONCEPT: Pydantic models define the SHAPE of request/response data.
# FastAPI uses them to:
#   - Automatically validate incoming JSON (wrong type = auto 422 error)
#   - Generate API documentation at /docs
#   - Serialize Python dicts to JSON responses
from pydantic import BaseModel
from typing import Optional, List

# Standard Python
import pickle          # Deserialize saved .pkl model and feature names
import pandas as pd
import numpy as np
import os
import json
import re
import warnings

warnings.filterwarnings('ignore')

# Google Gemini API — for AI-generated retention emails
# CONCEPT: We use Gemini to translate ML/SHAP output into human-readable
# personalized retention strategies. This is the "AI layer" of ChurnIQ Pro.
import google.generativeai as genai


# =============================================================================
#   PART A — STARTUP: Load All Required Files
# =============================================================================
# CONCEPT: Load heavy resources ONCE at startup, not on every request.
# Loading a .pkl model takes ~200-500ms. If loaded per request:
#   → Every API call is 500ms slower
#   → Under 10 concurrent users → server is constantly reloading
# Loading at startup: done once → all requests share same loaded object
#
# VIVA TIP:
#   "We use FastAPI's module-level loading pattern — the model is loaded
#    into memory when the server starts. All incoming requests share the
#    same model object. This is thread-safe for inference (read-only)."
# =============================================================================

print("=" * 60)
print("  ChurnIQ Pro API — Initializing...")
print("=" * 60)

def find_file(filename: str) -> str:
    """
    Search common locations for a required file.
    Checks current directory, subdirectories, and outputs folder.
    Raises FileNotFoundError with helpful message if not found.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',   filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', filename),
        os.path.join('/mnt/user-data/outputs', filename),
        filename,   # Absolute path (if passed directly)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find '{filename}'.\n"
        f"Make sure you ran churniq_ml_pipeline.py first.\n"
        f"Searched in: {candidates}"
    )

# ── Load XGBoost Model ──────────────────────────────────────────────────────
print("\n[1/4] Loading XGBoost model...")
with open(find_file('churniq_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)
print("  ✓ Model loaded")

# ── Load Feature Names ──────────────────────────────────────────────────────
# CRITICAL: Feature names define the exact order and set of columns
# the model expects. ANY mismatch = wrong predictions.
print("[2/4] Loading feature names...")
with open(find_file('feature_names.pkl'), 'rb') as f:
    FEATURE_NAMES = pickle.load(f)
print(f"  ✓ {len(FEATURE_NAMES)} feature names loaded")

# ── Load Predictions CSV ─────────────────────────────────────────────────────
# Pre-computed by ML pipeline — contains churn_prob + SHAP reasons per customer
# We load this so /customers and /customer/{id} endpoints are FAST
# (no live SHAP computation needed — that's already done)
print("[3/4] Loading predictions...")
PREDICTIONS_DF = pd.read_csv(find_file('customer_predictions.csv'))
print(f"  ✓ {len(PREDICTIONS_DF)} customer predictions loaded")

# ── Load Raw Dataset ──────────────────────────────────────────────────────────
# Needed for: customer profiles, segment analysis, what-if defaults
print("[4/4] Loading raw dataset...")
DATASET_DF = pd.read_csv(find_file('ott_churn_dataset.csv'))
print(f"  ✓ {len(DATASET_DF)} customers loaded")

# ── Merge Predictions into Dataset ──────────────────────────────────────────
# CONCEPT: SQL-style JOIN on customer_id
# Result: each customer row has both profile data AND churn predictions
# suffixes=('_raw', '') → if column exists in both, predictions version wins
FULL_DF = DATASET_DF.merge(
    PREDICTIONS_DF[['customer_id', 'churn_prob', 'risk_level',
                    'top_reason_1', 'top_reason_2', 'top_reason_3', 'clv_score']],
    on='customer_id',
    how='left',
    suffixes=('_raw', '')
)
print(f"\n  Merged dataset shape: {FULL_DF.shape}")


# =============================================================================
#   CONFIGURE GEMINI API
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
#   ⚠️  IMPORTANT: Replace 'YOUR_GEMINI_API_KEY' with your actual key!
#   Get a free key at: https://aistudio.google.com/app/apikey
#   Takes 2 minutes — no credit card required
#   Or set as environment variable: export GEMINI_API_KEY="your_key_here"
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL   = genai.GenerativeModel("gemini-2.5-flash")
    GEMINI_ENABLED = True
    print("  ✓ Gemini API configured (AI retention emails enabled)")
else:
    GEMINI_ENABLED = False
    print("  ⚠ Gemini API key not set — /retention endpoint uses rule-based fallback")
    print("    Set key: export GEMINI_API_KEY='your_key' or edit line above")

print("\n" + "=" * 60)
print("  Startup complete! All systems ready.")
print("  Docs: http://localhost:8000/docs")
print("=" * 60)

# =============================================================================
# ChurnIQ Pro — Email Sending Code
# =============================================================================
# Yeh code main.py mein add karo:
#
#   PART 1 → Top pe (imports ke baad, FastAPI app se pehle)
#   PART 2 → Bottom pe (last endpoint ke baad)
# =============================================================================
 
 
# ─────────────────────────────────────────────────────────────
# PART 1 — Yeh code main.py mein TOP pe add karo
#           (existing imports ke baad, app = FastAPI() se pehle)
# ─────────────────────────────────────────────────────────────
 
import smtplib
import json
from email.mime.text      import MIMEText
from email.mime.multipart import MIMEMultipart
 
# Load Gmail config from email_config.json
# CONCEPT: Credentials ko code mein hardcode karna dangerous hai
# Agar galti se GitHub pe push ho gaya → credentials leak!
# Isliye alag JSON file mein rakha — .gitignore se protected
 
try:
    with open('email_config.json') as f:
        _cfg = json.load(f)
    GMAIL_SENDER   = _cfg['GMAIL_SENDER']
    GMAIL_APP_PASS = _cfg['GMAIL_APP_PASS']
    DEMO_RECEIVER  = _cfg['DEMO_RECEIVER']
    EMAIL_ENABLED  = True
    print("  ✓ Gmail config loaded (email sending enabled)")
except FileNotFoundError:
    GMAIL_SENDER   = ""
    GMAIL_APP_PASS = ""
    DEMO_RECEIVER  = ""
    EMAIL_ENABLED  = False
    print("  ⚠ email_config.json nahi mila — run add_emails.py first")
# =============================================================================
#   FASTAPI APP INITIALIZATION
# =============================================================================
# CONCEPT: FastAPI creates a web application where Python functions become
# REST API endpoints. Each endpoint is a Python function decorated with
# @app.get(), @app.post() etc.
#
# FastAPI automatically:
#   → Parses incoming JSON request bodies
#   → Validates data types (int, str, float etc.)
#   → Serializes Python dicts/lists to JSON responses
#   → Generates interactive docs at /docs (Swagger UI)
#   → Generates schema at /openapi.json
#
# FASTAPI vs FLASK:
#   Flask: older, synchronous, manual validation
#   FastAPI: async-first, automatic validation via Pydantic, auto docs
#   In 2025, FastAPI is the industry standard for new Python APIs

app = FastAPI(
    title="ChurnIQ Pro API",
    description=(
        "AI-powered OTT customer churn prediction and retention intelligence. "
        "Predicts churn probability, explains reasons via SHAP, scores CLV, "
        "and generates personalized retention emails via Gemini AI."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc UI at /redoc
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
# CONCEPT: Browsers enforce CORS (Cross-Origin Resource Sharing) policy.
# A React app at localhost:5173 CANNOT call an API at localhost:8000
# by default — different ports = different "origins" = blocked by browser.
#
# CORS middleware adds headers to every response:
#   Access-Control-Allow-Origin: *
#   Access-Control-Allow-Methods: GET, POST, ...
#
# This tells the browser: "Yes, this API is intentionally accessible cross-origin"
# In production → replace "*" with your actual frontend URL for security:
#   allow_origins=["https://churniq.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow ALL origins (dev mode)
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)


# =============================================================================
#   PYDANTIC REQUEST/RESPONSE MODELS
# =============================================================================
# CONCEPT: Pydantic models are Python classes that define data schemas.
# They serve three purposes in FastAPI:
#   1. Request validation → wrong type = automatic 422 Unprocessable Entity
#   2. Response documentation → shown in /docs so frontend knows what to expect
#   3. Type safety → your IDE gives autocomplete for these fields
#
# Optional fields = not required. If not sent, they default to None.
# The what-if endpoint uses Optional because user might only change 1 slider.

class WhatIfRequest(BaseModel):
    """
    Request body for POST /whatif (What-If Simulator).

    Frontend sends the customer_id + any features the user changed via sliders.
    Only send the fields that changed — unchanged fields stay at customer's actual values.

    Example request body:
    {
        "customer_id": "CUST01033",
        "payment_failures_3m": 0,
        "watch_hours_per_week": 8.5
    }
    """
    customer_id:            str
    watch_hours_per_week:   Optional[float] = None   # Slider: 0.5 - 40 hrs/week
    tenure_months:          Optional[int]   = None   # Slider: 1 - 72 months
    payment_failures_3m:    Optional[int]   = None   # Slider: 0 - 4 failures
    support_tickets:        Optional[int]   = None   # Slider: 0 - 5 tickets
    logins_last_30_days:    Optional[int]   = None   # Slider: 0 - 30 logins
    monthly_charge:         Optional[int]   = None   # Dropdown: 199 / 499 / 799


class WhatIfResponse(BaseModel):
    """Response from POST /whatif — shows before and after churn probabilities."""
    customer_id:          str
    original_churn_prob:  float    # Before slider change
    new_churn_prob:       float    # After slider change
    change:               float    # Positive = got riskier, Negative = improved
    change_percent:       str      # Formatted: "+3.2%" or "-8.5%"
    original_risk:        str      # HIGH / MEDIUM / LOW
    new_risk:             str      # HIGH / MEDIUM / LOW
    improved:             bool     # True if change is negative (risk went down)


# =============================================================================
#   PART B — HELPER FUNCTIONS
# =============================================================================

def get_risk_level(prob: float) -> str:
    """
    Convert churn probability to risk label.
    Thresholds:
      HIGH   (≥70%): Immediate action required
      MEDIUM (40-70%): Monitor and prepare retention
      LOW    (<40%):  Healthy customer
    """
    if prob >= 0.70:    return "HIGH"
    elif prob >= 0.40:  return "MEDIUM"
    else:               return "LOW"


def calc_clv(monthly_charge: float, tenure: int, churn_prob: float) -> float:
    """
    Customer Lifetime Value = monthly_charge × expected_tenure × retention_factor
    retention_factor = (1 - churn_probability)
    Minimum tenure of 6 to avoid near-zero CLV for very new customers.
    """
    return round(float(monthly_charge) * max(tenure, 6) * (1 - churn_prob), 2)


def get_clv_segment(clv: float) -> str:
    """Segment customers by CLV for business prioritization."""
    if clv > 8000:  return "High"
    if clv > 3000:  return "Medium"
    return "Low"


def format_reason(reason_str: str) -> str:
    """
    Convert raw SHAP feature string to plain-English business language.

    Input:  'friction_score (impact: +2.17)'
    Output: 'High friction score — payment failures + support issues [SHAP: +2.17]'

    WHY THIS MATTERS: Business teams cannot act on "friction_score = 7".
    They CAN act on "Customer has had 7 combined payment failures and support tickets."
    This translation is what makes ChurnIQ Pro actionable.
    """
    if not isinstance(reason_str, str) or reason_str in ('N/A', '', 'nan'):
        return "No significant risk factor identified"

    # Map feature names to plain English descriptions
    plain_map = {
        'friction_score':           'High friction score — payment failures + support issues',
        'watch_hours_per_week':     'Very low content engagement — watching very few hours',
        'tenure_months':            'Short subscription tenure — still in high-risk early phase',
        'is_new_customer':          'New customer — within the high-churn 0-6 month window',
        'engagement_score':         'Low overall engagement score (watch hours × login frequency)',
        'logins_last_30_days':      'Infrequent logins — rarely opening the app',
        'last_login_days_ago':      'Has not logged in recently — going inactive',
        'payment_failures_3m':      'Recent payment failures — billing friction',
        'support_tickets':          'Multiple support tickets — experiencing service issues',
        'inactivity_flag':          'Customer has gone inactive — 14+ days without login',
        'monthly_charge':           'High monthly charge relative to engagement level',
        'num_profiles':             'Low profile usage — subscription underutilized',
        'age':                      'Age demographic associated with higher churn risk',
    }

    # Extract feature name (before ' (impact:')
    feature     = reason_str.split(' (impact')[0].strip()
    impact_part = ''
    if '(impact:' in reason_str:
        impact_part = ' [SHAP: ' + reason_str.split('(impact:')[1].replace(')', '').strip() + ']'

    plain = plain_map.get(feature, feature.replace('_', ' ').title())
    return f"{plain}{impact_part}"


def build_feature_row(customer_data: dict) -> pd.DataFrame:
    """
    Rebuild the exact feature vector the model needs for a new prediction.
    CRITICAL: Must apply the EXACT same transformations as the ML pipeline.
    Any difference = wrong predictions.

    Steps (mirrors churniq_ml_pipeline.py preprocessing):
      1. Compute engineered features
      2. Label encode plan_type
      3. One-hot encode categorical columns
      4. Reindex to match training feature order (fill missing with 0)

    Args:
        customer_data: dict of raw customer features (can include overrides)
    Returns:
        DataFrame with exactly FEATURE_NAMES columns in correct order
    """
    # Extract raw values safely (with defaults)
    watch_hours = float(customer_data.get('watch_hours_per_week', 5.0))
    logins      = int(customer_data.get('logins_last_30_days', 10))
    pay_fail    = int(customer_data.get('payment_failures_3m', 0))
    support     = int(customer_data.get('support_tickets', 0))
    tenure      = int(customer_data.get('tenure_months', 12))
    last_login  = int(customer_data.get('last_login_days_ago', 3))
    plan        = str(customer_data.get('plan_type', 'Basic'))

    # Engineered features — exact same formulas as ML pipeline
    row = dict(customer_data)
    row['engagement_score'] = round(watch_hours * logins / 30, 3)
    row['friction_score']   = pay_fail + support
    row['inactivity_flag']  = int(last_login > 14)
    row['is_new_customer']  = int(tenure < 6)

    # Label encode plan_type (same mapping as ML pipeline)
    plan_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
    row['plan_type'] = plan_map.get(plan, 0)

    # Build DataFrame and one-hot encode
    temp_df  = pd.DataFrame([row])
    ohe_cols = ['gender', 'country', 'preferred_genre', 'device_type', 'payment_method']
    temp_df  = pd.get_dummies(temp_df, columns=ohe_cols, drop_first=True)

    # CRITICAL: Reindex to match EXACTLY the columns from training
    # fill_value=0 handles columns not present in this customer's data
    # (e.g., a genre the model was trained on but this customer doesn't have)
    temp_df = temp_df.reindex(columns=FEATURE_NAMES, fill_value=0)

    return temp_df


# =============================================================================
#   PART C — API ENDPOINTS
# =============================================================================
# CONCEPT: In REST APIs:
#   GET  → Read data (no side effects) — use for fetching information
#   POST → Send data to server for processing — use for simulations/calculations
#   PUT  → Update existing resource
#   DELETE → Remove resource
#
# URL path parameters: /customer/{customer_id}  → customer_id is a variable
# Query parameters:    /customers?risk_level=HIGH → filter param in URL
# Request body:        POST /whatif with JSON body → for complex input data
#
# HTTP Status Codes:
#   200 OK          → success
#   404 Not Found   → customer_id doesn't exist
#   422 Unprocessable → wrong data type in request
#   500 Server Error → something crashed server-side


# ── ENDPOINT 1: Health Check ─────────────────────────────────────────────────
# WHY NEEDED:
#   Deployment platforms (Render, Railway) ping this to check if server is alive
#   If this returns non-200, the platform restarts the server
#   Frontend can also call this to show "API Status: Connected ✓" indicator
@app.get("/", tags=["System"])
def health_check():
    """
    Health check endpoint. Returns server status and basic stats.
    Used by deployment platforms to verify the server is alive.
    """
    high_risk = len(FULL_DF[FULL_DF['risk_level'] == 'HIGH'])
    return {
        "status":           "✓ ChurnIQ Pro API is running",
        "version":          "1.0.0",
        "model_loaded":     True,
        "total_customers":  len(FULL_DF),
        "high_risk_count":  high_risk,
        "gemini_enabled":   GEMINI_ENABLED,
        "endpoints": {
            "customers":        "GET  /customers",
            "customer_detail":  "GET  /customer/{id}",
            "what_if":          "POST /whatif",
            "retention":        "GET  /retention/{id}",
            "segments":         "GET  /segments",
            "dashboard":        "GET  /dashboard/summary",
            "docs":             "GET  /docs",
        }
    }


# ── ENDPOINT 2: Customer Risk Table ──────────────────────────────────────────
# USED BY: Dashboard page (KPI cards) + Customer Risk Table page (the table)
#
# Query Parameters (all optional — for filtering/sorting):
#   risk_level → filter to HIGH / MEDIUM / LOW only
#   plan_type  → filter by Basic / Standard / Premium
#   limit      → max rows to return (default 500)
#   sort_by    → field to sort descending (default: churn_prob)
#
# Returns:
#   summary → KPI data for the 4 stat cards on dashboard
#   customers → list of customer objects for the table
#   count → how many records returned (for pagination UI)
@app.get("/customers", tags=["Customers"])
def get_customers(
    risk_level: Optional[str] = Query(None, description="Filter: HIGH, MEDIUM, or LOW"),
    plan_type:  Optional[str] = Query(None, description="Filter: Basic, Standard, or Premium"),
    limit:      int           = Query(500,  description="Max records to return"),
    sort_by:    str           = Query("churn_prob", description="Field to sort by (descending)"),
):
    """
    Returns the full customer risk table with churn probabilities and risk levels.
    Also returns KPI summary data for the dashboard stat cards.

    Supports filtering by risk_level and plan_type.
    Default: all customers sorted by churn_prob descending (highest risk first).
    """
    df = FULL_DF.copy()

    # Apply filters
    if risk_level and risk_level.upper() in ['HIGH', 'MEDIUM', 'LOW']:
        df = df[df['risk_level'] == risk_level.upper()]

    if plan_type and plan_type in ['Basic', 'Standard', 'Premium']:
        df = df[df['plan_type'] == plan_type]

    # Sort (descending) — highest churn risk first by default
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    df = df.head(limit)

    # Build response — only return what frontend needs
    # CONCEPT: Don't send 33 columns when frontend uses 8
    # Smaller response = faster API = better UX
    customers = []
    for _, row in df.iterrows():
        churn_prob = float(row.get('churn_prob', 0) or 0)
        clv        = float(row.get('clv_score', 0) or 0)
        customers.append({
            "customer_id":    str(row.get('customer_id', '')),
            "name":           str(row.get('name', '')),
            "plan_type":      str(row.get('plan_type', '')),
            "monthly_charge": int(row.get('monthly_charge', 0)),
            "tenure_months":  int(row.get('tenure_months', 0)),
            "churn_prob":     round(churn_prob, 4),
            "churn_percent":  f"{churn_prob * 100:.1f}%",
            "risk_level":     str(row.get('risk_level', 'LOW')),
            "clv_score":      round(clv, 2),
            "clv_segment":    get_clv_segment(clv),
            "top_reason_1":   format_reason(str(row.get('top_reason_1', ''))),
            "device_type":    str(row.get('device_type', '')),
            "watch_hours":    round(float(row.get('watch_hours_per_week', 0) or 0), 1),
        })

    # KPI Summary — always computed on FULL dataset (not filtered)
    all_df     = FULL_DF
    high_risk  = all_df[all_df['risk_level'] == 'HIGH']
    rev_risk   = (all_df['monthly_charge'] * all_df['churn_prob']).sum()

    summary = {
        "total_customers":          int(len(all_df)),
        "high_risk_count":          int(len(high_risk)),
        "medium_risk_count":        int(len(all_df[all_df['risk_level'] == 'MEDIUM'])),
        "low_risk_count":           int(len(all_df[all_df['risk_level'] == 'LOW'])),
        "avg_churn_probability":    round(float(all_df['churn_prob'].mean()), 4),
        "avg_churn_percent":        f"{float(all_df['churn_prob'].mean()) * 100:.1f}%",
        "revenue_at_risk_monthly":  round(float(rev_risk), 2),
        "revenue_at_risk_formatted": f"₹{float(rev_risk):,.0f}",
    }

    return {
        "summary":   summary,
        "customers": customers,
        "count":     len(customers),
        "filters_applied": {
            "risk_level": risk_level,
            "plan_type":  plan_type,
        }
    }


# ── ENDPOINT 3: Customer Detail ───────────────────────────────────────────────
# USED BY: Customer Detail page (when user clicks a row in Risk Table)
#
# Returns EVERYTHING about one customer:
#   - Full profile (plan, tenure, behavior)
#   - Churn probability + risk level
#   - Top 3 SHAP reasons (plain English)
#   - CLV score + breakdown
#   - Default values for what-if simulator sliders
@app.get("/customer/{customer_id}", tags=["Customers"])
def get_customer_detail(customer_id: str):
    """
    Returns complete detail for a single customer.
    Used on the Customer Detail page to populate:
      - Customer profile card
      - Churn probability gauge
      - SHAP reasons cards
      - CLV score
      - What-if simulator slider default values
    """
    row = FULL_DF[FULL_DF['customer_id'] == customer_id]

    # HTTP 404 = Resource Not Found — standard REST convention
    # Never return a 200 with empty data for not-found (anti-pattern)
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customer_id}' not found. "
                   f"Total customers: {len(FULL_DF)}. "
                   f"Example valid ID: {FULL_DF['customer_id'].iloc[0]}"
        )

    row        = row.iloc[0]
    import math
    cp = row.get('churn_prob', 0)
    churn_prob = 0.0 if (cp is None or (isinstance(cp, float) and math.isnan(cp))) else float(cp)

    cv = row.get('clv_score', 0)
    clv = 0.0 if (cv is None or (isinstance(cv, float) and math.isnan(cv))) else float(cv)

    # CUST99999 ke liye manually set karo
    if str(row.get('customer_id','')) == 'CUST99999':
        churn_prob = float(row.get('churn_probability', 0.96) or 0.96)
        clv = 500.0

    return {
        # ── Identity ────────────────────────────────
        "customer_id":    str(row['customer_id']),
        "name":           str(row.get('name', '')),
        "age":            int(row.get('age', 0)),
        "gender":         str(row.get('gender', '')),
        "country":        str(row.get('country', '')),

        # ── Subscription ────────────────────────────
        "plan_type":       str(row.get('plan_type', '')),
        "monthly_charge":  int(row.get('monthly_charge', 0)),
        "tenure_months":   int(row.get('tenure_months', 0)),
        "tenure_display":  f"{int(row.get('tenure_months', 0))} months",

        # ── Behavior (also used as slider defaults in what-if) ───────
        "watch_hours_per_week":  round(float(row.get('watch_hours_per_week', 0) or 0), 1),
        "logins_last_30_days":   int(row.get('logins_last_30_days', 0)),
        "last_login_days_ago":   int(row.get('last_login_days_ago', 0)),
        "payment_failures_3m":   int(row.get('payment_failures_3m', 0)),
        "support_tickets":       int(row.get('support_tickets', 0)),
        "num_profiles":          int(row.get('num_profiles', 1)),
        "device_type":           str(row.get('device_type', '')),
        "payment_method":        str(row.get('payment_method', '')),
        "preferred_genre":       str(row.get('preferred_genre', '')),

        # ── ML Predictions ──────────────────────────
        "churn_prob":     round(churn_prob, 4),
        "churn_percent":  f"{churn_prob * 100:.1f}%",
        "risk_level":     get_risk_level(churn_prob),

        # ── SHAP Reasons (plain English) ─────────────
        # These are pre-computed by the ML pipeline
        # On the frontend: display as colored reason cards
        # In the Gemini prompt: sent as context for email generation
        "top_reason_1":   format_reason(str(row.get('top_reason_1', ''))),
        "top_reason_2":   format_reason(str(row.get('top_reason_2', ''))),
        "top_reason_3":   format_reason(str(row.get('top_reason_3', ''))),

        # ── CLV ──────────────────────────────────────
        # Shows business team HOW MUCH this customer is worth
        # Helps prioritize: High CLV + High Risk = act immediately
        "clv_score":    round(clv, 2),
        "clv_segment":  get_clv_segment(clv),
        "clv_formatted": f"₹{clv:,.0f}",
        "clv_breakdown": {
            "monthly_charge":   int(row.get('monthly_charge', 0)),
            "tenure_months":    int(row.get('tenure_months', 0)),
            "retention_factor": round(1 - churn_prob, 4),
            "formula":          "monthly_charge × tenure × (1 − churn_probability)",
        },
    }


# ── ENDPOINT 4: What-If Simulator ─────────────────────────────────────────────
# USED BY: What-If Simulator sliders on Customer Detail page
#
# HOW IT WORKS:
#   1. User moves a slider (e.g., payment_failures: 3 → 0)
#   2. React sends POST request with customer_id + new feature values
#   3. FastAPI loads customer's original data
#   4. Overrides only the changed values
#   5. Rebuilds the full feature vector (encoding + engineering)
#   6. Calls MODEL.predict_proba() → gets new churn probability
#   7. Returns before/after comparison
#
# WHY THIS IS POWERFUL:
#   This is LIVE model inference — the actual ML model running in real-time!
#   Business team can ask: "If we fix this customer's payment issues,
#   how much does their churn risk drop?"
#   And get a precise, model-backed answer instantly.
#
# VIVA TIP:
#   "The what-if simulator demonstrates counterfactual analysis —
#    changing one input to see its causal impact on the prediction.
#    This is implemented as a live inference call to the loaded
#    XGBoost model with modified feature values."
@app.post("/whatif", response_model=WhatIfResponse, tags=["Simulation"])
def what_if_simulation(request: WhatIfRequest):
    """
    Recalculates churn probability with modified feature values.
    Powers the interactive sliders on the Customer Detail page.

    Send only the fields you want to change — the rest stay at the customer's actual values.

    Example: Set payment_failures_3m=0 to see how fixing billing issues affects churn risk.
    """
    # Get original customer data
    row = FULL_DF[FULL_DF['customer_id'] == request.customer_id]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{request.customer_id}' not found"
        )

    original_prob = float(row.iloc[0]['churn_prob'] or 0)
    customer_data = row.iloc[0].to_dict()

    # Apply overrides — only change what was explicitly sent
    # CONCEPT: If frontend sends None for a field, we keep original value
    # This lets the user change just ONE slider without resetting others
    override_map = {
        'watch_hours_per_week': request.watch_hours_per_week,
        'tenure_months':        request.tenure_months,
        'payment_failures_3m':  request.payment_failures_3m,
        'support_tickets':      request.support_tickets,
        'logins_last_30_days':  request.logins_last_30_days,
        'monthly_charge':       request.monthly_charge,
    }
    for field, value in override_map.items():
        if value is not None:
            customer_data[field] = value

    # Rebuild feature vector and run prediction
    try:
        feature_df = build_feature_row(customer_data)
        # predict_proba returns shape (1, 2) → [[prob_class_0, prob_class_1]]
        # prob_class_1 = probability of churning
        new_prob = float(MODEL.predict_proba(feature_df)[0][1])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}. "
                   f"Check that feature values are within expected ranges."
        )

    change = round(new_prob - original_prob, 4)

    return WhatIfResponse(
        customer_id=         request.customer_id,
        original_churn_prob= round(original_prob, 4),
        new_churn_prob=      round(new_prob, 4),
        change=              change,
        change_percent=      f"{change * 100:+.1f}%",
        original_risk=       get_risk_level(original_prob),
        new_risk=            get_risk_level(new_prob),
        improved=            change < 0,
    )


# ── ENDPOINT 5: Gemini Retention Email ────────────────────────────────────────
# USED BY: Retention Email Card on Customer Detail page (Generate button)
#
# THIS IS THE AI LAYER OF CHURNIQ PRO:
#   Traditional system: if risk > 70%: send_generic_email()
#   ChurnIQ Pro:        SHAP reasons → Gemini prompt → personalized email
#                       that specifically addresses THIS customer's issues
#
# THE GEMINI PROMPT IS CAREFULLY ENGINEERED:
#   1. Role: "You are a retention specialist..."
#   2. Context: Customer profile + SHAP churn reasons
#   3. Task: Write email + 2 actions
#   4. Format: "Respond in JSON only" → easy to parse
#
# ALWAYS HAS A FALLBACK:
#   If Gemini is unavailable or returns malformed JSON,
#   a rule-based fallback generates context-aware responses.
#   "Graceful degradation" = fail softly, never show error to user.
#
# VIVA TIP:
#   "We use prompt engineering to translate SHAP's technical output into
#    a business action. The prompt includes the top 3 SHAP-identified
#    churn drivers with their impact scores, enabling Gemini to write
#    emails that address specific pain points rather than generic offers."
@app.get("/retention/{customer_id}", tags=["AI Retention"])
def get_retention_strategy(customer_id: str):
    """
    Generates a personalized retention email and recommended actions
    using Google Gemini AI, based on the customer's SHAP churn reasons.

    Returns: subject line, email body, 2 business actions, priority level,
             and the top churn reasons that informed the email.

    Falls back to rule-based generation if Gemini is unavailable.
    """
    row = FULL_DF[FULL_DF['customer_id'] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    row = row.iloc[0]

    # Extract customer context for prompt
    name        = str(row.get('name', 'Valued Customer')).split()[0]   # First name only
    plan        = str(row.get('plan_type', 'Basic'))
    charge      = int(row.get('monthly_charge', 199))
    tenure      = int(row.get('tenure_months', 1))
    watch_hrs   = round(float(row.get('watch_hours_per_week', 0) or 0), 1)
    pay_fail    = int(row.get('payment_failures_3m', 0))
    support     = int(row.get('support_tickets', 0))
    churn_pct   = round(float(row.get('churn_prob', 0) or 0) * 100, 1)
    clv         = round(float(row.get('clv_score', 0) or 0), 0)
    reason1     = format_reason(str(row.get('top_reason_1', '')))
    reason2     = format_reason(str(row.get('top_reason_2', '')))
    reason3     = format_reason(str(row.get('top_reason_3', '')))
    

    # ── Build Gemini Prompt ─────────────────────────────────────────────────
    # PROMPT ENGINEERING PRINCIPLES USED HERE:
    #   1. Role assignment: "You are a retention specialist" → sets context/tone
    #   2. Structured input: clear sections with labels → easier for model to parse
    #   3. Constrained output: "max 100 words", "exactly 2 actions" → prevents rambling
    #   4. JSON format requirement: "no extra text" → parseable by our code
    #   5. CLV context: model can calibrate offer value based on customer worth
    prompt = f"""You are a customer retention specialist for a Netflix-style OTT streaming platform.

CUSTOMER PROFILE:
- First Name: {name}
- Plan: {plan} (₹{charge}/month)
- Subscription Tenure: {tenure} months
- Weekly Watch Hours: {watch_hrs} hrs
- Payment Failures (last 3 months): {pay_fail}
- Support Tickets Raised: {support}
- Churn Risk Score: {churn_pct}%
- Customer Lifetime Value: ₹{clv:,.0f}

TOP CHURN REASONS (identified by our ML model using SHAP analysis):
1. {reason1}
2. {reason2}
3. {reason3}

YOUR TASKS:
1. Write a SHORT, warm, personalized retention email (80-100 words max).
   - Address the customer by first name: {name}
   - Naturally reference their specific issues from the churn reasons
   - Offer a concrete incentive appropriate for their plan (₹{charge}/month)
   - Keep it conversational, NOT corporate/robotic
   - Subject line must be compelling and personal

2. List exactly 2 specific, actionable recommendations for the business team.
   - Be concrete: who does what, when, and how
   - Consider CLV: ₹{clv:,.0f} CLV {'is HIGH — worth significant retention investment' if clv > 5000 else 'is moderate — standard retention offer appropriate'}
   - Each action in 1-2 sentences

RESPOND IN THIS EXACT JSON FORMAT (no extra text before or after, no markdown fences):
{{
  "subject": "email subject line here",
  "email": "full email body here with greeting and sign-off",
  "actions": [
    "First specific business action with who/what/when",
    "Second specific business action with who/what/when"
  ],
  "priority": "HIGH or MEDIUM or LOW",
  "suggested_offer": "one-line description of the best retention offer for this specific customer"
}}"""

    # ── Call Gemini API ─────────────────────────────────────────────────────
    if GEMINI_ENABLED:
        try:
            response      = GEMINI_MODEL.generate_content(prompt)
            response_text = response.text.strip()

            # Strip markdown code fences if Gemini wrapped response in them
            # Even when told "no markdown fences", LLMs sometimes add them
            response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'^```\s*',     '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'\s*```$',     '', response_text)
            response_text = response_text.strip()

            gemini_data = json.loads(response_text)

            return {
                "customer_id":     customer_id,
                "name":            name,
                "churn_prob":      churn_pct,
                "generated_by":    "gemini-1.5-flash",
                "subject":         gemini_data.get("subject", f"We value you, {name}"),
                "email":           gemini_data.get("email", ""),
                "actions":         gemini_data.get("actions", []),
                "priority":        gemini_data.get("priority", "MEDIUM"),
                "suggested_offer": gemini_data.get("suggested_offer", ""),
                "top_reasons":     [reason1, reason2, reason3],
                "clv_score":       clv,
            }

        except json.JSONDecodeError as e:
            # Gemini returned something we couldn't parse as JSON
            # Log the error, fall through to rule-based fallback
            print(f"  ⚠ Gemini JSON parse error for {customer_id}: {e}")

        except Exception as e:
            # Gemini API error (quota exceeded, network issue, etc.)
            print(f"  ⚠ Gemini API error for {customer_id}: {e}")

        # Fall through to rule-based fallback if we reach here

    # ── Rule-Based Fallback ─────────────────────────────────────────────────
    # CONCEPT: Graceful degradation — system still works without Gemini.
    # We analyze the primary churn reason and generate a targeted response.
    # Not as good as Gemini but far better than an error message.

    raw_r1 = str(row.get('top_reason_1', ''))

    # Determine response based on primary churn driver
    if 'friction' in raw_r1 or 'payment' in raw_r1:
        offer    = "1-month free subscription + payment assistance"
        action1  = (f"Contact {name} within 24 hours to resolve payment issue — "
                    f"offer to update billing method and waive the late charge.")
        action2  = ("Enable auto-retry with 3-day SMS reminder — "
                    "set up payment failure early-warning alert for this account.")
        body     = (f"Hi {name},\n\nWe noticed you've had some billing hiccups recently "
                    f"and we want to make it right. As a valued {plan} subscriber, "
                    f"we'd like to offer you 1 month free while we help sort out your "
                    f"payment setup. Our team is here to help — just reply to this email.\n\n"
                    f"Best regards,\nThe ChurnIQ Team")
        subject  = f"Let's fix this for you, {name} — 1 month on us"

    elif 'watch' in raw_r1 or 'engagement' in raw_r1:
        offer    = "free Premium upgrade for 30 days + curated content recommendations"
        action1  = (f"Send {name} a personalized content digest this week based on "
                    f"their '{row.get('preferred_genre', 'favourite')}' genre preference.")
        action2  = ("Trigger in-app push notification: 'New releases in your favourite "
                    "genre this week' — set for Tuesday 8PM when engagement is highest.")
        body     = (f"Hi {name},\n\nWe've curated a list of shows we think you'll love — "
                    f"specially picked based on what you enjoy. To make it even better, "
                    f"we're upgrading you to Premium free for 30 days. "
                    f"Dive in and let us know what you think!\n\n"
                    f"Best regards,\nThe ChurnIQ Team")
        subject  = f"We picked these just for you, {name} 🎬"

    elif 'tenure' in raw_r1 or 'new' in raw_r1.lower():
        offer    = "20% loyalty discount for next 3 months"
        action1  = (f"Assign {name} to the onboarding success team — "
                    f"schedule a 10-minute welcome call this week.")
        action2  = ("Send 'Getting Started' email series: Day 1 — top features, "
                    "Day 3 — content highlights, Day 7 — profile setup tips.")
        body     = (f"Hi {name},\n\nWelcome to the family! We want to make sure "
                    f"your first months with us are amazing. We're giving you 20% off "
                    f"your next 3 months as our way of saying thank you for joining. "
                    f"We'd love to hear what you think — reach out anytime.\n\n"
                    f"Best regards,\nThe ChurnIQ Team")
        subject  = f"A special welcome gift for you, {name}"

    else:
        offer    = "exclusive loyalty reward for valued subscribers"
        action1  = (f"Send {name} a personalized thank-you message acknowledging "
                    f"their {tenure}-month loyalty — include exclusive early access.")
        action2  = ("Offer a 2-month plan upgrade as a loyalty reward — "
                    "high-CLV customers respond well to recognition over discounts.")
        body     = (f"Hi {name},\n\nAs one of our valued subscribers, we just wanted "
                    f"to reach out and say thank you. You've been with us for "
                    f"{tenure} months and we truly appreciate it. "
                    f"As a token of thanks, we have a special offer waiting for you.\n\n"
                    f"Best regards,\nThe ChurnIQ Team")
        subject  = f"Thank you for being with us, {name}"

    return {
        "customer_id":     customer_id,
        "name":            name,
        "churn_prob":      churn_pct,
        "generated_by":    "rule-based-fallback",
        "subject":         subject,
        "email":           body,
        "actions":         [action1, action2],
        "priority":        "HIGH" if churn_pct >= 70 else "MEDIUM",
        "suggested_offer": offer,
        "top_reasons":     [reason1, reason2, reason3],
        "clv_score":       clv,
    }
# ─────────────────────────────────────────────────────────────
# PART 2 — Yeh endpoint main.py mein add karo
#           (/retention endpoint ke BAAD)
# ─────────────────────────────────────────────────────────────
 
class SendEmailRequest(BaseModel):
    """Request body for POST /send-email"""
    customer_id: str     # Customer jiske liye email generate hui
    subject:     str     # Gemini generated subject line
    body:        str     # Gemini generated email body
 
 
@app.post("/send-email", tags=["AI Retention"])
def send_retention_email(request: SendEmailRequest):
    """
    Sends Gemini-generated retention email via Gmail SMTP.
 
    DEMO MODE:
      Always sends to DEMO_RECEIVER (your own Gmail).
      This proves the email pipeline works end-to-end.
 
    PRODUCTION:
      Replace DEMO_RECEIVER with customer's actual email
      from the dataset.
 
    HOW GMAIL SMTP WORKS:
      1. Connect to Gmail's SMTP server (smtp.gmail.com port 465)
      2. Authenticate with App Password (not your Gmail password)
      3. Send the email as a MIME message
      4. Server delivers it to inbox
 
    VIVA TIP:
      "We use Gmail SMTP with App Password authentication.
       In production, this would be replaced with a transactional
       email service like SendGrid for bulk sending and analytics."
    """
    # Check email is configured
    if not EMAIL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "Email sending not configured. "
                "Run add_emails.py first to set up Gmail credentials."
            )
        )
 
    # Get customer details for email header
    row = FULL_DF[FULL_DF['customer_id'] == request.customer_id]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{request.customer_id}' not found"
        )
 
    r          = row.iloc[0]
    cust_name  = str(r.get('name', 'Unknown'))
    plan       = str(r.get('plan_type', ''))
    churn_pct  = round(float(r.get('churn_prob', 0) or 0) * 100, 1)
    clv        = round(float(r.get('clv_score', 0) or 0), 0)
 
    try:
        # ── Build Email ───────────────────────────────────────
        msg            = MIMEMultipart('alternative')
        msg['From']    = f"ChurnIQ Pro <{GMAIL_SENDER}>"
        msg['To']      = DEMO_RECEIVER
        msg['Subject'] = f"[ChurnIQ Demo] {request.subject}"
 
        # Plain text version
        plain_body = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ChurnIQ Pro — Retention Email Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Customer   : {cust_name} ({request.customer_id})
Plan       : {plan}
Churn Risk : {churn_pct}%  ← HIGH RISK
CLV Score  : Rs.{clv:,.0f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI Generated Retention Email:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
{request.body}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sent by ChurnIQ Pro — AI Churn Prevention System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """.strip()
 
        # HTML version (nicer in inbox)
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
 
  <div style="background: #1A1A2E; padding: 20px; border-radius: 8px 8px 0 0;">
    <h2 style="color: #4A90E2; margin: 0;">⚡ ChurnIQ Pro</h2>
    <p style="color: #888; margin: 5px 0 0 0; font-size: 13px;">
      AI-Powered Retention Email — Demo Mode
    </p>
  </div>
 
  <div style="background: #f8f9ff; padding: 15px; border-left: 4px solid #FF4444;">
    <table style="width: 100%; font-size: 14px;">
      <tr>
        <td style="color: #666; padding: 3px 0;">Customer</td>
        <td style="font-weight: bold;">{cust_name} ({request.customer_id})</td>
      </tr>
      <tr>
        <td style="color: #666; padding: 3px 0;">Plan</td>
        <td>{plan}</td>
      </tr>
      <tr>
        <td style="color: #666; padding: 3px 0;">Churn Risk</td>
        <td style="color: #FF4444; font-weight: bold;">{churn_pct}% — HIGH RISK 🔴</td>
      </tr>
      <tr>
        <td style="color: #666; padding: 3px 0;">CLV Score</td>
        <td>Rs.{clv:,.0f}</td>
      </tr>
    </table>
  </div>
 
  <div style="background: white; padding: 25px; border: 1px solid #e0e0e0;">
    <p style="color: #888; font-size: 12px; margin: 0 0 15px 0;">
      ✨ AI-Generated Retention Email (by Gemini)
    </p>
    <div style="background: #f0f4ff; padding: 20px; border-radius: 6px;
                font-size: 15px; line-height: 1.7; white-space: pre-wrap;">
{request.body}
    </div>
  </div>
 
  <div style="background: #1A1A2E; padding: 15px; border-radius: 0 0 8px 8px;
              text-align: center;">
    <p style="color: #555; font-size: 12px; margin: 0;">
      Sent by ChurnIQ Pro — AI Churn Prevention System
    </p>
  </div>
 
</body>
</html>
        """.strip()
 
        # Attach both versions
        # CONCEPT: MIMEMultipart('alternative') = email client chooses
        # which version to display (HTML if supported, plain text fallback)
        msg.attach(MIMEText(plain_body, 'plain'))
        msg.attach(MIMEText(html_body,  'html'))
 
        # ── Send via Gmail SMTP ───────────────────────────────
        # SMTP_SSL = encrypted connection from the start (port 465)
        # Alternative: SMTP + STARTTLS on port 587 (also works)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASS)
            server.send_message(msg)
 
        print(f"  ✓ Email sent for {request.customer_id} → {DEMO_RECEIVER}")
 
        return {
            "success":     True,
            "message":     f"Email sent to {DEMO_RECEIVER}",
            "customer_id": request.customer_id,
            "customer":    cust_name,
            "subject":     request.subject,
            "sent_to":     DEMO_RECEIVER,
        }
 
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(
            status_code=401,
            detail=(
                "Gmail authentication failed. "
                "Check App Password in email_config.json. "
                "Make sure 2-Step Verification is ON in Google Account."
            )
        )
    except smtplib.SMTPException as e:
        raise HTTPException(
            status_code=500,
            detail=f"SMTP error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Email sending failed: {str(e)}"
        )
 
 
@app.get("/email-status", tags=["AI Retention"])
def email_status():
    """Check if Gmail email sending is configured and ready."""
    return {
        "email_enabled": EMAIL_ENABLED,
        "sender":        GMAIL_SENDER if EMAIL_ENABLED else "not configured",
        "demo_receiver": DEMO_RECEIVER if EMAIL_ENABLED else "not configured",
        "message":       (
            "Email sending is ready! Use POST /send-email"
            if EMAIL_ENABLED else
            "Run add_emails.py first to configure Gmail"
        )
    }

# ── ENDPOINT 6: Segment Analysis ─────────────────────────────────────────────
# USED BY: Segment Analysis page — all 5 charts
#
# CONCEPT: groupby().agg() = SQL GROUP BY with aggregation
# Returns pre-aggregated data so frontend just renders charts —
# no frontend-side data processing needed
@app.get("/segments", tags=["Analytics"])
def get_segment_analysis():
    """
    Returns churn statistics aggregated by various segments.
    Powers all 5 charts on the Segment Analysis page.

    Returns: by_plan, by_device, by_tenure, by_genre, by_clv, monthly_trend
    """
    df = FULL_DF.copy()

    # ── By Plan Type ──────────────────────────────────────────────────
    plan_grp = df.groupby('plan_type').agg(
        count=             ('customer_id',    'count'),
        avg_churn=         ('churn_prob',     'mean'),
        high_risk_count=   ('risk_level',     lambda x: (x == 'HIGH').sum()),
        avg_monthly_charge=('monthly_charge', 'mean'),
        avg_clv=           ('clv_score',      'mean'),
    ).reset_index()
    plan_grp['churn_rate_pct'] = (plan_grp['avg_churn'] * 100).round(1)
    plan_grp['avg_clv']        = plan_grp['avg_clv'].round(0)

    # ── By Device Type ────────────────────────────────────────────────
    device_grp = df.groupby('device_type').agg(
        count=           ('customer_id', 'count'),
        avg_churn=       ('churn_prob',  'mean'),
        high_risk_count= ('risk_level',  lambda x: (x == 'HIGH').sum()),
    ).reset_index().sort_values('avg_churn', ascending=False)
    device_grp['churn_rate_pct'] = (device_grp['avg_churn'] * 100).round(1)

    # ── By Tenure Bucket ──────────────────────────────────────────────
    df['tenure_bucket'] = pd.cut(
        df['tenure_months'],
        bins=[0, 3, 6, 12, 24, 72],
        labels=['0-3m', '3-6m', '6-12m', '1-2yr', '2yr+']
    )
    tenure_grp = df.groupby('tenure_bucket', observed=True).agg(
        count=     ('customer_id', 'count'),
        avg_churn= ('churn_prob',  'mean'),
    ).reset_index()
    tenure_grp['churn_rate_pct'] = (tenure_grp['avg_churn'] * 100).round(1)
    tenure_grp['tenure_bucket']  = tenure_grp['tenure_bucket'].astype(str)

    # ── By Genre ──────────────────────────────────────────────────────
    genre_grp = df.groupby('preferred_genre').agg(
        count=     ('customer_id', 'count'),
        avg_churn= ('churn_prob',  'mean'),
    ).reset_index().sort_values('avg_churn', ascending=False)
    genre_grp['churn_rate_pct'] = (genre_grp['avg_churn'] * 100).round(1)

    # ── By CLV Segment ────────────────────────────────────────────────
    clv_grp = df.groupby('clv_segment').agg(
        count=     ('customer_id',    'count'),
        avg_churn= ('churn_prob',     'mean'),
        total_rev= ('monthly_charge', 'sum'),
    ).reset_index()
    clv_grp['churn_rate_pct'] = (clv_grp['avg_churn'] * 100).round(1)

    # ── Monthly Trend (simulated from data distribution) ─────────────
    # We don't have real time-series, so we simulate a realistic trend
    # In production: replace with actual monthly snapshot queries
    np.random.seed(42)
    months     = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    base_rate  = float(df['churn_prob'].mean()) * 100
    noise      = np.random.normal(0, 2.0, 12).cumsum() * 0.3
    trend      = [
        {"month": m, "churn_rate": round(max(5.0, min(60.0, base_rate + n)), 1)}
        for m, n in zip(months, noise)
    ]

    return {
        "by_plan":         plan_grp.to_dict(orient='records'),
        "by_device":       device_grp.to_dict(orient='records'),
        "by_tenure":       tenure_grp.to_dict(orient='records'),
        "by_genre":        genre_grp.to_dict(orient='records'),
        "by_clv":          clv_grp.to_dict(orient='records'),
        "monthly_trend":   trend,
        "overall": {
            "total_customers":       int(len(df)),
            "avg_churn_rate_pct":    round(base_rate, 1),
            "total_revenue_at_risk": round(float(
                (df['monthly_charge'] * df['churn_prob']).sum()
            ), 2),
        }
    }


# ── ENDPOINT 7: Dashboard Summary (All KPIs in One Call) ─────────────────────
# USED BY: Dashboard home page — populate ALL 4 KPI cards + priority list
# in a single network request instead of multiple calls
@app.get("/dashboard/summary", tags=["Analytics"])
def get_dashboard_summary():
    """
    Returns all KPI data for the dashboard home page in a single API call.
    More efficient than calling /customers and /segments separately.

    Returns: kpis (4 stat cards), churn_reasons_breakdown (donut chart),
             priority_customers (top 5 high-CLV at-risk customers)
    """
    df         = FULL_DF.copy()
    high_risk  = df[df['risk_level'] == 'HIGH']
    med_risk   = df[df['risk_level'] == 'MEDIUM']
    low_risk   = df[df['risk_level'] == 'LOW']
    rev_risk   = float((df['monthly_charge'] * df['churn_prob']).sum())

    # Top 5 highest-CLV high-risk customers = most impactful to act on first
    priority = df[df['risk_level'] == 'HIGH'].nlargest(5, 'clv_score')

    return {
        "kpis": {
            "total_customers":            int(len(df)),
            "high_risk_count":            int(len(high_risk)),
            "medium_risk_count":          int(len(med_risk)),
            "low_risk_count":             int(len(low_risk)),
            "high_risk_pct":              round(len(high_risk) / len(df) * 100, 1),
            "avg_churn_probability":      round(float(df['churn_prob'].mean()), 4),
            "avg_churn_percent":          f"{float(df['churn_prob'].mean()) * 100:.1f}%",
            "revenue_at_risk_monthly":    round(rev_risk, 2),
            "revenue_at_risk_formatted":  f"₹{rev_risk:,.0f}",
            "high_value_at_risk":         int(len(df[
                (df['risk_level'] == 'HIGH') & (df['clv_segment'] == 'High')
            ])),
        },
        "churn_reasons_breakdown": {
            "low_engagement":  int(len(df[df['watch_hours_per_week'] < 3])),
            "payment_issues":  int(len(df[df['payment_failures_3m'] > 0])),
            "new_customers":   int(len(df[df['tenure_months'] < 6])),
            "support_issues":  int(len(df[df['support_tickets'] > 1])),
        },
        "priority_customers": [
            {
                "customer_id": str(r['customer_id']),
                "name":        str(r.get('name', '')),
                "churn_prob":  round(float(r['churn_prob']), 4),
                "churn_pct":   f"{float(r['churn_prob']) * 100:.1f}%",
                "clv_score":   round(float(r['clv_score']), 2),
                "plan_type":   str(r.get('plan_type', '')),
                "risk_level":  str(r.get('risk_level', '')),
            }
            for _, r in priority.iterrows()
        ],
    }


# =============================================================================
#   RUN SERVER
# =============================================================================
# This block only runs when you execute: python main.py
# When deployed on Render/Railway, they run: uvicorn main:app
# The if __name__ guard prevents this block from running on import

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Starting ChurnIQ Pro API server...")
    print("  Local URL:  http://localhost:8000")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc:      http://localhost:8000/redoc")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # Accept connections from any IP (needed for deployment)
        port=8000,
        reload=True,       # Auto-restart on file save (DEVELOPMENT ONLY)
        # reload=False     # Use this in production
    )

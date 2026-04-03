# ChurnIQ Pro

> **AI-powered OTT customer churn prediction and retention intelligence system**

ChurnIQ Pro goes beyond basic churn prediction — it tells you **WHO** will churn, **WHY** they will churn, and **WHAT TO DO** about it, all in real time through a professional analytics dashboard.

---

## 🚀 Live Demo

| **Deployed** | `https://churn-iq-frontend-five.vercel.app/` |
---


##  What Makes This Different

| Traditional Churn Tools | ChurnIQ Pro |
|---|---|
| Outputs a probability number | Explains WHY with SHAP values |
| Generic retention offers | Personalized AI emails via Gemini |
| No prioritization | CLV scoring to prioritize who to save |
| Static reports | Live what-if simulator |
| Technical ML output | Plain English explanations |

---

##  Features

###  Churn Prediction Engine
- **XGBoost** gradient boosting model trained on 5,000 OTT customers
- **SMOTE** applied to handle class imbalance (training set only)
- Predicts churn probability from 0–100% per customer
- Achieves **ROC-AUC: 0.72** on held-out test set

###  Explainable AI (SHAP)
- **SHAP TreeExplainer** computes per-customer feature contributions
- Global beeswarm plot — which features matter most overall
- Individual waterfall plots — exactly why THIS customer will churn
- Top 3 churn reasons in plain English for every customer

###  Customer Lifetime Value (CLV)
- Formula: `monthly_charge × tenure × (1 − churn_probability)`
- Segments customers into High / Medium / Low value
- Helps prioritize: High CLV + High Risk = act immediately

###  Gemini AI Retention Emails
- SHAP reasons → structured Gemini prompt → personalized email
- Returns subject line, email body, 2 business actions, suggested offer
- Rule-based fallback if Gemini is unavailable
- Sent live via **Gmail SMTP** directly from the dashboard

###  Interactive Dashboard
- KPI cards: Total customers, High risk count, Revenue at risk, Avg churn %
- Churn trend line chart, Reasons donut chart, Plan type bar chart
- Sortable, filterable customer risk table with color-coded risk badges
- **What-if simulator**: move sliders → live churn % recalculation
- Segment analysis: by plan, device, tenure, genre
- Light / Dark mode toggle

---

##  Tech Stack

### Machine Learning
| Tool | Purpose |
|---|---|
| Python | Core language |
| XGBoost | Gradient boosting classifier |
| SHAP | Explainable AI (Shapley values) |
| Scikit-learn | Preprocessing, metrics, train-test split |
| imbalanced-learn | SMOTE for class balancing |
| Pandas + NumPy | Data manipulation |
| Faker | Synthetic dataset generation |
| Matplotlib + Seaborn | EDA visualization |

### Backend
| Tool | Purpose |
|---|---|
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pydantic | Request/response validation |
| Google Gemini API | AI retention email generation |
| Gmail SMTP | Live email delivery |

### Frontend
| Tool | Purpose |
|---|---|
| React 18 | UI framework |
| Vite | Build tool |
| Tailwind CSS | Utility-first styling |
| Recharts | Data visualization charts |
| React Router v6 | Client-side routing |
| Lucide React | Icons |

### Deployment
| Service | Purpose |
|---|---|
| Render | Backend hosting (free tier) |
| Vercel | Frontend hosting (free tier) |
| GitHub | Version control + CI/CD |

---

## 📁 Project Structure

```
ChurnIQ/
├── churniq_ml_pipeline.py    # Complete ML pipeline (Parts A–H)
├── main.py                   # FastAPI backend + Gemini + Email
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version for Render
├── add_emails.py             # Setup script for demo customer
├── fix_cust99999.py          # Adds demo customer to predictions
├── ott_churn_dataset.csv     # Synthetic OTT dataset (5000 rows)
├── customer_predictions.csv  # Precomputed predictions + SHAP reasons
├── churniq_model.pkl         # Trained XGBoost model
├── feature_names.pkl         # Feature column names for inference
└── frontend/
    ├── src/
    │   ├── App.jsx           # Main layout + routing + dark mode
    │   ├── api/churniq.js    # All API calls
    │   └── pages/
    │       ├── Dashboard.jsx
    │       ├── RiskTable.jsx
    │       ├── CustomerDetail.jsx
    │       ├── SegmentAnalysis.jsx
    │       └── ChurnPredict.jsx
    ├── package.json
    └── vite.config.js
```

---

## ⚙️ ML Pipeline (churniq_ml_pipeline.py)

```
Part A → Synthetic Dataset Generation  (5000 OTT customers, 21 features)
Part B → Exploratory Data Analysis      (6-chart EDA dashboard)
Part C → Preprocessing                  (encoding, feature engineering)
Part D → SMOTE                          (class imbalance fix, train only)
Part E → XGBoost Training               (200 trees, depth 5, lr 0.1)
Part F → Model Evaluation               (ROC-AUC, confusion matrix)
Part G → SHAP                           (global + per-customer explanations)
Part H → Predictions Table              (churn_prob + top 3 SHAP reasons)
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/customers` | Full risk table + KPI summary |
| GET | `/customer/{id}` | Single customer detail + SHAP |
| POST | `/whatif` | Live what-if simulation |
| GET | `/retention/{id}` | Gemini AI email + actions |
| GET | `/segments` | Churn stats by segment |
| GET | `/dashboard/summary` | All KPIs in one call |
| POST | `/send-email` | Send retention email via Gmail |
| GET | `/email-status` | Check email configuration |

---

##  Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Kd1880/ChurnIQ.git
cd ChurnIQ
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run ML pipeline (generates dataset + trains model)
```bash
python churniq_ml_pipeline.py
```

### 4. Configure environment
```bash
# Set your Gemini API key
set GEMINI_API_KEY=your_key_here   # Windows
export GEMINI_API_KEY=your_key_here # Mac/Linux

# Set up demo customer + Gmail (optional)
python add_emails.py
```

### 5. Start backend
```bash
uvicorn main:app --reload --port 8000
```

### 6. Start frontend
```bash
cd frontend
npm install
npm run dev
```

##  Key Engineered Features

| Feature | Formula | Why |
|---|---|---|
| `engagement_score` | `watch_hours × logins / 30` | Single number capturing overall activity |
| `friction_score` | `payment_failures + support_tickets` | Combined negative experience signal |
| `inactivity_flag` | `1 if last_login > 14 days` | Binary signal for customers going dark |
| `is_new_customer` | `1 if tenure < 6 months` | 0–3 month customers have 72% churn rate |



##  Acknowledgements

- [SHAP](https://github.com/slundberg/shap) — Explainable AI library
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient boosting framework
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python API framework
- [Google Gemini](https://aistudio.google.com/) — AI language model
- [Recharts](https://recharts.org/) — React charting library
---

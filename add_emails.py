# =============================================================================
# ChurnIQ Pro — Add Emails to Dataset + Create Demo Customer
# =============================================================================
# Yeh script:
#   1. Sabke liye fake emails generate karta hai dataset mein
#   2. Tumhara special customer add karta hai high churn risk ke saath
#   3. Gmail SMTP config set karta hai
#
# Run: python add_emails.py
# =============================================================================

import pandas as pd
import numpy as np
import random
import os
import pickle

# ─── Step 1: Tumse input lo (terminal mein — safe!) ──────────────────────────
print("=" * 55)
print("  ChurnIQ Pro — Demo Setup")
print("=" * 55)
print("\nYeh info sirf tumhare computer pe save hogi.\n")

your_name      = input("  Tumhara naam likho (e.g. Rahul):       ").strip()
your_email     = input("  Tumhari Gmail likho:                    ").strip()
your_app_pass  = input("  Gmail App Password (16 chars):          ").strip()
sender_email   = input("  Sender Gmail (same ya alag):            ").strip()

print("\n  Processing...")

# ─── Step 2: Dataset load karo ────────────────────────────────────────────────
def find_file(filename):
    candidates = [
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"'{filename}' nahi mila. Pehle churniq_ml_pipeline.py run karo.")

df = pd.read_csv(find_file('ott_churn_dataset.csv'))

# ─── Step 3: Sabke liye fake emails generate karo ────────────────────────────
# CONCEPT: Faker se realistic fake emails — demo ke liye kaafi hain
# Real project mein yeh actual customer database se aate

random.seed(42)
np.random.seed(42)

domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
           'rediffmail.com', 'ymail.com']

def make_fake_email(name, idx):
    """Name se realistic fake email banao."""
    parts = name.lower().split()
    first = parts[0] if parts else 'user'
    last  = parts[-1] if len(parts) > 1 else str(idx)
    # Remove special chars
    first = ''.join(c for c in first if c.isalnum())
    last  = ''.join(c for c in last  if c.isalnum())
    domain = random.choice(domains)
    style  = random.choice([
        f"{first}.{last}",
        f"{first}{last}{random.randint(1,99)}",
        f"{first}_{last}",
        f"{first}{random.randint(10,999)}",
    ])
    return f"{style}@{domain}"

print("  Generating fake emails for all customers...")
emails = [
    make_fake_email(row['name'], i)
    for i, row in df.iterrows()
]
df['email'] = emails
print(f"  ✓ {len(emails)} fake emails generated")

# ─── Step 4: Tumhara special customer add/update karo ────────────────────────
# High churn risk wala customer — tumhari real email ke saath
# Yeh demo star hai!

SPECIAL_ID = 'CUST99999'

# Check karo existing customer hai ya nahi
existing = df[df['customer_id'] == SPECIAL_ID]

special_customer = {
    'customer_id':          SPECIAL_ID,
    'name':                 your_name,
    'age':                  25,
    'gender':               'Male',
    'country':              'India',
    'plan_type':            'Basic',
    'monthly_charge':       199,
    'tenure_months':        2,           # New customer — high churn risk
    'watch_hours_per_week': 1.2,         # Very low engagement
    'num_profiles':         1,
    'logins_last_30_days':  2,           # Barely logging in
    'last_login_days_ago':  22,          # Gone dark
    'preferred_genre':      'Action',
    'device_type':          'Mobile',
    'payment_method':       'UPI',
    'payment_failures_3m':  3,           # Payment issues
    'support_tickets':      2,           # Support frustration
    'clv_score':            500.0,
    'clv_segment':          'Low',
    'churn_probability':    0.96,        # 96% churn risk!
    'churn':                1,
    'email':                your_email,  # TUMHARI REAL EMAIL
}

if existing.empty:
    # Naya row add karo
    df = pd.concat([df, pd.DataFrame([special_customer])], ignore_index=True)
    print(f"  ✓ Special demo customer added: {SPECIAL_ID}")
else:
    # Update karo
    for key, val in special_customer.items():
        df.loc[df['customer_id'] == SPECIAL_ID, key] = val
    print(f"  ✓ Special demo customer updated: {SPECIAL_ID}")

print(f"    Name:         {your_name}")
print(f"    Email:        {your_email}")
print(f"    Churn Risk:   96% (HIGH)")
print(f"    Plan:         Basic — Rs.199")

# ─── Step 5: Updated dataset save karo ───────────────────────────────────────
df.to_csv('ott_churn_dataset.csv', index=False)
print(f"\n  ✓ Dataset updated with emails → ott_churn_dataset.csv")

# ─── Step 6: Gmail config file save karo ─────────────────────────────────────
# Separate config file — never hardcode credentials in main.py!
# .gitignore mein add karo isko — kabhi GitHub pe mat push karo

config = {
    'GMAIL_SENDER':   sender_email,
    'GMAIL_APP_PASS': your_app_pass,
    'DEMO_RECEIVER':  your_email,
    'SPECIAL_CUST_ID': SPECIAL_ID,
}

import json
with open('email_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"  ✓ Gmail config saved → email_config.json")
print(f"  ⚠ email_config.json ko .gitignore mein add karo!")

# ─── Step 7: .gitignore update karo ──────────────────────────────────────────
gitignore_path = '.gitignore'
gitignore_content = """
# ChurnIQ Pro — Never push these to GitHub!
email_config.json
*.pkl
*.csv
__pycache__/
.env
"""
with open(gitignore_path, 'w') as f:
    f.write(gitignore_content.strip())
print(f"  ✓ .gitignore created (email_config.json protected)")

# ─── Step 8: Instructions print karo ─────────────────────────────────────────
print(f"""
{"=" * 55}
  SETUP COMPLETE!
{"=" * 55}

  Demo Customer Ready:
    ID:     {SPECIAL_ID}
    Name:   {your_name}
    Email:  {your_email}
    Risk:   96% HIGH ← tumhara star customer!

  Next Steps:
  1. main.py mein yeh add karo (top pe imports ke baad):

     import json
     with open('email_config.json') as f:
         _cfg = json.load(f)
     GMAIL_SENDER   = _cfg['GMAIL_SENDER']
     GMAIL_APP_PASS = _cfg['GMAIL_APP_PASS']
     DEMO_RECEIVER  = _cfg['DEMO_RECEIVER']

  2. Phir server restart karo:
     python -m uvicorn main:app --reload --port 8000

  3. Demo mein:
     → Risk Table mein {SPECIAL_ID} dhundo
     → Customer Detail pe jao
     → "Generate Email" click karo
     → "Send Email" click karo
     → Tumhare inbox mein email aayega! 🎉

{"=" * 55}
""")

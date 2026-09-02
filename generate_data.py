"""
Generates THREE linked CSVs simulating a real finance-ops chain:

  razorpay_payments.csv   (Hop 1: what the customer paid)
  settlements.csv         (Hop 2: what actually landed in the bank, after fees)
  internal_ledger.csv     (Hop 3: what the merchant's books say)

Unlike a simple 2-source reconciliation, breaks here are injected at a
SPECIFIC hop and labeled with WHERE they occurred, so the tracer's
accuracy at finding the break point can be scored against ground truth
— not just "matched or not."

ground_truth.csv columns:
  txn_id, settlement_id, order_id, break_hop, label

break_hop is one of: NONE, GATEWAY, SETTLEMENT, LEDGER
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(7)

N = 60
STANDARD_FEE_PCT = 0.02  # Razorpay-style 2% take rate, rounded to 2dp

FIRST_NAMES = ["Aditya", "Priya", "Rahul", "Sneha", "Vikram", "Ananya",
               "Karan", "Divya", "Arjun", "Neha", "Rohan", "Kavya"]
LAST_NAMES = ["Sharma", "Verma", "Reddy", "Iyer", "Nair", "Gupta",
              "Kapoor", "Menon", "Rao", "Singh"]

def rand_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def std_fee(amount):
    return round(amount * STANDARD_FEE_PCT, 2)

base_date = datetime(2026, 8, 1)

payments, settlements, ledger, ground_truth = [], [], [], []

for i in range(1, N + 1):
    txn_id = f"pay_{100000 + i}"
    settlement_id = f"stl_{200000 + i}"
    order_id = f"ORD-{2000 + i}"
    name = rand_name()
    amount = round(random.uniform(199, 4999), 2)
    order_date = base_date + timedelta(days=random.randint(0, 20))
    settle_date = order_date + timedelta(days=1)  # standard T+1 settlement

    roll = random.random()

    # ---- Hop 1: gateway payment always recorded (this is our starting universe) ----
    payments.append([txn_id, name, amount, order_date.strftime("%Y-%m-%d"), "captured"])

    if roll < 0.65:
        # --- CLEAN: settlement = amount - standard fee, ledger = amount, all match ---
        fee = std_fee(amount)
        settled = round(amount - fee, 2)
        settlements.append([settlement_id, txn_id, settled, fee, settle_date.strftime("%Y-%m-%d")])
        ledger.append([order_id, name, amount, order_date.strftime("%Y-%m-%d")])
        ground_truth.append([txn_id, settlement_id, order_id, "NONE", "CLEAN"])

    elif roll < 0.80:
        # --- BREAK AT SETTLEMENT: fee is wrong/unexpected (extra deduction, GST, etc) ---
        extra_fee = round(random.uniform(15, 60), 2)
        fee = std_fee(amount) + extra_fee
        settled = round(amount - fee, 2)
        settlements.append([settlement_id, txn_id, settled, fee, settle_date.strftime("%Y-%m-%d")])
        ledger.append([order_id, name, amount, order_date.strftime("%Y-%m-%d")])  # ledger still expects full amount
        ground_truth.append([txn_id, settlement_id, order_id, "SETTLEMENT", "FEE_VARIANCE"])

    elif roll < 0.92:
        # --- BREAK AT LEDGER: settlement is clean, but ledger was never updated / wrong amount ---
        fee = std_fee(amount)
        settled = round(amount - fee, 2)
        settlements.append([settlement_id, txn_id, settled, fee, settle_date.strftime("%Y-%m-%d")])
        ledger_error = round(random.uniform(20, 100), 2)
        wrong_ledger_amount = round(amount - ledger_error, 2)  # someone recorded the wrong figure
        ledger.append([order_id, name, wrong_ledger_amount, order_date.strftime("%Y-%m-%d")])
        ground_truth.append([txn_id, settlement_id, order_id, "LEDGER", "LEDGER_NOT_UPDATED"])

    elif roll < 0.97:
        # --- NO SETTLEMENT: payment captured, but never actually settled (stuck in gateway) ---
        # no settlement row at all
        ledger.append([order_id, name, amount, order_date.strftime("%Y-%m-%d")])  # merchant still expects it
        ground_truth.append([txn_id, None, order_id, "GATEWAY", "SETTLEMENT_MISSING"])

    else:
        # --- NO LEDGER ENTRY: payment + settlement fine, but no order was ever recorded internally ---
        fee = std_fee(amount)
        settled = round(amount - fee, 2)
        settlements.append([settlement_id, txn_id, settled, fee, settle_date.strftime("%Y-%m-%d")])
        # no ledger row
        ground_truth.append([txn_id, settlement_id, None, "LEDGER", "LEDGER_MISSING"])

random.shuffle(payments)
random.shuffle(settlements)
random.shuffle(ledger)

with open("razorpay_payments.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["txn_id", "customer_name", "amount", "payment_date", "status"])
    w.writerows(payments)

with open("settlements.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["settlement_id", "txn_id", "settled_amount", "fee", "settlement_date"])
    w.writerows(settlements)

with open("internal_ledger.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order_id", "customer_name", "amount", "order_date"])
    w.writerows(ledger)

with open("ground_truth.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["txn_id", "settlement_id", "order_id", "break_hop", "label"])
    w.writerows(ground_truth)

print(f"payments:    {len(payments)} rows")
print(f"settlements: {len(settlements)} rows")
print(f"ledger:      {len(ledger)} rows")
print(f"ground_truth:{len(ground_truth)} rows")

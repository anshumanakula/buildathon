"""
Deterministic reconciliation engine.
Exact match -> fuzzy scoring -> exception classification.
No LLM calls here — this stays fast, cheap, reproducible, auditable.
LLM reasoning (agent.py) only touches what THIS can't confidently resolve.
"""

import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
def _levenshtein(a, b):
    """Pure-Python Levenshtein distance — no external dependency needed."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                prev_row[j] + 1,       # deletion
                curr_row[j - 1] + 1,   # insertion
                prev_row[j - 1] + cost # substitution
            ))
        prev_row = curr_row
    return prev_row[-1]

AMOUNT_TOLERANCE = 25.0   # rupees
DATE_WINDOW_DAYS = 3
AUTO_MATCH_THRESHOLD = 0.85
REJECT_THRESHOLD = 0.50
DUPLICATE_GAP = 0.05      # candidates within this score gap -> ambiguous

WEIGHTS = {"amount": 0.4, "date": 0.3, "name": 0.3}


@dataclass
class Payment:
    txn_id: str
    name: str
    amount: float
    date: datetime
    status: str


@dataclass
class LedgerEntry:
    order_id: str
    name: str
    amount: float
    date: datetime


@dataclass
class MatchResult:
    txn_id: Optional[str]
    order_id: Optional[str]
    method: str                # MATCHED_EXACT / MATCHED_FUZZY / EXCEPTION
    exception_type: Optional[str] = None
    confidence: Optional[float] = None
    score_breakdown: dict = field(default_factory=dict)
    candidates_considered: int = 0
    agent_verdict: Optional[str] = None
    agent_reason: Optional[str] = None


def load_payments(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append(Payment(
                txn_id=r["txn_id"],
                name=r["customer_name"],
                amount=float(r["amount"]),
                date=datetime.strptime(r["payment_date"], "%Y-%m-%d"),
                status=r["status"],
            ))
    return out


def load_ledger(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append(LedgerEntry(
                order_id=r["order_id"],
                name=r["customer_name"],
                amount=float(r["amount"]),
                date=datetime.strptime(r["order_date"], "%Y-%m-%d"),
            ))
    return out


def amount_score(a, b):
    diff = abs(a - b)
    return max(0.0, 1 - min(diff / AMOUNT_TOLERANCE, 1))


def date_score(a, b):
    diff = abs((a - b).days)
    return max(0.0, 1 - min(diff / DATE_WINDOW_DAYS, 1))


def name_score(a, b):
    dist = _levenshtein(a.lower(), b.lower())
    return max(0.0, 1 - dist / max(len(a), len(b), 1))


def score_pair(p: Payment, l: LedgerEntry):
    a = amount_score(p.amount, l.amount)
    d = date_score(p.date, l.date)
    n = name_score(p.name, l.name)
    conf = a * WEIGHTS["amount"] + d * WEIGHTS["date"] + n * WEIGHTS["name"]
    return conf, {"amount_score": round(a, 3), "date_score": round(d, 3), "name_score": round(n, 3)}


def classify_exception(best_score, breakdown, n_close_candidates):
    if best_score is None or best_score < 0.05:
        return "NO_COUNTERPART"
    if n_close_candidates >= 2:
        return "DUPLICATE_AMBIGUOUS"
    if breakdown["amount_score"] < 0.3 and breakdown["date_score"] > 0.7 and breakdown["name_score"] > 0.7:
        return "AMOUNT_MISMATCH"
    if breakdown["date_score"] < 0.3 and breakdown["amount_score"] > 0.7 and breakdown["name_score"] > 0.7:
        return "DATE_OUTSIDE_WINDOW"
    if best_score < REJECT_THRESHOLD:
        return "NO_COUNTERPART"
    if REJECT_THRESHOLD <= best_score < AUTO_MATCH_THRESHOLD:
        return "LOW_CONFIDENCE"
    return "UNCLASSIFIED"


def reconcile(payments, ledger):
    results = []
    unmatched_payments = list(payments)
    unmatched_ledger = list(ledger)

    # ---- Pass 1: exact match ----
    ledger_index = {}
    for l in unmatched_ledger:
        key = (round(l.amount, 2), l.date.date(), l.name.lower())
        ledger_index.setdefault(key, []).append(l)

    still_unmatched_payments = []
    matched_ledger_ids = set()
    for p in unmatched_payments:
        key = (round(p.amount, 2), p.date.date(), p.name.lower())
        candidates = [l for l in ledger_index.get(key, []) if l.order_id not in matched_ledger_ids]
        if candidates:
            l = candidates[0]
            matched_ledger_ids.add(l.order_id)
            results.append(MatchResult(p.txn_id, l.order_id, "MATCHED_EXACT", confidence=1.0))
        else:
            still_unmatched_payments.append(p)

    still_unmatched_ledger = [l for l in unmatched_ledger if l.order_id not in matched_ledger_ids]

    # ---- Pass 2 + 3: fuzzy match + confidence scoring ----
    matched_ledger_ids2 = set()
    remaining_payments = []
    for p in still_unmatched_payments:
        scored = []
        for l in still_unmatched_ledger:
            if l.order_id in matched_ledger_ids2:
                continue
            conf, breakdown = score_pair(p, l)
            if conf > 0.05:
                scored.append((conf, l, breakdown))
        scored.sort(key=lambda x: -x[0])

        if not scored:
            results.append(MatchResult(p.txn_id, None, "EXCEPTION",
                                        exception_type="NO_COUNTERPART",
                                        confidence=0.0, candidates_considered=0))
            continue

        best_conf, best_l, best_breakdown = scored[0]
        close_candidates = [s for s in scored if best_conf - s[0] <= DUPLICATE_GAP]

        if best_conf >= AUTO_MATCH_THRESHOLD and len(close_candidates) == 1:
            matched_ledger_ids2.add(best_l.order_id)
            results.append(MatchResult(p.txn_id, best_l.order_id, "MATCHED_FUZZY",
                                        confidence=round(best_conf, 3),
                                        score_breakdown=best_breakdown,
                                        candidates_considered=len(scored)))
        else:
            exc_type = classify_exception(best_conf, best_breakdown, len(close_candidates))
            results.append(MatchResult(p.txn_id, best_l.order_id, "EXCEPTION",
                                        exception_type=exc_type,
                                        confidence=round(best_conf, 3),
                                        score_breakdown=best_breakdown,
                                        candidates_considered=len(scored)))

    remaining_ledger = [l for l in still_unmatched_ledger if l.order_id not in matched_ledger_ids2]
    for l in remaining_ledger:
        # only add if not already referenced as a "best candidate" in an exception above
        already_referenced = any(r.order_id == l.order_id for r in results)
        if not already_referenced:
            results.append(MatchResult(None, l.order_id, "EXCEPTION",
                                        exception_type="NO_COUNTERPART",
                                        confidence=0.0, candidates_considered=0))

    return results


if __name__ == "__main__":
    payments = load_payments("razorpay_payments.csv")
    ledger = load_ledger("internal_ledger.csv")
    results = reconcile(payments, ledger)

    matched = [r for r in results if r.method in ("MATCHED_EXACT", "MATCHED_FUZZY")]
    exceptions = [r for r in results if r.method == "EXCEPTION"]

    total = len(payments)
    print(f"Total payment records: {total}")
    print(f"Matched: {len(matched)} ({len(matched)/total*100:.1f}%)")
    print(f"  - Exact:  {len([r for r in matched if r.method=='MATCHED_EXACT'])}")
    print(f"  - Fuzzy:  {len([r for r in matched if r.method=='MATCHED_FUZZY'])}")
    print(f"Exceptions: {len(exceptions)}")
    for etype in set(r.exception_type for r in exceptions):
        count = len([r for r in exceptions if r.exception_type == etype])
        print(f"  - {etype}: {count}")

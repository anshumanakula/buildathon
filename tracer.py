"""
Chain tracer — the core differentiator over plain 2-source reconciliation.

Walks each payment through its full chain:
  Hop 1: Payment  ->  Hop 2: Settlement  ->  Hop 3: Ledger

For every payment, determines a per-hop status (OK / MISMATCH / MISSING)
and identifies the FIRST hop where the chain breaks — not just "unmatched."

This is what lets the product say "settlement is fine, the break is in
your ledger" instead of "these two things don't match," which is the
entire point of doing 3 sources instead of 2.
"""

import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


FEE_TOLERANCE_PCT = 0.005   # 0.5% — how much settlement fee can vary from standard before flagging
STANDARD_FEE_PCT = 0.02     # matches the generator's assumed take rate
AMOUNT_TOLERANCE = 5.0      # rupees, for ledger vs payment amount comparison
DATE_WINDOW_DAYS = 3


@dataclass
class Payment:
    txn_id: str
    name: str
    amount: float
    date: datetime
    status: str


@dataclass
class Settlement:
    settlement_id: str
    txn_id: str
    settled_amount: float
    fee: float
    date: datetime


@dataclass
class LedgerEntry:
    order_id: str
    name: str
    amount: float
    date: datetime


@dataclass
class ChainResult:
    txn_id: str
    settlement_id: Optional[str] = None
    order_id: Optional[str] = None

    payment_status: str = "OK"       # OK (always, payment is our starting point)
    settlement_status: str = "OK"    # OK / MISMATCH / MISSING
    ledger_status: str = "OK"        # OK / MISMATCH / MISSING

    break_hop: str = "NONE"          # NONE / GATEWAY / SETTLEMENT / LEDGER
    break_detail: dict = field(default_factory=dict)
    confidence: float = 1.0
    resolution: Optional[dict] = None


def load_payments(path):
    with open(path) as f:
        return load_payments_from_stream(f)


def load_payments_from_stream(file_obj):
    out = []
    for r in csv.DictReader(file_obj):
        out.append(Payment(r["txn_id"], r["customer_name"], float(r["amount"]),
                            datetime.strptime(r["payment_date"], "%Y-%m-%d"), r["status"]))
    return out


def load_settlements(path):
    with open(path) as f:
        return load_settlements_from_stream(f)


def load_settlements_from_stream(file_obj):
    out = []
    for r in csv.DictReader(file_obj):
        out.append(Settlement(r["settlement_id"], r["txn_id"], float(r["settled_amount"]),
                               float(r["fee"]), datetime.strptime(r["settlement_date"], "%Y-%m-%d")))
    return out


def load_ledger(path):
    with open(path) as f:
        return load_ledger_from_stream(f)


def load_ledger_from_stream(file_obj):
    out = []
    for r in csv.DictReader(file_obj):
        out.append(LedgerEntry(r["order_id"], r["customer_name"], float(r["amount"]),
                                datetime.strptime(r["order_date"], "%Y-%m-%d")))
    return out


def _name_similar(a, b, threshold=0.75):
    """Cheap similarity check — shared word overlap, good enough for name matching here."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / max(len(wa), len(wb))
    return overlap >= threshold


def trace_chain(payments, settlements, ledger):
    settlements_by_txn = {s.txn_id: s for s in settlements}

    # Ledger has no direct FK to txn_id, so we match it the same way the
    # 2-hop version did: amount + date + name proximity. Build a lookup
    # pool we consume as we go so ledger entries aren't reused across chains.
    ledger_pool = list(ledger)

    def find_ledger_match(payment):
        best, best_score = None, -1
        for l in ledger_pool:
            amount_diff = abs(l.amount - payment.amount)
            date_diff = abs((l.date - payment.date).days)
            name_ok = _name_similar(l.name, payment.name)
            # score favors close amount/date, name similarity as tiebreak
            score = 0
            if amount_diff <= AMOUNT_TOLERANCE:
                score += 3
            elif amount_diff <= 100:
                score += 1
            if date_diff <= DATE_WINDOW_DAYS:
                score += 2
            if name_ok:
                score += 2
            if score > best_score:
                best_score, best = score, l
        return best, best_score

    results = []

    for p in payments:
        result = ChainResult(txn_id=p.txn_id)

        # ---- Hop 2: payment -> settlement ----
        settlement = settlements_by_txn.get(p.txn_id)
        if settlement is None:
            result.settlement_status = "MISSING"
            result.break_hop = "GATEWAY"
            result.break_detail = {"reason": "Payment captured but no matching settlement record found"}
            # can't proceed meaningfully to ledger without settlement context,
            # but we still try a ledger match for completeness
            l, score = find_ledger_match(p)
            if l and score >= 5:
                result.order_id = l.order_id
                ledger_pool.remove(l)
            results.append(result)
            continue

        result.settlement_id = settlement.settlement_id
        expected_fee = round(p.amount * STANDARD_FEE_PCT, 2)
        fee_diff = settlement.fee - expected_fee
        fee_diff_pct = abs(fee_diff) / p.amount if p.amount else 0

        if fee_diff_pct > FEE_TOLERANCE_PCT:
            result.settlement_status = "MISMATCH"
            result.break_hop = "SETTLEMENT"
            result.break_detail = {
                "expected_fee": round(expected_fee, 2),
                "actual_fee": round(settlement.fee, 2),
                "variance": round(fee_diff, 2),
                "reason": f"Settlement fee ₹{round(settlement.fee, 2)} vs expected ₹{round(expected_fee, 2)} (standard {STANDARD_FEE_PCT*100:.0f}% rate)"
            }

        # ---- Hop 3: settlement -> ledger ----
        l, score = find_ledger_match(p)
        if l is None or score < 3:
            result.ledger_status = "MISSING"
            if result.break_hop == "NONE":
                result.break_hop = "LEDGER"
                result.break_detail = {"reason": "No matching ledger entry found for this payment"}
        else:
            ledger_pool.remove(l)
            result.order_id = l.order_id
            amount_diff = abs(l.amount - p.amount)
            if amount_diff > AMOUNT_TOLERANCE:
                result.ledger_status = "MISMATCH"
                if result.break_hop == "NONE":  # only override if settlement was clean
                    result.break_hop = "LEDGER"
                    result.break_detail = {
                        "ledger_amount": l.amount,
                        "expected_amount": p.amount,
                        "variance": round(p.amount - l.amount, 2),
                        "reason": f"Ledger shows ₹{l.amount} but payment was ₹{p.amount}"
                    }

        results.append(result)

    return results


def summarize(results):
    total = len(results)
    clean = len([r for r in results if r.break_hop == "NONE"])
    by_hop = {}
    for r in results:
        if r.break_hop != "NONE":
            by_hop[r.break_hop] = by_hop.get(r.break_hop, 0) + 1
    return {
        "total": total,
        "clean": clean,
        "clean_rate_pct": round(clean / total * 100, 2) if total else 0,
        "breaks_by_hop": by_hop,
    }


if __name__ == "__main__":
    payments = load_payments("razorpay_payments.csv")
    settlements = load_settlements("settlements.csv")
    ledger = load_ledger("internal_ledger.csv")

    results = trace_chain(payments, settlements, ledger)
    summary = summarize(results)

    print(f"Total payments: {summary['total']}")
    print(f"Clean end-to-end: {summary['clean']} ({summary['clean_rate_pct']}%)")
    print(f"Breaks by hop:")
    for hop, count in summary["breaks_by_hop"].items():
        print(f"  {hop}: {count}")

    print("\n--- Sample broken chains ---")
    shown = 0
    for r in results:
        if r.break_hop != "NONE" and shown < 5:
            icons = {
                "OK": "✅", "MISMATCH": "❌", "MISSING": "⬜"
            }
            print(f"{r.txn_id}: payment {icons['OK']} → settlement {icons[r.settlement_status]} → ledger {icons[r.ledger_status]}")
            print(f"  break at: {r.break_hop} | {r.break_detail.get('reason', '')}")
            shown += 1

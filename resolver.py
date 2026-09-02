"""
Resolution drafter — the "closer" layer.

Most reconciliation tools stop at detection: "these don't match."
This stops at RESOLUTION: for each break, draft the actual fix a
finance-ops person would need to approve and apply.

Three resolution types, matched to the three break hops:

  SETTLEMENT break (fee variance)
    -> draft a journal entry adjustment

  LEDGER break (amount mismatch or missing entry)
    -> draft a ledger correction with the confirmed source-of-truth value

  GATEWAY break (settlement never happened) or genuinely ambiguous cases
    -> draft a plain-English internal query for a human to answer

Every draft is generated, never auto-applied — this stays "explainable,
bounded and gated" per the track's bar. A human approves before anything
touches real books. The agent's job is to remove the blank-page problem,
not to make unsupervised changes to financial records.
"""

import os
import json
from datetime import datetime, timezone

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "openai/gpt-oss-120b"


def log(audit_log, event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    audit_log.append(event)


def build_prompt(chain_result, payment):
    hop = chain_result.break_hop
    detail = chain_result.break_detail

    if hop == "SETTLEMENT":
        task = f"""A settlement fee doesn't match the standard rate.
Transaction: {payment.txn_id}, payment amount ₹{payment.amount}
Expected fee: ₹{detail.get('expected_fee')}
Actual fee charged: ₹{detail.get('actual_fee')}
Variance: ₹{detail.get('variance')}

Draft a journal entry to record this fee variance. Respond ONLY with JSON:
{{"resolution_type": "JOURNAL_ENTRY", "draft": "Dr./Cr. lines as plain text, under 40 words", "requires_approval": true}}"""

    elif hop == "LEDGER":
        task = f"""A ledger entry doesn't match the confirmed payment.
Transaction: {payment.txn_id}, order: {chain_result.order_id or 'not found'}
Confirmed payment amount: ₹{payment.amount}
{f"Ledger currently shows: ₹{detail.get('ledger_amount')}" if 'ledger_amount' in detail else "No ledger entry exists for this payment."}

Draft a ledger correction action. Respond ONLY with JSON:
{{"resolution_type": "LEDGER_CORRECTION", "draft": "specific correction instruction, under 40 words", "requires_approval": true}}"""

    elif hop == "GATEWAY":
        task = f"""A payment was captured but never settled.
Transaction: {payment.txn_id}, amount ₹{payment.amount}, date {payment.date.date()}

Draft an internal query to investigate this. Respond ONLY with JSON:
{{"resolution_type": "INTERNAL_QUERY", "draft": "a specific question for the payments/ops team, under 30 words", "requires_approval": true}}"""

    else:
        task = f"""An ambiguous reconciliation case needs human judgment.
Transaction: {payment.txn_id}, amount ₹{payment.amount}
Detail: {detail}

Draft a clarifying question for a human reviewer. Respond ONLY with JSON:
{{"resolution_type": "INTERNAL_QUERY", "draft": "specific question, under 30 words", "requires_approval": true}}"""

    return task


def call_groq(prompt: str, retries: int = 1):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    last_error = None
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        if not text:
            last_error = ValueError("empty response from model")
            continue

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_error = e
            continue

    raise last_error


def fallback_draft(chain_result):
    """Deterministic template used when no Groq key is configured — keeps
    the pipeline fully runnable, clearly labeled as non-LLM."""
    hop = chain_result.break_hop
    detail = chain_result.break_detail

    if hop == "SETTLEMENT":
        variance = detail.get("variance", 0)
        return {
            "resolution_type": "JOURNAL_ENTRY",
            "draft": f"Dr. Payment Gateway Fees ₹{abs(variance)}, Cr. Accounts Receivable ₹{abs(variance)} — settlement fee variance on {chain_result.txn_id}.",
            "requires_approval": True,
        }
    if hop == "LEDGER":
        if "ledger_amount" in detail:
            return {
                "resolution_type": "LEDGER_CORRECTION",
                "draft": f"Update {chain_result.order_id or 'ledger entry'} amount from ₹{detail.get('ledger_amount')} to ₹{detail.get('expected_amount')} to match confirmed payment.",
                "requires_approval": True,
            }
        return {
            "resolution_type": "LEDGER_CORRECTION",
            "draft": f"Create missing ledger entry for {chain_result.txn_id} — no order record found matching this payment.",
            "requires_approval": True,
        }
    if hop == "GATEWAY":
        return {
            "resolution_type": "INTERNAL_QUERY",
            "draft": f"Payment {chain_result.txn_id} was captured but has no settlement record — confirm with payments team whether settlement is pending or failed.",
            "requires_approval": True,
        }
    return {
        "resolution_type": "INTERNAL_QUERY",
        "draft": f"Manually review {chain_result.txn_id} — automated classification was inconclusive.",
        "requires_approval": True,
    }


def draft_resolutions(chain_results, payments_by_id, audit_log=None):
    """Adds .resolution to every ChainResult with break_hop != NONE.
    Never modifies chain_result.break_hop or auto-resolves anything —
    purely additive drafting, always gated behind requires_approval."""
    if audit_log is None:
        audit_log = []
    using_llm = bool(GROQ_API_KEY)

    log(audit_log, {"event": "resolution_drafting_start", "using_llm": using_llm,
                     "model": MODEL if using_llm else "fallback_templates"})

    for r in chain_results:
        if r.break_hop == "NONE":
            continue

        payment = payments_by_id.get(r.txn_id)
        if not payment:
            continue

        try:
            if using_llm:
                resolution = call_groq(build_prompt(r, payment))
                source = "groq_llm"
            else:
                resolution = fallback_draft(r)
                source = "fallback_template"
        except Exception as e:
            resolution = {
                "resolution_type": "INTERNAL_QUERY",
                "draft": f"Automated drafting failed ({e}) — needs manual review.",
                "requires_approval": True,
            }
            source = "error_fallback"

        r.resolution = resolution

        log(audit_log, {
            "event": "resolution_drafted",
            "source": source,
            "txn_id": r.txn_id,
            "break_hop": r.break_hop,
            "resolution_type": resolution.get("resolution_type"),
            "draft": resolution.get("draft"),
        })

    log(audit_log, {"event": "resolution_drafting_end",
                     "drafted_count": len([r for r in chain_results if getattr(r, "resolution", None)])})

    return chain_results

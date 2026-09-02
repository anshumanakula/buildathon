"""
Agentic layer on top of the deterministic matcher.

The matcher (matcher.py) does all the arithmetic — exact match, fuzzy
scoring, exception bucketing. That part is deterministic and auditable
on purpose: judges need to trust the numbers.

This agent only touches what the matcher COULDN'T confidently resolve:
  - LOW_CONFIDENCE exceptions
  - DUPLICATE_AMBIGUOUS exceptions

For each, it asks an LLM (Groq) to reason over the score breakdown and
recommend one of: MATCH / REJECT / NEEDS_HUMAN — with a plain-English
reason. Every call and verdict is logged as an audit trail.

If no GROQ_API_KEY is set, the agent falls back to a deterministic
rule-based reasoner so the pipeline still runs end-to-end.
"""

import os
import json
from datetime import datetime, timezone
from matcher import load_payments, load_ledger, reconcile

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "openai/gpt-oss-120b"

AUDIT_LOG = []


def log(event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    AUDIT_LOG.append(event)


def build_prompt(result, payment, ledger_entry):
    return f"""You are a financial reconciliation reviewer. Decide whether this payment and ledger record are the SAME transaction.

Payment: txn_id={payment.txn_id}, name="{payment.name}", amount={payment.amount}, date={payment.date.date()}
Ledger entry: order_id={ledger_entry.order_id}, name="{ledger_entry.name}", amount={ledger_entry.amount}, date={ledger_entry.date.date()}

Deterministic scoring already computed:
amount_score={result.score_breakdown.get('amount_score')} date_score={result.score_breakdown.get('date_score')} name_score={result.score_breakdown.get('name_score')}
overall_confidence={result.confidence}
exception_type={result.exception_type}

Respond ONLY with JSON, no preamble, no markdown fences, no explanation outside the JSON. Keep "reason" under 15 words.
{{"verdict": "MATCH" | "REJECT" | "NEEDS_HUMAN", "reason": "short plain English reason"}}"""


def call_groq(prompt: str, retries: int = 1):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    last_error = None
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
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


def fallback_reasoner(result):
    """Deterministic stand-in when no Groq key is present — keeps the
    pipeline runnable end to end, clearly labeled as non-LLM."""
    b = result.score_breakdown
    if result.exception_type == "DUPLICATE_AMBIGUOUS":
        return {"verdict": "NEEDS_HUMAN",
                "reason": "Multiple candidates scored nearly identically — cannot safely auto-resolve without more context."}
    if b.get("amount_score", 0) < 0.4:
        return {"verdict": "REJECT",
                "reason": f"Amount similarity too low ({b.get('amount_score')}) to be the same transaction."}
    if result.confidence and result.confidence >= 0.65:
        return {"verdict": "NEEDS_HUMAN",
                "reason": f"Confidence {result.confidence} is borderline — close enough to review, not enough to auto-match."}
    return {"verdict": "REJECT",
            "reason": f"Overall confidence {result.confidence} too low to trust as a match."}


def resolve_exceptions_with_agent(results, payments_by_id, ledger_by_id):
    """Runs LLM (or fallback) reasoning only on LOW_CONFIDENCE and
    DUPLICATE_AMBIGUOUS exceptions. Everything else passes through untouched."""
    resolvable_types = {"LOW_CONFIDENCE", "DUPLICATE_AMBIGUOUS"}
    using_llm = bool(GROQ_API_KEY)

    log({"event": "agent_run_start", "using_llm": using_llm, "model": MODEL if using_llm else "fallback_rules"})

    for r in results:
        if r.method != "EXCEPTION" or r.exception_type not in resolvable_types:
            continue

        payment = payments_by_id.get(r.txn_id)
        ledger_entry = ledger_by_id.get(r.order_id)
        if not payment or not ledger_entry:
            continue

        prompt = build_prompt(r, payment, ledger_entry)

        try:
            if using_llm:
                verdict = call_groq(prompt)
                source = "groq_llm"
            else:
                verdict = fallback_reasoner(r)
                source = "fallback_rules"
        except Exception as e:
            verdict = {"verdict": "NEEDS_HUMAN", "reason": f"Agent error, defaulting to human review: {e}"}
            source = "error_fallback"

        r.agent_verdict = verdict["verdict"]
        r.agent_reason = verdict["reason"]

        log({
            "event": "exception_reviewed",
            "source": source,
            "txn_id": r.txn_id,
            "order_id": r.order_id,
            "exception_type": r.exception_type,
            "score_breakdown": r.score_breakdown,
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
        })

        # Bounded auto-action: only MATCH verdicts upgrade the result,
        # and only if confidence was already reasonably close (>=0.6).
        # This is the "gated" part — agent can recommend, can't override
        # low-signal cases into a match on reasoning alone.
        if verdict["verdict"] == "MATCH" and r.confidence and r.confidence >= 0.6:
            r.method = "MATCHED_AGENT_REVIEWED"

    log({"event": "agent_run_end", "exceptions_reviewed": sum(
        1 for r in results if r.method == "EXCEPTION" and r.exception_type in resolvable_types
        or getattr(r, "agent_verdict", None)
    )})

    return results


if __name__ == "__main__":
    payments = load_payments("razorpay_payments.csv")
    ledger = load_ledger("internal_ledger.csv")
    payments_by_id = {p.txn_id: p for p in payments}
    ledger_by_id = {l.order_id: l for l in ledger}

    results = reconcile(payments, ledger)
    results = resolve_exceptions_with_agent(results, payments_by_id, ledger_by_id)

    matched = [r for r in results if r.method.startswith("MATCHED")]
    exceptions = [r for r in results if r.method == "EXCEPTION"]

    total = len(payments)
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total: {total} | Matched: {len(matched)} ({len(matched)/total*100:.1f}%) | Exceptions: {len(exceptions)}")
    print(f"  MATCHED_EXACT:          {len([r for r in matched if r.method=='MATCHED_EXACT'])}")
    print(f"  MATCHED_FUZZY:          {len([r for r in matched if r.method=='MATCHED_FUZZY'])}")
    print(f"  MATCHED_AGENT_REVIEWED: {len([r for r in matched if r.method=='MATCHED_AGENT_REVIEWED'])}")
    print(f"\n=== REMAINING EXCEPTIONS ===")
    for r in exceptions:
        verdict = getattr(r, "agent_verdict", "—")
        reason = getattr(r, "agent_reason", "not reviewed by agent")
        print(f"  {r.txn_id or '—'} / {r.order_id or '—'} | {r.exception_type} | agent_verdict={verdict} | {reason}")

    with open("audit_log.json", "w") as f:
        json.dump(AUDIT_LOG, f, indent=2)
    print(f"\nAudit trail written: audit_log.json ({len(AUDIT_LOG)} events)")

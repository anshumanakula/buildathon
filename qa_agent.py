"""
Q&A agent — lets a human ask natural-language questions about a completed
trace run: "which breaks cost the most?", "summarize this for a CFO",
"which need urgent attention?"

Strictly read-only. It only ever answers/summarizes using the run's own
data as context — it cannot trigger a resolution, modify a record, or
take any action. This keeps it inside the same "explainable, bounded,
gated" principle as the rest of the app: the agent that DOES things
(resolver.py) is separate from the agent that EXPLAINS things (this).
"""

import os
import json

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "openai/gpt-oss-120b"

MAX_ROWS_IN_CONTEXT = 60  # keep prompt bounded even on larger batches


def build_context(summary, rows, metrics=None):
    """Compresses the run into a compact, LLM-friendly summary rather than
    dumping raw JSON — keeps token usage sane and keeps the model focused
    on breaks (the interesting part) rather than re-deriving stats itself."""

    breaks = [r for r in rows if r.get("break_hop") not in (None, "NONE")]
    breaks = breaks[:MAX_ROWS_IN_CONTEXT]

    lines = [
        f"RUN SUMMARY: {summary.get('total')} transactions traced, "
        f"{summary.get('clean')} clean end-to-end ({summary.get('clean_rate_pct')}%), "
        f"{len(breaks)} broken chains.",
        f"Breaks by hop: {json.dumps(summary.get('breaks_by_hop', {}))}",
    ]
    if metrics:
        lines.append(
            f"Verified accuracy: break_hop_accuracy={metrics.get('break_hop_accuracy')}%, "
            f"clean_precision={metrics.get('clean_precision')}%, "
            f"break_detection_rate={metrics.get('break_detection_rate')}%"
        )

    lines.append("\nBROKEN TRANSACTIONS:")
    for r in breaks:
        detail = r.get("break_detail", {}) or {}
        res = r.get("resolution", {}) or {}
        variance = detail.get("variance")
        lines.append(
            f"- {r.get('txn_id')} (order {r.get('order_id') or 'none'}): "
            f"broke at {r.get('break_hop')}"
            + (f", variance ₹{variance}" if variance is not None else "")
            + f". {detail.get('reason', '')}"
            + (f" | Drafted: [{res.get('resolution_type')}] {res.get('draft')}" if res else "")
        )

    return "\n".join(lines)


SYSTEM_INSTRUCTIONS = """You are a read-only assistant answering questions about a completed \
finance reconciliation run. You can only see the data provided below — you cannot look up \
anything else, cannot modify any record, and cannot trigger any action. If asked to do \
something outside answering/summarizing this data, say you can only discuss this run's results.

Be concise and concrete — use real numbers and transaction IDs from the data, never invent \
figures. If asked something the data can't answer, say so plainly."""


def call_groq_chat(question, context, retries=1):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"{SYSTEM_INSTRUCTIONS}\n\n--- RUN DATA ---\n{context}\n--- END RUN DATA ---\n\nQuestion: {question}"

    last_error = None
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        if text:
            return text
        last_error = ValueError("empty response from model")

    raise last_error


def fallback_answer(question, summary, rows):
    """No-LLM fallback: answers a few common question patterns with simple
    rules so the feature still works without a Groq key configured."""
    q = question.lower()
    breaks = [r for r in rows if r.get("break_hop") not in (None, "NONE")]

    if "summar" in q or "cfo" in q or "overview" in q:
        return (
            f"{summary.get('total')} transactions traced. "
            f"{summary.get('clean')} ({summary.get('clean_rate_pct')}%) reconciled cleanly end-to-end. "
            f"{len(breaks)} broke — see the breakdown by hop in the dashboard above. "
            f"Each break has a drafted resolution awaiting approval."
        )

    if "urgent" in q or "priority" in q or "biggest" in q or "largest" in q:
        scored = []
        for r in breaks:
            v = (r.get("break_detail") or {}).get("variance")
            if v is not None:
                scored.append((abs(v), r))
        scored.sort(key=lambda x: -x[0])
        if scored:
            top = scored[:3]
            lines = [f"- {r['txn_id']}: ₹{v} variance at {r['break_hop']}" for v, r in top]
            return "Largest variances by amount:\n" + "\n".join(lines)
        return "No variance amounts available to rank by size."

    if "gateway" in q:
        n = summary.get("breaks_by_hop", {}).get("GATEWAY", 0)
        return f"{n} transactions broke at the gateway hop — payment captured but never settled."

    if "settlement" in q:
        n = summary.get("breaks_by_hop", {}).get("SETTLEMENT", 0)
        return f"{n} transactions broke at the settlement hop — fee variance from the standard rate."

    if "ledger" in q:
        n = summary.get("breaks_by_hop", {}).get("LEDGER", 0)
        return f"{n} transactions broke at the ledger hop — amount mismatch or missing entry."

    return ("Groq key not configured, so I can only answer a few basic patterns "
            "(summary, urgent/largest, gateway/settlement/ledger counts). "
            "Set GROQ_API_KEY for full natural-language Q&A.")


def answer_question(question, summary, rows, metrics=None):
    using_llm = bool(GROQ_API_KEY)

    if not question or not question.strip():
        return {"answer": "Ask a question about this run — e.g. \"summarize this for a CFO\" or \"which breaks are most urgent?\"",
                "source": "validation"}

    if using_llm:
        try:
            context = build_context(summary, rows, metrics)
            answer = call_groq_chat(question, context)
            return {"answer": answer, "source": "groq_llm"}
        except Exception as e:
            return {"answer": f"Agent error, showing basic answer instead: {fallback_answer(question, summary, rows)}",
                    "source": "error_fallback"}
    else:
        return {"answer": fallback_answer(question, summary, rows), "source": "fallback_rules"}

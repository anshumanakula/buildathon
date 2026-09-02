"""
Flask API — 3-hop reconciliation with chain tracing and resolution drafting.

POST /api/trace
  multipart/form-data with three files: 'payments', 'settlements', 'ledger'
  optional fourth file 'ground_truth' to compute break-hop accuracy

  Pipeline:
    1. tracer.trace_chain()       — walks payment -> settlement -> ledger,
                                     finds the exact hop where each chain breaks
    2. resolver.draft_resolutions() — Groq drafts the actual fix per break
    3. scorer.score()             — break-hop accuracy if ground truth given

GET /api/sample
  Returns bundled sample data (4 CSVs) for the "use sample data" button.

GET /health
"""

import io
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from tracer import (
    load_payments_from_stream, load_settlements_from_stream, load_ledger_from_stream,
    trace_chain, summarize,
)
from resolver import draft_resolutions, GROQ_API_KEY
from scorer import load_ground_truth_from_stream, score
from qa_agent import answer_question

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)


@app.after_request
def add_cors_headers(response):
    # When opened as a local file:// page, browsers send Origin: null.
    # Returning "null" back is NOT treated as a wildcard — the browser still
    # blocks the request. We must return "*" in that case.
    origin = request.headers.get("Origin", "")
    allowed_origin = "*" if (not origin or origin == "null") else origin
    response.headers["Access-Control-Allow-Origin"] = allowed_origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/", defaults={"_any": ""}, methods=["OPTIONS"])
@app.route("/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    resp = app.make_response("")
    resp.status_code = 204
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "llm_configured": bool(GROQ_API_KEY)})


@app.route("/api/sample")
def sample():
    try:
        files = {}
        for name, fname in [
            ("payments_csv", "razorpay_payments.csv"),
            ("settlements_csv", "settlements.csv"),
            ("ledger_csv", "internal_ledger.csv"),
            ("ground_truth_csv", "ground_truth.csv"),
        ]:
            with open(os.path.join(SAMPLE_DIR, fname)) as f:
                files[name] = f.read()
    except FileNotFoundError as e:
        return jsonify({"error": f"sample data not found on server: {e}"}), 500
    return jsonify(files)


@app.route("/api/trace", methods=["POST"])
def api_trace():
    started = time.time()

    required = ["payments", "settlements", "ledger"]
    missing = [k for k in required if k not in request.files]
    if missing:
        return jsonify({"error": f"missing required files: {', '.join(missing)}"}), 400

    try:
        payments = load_payments_from_stream(
            io.StringIO(request.files["payments"].read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": f"could not parse payments CSV: {e}"}), 400

    try:
        settlements = load_settlements_from_stream(
            io.StringIO(request.files["settlements"].read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": f"could not parse settlements CSV: {e}"}), 400

    try:
        ledger = load_ledger_from_stream(
            io.StringIO(request.files["ledger"].read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": f"could not parse ledger CSV: {e}"}), 400

    if len(payments) < 1:
        return jsonify({"error": "payments file must contain at least one record"}), 400

    payments_by_id = {p.txn_id: p for p in payments}

    # ---- Stage 1: chain tracing ----
    results = trace_chain(payments, settlements, ledger)
    summary = summarize(results)

    # ---- Stage 2: resolution drafting on every break ----
    audit_log = []
    results = draft_resolutions(results, payments_by_id, audit_log)

    response = {
        "summary": {
            "total": summary["total"],
            "clean": summary["clean"],
            "clean_rate_pct": summary["clean_rate_pct"],
            "breaks_by_hop": summary["breaks_by_hop"],
            "using_llm": bool(GROQ_API_KEY),
            "runtime_seconds": round(time.time() - started, 2),
        },
        "rows": [
            {
                "txn_id": r.txn_id,
                "settlement_id": r.settlement_id,
                "order_id": r.order_id,
                "payment_status": r.payment_status,
                "settlement_status": r.settlement_status,
                "ledger_status": r.ledger_status,
                "break_hop": r.break_hop,
                "break_detail": r.break_detail,
                "resolution": r.resolution,
            }
            for r in results
        ],
        "audit_log": audit_log,
    }

    # ---- Stage 3: optional scoring against ground truth ----
    gt_file = request.files.get("ground_truth")
    if gt_file:
        try:
            truth = load_ground_truth_from_stream(
                io.StringIO(gt_file.read().decode("utf-8")))
            metrics = score(results, truth)
            response["metrics"] = {
                "break_hop_accuracy": metrics["break_hop_accuracy"],
                "clean_precision": metrics["clean_precision"],
                "break_detection_rate": metrics["break_detection_rate"],
            }
            outcome_by_txn = {r.txn_id: outcome for r, outcome in metrics["scored_rows"]}
            for row in response["rows"]:
                row["scoring_outcome"] = outcome_by_txn.get(row["txn_id"])
        except Exception as e:
            response["metrics_error"] = f"could not score against ground truth: {e}"

    return jsonify(response)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Read-only Q&A over a completed run's results. The frontend sends
    back the same summary/rows/metrics it received from /api/trace —
    this endpoint never re-runs the pipeline or touches any file, it only
    reasons over data that was already computed."""
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    summary = body.get("summary", {})
    rows = body.get("rows", [])
    metrics = body.get("metrics")

    if not summary or not rows:
        return jsonify({"error": "no run data provided — run a trace first"}), 400

    result = answer_question(question, summary, rows, metrics)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

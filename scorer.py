"""
Scores chain-trace results against ground_truth.csv.

The headline metric here is break_hop_accuracy — did the tracer correctly
identify WHERE in the chain (gateway/settlement/ledger) each break
happened, not just whether it flagged something as broken. This is what
separates hop-tracing from plain 2-source matching.
"""

import csv


def load_ground_truth(path):
    with open(path) as f:
        return load_ground_truth_from_stream(f)


def load_ground_truth_from_stream(file_obj):
    """Returns dict: txn_id -> {"break_hop": ..., "label": ...}"""
    truth = {}
    for r in csv.DictReader(file_obj):
        truth[r["txn_id"]] = {"break_hop": r["break_hop"], "label": r["label"]}
    return truth


def score(chain_results, truth):
    total = len(chain_results)
    hop_correct = 0
    clean_correct = 0     # correctly identified as clean (no break)
    break_correct = 0     # correctly identified AS broken (any hop) — coarse detection accuracy
    total_should_be_clean = 0
    total_should_be_broken = 0

    scored_rows = []

    for r in chain_results:
        gt = truth.get(r.txn_id)
        if gt is None:
            scored_rows.append((r, "NO_GROUND_TRUTH"))
            continue

        true_hop = gt["break_hop"]
        predicted_hop = r.break_hop

        hop_match = (true_hop == predicted_hop)
        if hop_match:
            hop_correct += 1

        if true_hop == "NONE":
            total_should_be_clean += 1
            if predicted_hop == "NONE":
                clean_correct += 1
                outcome = "CORRECT_CLEAN"
            else:
                outcome = "FALSE_BREAK"  # flagged a break that wasn't real
        else:
            total_should_be_broken += 1
            if predicted_hop != "NONE":
                break_correct += 1
                outcome = "CORRECT_HOP" if hop_match else "WRONG_HOP"
            else:
                outcome = "MISSED_BREAK"  # should have caught this, didn't

        scored_rows.append((r, outcome))

    return {
        "total": total,
        "break_hop_accuracy": round(hop_correct / total * 100, 2) if total else 0,
        "clean_precision": round(clean_correct / total_should_be_clean * 100, 2) if total_should_be_clean else 0,
        "break_detection_rate": round(break_correct / total_should_be_broken * 100, 2) if total_should_be_broken else 0,
        "hop_correct": hop_correct,
        "scored_rows": scored_rows,
    }

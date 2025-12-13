import json
import argparse
from collections import defaultdict

def load_jsonl(fname):
    return [json.loads(line) for line in open(fname)]

def safe_divide(num, denom):
    if denom == 0:
        return 0.0
    return num / denom

def compute_f1(correct, predicted, gold):
    precision = safe_divide(correct, predicted)
    recall = safe_divide(correct, gold)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1

def extract_data(claims):
    """
    Extract label and rationale data from claims list.
    Returns a dict: claim_id -> doc_id -> {label, sentences}
    """
    data = {}
    for claim in claims:
        claim_id = claim["id"]
        data[claim_id] = {}
        
        if "evidence" not in claim or claim["evidence"] is None:
            continue
            
        evidence_dict = claim["evidence"]
        
        # Normalize evidence format
        if not isinstance(evidence_dict, dict):
            continue
            
        for doc_id_str, evidence in evidence_dict.items():
            doc_id = int(doc_id_str)
            
            # Handle different formats
            label = "NEI"
            sentences = []
            
            if isinstance(evidence, dict):
                label = evidence.get("label", "NEI")
                sentences = evidence.get("sentences", [])
            elif isinstance(evidence, list):
                if len(evidence) > 0:
                    label = evidence[0].get("label", "NEI")
                    # Flatten sentences from multiple evidence sets if needed
                    for ev_item in evidence:
                        sentences.extend(ev_item.get("sentences", []))
            
            # Deduplicate sentences
            sentences = sorted(list(set(sentences)))
            
            data[claim_id][doc_id] = {
                "label": label,
                "sentences": sentences
            }
    return data

def evaluate_abstract_level(gold_data, pred_data, all_doc_ids=None):
    """
    Abstract-level (label-only) evaluation.
    Only checks if the document label is correct (ignores rationale sentences).
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Collect all (claim_id, doc_id) pairs to evaluate
    all_pairs = set()
    for c_id, docs in gold_data.items():
        for d_id in docs:
            all_pairs.add((c_id, d_id))
    for c_id, docs in pred_data.items():
        for d_id in docs:
            all_pairs.add((c_id, d_id))
    
    if all_doc_ids:
        for c_id, d_ids in all_doc_ids.items():
            for d_id in d_ids:
                all_pairs.add((c_id, d_id))

    total_gold_non_nei = 0
    total_pred_non_nei = 0
    
    for claim_id, doc_id in all_pairs:
        gold_entry = gold_data.get(claim_id, {}).get(doc_id, {"label": "NEI"})
        pred_entry = pred_data.get(claim_id, {}).get(doc_id, {"label": "NEI"})
        
        gold_label = gold_entry["label"]
        pred_label = pred_entry["label"]
        
        if gold_label != "NEI":
            total_gold_non_nei += 1
        if pred_label != "NEI":
            total_pred_non_nei += 1
            
        if gold_label != "NEI" and pred_label == gold_label:
            tp += 1
        elif gold_label != "NEI" and pred_label != gold_label:
            fn += 1
        elif gold_label == "NEI" and pred_label != "NEI":
            fp += 1
            
    precision, recall, f1 = compute_f1(tp, total_pred_non_nei, total_gold_non_nei)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_pred": total_pred_non_nei,
        "total_gold": total_gold_non_nei
    }

def evaluate_sentence_level(gold_data, pred_data):
    """
    Sentence-level (selection + label) evaluation.
    A predicted sentence is correct if:
    1. It is in the gold rationale set.
    2. The predicted label for the document matches the gold label.
    """
    correct_sentences = 0
    total_pred_sentences = 0
    total_gold_sentences = 0
    
    # Iterate over all claims in gold to count total gold sentences
    for claim_id, docs in gold_data.items():
        for doc_id, entry in docs.items():
            if entry["label"] != "NEI":
                total_gold_sentences += len(entry["sentences"])
                
    # Iterate over predictions
    for claim_id, docs in pred_data.items():
        for doc_id, pred_entry in docs.items():
            pred_label = pred_entry["label"]
            pred_sents = pred_entry["sentences"]
            
            if pred_label == "NEI":
                continue
                
            total_pred_sentences += len(pred_sents)
            
            # Check against gold
            gold_entry = gold_data.get(claim_id, {}).get(doc_id)
            
            if gold_entry and gold_entry["label"] != "NEI":
                # Label must match
                if pred_label == gold_entry["label"]:
                    gold_sents = set(gold_entry["sentences"])
                    # Count matching sentences
                    for sent_idx in pred_sents:
                        if sent_idx in gold_sents:
                            correct_sentences += 1
                            
    precision, recall, f1 = compute_f1(correct_sentences, total_pred_sentences, total_gold_sentences)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct_sentences,
        "total_pred": total_pred_sentences,
        "total_gold": total_gold_sentences
    }

def print_metrics(title, metrics, is_sentence=False):
    print(f"\n      - {title}")
    print("")
    print("============================================================")
    print("Evaluation Results")
    print("============================================================")
    print(f"Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:   {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print("")
    print("Detailed Statistics:")
    if not is_sentence:
        print(f"  Correct Predictions (TP):     {metrics['tp']}")
        print(f"  False Positives (FP):          {metrics['fp']}")
        print(f"  False Negatives (FN):           {metrics['fn']}")
        print(f"  Total Predicted (non-NEI):      {metrics['total_pred']}")
        print(f"  Total Gold (non-NEI):           {metrics['total_gold']}")
    else:
        print(f"  Correct Sentences:            {metrics['correct']}")
        print(f"  Total Predicted:              {metrics['total_pred']}")
        print(f"  Total Gold:                   {metrics['total_gold']}")
    print("============================================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, default=None)
    args = parser.parse_args()
    
    # Load data
    gold_claims = load_jsonl(args.gold_file)
    gold_data = extract_data(gold_claims)
    
    pred_claims = load_jsonl(args.pred_file)
    pred_data = extract_data(pred_claims)
    
    # Get all doc ids for exhaustive Abstract-level eval
    all_doc_ids = None
    if args.input_file:
        all_doc_ids = {}
        input_claims = load_jsonl(args.input_file)
        for c in input_claims:
            if "doc_ids" in c:
                all_doc_ids[c["id"]] = c["doc_ids"]
    
    # Compute Metrics
    abstract_metrics = evaluate_abstract_level(gold_data, pred_data, all_doc_ids)
    sentence_metrics = evaluate_sentence_level(gold_data, pred_data)
    
    # Print Results
    print_metrics("Abstract-level(label-only)", abstract_metrics, is_sentence=False)
    print_metrics("Sentence-level(selection+label)", sentence_metrics, is_sentence=True)

if __name__ == "__main__":
    main()

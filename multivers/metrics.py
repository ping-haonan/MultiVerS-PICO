"""
Implement metrics from the SciFact paper.
"""


import torch

from torchmetrics.metric import Metric

NEI = 1    # Label for NEI class.
MAX_ABSTRACT_SENTS = 3   # Max number of abstract sentences.


def safe_divide(num, denom):
    if denom == 0:
        return denom
    else:
        return num / denom


def compute_f1(relevant, retrieved, correct, prefix):
    precision = safe_divide(correct, retrieved)
    recall = safe_divide(correct, relevant)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1}


def has_correct_rationale_abstract(pred, gold):
    """
    Check if at least one correct rationale was predicted - for abstract
    evaluation. We only keep the first 3 predicted sentences for this.
    """
    pred_ix = pred.nonzero(as_tuple=True)[0][:MAX_ABSTRACT_SENTS]
    pred_allowed = torch.zeros_like(pred)
    pred_allowed[pred_ix] = 1
    return int(count_correct_rationales(pred_allowed, gold) > 0)


def count_correct_rationales(pred, gold):
    """
    Given vectors of predicted and gold rationales, count the number of correct
    predictions. A prediction is only correct if the whole gold set is ID'd.
    This is used for sentence-level evaluation.
    For this, we count all the predicted rationales (this is for sentence-level).
    """
    # Get all the ID's of gold rationales.
    correct_predictions = 0

    gold_ids = gold.unique().tolist()
    gold_ids = [entry for entry in gold_ids if entry > 0]
    for gold_id in gold_ids:
        ix = gold == gold_id  # Indices of this gold rationale set.
        this_pred = pred[ix]
        # If the model predicted all the sentences in this gold rationale set,
        # give it credit for them.
        if torch.all(this_pred.bool()):
            correct_predictions += len(this_pred)

    return correct_predictions


class SciFactMetrics(Metric):
    """
    Abstract and sentence-level metrics from the paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        names = ["correct_label",
                 "total_label",
                 "relevant_abstract_label",
                 "retrieved_abstract_label",
                 "correct_abstract_label",
                 "relevant_abstract_rationalized",
                 "retrieved_abstract_rationalized",
                 "correct_abstract_rationalized",
                 "relevant_sentence",
                 "retrieved_sentence_nei",
                 "retrieved_sentence",
                 "correct_sentence_nei",
                 "correct_sentence",
                 "correct_sentence_labeled"]
        for name in names:
            self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        pred_labs = preds["predicted_labels"]
        gold_labs = target["label"]
        pred_rats = preds["predicted_rationales"]
        gold_rats = target["rationale_sets"]
        rationale_mask = target["rationale_mask"]

        # Label accuracy.
        self.correct_label += torch.sum(pred_labs == gold_labs)
        self.total_label += len(gold_labs)

        # Abstract and sentence-level evaluation.
        zipped = zip(pred_labs, gold_labs, pred_rats, gold_rats, rationale_mask)
        # Loop over and evaluate the individual predictions.
        for pred_lab, gold_lab, pred_rat, gold_rat, rat_mask in zipped:

            # Abstract-level label.
            relevant_abstract = gold_lab != NEI
            retrieved_abstract = pred_lab != NEI
            correct_label = relevant_abstract & (gold_lab == pred_lab)

            self.relevant_abstract_label += relevant_abstract
            self.retrieved_abstract_label += retrieved_abstract
            self.correct_abstract_label += correct_label

            # Rationalized scores. Can only do this if the doc has rationale
            # annotations.
            if rat_mask:
                has_correct_rat_abstract = has_correct_rationale_abstract(
                    pred_rat, gold_rat)
                correct_abstract_rat = correct_label * has_correct_rat_abstract

                self.relevant_abstract_rationalized += relevant_abstract
                self.retrieved_abstract_rationalized += retrieved_abstract
                self.correct_abstract_rationalized += correct_abstract_rat

                # Sentence_level. Only do this if we have rationale annotations.
                retrieved_sentence_nei = torch.sum(pred_rat > 0)
                n_correct_rat = count_correct_rationales(pred_rat, gold_rat)

                self.relevant_sentence += torch.sum(gold_rat > 0)
                self.retrieved_sentence_nei += retrieved_sentence_nei
                self.retrieved_sentence += retrieved_sentence_nei * retrieved_abstract
                self.correct_sentence_nei += n_correct_rat
                self.correct_sentence += n_correct_rat * retrieved_abstract
                self.correct_sentence_labeled += n_correct_rat * correct_label

    def compute(self):
        res = {}
        res.update({"label_accuracy": (self.correct_label / self.total_label)})
        res.update(compute_f1(
            self.relevant_abstract_label, self.retrieved_abstract_label,
            self.correct_abstract_label, "abstract_label_only"))
        res.update(compute_f1(
            self.relevant_abstract_rationalized, self.retrieved_abstract_rationalized,
            self.correct_abstract_rationalized, "abstract_rationalized"))
        
        # === DIAGNOSTIC: Add sample count for sentence-level metrics ===
        # This helps diagnose if metrics are computed on too few samples
        # NOTE: Don't use _ prefix as PyTorch Lightning may ignore it
        res["sentence_relevant_count"] = self.relevant_sentence.item()
        res["sentence_retrieved_nei_count"] = self.retrieved_sentence_nei.item()
        res["sentence_retrieved_count"] = self.retrieved_sentence.item()
        res["sentence_correct_nei_count"] = self.correct_sentence_nei.item()
        res["sentence_correct_count"] = self.correct_sentence.item()
        
        # Warn if very few samples are used for sentence-level metrics
        if self.relevant_sentence.item() < 10:
            import warnings
            warnings.warn(
                f"⚠️  WARNING: Only {self.relevant_sentence.item()} sentences used for "
                f"sentence-level metrics. This may lead to unreliable metrics. "
                f"Check if rationale_mask is set correctly and if validation set has rationale annotations.",
                UserWarning
            )
        
        res.update(compute_f1(
            self.relevant_sentence, self.retrieved_sentence_nei,
            self.correct_sentence_nei, "sentence_nei"))
        res.update(compute_f1(
            self.relevant_sentence, self.retrieved_sentence,
            self.correct_sentence, "sentence_selection"))
        res.update(compute_f1(
            self.relevant_sentence, self.retrieved_sentence,
            self.correct_sentence_labeled, "sentence_label"))

        return res


class ClaimDocumentMetrics(Metric):
    """
    Metrics computed exactly as in eval_multivers.py:
    Based on (claim_id, doc_id) pairs with label matching.
    This aligns with the evaluation script used for scifact-open.
    
    Calculation method:
    - For each (claim_id, doc_id) pair:
      - If gold_label != "NEI" and pred_label == gold_label: correct_predictions++, TP++
      - If gold_label != "NEI" and pred_label != gold_label: FN++
      - If gold_label == "NEI" and pred_label != "NEI": FP++
    - Precision = correct_predictions / total_predicted_non_nei
    - Recall = correct_predictions / total_gold_non_nei
    - F1 = 2 * precision * recall / (precision + recall)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        names = [
            "correct_predictions",
            "total_predicted_non_nei",
            "total_gold_non_nei",
            "true_positives",
            "false_positives",
            "false_negatives",
        ]
        for name in names:
            self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        """
        Update metrics based on predictions and targets.
        preds: dict with "predicted_labels"
        target: dict with "label", "claim_id", "abstract_id"
        """
        pred_labs = preds["predicted_labels"]  # [batch_size]
        gold_labs = target["label"]  # [batch_size]
        
        # Convert labels: 0=REFUTES, 1=NEI, 2=SUPPORTS -> "REFUTES", "NEI", "SUPPORTS"
        label_map = {0: "REFUTES", 1: "NEI", 2: "SUPPORTS"}
        
        for i in range(len(pred_labs)):
            pred_label = label_map[pred_labs[i].item()]
            gold_label = label_map[gold_labs[i].item()]
            
            # Count statistics exactly as in eval_multivers.py
            if gold_label != "NEI":
                self.total_gold_non_nei += 1
            
            if pred_label != "NEI":
                self.total_predicted_non_nei += 1
            
            if gold_label != "NEI" and pred_label == gold_label:
                self.correct_predictions += 1
                self.true_positives += 1
            elif gold_label != "NEI" and pred_label != gold_label:
                self.false_negatives += 1
            elif gold_label == "NEI" and pred_label != "NEI":
                self.false_positives += 1

    def compute(self):
        """
        Compute final metrics exactly as in eval_multivers.py
        """
        if self.total_predicted_non_nei == 0:
            precision = torch.tensor(0.0)
        else:
            precision = self.correct_predictions.float() / self.total_predicted_non_nei.float()
        
        if self.total_gold_non_nei == 0:
            recall = torch.tensor(0.0)
        else:
            recall = self.correct_predictions.float() / self.total_gold_non_nei.float()
        
        if precision + recall == 0:
            f1 = torch.tensor(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": self.true_positives.float(),
            "false_positives": self.false_positives.float(),
            "false_negatives": self.false_negatives.float(),
            "total_predicted_non_nei": self.total_predicted_non_nei.float(),
            "total_gold_non_nei": self.total_gold_non_nei.float(),
            "correct_predictions": self.correct_predictions.float(),
        }

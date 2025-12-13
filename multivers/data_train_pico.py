"""
Extended data loader with PICO feature support.
Modified from data_train.py to load precomputed PICO features.
"""

import os
import pathlib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from torch.utils.data.dataset import ConcatDataset
import random

from pytorch_lightning import LightningDataModule

from multivers.data_verisci import GoldDataset
from transformers import AutoTokenizer, BatchEncoding

import multivers.util as util

# === PICO: Import original components we'll reuse ===
from multivers.data_train import (
    get_tokenizer,
    SciFactDataset as BaseSciFactDataset,
    FactCheckingReader,
    SciFactOriginalReader,
    SciFactOpenReader,
    SciFact10Reader,
    SciFact20Reader,
    HealthVerReader,
    CovidFactReader,
    FEVERReader,
    PubMedQAReader,
    EvidenceInferenceReader
)


# === PICO: Extended Collator ===
class SciFactPICOCollator:
    """Extended collator that handles PICO features in addition to standard fields."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """Collate all the data together into padded batch tensors."""
        res = {
            "claim_id": self._collate_scalar(batch, "claim_id"),
            "abstract_id": self._collate_scalar(batch, "abstract_id"),
            "negative_sample_id": self._collate_scalar(batch, "negative_sample_id"),
            "dataset": [x["dataset"] for x in batch],
            "tokenized": self._pad_tokenized([x["tokenized"] for x in batch]),
            "abstract_sent_idx": self._pad_field(batch, "abstract_sent_idx", 0),
            "label": self._collate_scalar(batch, "label"),
            "rationale": self._pad_field(batch, "rationale", -1),
            "rationale_sets": self._pad_field(batch, "rationale_sets", -1),
            "weight": self._collate_scalar(batch, "weight"),
            "rationale_mask": self._collate_scalar(batch, "rationale_mask"),
        }
        
        # === PICO: Add PICO features if present ===
        if "pico_span_features" in batch[0] and batch[0]["pico_span_features"] is not None:
            res["pico_span_features"] = self._pad_pico_spans(batch, "pico_span_features")
            res["pico_span_mask"] = self._pad_pico_spans(batch, "pico_span_mask")
        
        if "claim_span_features" in batch[0] and batch[0]["claim_span_features"] is not None:
            res["claim_span_features"] = self._pad_pico_spans(batch, "claim_span_features")
            res["claim_span_mask"] = self._pad_pico_spans(batch, "claim_span_mask")
        
        return res

    @staticmethod
    def _collate_scalar(batch, field):
        return torch.tensor([x[field] for x in batch])

    def _pad_tokenized(self, tokenized):
        fields = ["input_ids", "attention_mask", "global_attention_mask"]
        pad_values = [self.tokenizer.pad_token_id, 0, 0]
        tokenized_padded = {}
        for field, pad_value in zip(fields, pad_values):
            tokenized_padded[field] = self._pad_field(tokenized, field, pad_value)
        return tokenized_padded

    def _pad_field(self, entries, field_name, pad_value):
        xxs = [entry[field_name] for entry in entries]
        return self._pad(xxs, pad_value)

    @staticmethod
    def _pad(xxs, pad_value):
        res = []
        max_length = max(map(len, xxs))
        for entry in xxs:
            to_append = [pad_value] * (max_length - len(entry))
            padded = entry + to_append
            res.append(padded)
        return torch.tensor(res)
    
    # === PICO: New method for padding span features ===
    def _pad_pico_spans(self, batch, field_name):
        """
        Pad PICO span features: [batch, n_sentences, n_spans, dim]
        """
        spans_list = [x[field_name] for x in batch]
        
        # Find max dimensions
        max_sents = max(s.size(0) for s in spans_list)
        max_spans = max(s.size(1) for s in spans_list)
        
        if len(spans_list[0].shape) == 3:  # span_features [n_sent, n_span, dim]
            feat_dim = spans_list[0].size(2)
            padded = torch.zeros(len(spans_list), max_sents, max_spans, feat_dim)
        else:  # span_mask [n_sent, n_span]
            padded = torch.zeros(len(spans_list), max_sents, max_spans)
        
        for i, spans in enumerate(spans_list):
            n_sent, n_span = spans.size(0), spans.size(1)
            if len(spans.shape) == 3:
                padded[i, :n_sent, :n_span, :] = spans
            else:
                padded[i, :n_sent, :n_span] = spans
        
        return padded


# === PICO: Extended Dataset ===
class SciFactPICODataset(BaseSciFactDataset):
    """Extended dataset that loads PICO features from precomputed cache."""
    
    # Class-level cache to avoid reloading the same files
    _claims_cache = {}
    _corpus_cache = {}
    
    def __init__(self, entries, tokenizer, dataset_name, rationale_mask, 
                 pico_claims_cache=None, pico_corpus_cache=None):
        super().__init__(entries, tokenizer, dataset_name, rationale_mask)
        
        # === PICO: Load cached features ===
        self.pico_claims = {}
        self.pico_corpus = {}
        
        if pico_claims_cache and os.path.exists(pico_claims_cache):
            # Check if already in class-level cache
            if pico_claims_cache not in self._claims_cache:
                print(f"Loading claim PICO features from {pico_claims_cache}")
                self._claims_cache[pico_claims_cache] = torch.load(pico_claims_cache)
                print(f"Loaded {len(self._claims_cache[pico_claims_cache])} claim PICO features")
            self.pico_claims = self._claims_cache[pico_claims_cache]
        
        if pico_corpus_cache and os.path.exists(pico_corpus_cache):
            # Check if already in class-level cache
            if pico_corpus_cache not in self._corpus_cache:
                print(f"Loading corpus PICO features from {pico_corpus_cache}")
                self._corpus_cache[pico_corpus_cache] = torch.load(pico_corpus_cache)
                print(f"Loaded {len(self._corpus_cache[pico_corpus_cache])} document PICO features")
            self.pico_corpus = self._corpus_cache[pico_corpus_cache]
    
    def __getitem__(self, idx):
        """Extended getitem that includes PICO features."""
        res = super().__getitem__(idx)
        
        claim_id = res["claim_id"]
        doc_id = res["abstract_id"]
        
        # === PICO: Fetch claim features ===
        claim_feats = self.pico_claims.get(claim_id)
        if claim_feats is not None:
            # Reshape to [1, n_spans, dim] for consistency
            span_emb = claim_feats["span_embeddings"]  # [n_spans, 768]
            if span_emb.size(0) > 0:
                res["claim_span_features"] = span_emb.unsqueeze(0)  # [1, n_spans, 768]
                res["claim_span_mask"] = torch.ones(1, span_emb.size(0))
            else:
                # No PICO spans - use dummy span with mask=1 to avoid softmax(all -inf)=nan
                res["claim_span_features"] = torch.zeros(1, 1, 768)
                res["claim_span_mask"] = torch.ones(1, 1)  # Changed from zeros to ones!
        else:
            # No PICO features for this claim - use dummy span
            res["claim_span_features"] = torch.zeros(1, 1, 768)
            res["claim_span_mask"] = torch.ones(1, 1)  # Changed from zeros to ones!
        
        # === PICO: Fetch document sentence features ===
        doc_feats = self.pico_corpus.get(doc_id)
        if doc_feats is not None:
            # doc_feats is a list of sentence features
            n_sents = len(doc_feats)
            span_features_list = []
            span_mask_list = []
            
            for sent_feat in doc_feats:
                if sent_feat is not None:
                    span_emb = sent_feat["span_embeddings"]  # [n_spans, 768]
                    if span_emb.size(0) > 0:
                        span_features_list.append(span_emb)
                        span_mask_list.append(torch.ones(span_emb.size(0)))
                    else:
                        # No PICO spans - use dummy span with mask=1
                        span_features_list.append(torch.zeros(1, 768))
                        span_mask_list.append(torch.ones(1))  # Changed from zeros to ones!
                else:
                    # No features - use dummy span with mask=1
                    span_features_list.append(torch.zeros(1, 768))
                    span_mask_list.append(torch.ones(1))  # Changed from zeros to ones!
            
            # Pad to same number of spans per sentence
            max_spans = max(s.size(0) for s in span_features_list)
            padded_features = torch.zeros(n_sents, max_spans, 768)
            padded_mask = torch.zeros(n_sents, max_spans)
            
            for i, (feats, mask) in enumerate(zip(span_features_list, span_mask_list)):
                n_span = feats.size(0)
                padded_features[i, :n_span, :] = feats
                padded_mask[i, :n_span] = mask
            
            res["pico_span_features"] = padded_features
            res["pico_span_mask"] = padded_mask
        else:
            # No PICO features for this doc - use dummy spans with mask=1
            n_sents = len(res["rationale"])  # Use rationale length as proxy
            res["pico_span_features"] = torch.zeros(n_sents, 1, 768)
            res["pico_span_mask"] = torch.ones(n_sents, 1)  # Changed from zeros to ones!
        
        return res


# === PICO: Extended DataModule ===
class ConcatDataModulePICO(LightningDataModule):
    """Extended data module that passes PICO cache paths to datasets."""
    
    def __init__(self, hparams):
        super().__init__()
        self.tokenizer = get_tokenizer(hparams)
        self.num_workers = hparams.num_workers
        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.collator = SciFactPICOCollator(self.tokenizer)
        self.shuffle = not hparams.no_shuffle
        self.reweight_labels = not hparams.no_reweight_labels
        self.reweight_datasets = not hparams.no_reweight_datasets
        self.max_label_weight = hparams.max_label_weight
        self.max_dataset_weight = hparams.max_dataset_weight
        self.cap_fever_nsamples = hparams.cap_fever_nsamples
        self.debug = getattr(hparams, "debug", False) or getattr(hparams, "fast_dev_run", False)
        self.fewshot = getattr(hparams, "fewshot", False)
        
        # === PICO: Get PICO cache paths ===
        self.pico_feature_dir = getattr(hparams, "pico_feature_dir", None)
        
        # Reader lookup (same as original)
        self.reader_lookup = {
            "scifact": SciFactOriginalReader,
            "scifact_open": SciFactOpenReader,
            "scifact_20": SciFact20Reader,
            "scifact_10": SciFact10Reader,
            "healthver": HealthVerReader,
            "covidfact": CovidFactReader,
            "fever": FEVERReader,
            "pubmedqa": PubMedQAReader,
            "evidence_inference": EvidenceInferenceReader,
        }
        
        self.dataset_weights = {
            "SciFact": hparams.scifact_weight,
            "SciFactOpen": hparams.scifact_open_weight,
            "HealthVer": hparams.healthver_weight,
            "CovidFact": hparams.covidfact_weight,
            "FEVER": hparams.fever_weight,
            "PubMedQA": hparams.pubmedqa_weight,
            "EvidenceInference": hparams.evidence_inference_weight,
        }
        
        self.datasets_with_test = ["scifact", "healthver", "covidfact"]
        
        self.dataset_names = hparams.datasets.split(",")
        for name in self.dataset_names:
            assert name in self.reader_lookup
        
        self.readers = []
        for name in self.dataset_names:
            reader_class = self.reader_lookup[name]
            # Only subclasses of SciFactReader take the `fewshot` argument.
            from .data_train import SciFactReader
            if issubclass(reader_class, SciFactReader):
                reader = reader_class(self.fewshot, self.debug)
            else:
                reader = reader_class(self.debug)
            self.readers.append(reader)
        
        # Cache to avoid reloading data
        self._setup_done = False
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--datasets", type=str, required=True)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--no_shuffle", action="store_true")
        parser.add_argument("--no_reweight_labels", action="store_true")
        parser.add_argument("--no_reweight_datasets", action="store_true")
        parser.add_argument("--max_label_weight", type=float, default=3.0)
        parser.add_argument("--max_dataset_weight", type=float, default=10.0)
        parser.add_argument("--cap_fever_nsamples", action="store_true")
        parser.add_argument("--scifact_weight", type=float, default=1.0)
        parser.add_argument("--scifact_open_weight", type=float, default=1.0)
        parser.add_argument("--healthver_weight", type=float, default=1.0)
        parser.add_argument("--covidfact_weight", type=float, default=1.0)
        parser.add_argument("--fever_weight", type=float, default=1.0)
        parser.add_argument("--pubmedqa_weight", type=float, default=1.0)
        parser.add_argument("--evidence_inference_weight", type=float, default=1.0)
        
        # === PICO: New argument for PICO feature directory ===
        parser.add_argument("--pico_feature_dir", type=str, default=None,
                           help="Directory containing claims_pico.pt and corpus_pico.pt")
        
        return parser
    
    def setup(self, stage=None):
        """Setup datasets with PICO features - follows original ConcatDataModule logic."""
        # Avoid reloading data if already done
        if self._setup_done:
            return
        
        print(f"Setting up data for stage: {stage}")
        
        # === PICO: Construct paths to PICO cache files ===
        pico_claims_cache = None
        pico_corpus_cache = None
        
        if self.pico_feature_dir:
            # Assume structure: pico_feature_dir/claims_{fold}_pico.pt, corpus_pico.pt
            pico_corpus_cache = os.path.join(self.pico_feature_dir, "corpus_pico.pt")
        
        # Not all datasets have test sets (e.g., original SciFact lacks claims_test.jsonl).
        if self._has_test_split():
            test_fold = self._process_fold("test", pico_claims_cache, pico_corpus_cache)
        else:
            test_fold = None

        self.folds = {
            "train": self._process_fold("train", pico_claims_cache, pico_corpus_cache),
            "val": self._process_fold("val", pico_claims_cache, pico_corpus_cache),
            "test": test_fold,
        }
        
        self._setup_done = True
        print(f"✓ Data setup completed")
    
    def _has_test_split(self):
        """Check whether any reader actually has a test split on disk."""
        from .data_train import SciFactReader
        for reader in self.readers:
            if isinstance(reader, SciFactReader):
                test_file = reader.data_dir / "claims_test.jsonl"
                if test_file.exists():
                    return True
        return False
    
    def _process_fold(self, fold, pico_claims_cache, pico_corpus_cache):
        """Get the data from all the data readers."""
        # Update claims cache path based on fold
        if self.pico_feature_dir:
            fold_map = {"train": "train", "val": "dev", "test": "test"}
            claims_fname = f"claims_{fold_map[fold]}_pico.pt"
            pico_claims_cache = os.path.join(self.pico_feature_dir, claims_fname)
        
        datasets = []
        for reader in self.readers:
            # Only subclasses of SciFactReader have a test set.
            # Import here to avoid circular dependency
            from .data_train import SciFactReader
            if fold == "test" and not isinstance(reader, SciFactReader):
                continue
            
            # Get the base dataset from reader
            base_dataset = reader.get_fold(fold, self.tokenizer)
            
            # === CRITICAL FIX: Convert to PICO dataset ===
            # The readers return BaseSciFactDataset, but we need SciFactPICODataset
            # to load PICO features. Wrap the base dataset with PICO support.
            pico_dataset = SciFactPICODataset(
                entries=base_dataset.entries,
                tokenizer=self.tokenizer,
                dataset_name=base_dataset.dataset_name,
                rationale_mask=base_dataset.rationale_mask,
                pico_claims_cache=pico_claims_cache,
                pico_corpus_cache=pico_corpus_cache
            )
            datasets.append(pico_dataset)

        # Add instance weights (same as original)
        datasets = self._add_instance_weights(datasets)
        datasets = self._sample_instances(datasets, fold)

        return ConcatDataset(datasets)
    
    def _add_instance_weights(self, datasets):
        """Add instance weights for label classes and datasets."""
        import pandas as pd
        import numpy as np
        
        if not self.reweight_datasets:
            dataset_weights = [1.0] * len(datasets)
        else:
            dataset_lengths = [len(x) for x in datasets]
            # Handle empty datasets gracefully
            if not dataset_lengths or max(dataset_lengths) == 0:
                max_len = 1.0
            else:
                max_len = max(dataset_lengths)
            
            dataset_weights = []
            for x in dataset_lengths:
                if x > 0:
                    dataset_weights.append(max_len / x)
                else:
                    dataset_weights.append(0.0)
            
            dataset_weights = [min(x, self.max_dataset_weight) for x in dataset_weights]

        for ds_weight_prelim, dataset in zip(dataset_weights, datasets):
            # Re-weight by our "prior" on dataset quality.
            ds_weight = ds_weight_prelim * self.dataset_weights[dataset.dataset_name]
            entries = dataset.entries
            # If we're not reweighting by label category, just give it the
            # weight of the dataset.
            if not self.reweight_labels:
                for entry in entries:
                    entry["weight"] = ds_weight
            # Otherwise, reweight by label frequency.
            else:
                # Re-weight so that supports and refutes are even, but don't
                # downweight the NEI's,
                if not entries:
                    continue

                labels = [x["to_tensorize"]["label"] for x in entries]
                counts = pd.Series(labels).value_counts()
                
                # Check if SUPPORTS/REFUTES exist
                target_labels = ["SUPPORTS", "REFUTES"]
                present_labels = [l for l in target_labels if l in counts.index]
                
                if not present_labels:
                    # Fallback if no positive labels found
                    for entry in entries:
                        entry["weight"] = ds_weight
                    continue

                label_counts = counts.loc[present_labels]
                max_label_count = label_counts.max()
                label_weights = max_label_count / label_counts
                label_weights = np.minimum(label_weights, self.max_label_weight)
                label_weights["NOT ENOUGH INFO"] = 1.0
                label_weight_dict = label_weights.to_dict()
                for entry in entries:
                    label = entry["to_tensorize"]["label"]
                    # Handle case where label might be in entries but not in target_labels (e.g. NEI)
                    weight = label_weight_dict.get(label, 1.0)
                    entry["weight"] = weight * ds_weight

        return datasets
    
    def _sample_instances(self, datasets, fold):
        """
        If `cap_fever_nsamples` is True, cap PQA at 50K and have FEVER make up
        the rest.
        """
        import random
        
        # If not capping, just return. Also, don't cap for evaluation data.
        if not self.cap_fever_nsamples or (fold != "train"):
            return datasets

        # Otherwise, do some subsampling.
        original_lengths = {dataset.dataset_name: len(dataset) for dataset in datasets}
        if "FEVER" not in original_lengths:
            return datasets

        total_length = original_lengths["FEVER"]

        new_lengths = {k: v for k, v in original_lengths.items()}

        total_non_fever = sum([v for k, v in new_lengths.items() if k != "FEVER"])
        new_fever = total_length - total_non_fever
        new_lengths["FEVER"] = new_fever

        assert sum(new_lengths.values()) == total_length

        # Get new datasets by sampling the originals.
        for name, n_samples in new_lengths.items():
            this_dataset = [entry for entry in datasets if entry.dataset_name == name]
            assert len(this_dataset) == 1
            this_dataset = this_dataset[0]
            new_entries = random.sample(this_dataset.entries, n_samples)
            # Re-set the entries for the dataset
            this_dataset.entries = new_entries

        return datasets
    
    def train_dataloader(self):
        return DataLoader(
            self.folds["train"],
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.folds["val"],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=False,  # Don't keep workers alive between epochs
        )
    
    def test_dataloader(self):
        # If no test set is available (e.g. original SciFact), use validation set for testing
        dataset = self.folds["test"]
        if dataset is None:
            print("Warning: No test set found. Using validation set for testing.")
            dataset = self.folds["val"]
            
        if dataset is None:
            return None
            
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
"""
MultiVerS model enhanced with PICO features using Attention-based aggregation.

This version uses a Scaled Dot-Product Attention mechanism to aggregate PICO span features.
It is designed to be more effective when training with larger negative sample ratios (e.g., scifact_20),
as it allows the model to dynamically focus on the most relevant PICO spans for each claim/sentence.

IMPORTANT: The Longformer encoder is automatically frozen in __init__ to prevent inplace operation
errors during backpropagation. This is a known issue with Longformer's attention mechanism. Only
task-specific layers (classifiers, PICO projection layers) will be trained.
"""

import torch
from torch import nn
from torch.nn import functional as F
import math

from multivers.model import MultiVerSModel
from multivers.allennlp_nn_util import batched_index_select
from multivers.allennlp_feedforward import FeedForward
from .metrics import ClaimDocumentMetrics


class MultiVerSPICOAttnModel(MultiVerSModel):
    """
    Multi-task SciFact model enhanced with BioELECTRA-PICO features.
    Uses Dot-Product Attention for span aggregation.
    """
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MultiVerSModel.add_model_specific_args(parent_parser)

        # === NEW: Hyper-parameters for PICO feature fusion ===
        parser.add_argument("--pico_num_tags", type=int, default=5,
                            help="Number of PICO tag ids (default assumes O + P/I/C/O).")
        parser.add_argument("--pico_token_dim", type=int, default=32,
                            help="Embedding dimension for token-level PICO tags. Set 0 to disable.")
        # PICO Span features
        parser.add_argument("--pico_span_feature_dim", type=int, default=768,
                            help="Dimension of raw span semantic embeddings (e.g., from BioELECTRA).")
        # Note: Projection dim is removed/fixed to hidden_size for dot-product attention
        
        # Statistical features
        parser.add_argument("--pico_sentence_feature_dim", type=int, default=16,
                            help="Dimensionality of sentence-level PICO stats (e.g., counts/scores).")
        parser.add_argument("--pico_claim_feature_dim", type=int, default=16,
                            help="Dimensionality of claim-level PICO stats.")
        parser.add_argument("--pico_dropout", type=float, default=0.1,
                            help="Dropout applied to PICO embeddings/features.")
        parser.add_argument("--rationale_pos_weight", type=float, default=1.0,
                            help="Positive weight for rationale loss (recall booster).")
        parser.add_argument("--label_pos_weight", type=float, default=1.0,
                            help="Positive weight for label loss (boost SUPPORTS/REFUTES vs NEI).")

        return parser

    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.rationale_pos_weight = getattr(hparams, "rationale_pos_weight", 1.0)
        self.label_pos_weight = getattr(hparams, "label_pos_weight", 1.0)

        # === CRITICAL: Freeze encoder to avoid inplace operation errors ===
        # Longformer has known issues with inplace operations during backpropagation
        # Freezing the encoder prevents gradient computation through it, avoiding the error
        print("=" * 60)
        print("Freezing Longformer encoder to avoid inplace operation errors")
        print("Only task-specific layers (classifiers, PICO layers) will be trained")
        print("=" * 60)
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"✓ Encoder frozen: {sum(p.numel() for p in self.encoder.parameters()):,} parameters")
        print(f"✓ Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

        hidden_size = self.encoder.config.hidden_size
        pico_dropout = getattr(hparams, "pico_dropout", 0.1)

        # 1. Token-level PICO Embedding (Optional)
        self.use_pico_tokens = hparams.pico_token_dim > 0 and hparams.pico_num_tags > 0
        if self.use_pico_tokens:
            self.pico_token_embedding = nn.Embedding(
                num_embeddings=hparams.pico_num_tags,
                embedding_dim=hparams.pico_token_dim,
                padding_idx=0
            )
            self.pico_token_dropout = nn.Dropout(pico_dropout)
            self.pico_token_projection = nn.Linear(hparams.pico_token_dim, hidden_size)
        else:
            self.pico_token_embedding = None

        # 2. Span-level PICO Attention Projection
        # We project raw span features (768) to hidden_size to match the Query (Sentence/Claim embedding)
        self.use_span_features = hparams.pico_span_feature_dim > 0
        if self.use_span_features:
            self.pico_span_proj = nn.Sequential(
                nn.Linear(hparams.pico_span_feature_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(pico_dropout)
            )
            # Output projection after attention aggregation (optional but good for mixing)
            self.pico_span_out_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(pico_dropout)
            )
        else:
            self.pico_span_proj = None

        # 3. Statistical Features Projection
        self.use_sentence_features = hparams.pico_sentence_feature_dim > 0
        if self.use_sentence_features:
            self.pico_sentence_stat_proj = nn.Sequential(
                nn.Linear(hparams.pico_sentence_feature_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(pico_dropout)
            )
        else:
            self.pico_sentence_stat_proj = None

        self.use_claim_features = hparams.pico_claim_feature_dim > 0
        if self.use_claim_features:
            self.pico_claim_stat_proj = nn.Sequential(
                nn.Linear(hparams.pico_claim_feature_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(pico_dropout)
            )
        else:
            self.pico_claim_stat_proj = None

        # === Calculate input dimensions for classifiers ===
        # Rationale Head
        rationale_input_dim = hidden_size * 2
        if self.use_span_features:
            rationale_input_dim += hidden_size
        if self.use_sentence_features:
            rationale_input_dim += hidden_size

        self.rationale_classifier = FeedForward(
            input_dim=rationale_input_dim,
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=[nn.GELU(), nn.Identity()],
            dropout=[self.dropout.p, 0]
        )

        # Label Head
        label_input_dim = hidden_size
        if self.use_span_features:
            label_input_dim += hidden_size
        if self.use_claim_features:
            label_input_dim += hidden_size

        num_labels = getattr(self.hparams, "num_labels", 3)
        self.label_classifier = FeedForward(
            input_dim=label_input_dim,
            num_layers=2,
            hidden_dims=[hidden_size, num_labels],
            activations=[nn.GELU(), nn.Identity()],
            dropout=[self.dropout.p, 0]
        )

        # Buffers for zero-padding
        self.register_buffer("_zero_vec", torch.zeros(1, 1, hidden_size), persistent=False)
        
        # === NEW: Add ClaimDocumentMetrics for alignment with eval_multivers.py ===
        # Only keep train and valid metrics (as requested, remove test and old metrics)
        fold_names = ["train", "valid"]
        claim_doc_metrics = {}
        for name in fold_names:
            claim_doc_metrics[f"claim_doc_metrics_{name}"] = ClaimDocumentMetrics(compute_on_step=False)
        self.claim_doc_metrics = nn.ModuleDict(claim_doc_metrics)

    def _attention_aggregate(self, query, span_features, span_mask):
        """
        Attention-based aggregation of PICO spans.
        """
        b, s, n, d_raw = span_features.shape
        hidden_size = query.size(-1)
        
        # Clone to avoid view issues
        span_features = span_features.contiguous()
        
        # 1. Project Spans
        flat_feats = span_features.reshape(b * s, n, d_raw)
        keys = self.pico_span_proj(flat_feats)  # [b*s, n, hidden_size]
        values = keys
        
        # 2. Prepare Query
        flat_query = query.reshape(b * s, 1, hidden_size)
        
        # 3. Calculate Attention Scores
        scores = torch.bmm(flat_query, keys.transpose(1, 2))
        scores = scores / math.sqrt(hidden_size)
        
        # 4. Apply Mask (FIXED: clamp scores to prevent overflow in softmax)
        if span_mask is not None:
            mask = span_mask.view(b * s, 1, n)
            scores = torch.where(mask > 0, scores, torch.tensor(-1e4, device=scores.device, dtype=scores.dtype))
        
        # Clamp scores to safe range for FP16
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        
        # 5. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 6. Weighted Sum
        context = torch.bmm(attn_weights, values)
        
        # 7. Project
        context = context.view(b, s, hidden_size)
        context = self.pico_span_out_proj(context)
        
        return context

    def forward(
        self,
        tokenized,
        abstract_sent_idx,
        # === NEW Inputs ===
        pico_token_ids=None,            # [batch, seq_len]
        pico_sentence_features=None,    # [batch, max_sent, dim]
        claim_pico_features=None,       # [batch, dim]
        pico_span_features=None,        # [batch, max_sent, max_spans, dim]
        pico_span_mask=None,            # [batch, max_sent, max_spans]
        claim_span_features=None,       # [batch, 1, max_spans, dim]
        claim_span_mask=None,           # [batch, 1, max_spans]
        **kwargs
    ):
        # Clone tensor inputs to avoid shared-storage views that can trigger
        # Longformer autograd complaining about inplace modifications.
        tokenized = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in tokenized.items()
        }

        # 1. Encoder Pass
        # Note: encoder is frozen, so no gradients will flow through it
        # This prevents inplace operation errors in Longformer's attention mechanism
        encoded = self.encoder(**tokenized)
        # Immediately extract and clone to avoid any view-related issues
        hidden_states = encoded.last_hidden_state.clone()
        pooled_output = encoded.pooler_output.clone() if encoded.pooler_output is not None else None

        if self.use_pico_tokens and pico_token_ids is not None:
            pico_emb = self.pico_token_embedding(pico_token_ids)
            pico_emb = self.pico_token_dropout(pico_emb)
            pico_emb = self.pico_token_projection(pico_emb)
            # Use non-inplace addition to avoid autograd issues
            hidden_states = hidden_states + pico_emb

        # Apply dropout (non-inplace operations)
        hidden_states = self.dropout(hidden_states).contiguous()
        if pooled_output is not None:
            pooled_output = self.dropout(pooled_output)

        # 2. Prepare Features for Rationale Classifier (Sentence-level)
        # Ensure hidden_states is not modified inplace before this
        sentence_states = batched_index_select(hidden_states, abstract_sent_idx)
        # [batch, n_sentences, hidden_size]
        # Use clone() to ensure we have a separate tensor for pooled_rep
        if pooled_output is not None:
            pooled_rep = pooled_output.unsqueeze(1).expand_as(sentence_states)
        else:
            # Fallback: use mean pooling if pooler_output is None
            pooled_rep = hidden_states.mean(dim=1, keepdim=True).expand_as(sentence_states)

        rationale_inputs = [pooled_rep, sentence_states]

        # Feature: Span Aggregation (Attention)
        if self.use_span_features:
            if pico_span_features is not None:
                # Align pico_span_features with sentence_states length
                # sentence_states: [batch, n_sents_query, hidden]
                # pico_span_features: [batch, n_sents_pico, n_spans, dim]
                
                n_sents_query = sentence_states.size(1)
                n_sents_pico = pico_span_features.size(1)
                
                if n_sents_query != n_sents_pico:
                    # print(f"Aligning PICO features: query={n_sents_query}, pico={n_sents_pico}")
                    if n_sents_pico > n_sents_query:
                        # Truncate PICO
                        pico_span_features = pico_span_features[:, :n_sents_query, :, :]
                        if pico_span_mask is not None:
                            pico_span_mask = pico_span_mask[:, :n_sents_query, :]
                    else:
                        # Pad PICO
                        diff = n_sents_query - n_sents_pico
                        b, _, n_spans, dim = pico_span_features.shape
                        
                        padding_feats = torch.zeros(b, diff, n_spans, dim, device=pico_span_features.device, dtype=pico_span_features.dtype)
                        pico_span_features = torch.cat([pico_span_features, padding_feats], dim=1)
                        
                        if pico_span_mask is not None:
                            padding_mask = torch.zeros(b, diff, n_spans, device=pico_span_mask.device, dtype=pico_span_mask.dtype)
                            pico_span_mask = torch.cat([pico_span_mask, padding_mask], dim=1)

                # Query: sentence_states (Focus on spans relevant to this sentence's meaning)
                # Ideally, we might want to attend based on Claim info too, but sentence_state
                # implicitly contains some claim info due to self-attention in the encoder.
                # Better yet: Query = sentence_states + pooled_rep (Claim+Doc info)
                # Let's try Query = sentence_states for simplicity and direct relevance.
                span_context = self._attention_aggregate(sentence_states, pico_span_features, pico_span_mask)
            else:
                span_context = self._zero_vec.expand_as(sentence_states)
            rationale_inputs.append(span_context)

        if self.use_sentence_features:
            if pico_sentence_features is not None:
                stat_context = self.pico_sentence_stat_proj(pico_sentence_features)
            else:
                stat_context = self._zero_vec.expand_as(sentence_states)
            rationale_inputs.append(stat_context)

        # Predict Rationales
        rationale_cat = torch.cat(rationale_inputs, dim=2)
        rationale_logits = self.rationale_classifier(rationale_cat).squeeze(2)
        rationale_probs = torch.sigmoid(rationale_logits).detach()
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)

        # 3. Prepare Features for Label Classifier (Document-level)
        if pooled_output is not None:
            label_inputs = [pooled_output]
        else:
            # Fallback: use mean pooling if pooler_output is None
            label_inputs = [hidden_states.mean(dim=1)]

        # Feature: Claim Span Aggregation (Attention)
        if self.use_span_features:
            if claim_span_features is not None:
                # Query: pooled_output (Focus on claim spans relevant to the whole document context)
                # Note: pooled_output is [batch, hidden_size], reshape to [batch, 1, hidden_size]
                if pooled_output is not None:
                    query_claim = pooled_output.unsqueeze(1)
                else:
                    query_claim = hidden_states.mean(dim=1, keepdim=True)
                # claim_span_features is [batch, 1, n_spans, dim]
                claim_span_ctx = self._attention_aggregate(query_claim, claim_span_features, claim_span_mask).squeeze(1)
            else:
                if pooled_output is not None:
                    claim_span_ctx = self._zero_vec.squeeze(1).expand_as(pooled_output)
                else:
                    claim_span_ctx = self._zero_vec.squeeze(1).expand_as(label_inputs[0])
            label_inputs.append(claim_span_ctx)

        if self.use_claim_features:
            if claim_pico_features is not None:
                claim_stat_ctx = self.pico_claim_stat_proj(claim_pico_features)
            else:
                if pooled_output is not None:
                    claim_stat_ctx = self._zero_vec.squeeze(1).expand_as(pooled_output)
                else:
                    claim_stat_ctx = self._zero_vec.squeeze(1).expand_as(label_inputs[0])
            label_inputs.append(claim_stat_ctx)

        # Predict Labels
        label_cat = torch.cat(label_inputs, dim=1)
        label_logits = self.label_classifier(label_cat)
        label_probs = F.softmax(label_logits, dim=1).detach()

        if self.label_threshold is None:
            predicted_labels = label_logits.argmax(dim=1)
        else:
            label_probs_truncated = label_probs.clone()
            label_probs_truncated[:, self.nei_label] = self.label_threshold
            predicted_labels = label_probs_truncated.argmax(dim=1)

        return {
            "label_logits": label_logits,
            "rationale_logits": rationale_logits,
            "label_probs": label_probs,
            "rationale_probs": rationale_probs,
            "predicted_labels": predicted_labels,
            "predicted_rationales": predicted_rationales,
        }

    def training_step(self, batch, batch_idx):
        """Override to pass PICO features from batch."""
        # Extract PICO features from batch if available
        pico_kwargs = {}
        for key in ["pico_span_features", "pico_span_mask", "claim_span_features", 
                    "claim_span_mask", "pico_sentence_features", "claim_pico_features", "pico_token_ids"]:
            if key in batch:
                pico_kwargs[key] = batch[key]
        
        # Call forward with PICO features
        res = self(batch["tokenized"], batch["abstract_sent_idx"], **pico_kwargs)
        
        # Calculate losses
        import torch.nn.functional as F
        
        # Loss for label prediction
        # We use label_pos_weight to penalize predicting NEI (index 1)
        # or rather, to boost SUPPORTS (2) and REFUTES (0).
        # Weights: [pos_weight, 1.0, pos_weight]
        
        label_weights = torch.tensor([self.label_pos_weight, 1.0, self.label_pos_weight], 
                                     device=batch["label"].device)
        
        label_loss_per_sample = F.cross_entropy(
            res["label_logits"], batch["label"], weight=label_weights, reduction="none")
        label_loss = (batch["weight"] * label_loss_per_sample).sum()
        
        # Loss for rationale selection with POS_WEIGHT
        # Custom implementation of masked_binary_cross_entropy_with_logits with pos_weight
        
        logits = res["rationale_logits"]
        targets = batch["rationale"].float() # [batch, n_sents], contains -1 for padding
        weights = batch["weight"]            # [batch]
        dataset_rationale_mask = batch["rationale_mask"].float() # [batch]
        
        # 1. Create sentence-level mask (ignore padding -1)
        sentence_mask = (targets != -1).float() # [batch, n_sents]
        
        # 2. Clamp targets to [0, 1] for BCE (replace -1 with 0) to avoid NaN
        # We multiply by sentence_mask later so these 0s won't affect loss
        clean_targets = targets.clone()
        clean_targets[clean_targets == -1] = 0
        
        # 3. Create pos_weight tensor
        pos_weight = torch.tensor(self.rationale_pos_weight, device=logits.device)
        
        # 4. BCE with logits and pos_weight (reduction='none')
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, clean_targets, pos_weight=pos_weight, reduction="none"
        ) # [batch, n_sents]
        
        # 5. Apply masks
        # sentence_mask: [batch, n_sents] - masks out padding tokens
        # dataset_rationale_mask: [batch] - masks out datasets without rationale annotations
        # We need to unsqueeze dataset_rationale_mask to [batch, 1] for broadcasting
        
        final_mask = sentence_mask * dataset_rationale_mask.unsqueeze(1)
        weighted_loss = bce_loss * final_mask
        
        # 6. Sum over sentences for each instance
        instance_loss = weighted_loss.sum(dim=1) # [batch]
        
        # 7. Apply instance weights and sum batch
        rationale_loss = (instance_loss * weights).sum()
        
        # Total loss (weighted sum)
        # CRITICAL: Keep as SUM to match original MultiVerS optimization dynamics.
        # Dividing by batch_size reduces gradient scale and effectively reduces LR by 4x.
        # We want high-weight batches (difficult examples) to contribute LARGER gradients.
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        
        # For display/logging only: calculate mean
        batch_size = len(batch["label"])
        loss_mean = loss / batch_size
        
        # DEBUG PRINT: Show breakdown occasionally
        if batch_idx % 50 == 0:
            n_nei = (batch["label"] == 1).sum().item()
            n_evidence = batch_size - n_nei
            
            print(f"\n[Batch {batch_idx}] Sum: {loss.item():.2f} (Mean: {loss_mean.item():.2f}) "
                  f"| LabelSum: {label_loss.item():.2f} | RationaleSum: {rationale_loss.item():.2f}")
            if n_evidence > 0:
                print(f"  - Contains {n_evidence} evidence samples (High loss expected)")
        
        # Log losses (Log the SUM to match original metric scale)
        self.log("train_label_loss", label_loss, on_step=False, on_epoch=True)
        self.log("train_rationale_loss", rationale_loss, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Invoke metrics
        self._invoke_metrics(res, batch, "train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Override to pass PICO features from batch during validation and compute loss."""
        # Extract PICO features from batch if available
        pico_kwargs = {}
        for key in ["pico_span_features", "pico_span_mask", "claim_span_features", 
                    "claim_span_mask", "pico_sentence_features", "claim_pico_features", "pico_token_ids"]:
            if key in batch:
                pico_kwargs[key] = batch[key]
        
        # Call forward with PICO features (same as training_step)
        res = self(batch["tokenized"], batch["abstract_sent_idx"], **pico_kwargs)
        
        # Calculate validation loss (same as training_step)
        import torch.nn.functional as F
        
        # Loss for label prediction
        label_weights = torch.tensor([self.label_pos_weight, 1.0, self.label_pos_weight], 
                                     device=batch["label"].device)
        label_loss = F.cross_entropy(
            res["label_logits"], batch["label"], weight=label_weights, reduction="none")
        label_loss = (batch["weight"] * label_loss).sum()
        
        # Loss for rationale selection with POS_WEIGHT (Same as training)
        logits = res["rationale_logits"]
        targets = batch["rationale"].float()
        weights = batch["weight"]
        dataset_rationale_mask = batch["rationale_mask"].float()
        
        # 1. Create sentence-level mask (ignore padding -1)
        sentence_mask = (targets != -1).float()
        
        # 2. Clean targets
        clean_targets = targets.clone()
        clean_targets[clean_targets == -1] = 0
        
        # 3. Pos weight
        pos_weight = torch.tensor(self.rationale_pos_weight, device=logits.device)
        
        # 4. BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, clean_targets, pos_weight=pos_weight, reduction="none"
        )
        
        # 5. Apply masks
        final_mask = sentence_mask * dataset_rationale_mask.unsqueeze(1)
        weighted_loss = bce_loss * final_mask
        
        # 6. Sum
        instance_loss = weighted_loss.sum(dim=1)
        rationale_loss = (instance_loss * weights).sum()
        
        # Total loss (weighted sum)
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        
        # Log validation losses (Sum, consistent with training)
        self.log("val_label_loss", label_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rationale_loss", rationale_loss, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Invoke metrics
        self._invoke_metrics(res, batch, "valid")
    
    def test_step(self, batch, batch_idx):
        """Override to pass PICO features, but skip metrics (we only track train/valid)."""
        # Extract PICO features from batch if available
        pico_kwargs = {}
        for key in ["pico_span_features", "pico_span_mask", "claim_span_features", 
                    "claim_span_mask", "pico_sentence_features", "claim_pico_features", "pico_token_ids"]:
            if key in batch:
                pico_kwargs[key] = batch[key]
        
        # Call forward with PICO features, but don't compute metrics
        pred = self(batch["tokenized"], batch["abstract_sent_idx"], **pico_kwargs)
        # Skip metrics for test (we only track train/valid)
    
    def _invoke_metrics(self, pred, batch, fold):
        """
        Override to only use ClaimDocumentMetrics (aligned with eval_multivers.py).
        Removed old SciFactMetrics to keep only the metrics we need.
        """
        # Only invoke ClaimDocumentMetrics for train and valid
        if fold in ["train", "valid"]:
            detached = {k: v.detach() for k, v in pred.items()}
            self.claim_doc_metrics[f"claim_doc_metrics_{fold}"](detached, batch)
    
    def _log_metrics(self, fold):
        """
        Override to only log ClaimDocumentMetrics (aligned with eval_multivers.py).
        Removed old SciFactMetrics to keep only the metrics we need.
        """
        # Only log ClaimDocumentMetrics for train and valid
        if fold in ["train", "valid"]:
            the_metric = self.claim_doc_metrics[f"claim_doc_metrics_{fold}"]
            to_log = the_metric.compute()
            the_metric.reset()
            for k, v in to_log.items():
                # Log without "claim_doc_" prefix to match eval_multivers.py output names
                self.log(f"{fold}_{k}", v)


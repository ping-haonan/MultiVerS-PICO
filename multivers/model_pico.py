"""
Extended MultiVerS model with PICO-aware feature fusion.

This module adds token-level, sentence-level, and claim-level PICO signals
(predicted by BioELECTRA-PICO) into the original MultiVerS architecture.
"""

import torch
from torch import nn
from torch.nn import functional as F

from multivers.model import MultiVerSModel
from multivers.allennlp_nn_util import batched_index_select
from multivers.allennlp_feedforward import FeedForward


class MultiVerSPICOModel(MultiVerSModel):
    """
    Multi-task SciFact model enhanced with BioELECTRA-PICO features.
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MultiVerSModel.add_model_specific_args(parent_parser)

        # === NEW: hyper-parameters for PICO feature fusion ===
        parser.add_argument("--pico_num_tags", type=int, default=5,
                            help="Number of PICO tag ids (default assumes O + P/I/C/O).")
        parser.add_argument("--pico_token_dim", type=int, default=32,
                            help="Embedding dimension for token-level PICO tags.")
        parser.add_argument("--pico_sentence_feature_dim", type=int, default=16,
                            help="Dimensionality of sentence-level PICO features per sentence.")
        parser.add_argument("--pico_claim_feature_dim", type=int, default=16,
                            help="Dimensionality of claim-level PICO feature vector.")
        parser.add_argument("--pico_dropout", type=float, default=0.1,
                            help="Dropout applied to PICO embeddings/features.")

        return parser

    def __init__(self, hparams):
        super().__init__(hparams)

        hidden_size = self.encoder.config.hidden_size
        pico_dropout_p = getattr(hparams, "pico_dropout", 0.1)

        # === NEW: token-level PICO embedding (aligned with Longformer tokens) ===
        self.pico_token_embedding = nn.Embedding(
            num_embeddings=hparams.pico_num_tags,
            embedding_dim=hparams.pico_token_dim,
            padding_idx=0
        )
        self.pico_token_dropout = nn.Dropout(pico_dropout_p)
        self.pico_token_projection = nn.Linear(
            hparams.pico_token_dim,
            hidden_size
        )

        # === NEW: sentence-level PICO feature projection ===
        self.pico_sentence_proj = nn.Sequential(
            nn.Linear(hparams.pico_sentence_feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(pico_dropout_p),
        )

        # === NEW: claim-level PICO feature projection ===
        self.pico_claim_proj = nn.Sequential(
            nn.Linear(hparams.pico_claim_feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(pico_dropout_p),
        )

        # === NEW: re-initialize classifiers to match extended input dims ===
        self.label_classifier = FeedForward(
            input_dim=hidden_size * 2,  # pooled output + claim PICO projection
            num_layers=2,
            hidden_dims=[hidden_size, self.hparams.num_labels],
            activations=[nn.GELU(), nn.Identity()],
            dropout=[self.dropout.p, 0],
        )

        self.rationale_classifier = FeedForward(
            input_dim=hidden_size * 3,  # pooled rep + sentence rep + sentence PICO projection
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=[nn.GELU(), nn.Identity()],
            dropout=[self.dropout.p, 0],
        )

        # === NEW: default buffers when PICO features are missing ===
        self.register_buffer(
            "_pico_sentence_pad",
            torch.zeros(1, 1, hidden_size),
            persistent=False
        )
        self.register_buffer(
            "_pico_claim_pad",
            torch.zeros(1, hidden_size),
            persistent=False
        )

    def forward(
        self,
        tokenized,
        abstract_sent_idx,
        pico_token_ids=None,
        pico_sentence_features=None,
        claim_pico_features=None,
    ):
        """
        Arguments:
            tokenized: standard tokenizer outputs for claim/abstract pair.
            abstract_sent_idx: indices of </s> tokens representing each sentence.
            pico_token_ids: Longformer-length tensor with PICO tag ids.
            pico_sentence_features: [batch, max_sent, feature_dim] sentence features.
            claim_pico_features: [batch, feature_dim] claim-level PICO vector.
        """
        encoded = self.encoder(**tokenized)

        # === NEW: inject token-level PICO signals into hidden states ===
        hidden_states = encoded.last_hidden_state
        if pico_token_ids is not None:
            pico_emb = self.pico_token_embedding(pico_token_ids)
            pico_emb = self.pico_token_dropout(pico_emb)
            pico_emb = self.pico_token_projection(pico_emb)
            hidden_states = hidden_states + pico_emb
        hidden_states = self.dropout(hidden_states).contiguous()

        # Label prediction branch -------------------------------------------------
        pooled_output = self.dropout(encoded.pooler_output)

        # === NEW: fuse claim-level PICO features ===
        if claim_pico_features is not None:
            claim_pico_proj = self.pico_claim_proj(claim_pico_features)
        else:
            claim_pico_proj = self._pico_claim_pad.expand(pooled_output.size(0), -1)

        pooled_fused = torch.cat([pooled_output, claim_pico_proj], dim=1)
        label_logits = self.label_classifier(pooled_fused)
        label_probs = F.softmax(label_logits, dim=1).detach()

        if self.label_threshold is None:
            predicted_labels = label_logits.argmax(dim=1)
        else:
            label_probs_truncated = label_probs.clone()
            label_probs_truncated[:, self.nei_label] = self.label_threshold
            predicted_labels = label_probs_truncated.argmax(dim=1)

        # Rationale prediction branch ---------------------------------------------
        sentence_states = batched_index_select(hidden_states, abstract_sent_idx)
        pooled_rep = pooled_output.unsqueeze(1).expand_as(sentence_states)

        # === NEW: fuse sentence-level PICO features ===
        if pico_sentence_features is not None:
            pico_sentence_proj = self.pico_sentence_proj(pico_sentence_features)
        else:
            pico_sentence_proj = self._pico_sentence_pad.expand(
                sentence_states.size(0),
                sentence_states.size(1),
                -1,
            )

        rationale_input = torch.cat(
            [pooled_rep, sentence_states, pico_sentence_proj],
            dim=2,
        )
        rationale_logits = self.rationale_classifier(rationale_input).squeeze(2)
        rationale_probs = torch.sigmoid(rationale_logits).detach()
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)

        return {
            "label_logits": label_logits,
            "rationale_logits": rationale_logits,
            "label_probs": label_probs,
            "rationale_probs": rationale_probs,
            "predicted_labels": predicted_labels,
            "predicted_rationales": predicted_rationales,
        }
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class _LoRALinear(nn.Module):
    """Minimal LoRA wrapper around a Linear layer."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A = nn.Parameter(torch.zeros(base.out_features, r))
        self.B = nn.Parameter(torch.zeros(r, base.in_features))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base_out = self.base(x)
        if self.r == 0:
            return base_out
        lora_out = self.dropout(x) @ self.B.t()
        lora_out = lora_out @ self.A.t() * self.scaling
        return base_out + lora_out


class ChemBERTaEncoder(nn.Module):
    """Wrapper around a ChemBERTa-like HF model with partial freezing and LoRA."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        freeze_lower_layers: int = 6,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

        encoder = getattr(self.backbone, "encoder", None)
        if encoder is None and hasattr(self.backbone, "transformer"):
            encoder = self.backbone.transformer
        self._encoder = encoder

        if self._encoder is not None and hasattr(self._encoder, "layer"):
            for layer in self._encoder.layer[:freeze_lower_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
            self._inject_lora(start_layer=freeze_lower_layers, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

        self.out_dim = self.backbone.config.hidden_size

    def _inject_lora(self, start_layer: int, rank: int, alpha: int, dropout: float) -> None:
        if self._encoder is None or not hasattr(self._encoder, "layer"):
            return
        for idx, layer in enumerate(self._encoder.layer[start_layer:], start=start_layer):
            attn = getattr(layer, "attention", None)
            self_attn = getattr(attn, "self", None) if attn is not None else None
            if self_attn is None:
                continue
            for proj_name in ("query", "key", "value"):
                proj = getattr(self_attn, proj_name, None)
                if isinstance(proj, nn.Linear):
                    setattr(self_attn, proj_name, _LoRALinear(proj, rank, alpha, dropout))

    def encode_smiles(self, smiles: Sequence[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            list(smiles),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.backbone(**encoded)
        if getattr(out, "pooler_output", None) is not None:
            emb = out.pooler_output
        else:
            emb = out.last_hidden_state[:, 0, :]
        return emb

    def forward(self, smiles: Sequence[str], device: torch.device) -> torch.Tensor:  # type: ignore[override]
        return self.encode_smiles(smiles, device=device)

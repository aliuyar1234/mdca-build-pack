from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import torch
import torch.nn.functional as F

from src.core.schemas import Site

FAMILY_CANONICAL = "canonical"
FAMILY_SHIFT = "shift"
INTERVENTION_VARS = ("N1", "N2", "R")
NAME_VOCAB = ("Alice", "Bob", "Carol", "Dave")
OTHER_TOKEN = "OTHER"
CANONICAL_TEMPLATE = ("When", "{N1}", "and", "{N2}", "met,", "{SUBJ}", "gave", "a", "gift", "to")
SHIFT_TEMPLATE = ("After", "{N1}", "and", "{N2}", "spoke,", "{SUBJ}", "handed", "a", "note", "to")
POSITION_NAMES = {
    0: "lead",
    1: "N1_in",
    2: "and",
    3: "N2_in",
    4: "context",
    5: "SUBJ_in",
    6: "action",
    7: "article",
    8: "object",
    9: "query",
}
N_LAYERS = 2
N_HEADS = 4
D_MODEL = 128
D_MLP = 512
CONTEXT_LENGTH = len(CANONICAL_TEMPLATE)


def stable_example_id(*parts: str) -> str:
    return sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]


def _site_sort_key(site: Site) -> tuple[int, int]:
    return (site.layer_index, site.token_index)


@dataclass(frozen=True, slots=True)
class MiniIOILatents:
    n1_index: int
    n2_index: int
    r_value: int
    family: str

    def __post_init__(self) -> None:
        if not (0 <= self.n1_index < len(NAME_VOCAB)):
            raise ValueError("n1_index out of range")
        if not (0 <= self.n2_index < len(NAME_VOCAB)):
            raise ValueError("n2_index out of range")
        if self.n1_index == self.n2_index:
            raise ValueError("n1_index and n2_index must be distinct")
        if self.r_value not in (1, 2):
            raise ValueError("r_value must be 1 or 2")
        if self.family not in {FAMILY_CANONICAL, FAMILY_SHIFT}:
            raise ValueError("family must be canonical or shift")

    @property
    def n1_token(self) -> str:
        return NAME_VOCAB[self.n1_index]

    @property
    def n2_token(self) -> str:
        return NAME_VOCAB[self.n2_index]

    @property
    def subj_token(self) -> str:
        return self.n1_token if self.r_value == 1 else self.n2_token

    @property
    def target_token(self) -> str:
        return self.n2_token if self.r_value == 1 else self.n1_token

    def to_dict(self) -> dict[str, int | str]:
        return {
            "N1_index": self.n1_index,
            "N1_token": self.n1_token,
            "N2_index": self.n2_index,
            "N2_token": self.n2_token,
            "R": self.r_value,
            "SUBJ_token": self.subj_token,
            "target_token": self.target_token,
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "MiniIOILatents":
        return cls(
            n1_index=int(data["N1_index"]),
            n2_index=int(data["N2_index"]),
            r_value=int(data["R"]),
            family=str(data["family"]),
        )

    def abstract_difference(self, other: "MiniIOILatents") -> list[str]:
        diffs: list[str] = []
        if self.n1_index != other.n1_index:
            diffs.append("N1")
        if self.n2_index != other.n2_index:
            diffs.append("N2")
        if self.r_value != other.r_value:
            diffs.append("R")
        return diffs


@dataclass(frozen=True, slots=True)
class MiniIOIRun:
    latents: MiniIOILatents
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    activations: tuple[torch.Tensor, ...]
    output_index: int
    output_logits: torch.Tensor
    tokenizer_vocab: tuple[str, ...]

    @property
    def output_token(self) -> str:
        return self.tokenizer_vocab[self.output_index]

    def site_activation(self, site: Site) -> torch.Tensor:
        return self.activations[site.layer_index][site.token_index].clone()


class MiniIOITokenizer:
    def __init__(self, *, name_vocab: tuple[str, ...] = NAME_VOCAB) -> None:
        fixed_words = (
            "When",
            "and",
            "met,",
            "gave",
            "a",
            "gift",
            "to",
            "After",
            "spoke,",
            "handed",
            "note",
        )
        vocab = tuple(name_vocab) + fixed_words
        seen: set[str] = set()
        ordered_vocab: list[str] = []
        for token in vocab:
            if token in seen:
                continue
            seen.add(token)
            ordered_vocab.append(token)
        self.vocab = tuple(ordered_vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.name_vocab = tuple(name_vocab)

    def encode(self, tokens: Iterable[str]) -> tuple[int, ...]:
        token_ids: list[int] = []
        for token in tokens:
            if token not in self.token_to_id:
                raise KeyError(f"Unknown token: {token}")
            token_ids.append(self.token_to_id[token])
        return tuple(token_ids)

    def decode(self, token_ids: Iterable[int]) -> tuple[str, ...]:
        return tuple(self.id_to_token[int(token_id)] for token_id in token_ids)


class DecoderBlock(torch.nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, d_mlp: int) -> None:
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_mlp),
            torch.nn.ReLU(),
            torch.nn.Linear(d_mlp, d_model),
        )

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        normalized = self.ln1(hidden)
        attn_output, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden = residual + attn_output
        hidden = hidden + self.mlp(self.ln2(hidden))
        return hidden


class MiniIOITransformer(torch.nn.Module):
    def __init__(
        self,
        *,
        model_seed: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        torch.manual_seed(model_seed)
        self.model_seed = model_seed
        self.dtype = dtype
        self.name_vocab = NAME_VOCAB
        self.tokenizer = MiniIOITokenizer(name_vocab=self.name_vocab)
        self.sequence_length = CONTEXT_LENGTH
        self.n_layers = N_LAYERS
        self.n_heads = N_HEADS
        self.d_model = D_MODEL
        self.d_mlp = D_MLP
        self.position_names = POSITION_NAMES
        self.site_universe = tuple(
            Site(layer_index=layer_index, token_index=token_index)
            for layer_index in range(self.n_layers + 1)
            for token_index in range(self.sequence_length)
        )
        self.site_index = {
            (site.layer_index, site.token_index): idx
            for idx, site in enumerate(self.site_universe)
        }

        self.token_embeddings = torch.nn.Embedding(len(self.tokenizer.vocab), self.d_model)
        self.position_embeddings = torch.nn.Embedding(self.sequence_length, self.d_model)
        self.blocks = torch.nn.ModuleList(
            [
                DecoderBlock(d_model=self.d_model, n_heads=self.n_heads, d_mlp=self.d_mlp)
                for _ in range(self.n_layers)
            ]
        )
        self.final_ln = torch.nn.LayerNorm(self.d_model)
        self.lm_head = torch.nn.Linear(self.d_model, len(self.tokenizer.vocab), bias=False)
        self.to(dtype=self.dtype)

    def render_prompt_tokens(self, latents: MiniIOILatents) -> tuple[str, ...]:
        template = CANONICAL_TEMPLATE if latents.family == FAMILY_CANONICAL else SHIFT_TEMPLATE
        rendered: list[str] = []
        for token in template:
            if token == "{N1}":
                rendered.append(latents.n1_token)
            elif token == "{N2}":
                rendered.append(latents.n2_token)
            elif token == "{SUBJ}":
                rendered.append(latents.subj_token)
            else:
                rendered.append(token)
        return tuple(rendered)

    def encode_prompt(self, latents: MiniIOILatents) -> tuple[int, ...]:
        return self.tokenizer.encode(self.render_prompt_tokens(latents))

    def site_id(self, site: Site) -> int:
        return self.site_index[(site.layer_index, site.token_index)]

    def site_table_records(self) -> list[dict[str, int | str]]:
        return [
            {
                "site_id": self.site_id(site),
                "layer_index": site.layer_index,
                "token_index": site.token_index,
                "position_name": self.position_names[site.token_index],
            }
            for site in self.site_universe
        ]

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def _apply_patches(
        self,
        hidden: torch.Tensor,
        *,
        layer_index: int,
        patch_sites_by_layer: dict[int, tuple[Site, ...]],
        source_run: MiniIOIRun | None,
    ) -> torch.Tensor:
        if source_run is None:
            return hidden
        patched = hidden.clone()
        for site in patch_sites_by_layer.get(layer_index, ()):
            patched[0, site.token_index] = source_run.site_activation(site)
        return patched

    def _forward_hidden(
        self,
        token_ids: tuple[int, ...],
        *,
        latents: MiniIOILatents,
        source_run: MiniIOIRun | None,
        patch_sites: tuple[Site, ...],
    ) -> MiniIOIRun:
        device = next(self.parameters()).device
        inputs = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        positions = torch.arange(len(token_ids), device=device, dtype=torch.long).unsqueeze(0)
        patch_sites_by_layer: dict[int, tuple[Site, ...]] = {}
        for site in sorted(set(patch_sites), key=_site_sort_key):
            patch_sites_by_layer.setdefault(site.layer_index, tuple())
            patch_sites_by_layer[site.layer_index] = tuple(
                sorted(
                    (*patch_sites_by_layer[site.layer_index], site),
                    key=_site_sort_key,
                )
            )

        hidden = self.token_embeddings(inputs) + self.position_embeddings(positions)
        hidden = self._apply_patches(
            hidden,
            layer_index=0,
            patch_sites_by_layer=patch_sites_by_layer,
            source_run=source_run,
        )
        activations = [hidden[0].detach().cpu().clone().to(dtype=self.dtype)]

        attn_mask = self._causal_mask(len(token_ids), inputs.device)
        for layer_index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden, attn_mask)
            hidden = self._apply_patches(
                hidden,
                layer_index=layer_index,
                patch_sites_by_layer=patch_sites_by_layer,
                source_run=source_run,
            )
            activations.append(hidden[0].detach().cpu().clone().to(dtype=self.dtype))

        final_hidden = self.final_ln(hidden[:, -1, :])
        logits = self.lm_head(final_hidden)[0]
        output_index = int(torch.argmax(logits).item())
        return MiniIOIRun(
            latents=latents,
            tokens=self.render_prompt_tokens(latents),
            token_ids=token_ids,
            activations=tuple(activations),
            output_index=output_index,
            output_logits=logits.detach().cpu().clone().to(dtype=self.dtype),
            tokenizer_vocab=self.tokenizer.vocab,
        )

    def run_clean(self, latents: MiniIOILatents) -> MiniIOIRun:
        self.eval()
        with torch.no_grad():
            return self._forward_hidden(
                self.encode_prompt(latents),
                latents=latents,
                source_run=None,
                patch_sites=(),
            )

    def patch_and_run(
        self,
        *,
        base_latents: MiniIOILatents,
        source_latents: MiniIOILatents,
        intervention_type: str,
        patch_sites: Iterable[Site],
        allow_misaligned_source: bool = False,
    ) -> MiniIOIRun:
        if intervention_type not in INTERVENTION_VARS:
            raise ValueError(f"Unknown intervention_type: {intervention_type}")
        diffs = base_latents.abstract_difference(source_latents)
        if not allow_misaligned_source and diffs != [intervention_type]:
            raise ValueError("base and source must differ in exactly the intervention variable")
        if base_latents.family != source_latents.family:
            raise ValueError("base and source must share the same template family")
        source_run = self.run_clean(source_latents)
        self.eval()
        with torch.no_grad():
            return self._forward_hidden(
                self.encode_prompt(base_latents),
                latents=base_latents,
                source_run=source_run,
                patch_sites=tuple(sorted(set(patch_sites), key=_site_sort_key)),
            )

    def predict_logits_batch(self, token_batch: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_batch.shape[1], device=token_batch.device).unsqueeze(0)
        hidden = self.token_embeddings(token_batch) + self.position_embeddings(positions)
        attn_mask = self._causal_mask(token_batch.shape[1], token_batch.device)
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        final_hidden = self.final_ln(hidden[:, -1, :])
        return self.lm_head(final_hidden)

    def train_on_canonical_prompts(
        self,
        *,
        canonical_latents: Iterable[MiniIOILatents],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
    ) -> dict[str, object]:
        device = next(self.parameters()).device
        examples = sorted(
            canonical_latents,
            key=lambda latents: (latents.n1_index, latents.n2_index, latents.r_value),
        )
        inputs = torch.tensor(
            [self.encode_prompt(latents) for latents in examples],
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            [self.tokenizer.token_to_id[latents.target_token] for latents in examples],
            dtype=torch.long,
            device=device,
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        history: list[dict[str, float | int]] = []
        for epoch in range(epochs):
            permutation = torch.randperm(inputs.shape[0], device=device)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_count = 0
            for start in range(0, inputs.shape[0], batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_targets = targets[batch_indices]
                optimizer.zero_grad()
                logits = self.predict_logits_batch(batch_inputs)
                loss = F.cross_entropy(logits, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * int(batch_targets.shape[0])
                epoch_correct += int((torch.argmax(logits, dim=1) == batch_targets).sum().item())
                epoch_count += int(batch_targets.shape[0])
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss / max(epoch_count, 1),
                    "accuracy": epoch_correct / max(epoch_count, 1),
                }
            )

        self.eval()
        with torch.no_grad():
            logits = self.predict_logits_batch(inputs)
            predictions = torch.argmax(logits, dim=1)
            final_loss = float(F.cross_entropy(logits, targets).item())
            final_accuracy = float((predictions == targets).float().mean().item())
        return {
            "n_training_examples": int(inputs.shape[0]),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "history_tail": history[-10:],
        }

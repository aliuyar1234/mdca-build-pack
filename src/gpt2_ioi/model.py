from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar, Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

from src.core.schemas import Site

FAMILY_CANONICAL = "canonical"
FAMILY_SHIFT = "shift"
INTERVENTION_VARS = ("N1", "N2", "R")
MODEL_NAME = "gpt2"
GPT2_NAME_CANDIDATES = (
    "Alice",
    "Bob",
    "Carol",
    "Dave",
    "Erin",
    "John",
    "Mary",
    "Anna",
    "James",
    "Sarah",
    "Tom",
    "Emma",
    "Liam",
    "Noah",
)
DEFAULT_NAME_VOCAB = ("Alice", "Bob", "Carol")
NAME_VOCAB = DEFAULT_NAME_VOCAB
OTHER_TOKEN = "OTHER"
CANONICAL_TEMPLATE = "When {N1} and {N2} met, {SUBJ} gave a gift to"
SHIFT_TEMPLATE = "After {N1} and {N2} spoke, {SUBJ} handed a note to"


def stable_example_id(*parts: str) -> str:
    return sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]


def _site_sort_key(site: Site) -> tuple[int, int]:
    return (site.layer_index, site.token_index)


@dataclass(frozen=True, slots=True)
class GPT2IOILatents:
    NAME_VOCAB: ClassVar[tuple[str, ...]] = DEFAULT_NAME_VOCAB
    n1_index: int
    n2_index: int
    r_value: int
    family: str

    def __post_init__(self) -> None:
        if not (0 <= self.n1_index < len(self.NAME_VOCAB)):
            raise ValueError("n1_index out of range")
        if not (0 <= self.n2_index < len(self.NAME_VOCAB)):
            raise ValueError("n2_index out of range")
        if self.n1_index == self.n2_index:
            raise ValueError("n1_index and n2_index must be distinct")
        if self.r_value not in (1, 2):
            raise ValueError("r_value must be 1 or 2")
        if self.family not in {FAMILY_CANONICAL, FAMILY_SHIFT}:
            raise ValueError("family must be canonical or shift")

    @property
    def n1_token(self) -> str:
        return self.NAME_VOCAB[self.n1_index]

    @property
    def n2_token(self) -> str:
        return self.NAME_VOCAB[self.n2_index]

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
    def from_dict(cls, data: dict[str, object]) -> "GPT2IOILatents":
        return cls(
            n1_index=int(data["N1_index"]),
            n2_index=int(data["N2_index"]),
            r_value=int(data["R"]),
            family=str(data["family"]),
        )

    @classmethod
    def set_name_vocab(cls, name_vocab: tuple[str, ...]) -> None:
        cls.NAME_VOCAB = tuple(name_vocab)

    def abstract_difference(self, other: "GPT2IOILatents") -> list[str]:
        diffs: list[str] = []
        if self.n1_index != other.n1_index:
            diffs.append("N1")
        if self.n2_index != other.n2_index:
            diffs.append("N2")
        if self.r_value != other.r_value:
            diffs.append("R")
        return diffs


@dataclass(frozen=True, slots=True)
class GPT2IOIRun:
    latents: GPT2IOILatents
    prompt_text: str
    prompt_tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    activations: tuple[torch.Tensor, ...]
    output_index: int
    output_token_raw: str
    output_token_normalized: str
    output_logits: torch.Tensor

    @property
    def output_token(self) -> str:
        return self.output_token_normalized

    def site_activation(self, site: Site) -> torch.Tensor:
        return self.activations[site.layer_index][site.token_index].clone()


def normalize_output_token(token: str) -> str:
    return token.strip()


def select_name_vocab(
    tokenizer: AutoTokenizer,
    *,
    candidate_names: tuple[str, ...] = GPT2_NAME_CANDIDATES,
    target_size: int = 3,
) -> tuple[str, ...]:
    valid: list[str] = []
    for name in candidate_names:
        token_ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(token_ids) == 1:
            valid.append(name)
        if len(valid) == target_size:
            return tuple(valid)
    raise ValueError("No tokenizer-valid single-token name list found for GPT-2-small")


def validate_fixed_length_prompts(
    tokenizer: AutoTokenizer,
    *,
    name_vocab: tuple[str, ...],
) -> dict[str, object]:
    name_token_lengths = {
        name: len(tokenizer.encode(" " + name, add_special_tokens=False))
        for name in name_vocab
    }
    if any(length != 1 for length in name_token_lengths.values()):
        raise ValueError("Selected GPT-2 names are not all single tokens")

    prompt_lengths: dict[str, list[int]] = {
        FAMILY_CANONICAL: [],
        FAMILY_SHIFT: [],
    }
    example_prompts: dict[str, str] = {}
    for family in (FAMILY_CANONICAL, FAMILY_SHIFT):
        for n1 in name_vocab:
            for n2 in name_vocab:
                if n1 == n2:
                    continue
                for subj in (n1, n2):
                    prompt = render_prompt_text(n1=n1, n2=n2, subj=subj, family=family)
                    prompt_lengths[family].append(
                        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                    )
                    example_prompts.setdefault(family, prompt)
    canonical_lengths = sorted(set(prompt_lengths[FAMILY_CANONICAL]))
    shift_lengths = sorted(set(prompt_lengths[FAMILY_SHIFT]))
    if len(canonical_lengths) != 1 or len(shift_lengths) != 1:
        raise ValueError("GPT-2 prompt lengths vary within canonical or shift family")
    if canonical_lengths[0] != shift_lengths[0]:
        raise ValueError("GPT-2 canonical and shift families do not share a fixed length")
    return {
        "model_name": MODEL_NAME,
        "name_vocab": list(name_vocab),
        "name_token_lengths": name_token_lengths,
        "canonical_prompt_lengths": canonical_lengths,
        "shift_prompt_lengths": shift_lengths,
        "fixed_length": canonical_lengths[0],
        "lexical_adjustment_applied": False,
        "example_prompts": example_prompts,
    }


def render_prompt_text(*, n1: str, n2: str, subj: str, family: str) -> str:
    template = CANONICAL_TEMPLATE if family == FAMILY_CANONICAL else SHIFT_TEMPLATE
    return template.format(N1=n1, N2=n2, SUBJ=subj)


class GPT2IOIModel:
    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        pretrained: bool = True,
        model_seed: int = 0,
        name_vocab: tuple[str, ...] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.name_vocab = name_vocab or select_name_vocab(self.tokenizer, target_size=3)
        GPT2IOILatents.set_name_vocab(self.name_vocab)
        self.validation_summary = validate_fixed_length_prompts(
            self.tokenizer,
            name_vocab=self.name_vocab,
        )
        if pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        else:
            torch.manual_seed(model_seed)
            config = AutoConfig.from_pretrained(MODEL_NAME)
            self.model = GPT2LMHeadModel(config)
        self.model.to(self.device)
        self.model.eval()
        self.n_layers = int(self.model.config.n_layer)
        self.sequence_length = int(self.validation_summary["fixed_length"])
        self.d_model = int(self.model.config.n_embd)
        self.position_names = self._build_position_names()
        self.site_universe = tuple(
            Site(layer_index=layer_index, token_index=token_index)
            for layer_index in range(self.n_layers + 1)
            for token_index in range(self.sequence_length)
        )
        self.site_index = {
            (site.layer_index, site.token_index): idx
            for idx, site in enumerate(self.site_universe)
        }

    def _build_position_names(self) -> dict[int, str]:
        sample = render_prompt_text(
            n1=self.name_vocab[0],
            n2=self.name_vocab[1],
            subj=self.name_vocab[0],
            family=FAMILY_CANONICAL,
        )
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(sample, add_special_tokens=False)["input_ids"]
        )
        return {index: token for index, token in enumerate(tokens)}

    def render_prompt_text(self, latents: GPT2IOILatents) -> str:
        return render_prompt_text(
            n1=self.name_vocab[latents.n1_index],
            n2=self.name_vocab[latents.n2_index],
            subj=self.name_vocab[latents.n1_index if latents.r_value == 1 else latents.n2_index],
            family=latents.family,
        )

    def encode_prompt(self, latents: GPT2IOILatents) -> tuple[int, ...]:
        prompt = self.render_prompt_text(latents)
        token_ids = tuple(
            self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        if len(token_ids) != self.sequence_length:
            raise ValueError("Prompt length drifted after GPT-2 validation")
        return token_ids

    def prompt_tokens(self, latents: GPT2IOILatents) -> tuple[str, ...]:
        return tuple(
            self.tokenizer.convert_ids_to_tokens(self.encode_prompt(latents))
        )

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

    def _apply_patch(
        self,
        hidden: torch.Tensor,
        *,
        layer_index: int,
        patch_sites_by_layer: dict[int, tuple[Site, ...]],
        source_run: GPT2IOIRun | None,
    ) -> torch.Tensor:
        if source_run is None:
            return hidden
        patched = hidden.clone()
        for site in patch_sites_by_layer.get(layer_index, ()):
            source_value = source_run.site_activation(site).to(
                device=patched.device,
                dtype=patched.dtype,
            )
            patched[0, site.token_index] = source_value
        return patched

    def _run(
        self,
        *,
        latents: GPT2IOILatents,
        source_run: GPT2IOIRun | None,
        patch_sites: tuple[Site, ...],
    ) -> GPT2IOIRun:
        token_ids = self.encode_prompt(latents)
        prompt_text = self.render_prompt_text(latents)
        prompt_tokens = self.prompt_tokens(latents)
        inputs = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        activations: dict[int, torch.Tensor] = {}

        patch_sites_by_layer: dict[int, tuple[Site, ...]] = {}
        for site in sorted(set(patch_sites), key=_site_sort_key):
            existing = patch_sites_by_layer.get(site.layer_index, tuple())
            patch_sites_by_layer[site.layer_index] = tuple(sorted((*existing, site), key=_site_sort_key))

        handles = []
        transformer = self.model.transformer

        def pre_hook(_module, inputs_tuple):
            hidden = inputs_tuple[0]
            patched = self._apply_patch(
                hidden,
                layer_index=0,
                patch_sites_by_layer=patch_sites_by_layer,
                source_run=source_run,
            )
            activations[0] = patched[0].detach().cpu().clone().to(dtype=self.dtype)
            return (patched, *inputs_tuple[1:])

        def make_block_hook(block_index: int):
            layer_index = block_index + 1

            def hook(_module, _inputs, output):
                hidden = output[0]
                patched = self._apply_patch(
                    hidden,
                    layer_index=layer_index,
                    patch_sites_by_layer=patch_sites_by_layer,
                    source_run=source_run,
                )
                activations[layer_index] = patched[0].detach().cpu().clone().to(dtype=self.dtype)
                if isinstance(output, tuple):
                    return (patched, *output[1:])
                return patched

            return hook

        handles.append(transformer.h[0].register_forward_pre_hook(pre_hook, with_kwargs=False))
        for block_index, block in enumerate(transformer.h):
            handles.append(block.register_forward_hook(make_block_hook(block_index), with_kwargs=False))

        try:
            with torch.no_grad():
                outputs = self.model(inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        logits = outputs.logits[0, -1].detach().cpu().clone().to(dtype=self.dtype)
        output_index = int(torch.argmax(logits).item())
        output_token_raw = self.tokenizer.decode([output_index])
        output_token_normalized = normalize_output_token(output_token_raw)
        ordered_activations = tuple(activations[layer] for layer in range(self.n_layers + 1))
        return GPT2IOIRun(
            latents=latents,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            token_ids=token_ids,
            activations=ordered_activations,
            output_index=output_index,
            output_token_raw=output_token_raw,
            output_token_normalized=output_token_normalized,
            output_logits=logits,
        )

    def run_clean(self, latents: GPT2IOILatents) -> GPT2IOIRun:
        return self._run(latents=latents, source_run=None, patch_sites=())

    def patch_and_run(
        self,
        *,
        base_latents: GPT2IOILatents,
        source_latents: GPT2IOILatents,
        intervention_type: str,
        patch_sites: Iterable[Site],
        allow_misaligned_source: bool = False,
    ) -> GPT2IOIRun:
        if intervention_type not in INTERVENTION_VARS:
            raise ValueError(f"Unknown intervention_type: {intervention_type}")
        diffs = base_latents.abstract_difference(source_latents)
        if not allow_misaligned_source and diffs != [intervention_type]:
            raise ValueError("base and source must differ in exactly the intervention variable")
        if base_latents.family != source_latents.family:
            raise ValueError("base and source must share the same template family")
        source_run = self.run_clean(source_latents)
        return self._run(
            latents=base_latents,
            source_run=source_run,
            patch_sites=tuple(sorted(set(patch_sites), key=_site_sort_key)),
        )

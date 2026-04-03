from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from src.core.schemas import BaseSourceTuple, TupleMetadata

from .model import (
    FAMILY_CANONICAL,
    FAMILY_SHIFT,
    INTERVENTION_VARS,
    GPT2IOILatents,
    GPT2IOIModel,
)


def _stable_key(*parts: str) -> str:
    return sha256("||".join(parts).encode("utf-8")).hexdigest()


def _group_key(
    *,
    setting_id: str,
    family: str,
    base: GPT2IOILatents,
    source: GPT2IOILatents,
    intervention: str,
) -> str:
    return _stable_key(
        setting_id,
        family,
        f"base:{base.n1_index},{base.n2_index},{base.r_value}",
        f"source:{source.n1_index},{source.n2_index},{source.r_value}",
        intervention,
    )


def _prompt_payload(model: GPT2IOIModel, latents: GPT2IOILatents) -> dict[str, object]:
    return {
        "kind": "gpt2_prompt",
        "prompt_text": model.render_prompt_text(latents),
        "prompt_tokens": list(model.prompt_tokens(latents)),
        "token_ids": list(model.encode_prompt(latents)),
        "latents": latents.to_dict(),
    }


def _prompt_id(setting_id: str, latents: GPT2IOILatents) -> str:
    return _stable_key(
        setting_id,
        latents.family,
        str(latents.n1_index),
        str(latents.n2_index),
        str(latents.r_value),
    )[:16]


def _make_tuple(
    *,
    setting_id: str,
    model: GPT2IOIModel,
    base: GPT2IOILatents,
    source: GPT2IOILatents,
    intervention: str,
) -> BaseSourceTuple:
    group_id = _group_key(
        setting_id=setting_id,
        family=base.family,
        base=base,
        source=source,
        intervention=intervention,
    )
    metadata = TupleMetadata(
        group_id=group_id,
        template_family=base.family,
        latent_base=base.to_dict(),
        latent_source=source.to_dict(),
        prompt_id_base=_prompt_id(setting_id, base),
        prompt_id_source=_prompt_id(setting_id, source),
        extra={"setting_id": setting_id},
    )
    return BaseSourceTuple(
        tuple_id=f"{setting_id}_{group_id}",
        base_input=_prompt_payload(model, base),
        source_input=_prompt_payload(model, source),
        intervention_type=intervention,
        metadata=metadata,
    )


def _source_candidates(base: GPT2IOILatents, intervention: str, *, n_names: int) -> Iterable[GPT2IOILatents]:
    if intervention == "N1":
        for candidate_index in range(n_names):
            if candidate_index not in {base.n1_index, base.n2_index}:
                yield GPT2IOILatents(
                    n1_index=candidate_index,
                    n2_index=base.n2_index,
                    r_value=base.r_value,
                    family=base.family,
                )
    elif intervention == "N2":
        for candidate_index in range(n_names):
            if candidate_index not in {base.n1_index, base.n2_index}:
                yield GPT2IOILatents(
                    n1_index=base.n1_index,
                    n2_index=candidate_index,
                    r_value=base.r_value,
                    family=base.family,
                )
    elif intervention == "R":
        yield GPT2IOILatents(
            n1_index=base.n1_index,
            n2_index=base.n2_index,
            r_value=2 if base.r_value == 1 else 1,
            family=base.family,
        )
    else:
        raise ValueError(f"Unknown intervention: {intervention}")


def generate_all_s3_tuples(model: GPT2IOIModel, *, setting_id: str = "gpt2_ioi") -> list[BaseSourceTuple]:
    tuples: list[BaseSourceTuple] = []
    n_names = len(model.name_vocab)
    for family in (FAMILY_CANONICAL, FAMILY_SHIFT):
        for n1_index in range(n_names):
            for n2_index in range(n_names):
                if n1_index == n2_index:
                    continue
                for r_value in (1, 2):
                    base = GPT2IOILatents(
                        n1_index=n1_index,
                        n2_index=n2_index,
                        r_value=r_value,
                        family=family,
                    )
                    for intervention in INTERVENTION_VARS:
                        for source in _source_candidates(base, intervention, n_names=n_names):
                            tuples.append(
                                _make_tuple(
                                    setting_id=setting_id,
                                    model=model,
                                    base=base,
                                    source=source,
                                    intervention=intervention,
                                )
                            )
    return tuples


def assign_splits(tuple_records: list[BaseSourceTuple]) -> dict[str, str]:
    canonical = [
        record
        for record in tuple_records
        if record.metadata.template_family == FAMILY_CANONICAL
    ]
    shift = [
        record
        for record in tuple_records
        if record.metadata.template_family == FAMILY_SHIFT
    ]
    ordered = sorted(canonical, key=lambda record: _stable_key(record.metadata.group_id))
    n_total = len(ordered)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)

    split_map: dict[str, str] = {}
    for index, record in enumerate(ordered):
        if index < n_train:
            split_map[record.metadata.group_id] = "train"
        elif index < n_train + n_val:
            split_map[record.metadata.group_id] = "val"
        else:
            split_map[record.metadata.group_id] = "test"
    for record in shift:
        split_map[record.metadata.group_id] = "shift"
    return split_map


def split_manifest_records(tuple_records: list[BaseSourceTuple]) -> list[dict[str, object]]:
    split_map = assign_splits(tuple_records)
    records: list[dict[str, object]] = []
    for tuple_record in sorted(tuple_records, key=lambda item: item.metadata.group_id):
        records.append(
            {
                "group_id": tuple_record.metadata.group_id,
                "tuple_id": tuple_record.tuple_id,
                "split": split_map[tuple_record.metadata.group_id],
                "template_family": tuple_record.metadata.template_family,
                "intervention_type": tuple_record.intervention_type,
                "latent_base": tuple_record.metadata.latent_base,
                "latent_source": tuple_record.metadata.latent_source,
            }
        )
    return records


@dataclass(frozen=True, slots=True)
class S3DatasetBundle:
    tuples: tuple[BaseSourceTuple, ...]
    split_by_group: dict[str, str]

    @property
    def tuples_by_split(self) -> dict[str, list[BaseSourceTuple]]:
        grouped = {
            "train": [],
            "val": [],
            "test": [],
            "shift": [],
        }
        for tuple_record in self.tuples:
            grouped[self.split_by_group[tuple_record.metadata.group_id]].append(tuple_record)
        return grouped

    def dataset_stats(self) -> dict[str, object]:
        tuples_by_split = self.tuples_by_split
        example_name_vocab = sorted(
            {
                str(tuple_record.metadata.latent_base["N1_token"])
                for tuple_record in self.tuples
            }
            | {
                str(tuple_record.metadata.latent_base["N2_token"])
                for tuple_record in self.tuples
            }
        )
        counts_by_intervention = {
            split_name: {
                intervention: sum(
                    1
                    for tuple_record in split_records
                    if tuple_record.intervention_type == intervention
                )
                for intervention in INTERVENTION_VARS
            }
            for split_name, split_records in tuples_by_split.items()
        }
        return {
            "name_vocab": example_name_vocab,
            "n_total_tuples": len(self.tuples),
            "n_groups": len(self.split_by_group),
            "counts_by_split": {
                split_name: len(split_records)
                for split_name, split_records in tuples_by_split.items()
            },
            "counts_by_intervention": counts_by_intervention,
            "counts_by_family": {
                FAMILY_CANONICAL: sum(
                    1
                    for tuple_record in self.tuples
                    if tuple_record.metadata.template_family == FAMILY_CANONICAL
                ),
                FAMILY_SHIFT: sum(
                    1
                    for tuple_record in self.tuples
                    if tuple_record.metadata.template_family == FAMILY_SHIFT
                ),
            },
        }


def build_s3_dataset_bundle(model: GPT2IOIModel) -> S3DatasetBundle:
    tuple_records = generate_all_s3_tuples(model)
    split_map = assign_splits(tuple_records)
    return S3DatasetBundle(tuples=tuple(tuple_records), split_by_group=split_map)

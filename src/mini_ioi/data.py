from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from src.core.schemas import BaseSourceTuple, TupleMetadata

from .model import (
    FAMILY_CANONICAL,
    FAMILY_SHIFT,
    INTERVENTION_VARS,
    NAME_VOCAB,
    MiniIOILatents,
    MiniIOITransformer,
)


def _stable_key(*parts: str) -> str:
    return sha256("||".join(parts).encode("utf-8")).hexdigest()


def _group_key(
    *,
    setting_id: str,
    family: str,
    base: MiniIOILatents,
    source: MiniIOILatents,
    intervention: str,
) -> str:
    return _stable_key(
        setting_id,
        family,
        f"base:{base.n1_index},{base.n2_index},{base.r_value}",
        f"source:{source.n1_index},{source.n2_index},{source.r_value}",
        intervention,
    )


def _prompt_payload(model: MiniIOITransformer, latents: MiniIOILatents) -> dict[str, object]:
    tokens = model.render_prompt_tokens(latents)
    token_ids = model.encode_prompt(latents)
    return {
        "kind": "word_level_prompt",
        "tokens": list(tokens),
        "token_ids": list(token_ids),
        "latents": latents.to_dict(),
    }


def _prompt_id(setting_id: str, latents: MiniIOILatents) -> str:
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
    model: MiniIOITransformer,
    base: MiniIOILatents,
    source: MiniIOILatents,
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


def _source_candidates(base: MiniIOILatents, intervention: str) -> Iterable[MiniIOILatents]:
    if intervention == "N1":
        for candidate_index in range(len(NAME_VOCAB)):
            if candidate_index not in {base.n1_index, base.n2_index}:
                yield MiniIOILatents(
                    n1_index=candidate_index,
                    n2_index=base.n2_index,
                    r_value=base.r_value,
                    family=base.family,
                )
    elif intervention == "N2":
        for candidate_index in range(len(NAME_VOCAB)):
            if candidate_index not in {base.n1_index, base.n2_index}:
                yield MiniIOILatents(
                    n1_index=base.n1_index,
                    n2_index=candidate_index,
                    r_value=base.r_value,
                    family=base.family,
                )
    elif intervention == "R":
        yield MiniIOILatents(
            n1_index=base.n1_index,
            n2_index=base.n2_index,
            r_value=2 if base.r_value == 1 else 1,
            family=base.family,
        )
    else:
        raise ValueError(f"Unknown intervention: {intervention}")


def generate_training_latents() -> list[MiniIOILatents]:
    latents: list[MiniIOILatents] = []
    for n1_index in range(len(NAME_VOCAB)):
        for n2_index in range(len(NAME_VOCAB)):
            if n1_index == n2_index:
                continue
            for r_value in (1, 2):
                latents.append(
                    MiniIOILatents(
                        n1_index=n1_index,
                        n2_index=n2_index,
                        r_value=r_value,
                        family=FAMILY_CANONICAL,
                    )
                )
    return latents


def generate_all_s2_tuples(
    model: MiniIOITransformer,
    *,
    setting_id: str = "mini_ioi",
) -> list[BaseSourceTuple]:
    tuples: list[BaseSourceTuple] = []
    for family in (FAMILY_CANONICAL, FAMILY_SHIFT):
        for n1_index in range(len(NAME_VOCAB)):
            for n2_index in range(len(NAME_VOCAB)):
                if n1_index == n2_index:
                    continue
                for r_value in (1, 2):
                    base = MiniIOILatents(
                        n1_index=n1_index,
                        n2_index=n2_index,
                        r_value=r_value,
                        family=family,
                    )
                    for intervention in INTERVENTION_VARS:
                        for source in _source_candidates(base, intervention):
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
class S2DatasetBundle:
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
            "name_vocab": list(NAME_VOCAB),
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


def build_s2_dataset_bundle(model: MiniIOITransformer) -> S2DatasetBundle:
    tuple_records = generate_all_s2_tuples(model)
    split_map = assign_splits(tuple_records)
    return S2DatasetBundle(tuples=tuple(tuple_records), split_by_group=split_map)

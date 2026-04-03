from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any, Iterable

import torch

from src.core.codebooks import (
    PRIMARY_CODEBOOK_ID,
    build_code_length_breakdown,
    validate_codebook_id,
)
from src.core.schemas import BaseSourceTuple, CodeLengthBreakdown, ResidualContribution, Site

from .data import S1DatasetBundle
from .high_level import HIGH_LEVEL_MODEL_ORDER, HIGH_LEVEL_MODELS
from .model import NAME_VOCAB, PlantedLatents, PlantedRun, S1PlantedModel
from .readouts import (
    MAP_FAMILY_HYPERGRIDS,
    MAP_FAMILY_ORDER,
    ReadoutFitResult,
    fit_readout,
    hyperparameter_id,
    parse_map_family_grid,
)

VARIABLE_ORDER = ("N1", "N2", "R")
SITE_BUDGET_CHOICES = (1, 2, 4)
CLASS_COUNTS = {
    "N1": len(NAME_VOCAB),
    "N2": len(NAME_VOCAB),
    "R": 2,
}
TOP_Q = 8
TOP_R = 4


def _stable_digest(*parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()


def _stable_int(*parts: object) -> int:
    return int(_stable_digest(*parts)[:12], 16)


def _site_sort_key(site: Site) -> tuple[int, int]:
    return (site.layer_index, site.token_index)


def _normalize_sites(sites: Iterable[Site]) -> tuple[Site, ...]:
    return tuple(sorted(set(sites), key=_site_sort_key))


def _sites_key(sites: Iterable[Site]) -> tuple[tuple[int, int], ...]:
    return tuple((site.layer_index, site.token_index) for site in _normalize_sites(sites))


def _site_dicts(sites: Iterable[Site]) -> list[dict[str, int]]:
    return [site.to_dict() for site in _normalize_sites(sites)]


def _site_groups_key(
    site_groups: dict[str, tuple[Site, ...]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(_sites_key(site_groups[var_name]) for var_name in VARIABLE_ORDER)


def _serialize_site_groups(
    site_groups: dict[str, tuple[Site, ...]],
) -> dict[str, list[dict[str, int]]]:
    return {
        var_name: _site_dicts(site_groups[var_name])
        for var_name in VARIABLE_ORDER
    }


def _log2_comb(n: int, k: int) -> float:
    return math.log2(math.comb(n, k))


def _latents_key(latents: PlantedLatents) -> tuple[int, int, int, str]:
    return (latents.n1_index, latents.n2_index, latents.r_value, latents.family)


def _label_for_variable(latents: PlantedLatents, variable_name: str) -> int:
    if variable_name == "N1":
        return latents.n1_index
    if variable_name == "N2":
        return latents.n2_index
    if variable_name == "R":
        return latents.r_value - 1
    raise ValueError(f"Unknown variable_name: {variable_name}")


def _decode_state(variable_name: str, class_index: int) -> int:
    if variable_name == "R":
        return class_index + 1
    return class_index


def _is_disjoint(site_groups: dict[str, tuple[Site, ...]]) -> bool:
    merged: list[Site] = []
    for var_name in VARIABLE_ORDER:
        merged.extend(site_groups[var_name])
    return len(merged) == len(set(merged))


@dataclass(frozen=True, slots=True)
class RunExample:
    split: str
    role: str
    latents: PlantedLatents
    run: PlantedRun


@dataclass(frozen=True, slots=True)
class GroupProposal:
    sites: tuple[Site, ...]
    val_nll: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sites": _site_dicts(self.sites),
            "val_nll": self.val_nll,
        }


@dataclass(frozen=True, slots=True)
class TupleBehavior:
    tuple_record: BaseSourceTuple
    observed_output_token: str
    base_states: dict[str, int]
    source_states: dict[str, int]


@dataclass(slots=True)
class CandidateBehavior:
    map_family_id: str
    hyperparameter_value: object
    site_groups: dict[str, tuple[Site, ...]]
    readout_fits: dict[str, ReadoutFitResult]
    tuple_behaviors_by_split: dict[str, tuple[TupleBehavior, ...]]
    parameter_count: int


@dataclass(frozen=True, slots=True)
class SearchSpec:
    high_level_model_ids: tuple[str, ...]
    site_budgets: tuple[int, ...]
    map_family_grid: dict[str, tuple[object, ...]]
    linear_epochs: int
    mlp_epochs: int
    learning_rate: float

    @classmethod
    def from_config_extras(cls, extras: dict[str, Any]) -> "SearchSpec":
        search_config = extras.get("candidate_pool", {})
        if not isinstance(search_config, dict):
            raise TypeError("extras.candidate_pool must be a mapping when present")

        raw_high_level_models = search_config.get(
            "high_level_models",
            list(HIGH_LEVEL_MODEL_ORDER),
        )
        if not isinstance(raw_high_level_models, list):
            raise TypeError("candidate_pool.high_level_models must be a list")
        high_level_model_ids = tuple(str(model_id) for model_id in raw_high_level_models)
        for model_id in high_level_model_ids:
            if model_id not in HIGH_LEVEL_MODELS:
                raise ValueError(f"Unknown high-level model id: {model_id}")

        raw_site_budgets = search_config.get("site_budgets", list(SITE_BUDGET_CHOICES))
        if not isinstance(raw_site_budgets, list):
            raise TypeError("candidate_pool.site_budgets must be a list")
        site_budgets = tuple(int(value) for value in raw_site_budgets)
        for site_budget in site_budgets:
            if site_budget not in SITE_BUDGET_CHOICES:
                raise ValueError(f"Unsupported site budget: {site_budget}")

        map_family_grid = parse_map_family_grid(search_config.get("map_families"))

        train_config = extras.get("training", {})
        if not isinstance(train_config, dict):
            raise TypeError("extras.training must be a mapping when present")
        linear_epochs = int(train_config.get("linear_epochs", 40))
        mlp_epochs = int(train_config.get("mlp_epochs", 60))
        learning_rate = float(train_config.get("learning_rate", 0.05))

        return cls(
            high_level_model_ids=high_level_model_ids,
            site_budgets=site_budgets,
            map_family_grid=map_family_grid,
            linear_epochs=linear_epochs,
            mlp_epochs=mlp_epochs,
            learning_rate=learning_rate,
        )


class CandidateSearchEngine:
    def __init__(
        self,
        *,
        model: S1PlantedModel,
        dataset_bundle: S1DatasetBundle,
        codebook_id: str = PRIMARY_CODEBOOK_ID,
        search_seed: int = 0,
        linear_epochs: int = 40,
        mlp_epochs: int = 60,
        learning_rate: float = 0.05,
    ) -> None:
        self.model = model
        self.dataset_bundle = dataset_bundle
        self.codebook_id = validate_codebook_id(codebook_id)
        self.search_seed = search_seed
        self.linear_epochs = linear_epochs
        self.mlp_epochs = mlp_epochs
        self.learning_rate = learning_rate
        self.tuples_by_split = {
            split_name: tuple(
                sorted(split_records, key=lambda record: record.metadata.group_id)
            )
            for split_name, split_records in dataset_bundle.tuples_by_split.items()
        }
        self.n_train_tuples = len(self.tuples_by_split["train"])
        self.n_val_tuples = len(self.tuples_by_split["val"])
        self.clean_run_cache = self._build_clean_run_cache()
        self.feature_cache: dict[
            tuple[tuple[int, int, int, str], tuple[tuple[int, int], ...]],
            torch.Tensor,
        ] = {}
        self.patch_output_cache: dict[tuple[str, tuple[tuple[int, int], ...]], str] = {}
        self.group_fit_cache: dict[
            tuple[str, tuple[tuple[int, int], ...], str, str],
            ReadoutFitResult,
        ] = {}
        self.group_proposal_cache: dict[
            tuple[str, int, str, str],
            tuple[GroupProposal, ...],
        ] = {}
        self.candidate_behavior_cache: dict[
            tuple[str, str, tuple[tuple[tuple[int, int], ...], ...]],
            CandidateBehavior,
        ] = {}

    def _build_clean_run_cache(self) -> dict[tuple[int, int, int, str], PlantedRun]:
        cache: dict[tuple[int, int, int, str], PlantedRun] = {}
        for tuple_record in self.dataset_bundle.tuples:
            for latents_dict in (
                tuple_record.metadata.latent_base,
                tuple_record.metadata.latent_source,
            ):
                latents = PlantedLatents.from_dict(latents_dict)
                cache.setdefault(_latents_key(latents), self.model.run_clean(latents))
        return cache

    def _tuple_conditioned_latents(
        self,
        tuple_record: BaseSourceTuple,
        variable_name: str,
    ) -> PlantedLatents:
        latents_dict = (
            tuple_record.metadata.latent_source
            if tuple_record.intervention_type == variable_name
            else tuple_record.metadata.latent_base
        )
        return PlantedLatents.from_dict(latents_dict)

    def _feature_vector(self, run: PlantedRun, sites: tuple[Site, ...]) -> torch.Tensor:
        cache_key = (_latents_key(run.latents), _sites_key(sites))
        cached = self.feature_cache.get(cache_key)
        if cached is not None:
            return cached
        feature = torch.cat(
            [run.site_activation(site).reshape(-1) for site in _normalize_sites(sites)],
            dim=0,
        ).to(dtype=self.model.dtype)
        self.feature_cache[cache_key] = feature
        return feature

    def _dataset_for_variable(
        self,
        *,
        split_name: str,
        variable_name: str,
        sites: tuple[Site, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tuple_records = self.tuples_by_split[split_name]
        examples = [
            self._tuple_conditioned_latents(tuple_record, variable_name)
            for tuple_record in tuple_records
        ]
        features = torch.stack(
            [
                self._feature_vector(
                    self.clean_run_cache[_latents_key(latents)],
                    sites,
                )
                for latents in examples
            ],
            dim=0,
        )
        labels = torch.tensor(
            [_label_for_variable(latents, variable_name) for latents in examples],
            dtype=torch.long,
        )
        return features, labels

    def _fit_variable_readout(
        self,
        *,
        variable_name: str,
        sites: tuple[Site, ...],
        map_family_id: str,
        hyperparameter_value: object,
    ) -> ReadoutFitResult:
        normalized_sites = _normalize_sites(sites)
        hyper_id = hyperparameter_id(map_family_id, hyperparameter_value)
        cache_key = (variable_name, _sites_key(normalized_sites), map_family_id, hyper_id)
        cached = self.group_fit_cache.get(cache_key)
        if cached is not None:
            return cached

        train_features, train_labels = self._dataset_for_variable(
            split_name="train",
            variable_name=variable_name,
            sites=normalized_sites,
        )
        val_features, val_labels = self._dataset_for_variable(
            split_name="val",
            variable_name=variable_name,
            sites=normalized_sites,
        )
        seed = _stable_int(
            self.search_seed,
            "fit",
            variable_name,
            map_family_id,
            hyper_id,
            cache_key[1],
        )
        fit_result = fit_readout(
            variable_name=variable_name,
            map_family_id=map_family_id,
            hyperparameter_value=hyperparameter_value,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            output_dim=CLASS_COUNTS[variable_name],
            seed=seed,
            linear_epochs=self.linear_epochs,
            mlp_epochs=self.mlp_epochs,
            learning_rate=self.learning_rate,
        )
        self.group_fit_cache[cache_key] = fit_result
        return fit_result

    def _single_site_rankings(
        self,
        *,
        variable_name: str,
        map_family_id: str,
        hyperparameter_value: object,
    ) -> tuple[tuple[Site, float], ...]:
        scored_sites: list[tuple[Site, float]] = []
        for site in self.model.site_universe:
            fit_result = self._fit_variable_readout(
                variable_name=variable_name,
                sites=(site,),
                map_family_id=map_family_id,
                hyperparameter_value=hyperparameter_value,
            )
            scored_sites.append((site, fit_result.val_nll))
        scored_sites.sort(key=lambda item: (item[1], self.model.site_id(item[0])))
        return tuple(scored_sites)

    def build_group_proposals(
        self,
        *,
        variable_name: str,
        site_budget: int,
        map_family_id: str,
        hyperparameter_value: object,
    ) -> tuple[GroupProposal, ...]:
        hyper_id = hyperparameter_id(map_family_id, hyperparameter_value)
        cache_key = (variable_name, site_budget, map_family_id, hyper_id)
        cached = self.group_proposal_cache.get(cache_key)
        if cached is not None:
            return cached

        ranked_sites = self._single_site_rankings(
            variable_name=variable_name,
            map_family_id=map_family_id,
            hyperparameter_value=hyperparameter_value,
        )
        top_sites = [site for site, _ in ranked_sites[:TOP_Q]]
        seed_sites = top_sites[:TOP_R]

        proposals: list[GroupProposal] = []
        seen_keys: set[tuple[tuple[int, int], ...]] = set()
        for seed_site in seed_sites:
            current_sites = [seed_site]
            while len(current_sites) < site_budget:
                best_candidate_sites: tuple[Site, ...] | None = None
                best_candidate_nll = float("inf")
                for candidate_site in top_sites:
                    if candidate_site in current_sites:
                        continue
                    candidate_sites = _normalize_sites((*current_sites, candidate_site))
                    fit_result = self._fit_variable_readout(
                        variable_name=variable_name,
                        sites=candidate_sites,
                        map_family_id=map_family_id,
                        hyperparameter_value=hyperparameter_value,
                    )
                    if best_candidate_sites is None or (
                        fit_result.val_nll < best_candidate_nll
                        or (
                            math.isclose(fit_result.val_nll, best_candidate_nll)
                            and _sites_key(candidate_sites) < _sites_key(best_candidate_sites)
                        )
                    ):
                        best_candidate_sites = candidate_sites
                        best_candidate_nll = fit_result.val_nll
                if best_candidate_sites is None:
                    break
                current_sites = list(best_candidate_sites)

            normalized_sites = _normalize_sites(current_sites)
            if len(normalized_sites) != site_budget:
                continue
            group_key = _sites_key(normalized_sites)
            if group_key in seen_keys:
                continue
            seen_keys.add(group_key)
            fit_result = self._fit_variable_readout(
                variable_name=variable_name,
                sites=normalized_sites,
                map_family_id=map_family_id,
                hyperparameter_value=hyperparameter_value,
            )
            proposals.append(GroupProposal(sites=normalized_sites, val_nll=fit_result.val_nll))

        proposals.sort(key=lambda proposal: (proposal.val_nll, _sites_key(proposal.sites)))
        cached_value = tuple(proposals[:TOP_R])
        self.group_proposal_cache[cache_key] = cached_value
        return cached_value

    def _observed_output_token(
        self,
        *,
        tuple_record: BaseSourceTuple,
        patch_sites: tuple[Site, ...],
    ) -> str:
        cache_key = (tuple_record.tuple_id, _sites_key(patch_sites))
        cached = self.patch_output_cache.get(cache_key)
        if cached is not None:
            return cached

        base_latents = PlantedLatents.from_dict(tuple_record.metadata.latent_base)
        source_latents = PlantedLatents.from_dict(tuple_record.metadata.latent_source)
        patched_run = self.model.patch_and_run(
            base_latents=base_latents,
            source_latents=source_latents,
            intervention_type=tuple_record.intervention_type,
            patch_sites=patch_sites,
            allow_misaligned_source=(
                "shuffled_pair_original_group_id" in tuple_record.metadata.extra
            ),
        )
        self.patch_output_cache[cache_key] = patched_run.output_token
        return patched_run.output_token

    def build_candidate_behavior(
        self,
        *,
        map_family_id: str,
        hyperparameter_value: object,
        site_groups: dict[str, tuple[Site, ...]],
    ) -> CandidateBehavior:
        normalized_site_groups = {
            var_name: _normalize_sites(site_groups[var_name])
            for var_name in VARIABLE_ORDER
        }
        hyper_id = hyperparameter_id(map_family_id, hyperparameter_value)
        cache_key = (map_family_id, hyper_id, _site_groups_key(normalized_site_groups))
        cached = self.candidate_behavior_cache.get(cache_key)
        if cached is not None:
            return cached

        readout_fits = {
            var_name: self._fit_variable_readout(
                variable_name=var_name,
                sites=normalized_site_groups[var_name],
                map_family_id=map_family_id,
                hyperparameter_value=hyperparameter_value,
            )
            for var_name in VARIABLE_ORDER
        }
        parameter_count = sum(fit.parameter_count for fit in readout_fits.values())

        state_cache: dict[tuple[int, int, int, str], dict[str, int]] = {}
        for latent_key, run in self.clean_run_cache.items():
            predicted_state_by_var: dict[str, int] = {}
            for var_name in VARIABLE_ORDER:
                features = self._feature_vector(
                    run,
                    normalized_site_groups[var_name],
                ).unsqueeze(0)
                predicted_class = int(readout_fits[var_name].predict_classes(features)[0].item())
                predicted_state_by_var[var_name] = _decode_state(var_name, predicted_class)
            state_cache[latent_key] = predicted_state_by_var

        tuple_behaviors_by_split: dict[str, tuple[TupleBehavior, ...]] = {}
        for split_name in ("val", "test", "shift"):
            behaviors: list[TupleBehavior] = []
            for tuple_record in self.tuples_by_split[split_name]:
                base_latents = PlantedLatents.from_dict(tuple_record.metadata.latent_base)
                source_latents = PlantedLatents.from_dict(tuple_record.metadata.latent_source)
                observed_output_token = self._observed_output_token(
                    tuple_record=tuple_record,
                    patch_sites=normalized_site_groups[tuple_record.intervention_type],
                )
                behaviors.append(
                    TupleBehavior(
                        tuple_record=tuple_record,
                        observed_output_token=observed_output_token,
                        base_states=dict(state_cache[_latents_key(base_latents)]),
                        source_states=dict(state_cache[_latents_key(source_latents)]),
                    )
                )
            tuple_behaviors_by_split[split_name] = tuple(behaviors)

        behavior = CandidateBehavior(
            map_family_id=map_family_id,
            hyperparameter_value=hyperparameter_value,
            site_groups=normalized_site_groups,
            readout_fits=readout_fits,
            tuple_behaviors_by_split=tuple_behaviors_by_split,
            parameter_count=parameter_count,
        )
        self.candidate_behavior_cache[cache_key] = behavior
        return behavior

    def _structural_code_lengths(
        self,
        *,
        site_budget: int,
        map_family_id: str,
        site_groups: dict[str, tuple[Site, ...]],
        parameter_count: int,
    ) -> CodeLengthBreakdown:
        site_universe_size = len(self.model.site_universe)
        if not _is_disjoint(site_groups):
            raise ValueError("structural code requested for non-disjoint site groups")
        return build_code_length_breakdown(
            high_level_bits=math.log2(4),
            budget_bits=math.log2(3),
            site_bits=(
                _log2_comb(site_universe_size, site_budget)
                + _log2_comb(site_universe_size - site_budget, site_budget)
                + _log2_comb(site_universe_size - 2 * site_budget, site_budget)
            ),
            family_bits=math.log2(4),
            hyperparameter_bits=math.log2(len(MAP_FAMILY_HYPERGRIDS[map_family_id])),
            parameter_count_eff=parameter_count,
            n_train_tuples=self.n_train_tuples,
            codebook_id=self.codebook_id,
        )

    def score_candidate_behavior(
        self,
        *,
        high_level_model_id: str,
        site_budget: int,
        behavior: CandidateBehavior,
    ) -> dict[str, Any]:
        high_level_model = HIGH_LEVEL_MODELS[high_level_model_id]

        split_predictions: dict[str, list[tuple[TupleBehavior, str]]] = {}
        for split_name, tuple_behaviors in behavior.tuple_behaviors_by_split.items():
            predictions_for_split: list[tuple[TupleBehavior, str]] = []
            for tuple_behavior in tuple_behaviors:
                effective_states = dict(tuple_behavior.base_states)
                intervention_type = tuple_behavior.tuple_record.intervention_type
                effective_states[intervention_type] = tuple_behavior.source_states[intervention_type]
                predicted_index = high_level_model.predict_index(
                    effective_states["N1"],
                    effective_states["N2"],
                    effective_states["R"],
                )
                predictions_for_split.append(
                    (tuple_behavior, NAME_VOCAB[predicted_index])
                )
            split_predictions[split_name] = predictions_for_split

        val_predictions = split_predictions["val"]
        val_error_rate = (
            sum(
                1
                for tuple_behavior, predicted_token in val_predictions
                if tuple_behavior.observed_output_token != predicted_token
            )
            / max(len(val_predictions), 1)
        )
        epsilon_min = max(1e-4, 1.0 / (10 * max(self.n_val_tuples, 1)))
        epsilon = min(
            max(val_error_rate, epsilon_min),
            1 - (1 / len(NAME_VOCAB)) - epsilon_min,
        )

        candidate_id = _stable_digest(
            "candidate",
            high_level_model_id,
            behavior.map_family_id,
            hyperparameter_id(behavior.map_family_id, behavior.hyperparameter_value),
            site_budget,
            _site_groups_key(behavior.site_groups),
        )[:16]
        residual_totals: dict[str, float] = {}
        residual_per_example: dict[str, float] = {}
        residual_contributions: dict[str, list[dict[str, Any]]] = {}
        for split_name, predictions_for_split in split_predictions.items():
            total_bits = 0.0
            contributions: list[dict[str, Any]] = []
            for tuple_behavior, predicted_token in predictions_for_split:
                if predicted_token == tuple_behavior.observed_output_token:
                    residual_bits = -math.log2(1 - epsilon)
                else:
                    residual_bits = -math.log2(epsilon / (len(NAME_VOCAB) - 1))
                total_bits += residual_bits
                contributions.append(
                    ResidualContribution(
                        candidate_id=candidate_id,
                        split=split_name,
                        group_id=tuple_behavior.tuple_record.metadata.group_id,
                        residual_bits=residual_bits,
                        n_examples=1,
                    ).to_dict()
                )
            residual_totals[split_name] = total_bits
            residual_per_example[split_name] = total_bits / max(len(predictions_for_split), 1)
            residual_contributions[split_name] = contributions

        code_lengths = self._structural_code_lengths(
            site_budget=site_budget,
            map_family_id=behavior.map_family_id,
            site_groups=behavior.site_groups,
            parameter_count=behavior.parameter_count,
        )
        return {
            "candidate_id": candidate_id,
            "status": "evaluable",
            "high_level_model_id": high_level_model_id,
            "map_family_id": behavior.map_family_id,
            "hyperparameter_id": hyperparameter_id(
                behavior.map_family_id,
                behavior.hyperparameter_value,
            ),
            "hyperparameter_value": behavior.hyperparameter_value,
            "site_budget": site_budget,
            "site_groups": _serialize_site_groups(behavior.site_groups),
            "parameter_count_eff": behavior.parameter_count,
            "val_error_rate": val_error_rate,
            "epsilon": epsilon,
            "code_lengths": code_lengths.to_dict(),
            "residual_bits": residual_totals,
            "residual_bits_per_example": residual_per_example,
            "test_total_bits": code_lengths.total_structural_bits + residual_totals["test"],
            "test_total_bits_per_example": (
                code_lengths.total_structural_bits + residual_totals["test"]
            )
            / max(len(split_predictions["test"]), 1),
            "residual_contributions": residual_contributions,
            "split_sizes": {
                split_name: len(split_predictions[split_name])
                for split_name in ("val", "test", "shift")
            },
        }

    def score_fixed_candidate(
        self,
        *,
        high_level_model_id: str,
        map_family_id: str,
        hyperparameter_value: object,
        site_groups: dict[str, tuple[Site, ...]],
    ) -> dict[str, Any]:
        normalized_site_groups = {
            var_name: _normalize_sites(site_groups[var_name])
            for var_name in VARIABLE_ORDER
        }
        site_budget = len(normalized_site_groups["N1"])
        for variable_name in VARIABLE_ORDER:
            if len(normalized_site_groups[variable_name]) != site_budget:
                raise ValueError("fixed candidate site groups must have equal budgets")
        if not _is_disjoint(normalized_site_groups):
            raise ValueError("fixed candidate site groups must be pairwise disjoint")

        behavior = self.build_candidate_behavior(
            map_family_id=map_family_id,
            hyperparameter_value=hyperparameter_value,
            site_groups=normalized_site_groups,
        )
        return self.score_candidate_behavior(
            high_level_model_id=high_level_model_id,
            site_budget=site_budget,
            behavior=behavior,
        )

    def run_search(
        self,
        *,
        high_level_model_ids: tuple[str, ...],
        site_budgets: tuple[int, ...],
        map_family_grid: dict[str, tuple[object, ...]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        candidate_records: list[dict[str, Any]] = []
        proposal_logs: list[dict[str, Any]] = []

        for map_family_id in MAP_FAMILY_ORDER:
            if map_family_id not in map_family_grid:
                continue
            for hyperparameter_value in map_family_grid[map_family_id]:
                hyper_id = hyperparameter_id(map_family_id, hyperparameter_value)
                for site_budget in site_budgets:
                    group_proposals_by_variable = {
                        variable_name: self.build_group_proposals(
                            variable_name=variable_name,
                            site_budget=site_budget,
                            map_family_id=map_family_id,
                            hyperparameter_value=hyperparameter_value,
                        )
                        for variable_name in VARIABLE_ORDER
                    }
                    top8_by_variable = {
                        variable_name: self._single_site_rankings(
                            variable_name=variable_name,
                            map_family_id=map_family_id,
                            hyperparameter_value=hyperparameter_value,
                        )[:TOP_Q]
                        for variable_name in VARIABLE_ORDER
                    }
                    raw_full_proposals = [
                        {
                            "N1": proposal_n1.sites,
                            "N2": proposal_n2.sites,
                            "R": proposal_r.sites,
                        }
                        for proposal_n1, proposal_n2, proposal_r in product(
                            group_proposals_by_variable["N1"],
                            group_proposals_by_variable["N2"],
                            group_proposals_by_variable["R"],
                        )
                    ]
                    valid_full_proposals = [
                        site_groups
                        for site_groups in raw_full_proposals
                        if _is_disjoint(site_groups)
                    ]

                    for high_level_model_id in high_level_model_ids:
                        proposal_logs.append(
                            self._score_cell(
                                candidate_records=candidate_records,
                                high_level_model_id=high_level_model_id,
                                map_family_id=map_family_id,
                                hyperparameter_value=hyperparameter_value,
                                hyperparameter_id_value=hyper_id,
                                site_budget=site_budget,
                                group_proposals_by_variable=group_proposals_by_variable,
                                top8_by_variable=top8_by_variable,
                                raw_full_proposals=raw_full_proposals,
                                valid_full_proposals=valid_full_proposals,
                            )
                        )

        candidate_records.sort(key=lambda record: (record["test_total_bits"], record["candidate_id"]))
        proposal_logs.sort(key=lambda record: record["cell_id"])
        return candidate_records, proposal_logs

    def _score_cell(
        self,
        *,
        candidate_records: list[dict[str, Any]],
        high_level_model_id: str,
        map_family_id: str,
        hyperparameter_value: object,
        hyperparameter_id_value: str,
        site_budget: int,
        group_proposals_by_variable: dict[str, tuple[GroupProposal, ...]],
        top8_by_variable: dict[str, tuple[tuple[Site, float], ...]],
        raw_full_proposals: list[dict[str, tuple[Site, ...]]],
        valid_full_proposals: list[dict[str, tuple[Site, ...]]],
    ) -> dict[str, Any]:
        cell_id = _stable_digest(
            "cell",
            high_level_model_id,
            map_family_id,
            hyperparameter_id_value,
            site_budget,
        )[:16]
        proposal_log: dict[str, Any] = {
            "cell_id": cell_id,
            "status": "evaluable",
            "high_level_model_id": high_level_model_id,
            "map_family_id": map_family_id,
            "hyperparameter_id": hyperparameter_id_value,
            "hyperparameter_value": hyperparameter_value,
            "site_budget": site_budget,
            "search_budget": {
                "top_q": TOP_Q,
                "top_r": TOP_R,
                "max_full_proposals": TOP_R ** len(VARIABLE_ORDER),
                "local_refinement": "one_full_sweep",
            },
            "top8_sites": {
                variable_name: [
                    {"site": site.to_dict(), "val_nll": val_nll}
                    for site, val_nll in top8_by_variable[variable_name]
                ]
                for variable_name in VARIABLE_ORDER
            },
            "group_proposals": {
                variable_name: [
                    proposal.to_dict()
                    for proposal in group_proposals_by_variable[variable_name]
                ]
                for variable_name in VARIABLE_ORDER
            },
            "n_raw_full_proposals": len(raw_full_proposals),
            "n_valid_full_proposals": len(valid_full_proposals),
        }

        if not valid_full_proposals:
            proposal_log["status"] = "unevaluable_due_to_disjointness"
            proposal_log["full_proposals"] = []
            proposal_log["local_refinement"] = {
                "applied": False,
                "reason": "no_valid_full_proposals",
            }
            return proposal_log

        full_proposal_scores: list[tuple[dict[str, tuple[Site, ...]], CandidateBehavior, dict[str, Any]]] = []
        for site_groups in valid_full_proposals:
            behavior = self.build_candidate_behavior(
                map_family_id=map_family_id,
                hyperparameter_value=hyperparameter_value,
                site_groups=site_groups,
            )
            candidate_record = self.score_candidate_behavior(
                high_level_model_id=high_level_model_id,
                site_budget=site_budget,
                behavior=behavior,
            )
            full_proposal_scores.append((site_groups, behavior, candidate_record))

        full_proposal_scores.sort(
            key=lambda item: (item[2]["residual_bits"]["val"], _site_groups_key(item[0]))
        )
        best_site_groups, best_behavior, best_record = full_proposal_scores[0]

        refined_site_groups = best_site_groups
        refined_record = best_record
        refinement_update: dict[str, Any] = {
            "applied": False,
            "starting_val_residual_bits": best_record["residual_bits"]["val"],
        }
        for variable_name in VARIABLE_ORDER:
            current_sites = list(refined_site_groups[variable_name])
            for site_index in range(site_budget):
                for candidate_site, _ in top8_by_variable[variable_name]:
                    if candidate_site == current_sites[site_index]:
                        continue
                    swapped_sites = list(current_sites)
                    swapped_sites[site_index] = candidate_site
                    normalized_swapped_sites = _normalize_sites(swapped_sites)
                    if len(normalized_swapped_sites) != site_budget:
                        continue
                    candidate_site_groups = dict(refined_site_groups)
                    candidate_site_groups[variable_name] = normalized_swapped_sites
                    if not _is_disjoint(candidate_site_groups):
                        continue
                    behavior = self.build_candidate_behavior(
                        map_family_id=map_family_id,
                        hyperparameter_value=hyperparameter_value,
                        site_groups=candidate_site_groups,
                    )
                    candidate_record = self.score_candidate_behavior(
                        high_level_model_id=high_level_model_id,
                        site_budget=site_budget,
                        behavior=behavior,
                    )
                    if candidate_record["residual_bits"]["val"] < refined_record["residual_bits"]["val"]:
                        refined_site_groups = candidate_site_groups
                        refined_record = candidate_record
                        refinement_update = {
                            "applied": True,
                            "variable_name": variable_name,
                            "site_index": site_index,
                            "replacement_site": candidate_site.to_dict(),
                            "starting_val_residual_bits": best_record["residual_bits"]["val"],
                            "ending_val_residual_bits": candidate_record["residual_bits"]["val"],
                        }

        proposal_log["full_proposals"] = [
            {
                "site_groups": _serialize_site_groups(site_groups),
                "val_residual_bits": candidate_record["residual_bits"]["val"],
            }
            for site_groups, _, candidate_record in full_proposal_scores
        ]
        proposal_log["best_full_proposal"] = {
            "site_groups": _serialize_site_groups(best_site_groups),
            "val_residual_bits": best_record["residual_bits"]["val"],
        }
        proposal_log["local_refinement"] = refinement_update

        refined_record["cell_id"] = cell_id
        refined_record["search_status"] = proposal_log["status"]
        refined_record["n_raw_full_proposals"] = len(raw_full_proposals)
        refined_record["n_valid_full_proposals"] = len(valid_full_proposals)
        candidate_records.append(refined_record)
        return proposal_log


def candidate_table_rows(candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in candidate_records:
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "high_level_model_id": record["high_level_model_id"],
                "map_family_id": record["map_family_id"],
                "hyperparameter_id": record["hyperparameter_id"],
                "site_budget": record["site_budget"],
                "site_groups": record["site_groups"],
                "parameter_count_eff": record["parameter_count_eff"],
                "val_error_rate": record["val_error_rate"],
                "epsilon": record["epsilon"],
                "structural_bits": record["code_lengths"]["total_structural_bits"],
                "val_residual_bits": record["residual_bits"]["val"],
                "test_residual_bits": record["residual_bits"]["test"],
                "shift_residual_bits": record["residual_bits"]["shift"],
                "test_total_bits": record["test_total_bits"],
                "test_total_bits_per_example": record["test_total_bits_per_example"],
                "cell_id": record["cell_id"],
                "n_raw_full_proposals": record["n_raw_full_proposals"],
                "n_valid_full_proposals": record["n_valid_full_proposals"],
            }
        )
    return rows

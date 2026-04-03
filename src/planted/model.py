from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import torch

from src.core.schemas import Site

FAMILY_CANONICAL = "canonical"
FAMILY_SHIFT = "shift"
INTERVENTION_VARS = ("N1", "N2", "R")
NAME_VOCAB = tuple(f"sym_{index}" for index in range(8))
POSITION_NAMES = {
    0: "N1_in",
    1: "N2_in",
    2: "R_in",
    3: "F_in",
    4: "relay_N1",
    5: "relay_N2",
    6: "relay_R",
    7: "nuis",
    8: "query",
}
RESIDUAL_WIDTH = 32
N_BLOCKS = 3
SEQUENCE_LENGTH = len(POSITION_NAMES)
S1_PLANT_SEED = 0

S1_TRUE_SITE_GROUPS = {
    "N1": (Site(layer_index=1, token_index=4), Site(layer_index=2, token_index=4)),
    "N2": (Site(layer_index=1, token_index=5), Site(layer_index=2, token_index=5)),
    "R": (Site(layer_index=1, token_index=6), Site(layer_index=2, token_index=6)),
}


def _site_sort_key(site: Site) -> tuple[int, int]:
    return (site.layer_index, site.token_index)


def _sample_square_full_rank(
    *,
    size: int,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    while True:
        candidate = torch.randn(size, size, generator=generator, dtype=dtype)
        if int(torch.linalg.matrix_rank(candidate).item()) == size:
            return candidate


def _sample_rect(
    *,
    rows: int,
    cols: int,
    generator: torch.Generator,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    return scale * torch.randn(rows, cols, generator=generator, dtype=dtype)


@dataclass(frozen=True, slots=True)
class PlantedLatents:
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
            raise ValueError(
                f"family must be {FAMILY_CANONICAL!r} or {FAMILY_SHIFT!r}"
            )

    @property
    def n1_token(self) -> str:
        return NAME_VOCAB[self.n1_index]

    @property
    def n2_token(self) -> str:
        return NAME_VOCAB[self.n2_index]

    def to_dict(self) -> dict[str, int | str]:
        return {
            "N1_index": self.n1_index,
            "N1_token": self.n1_token,
            "N2_index": self.n2_index,
            "N2_token": self.n2_token,
            "R": self.r_value,
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PlantedLatents":
        return cls(
            n1_index=int(data["N1_index"]),
            n2_index=int(data["N2_index"]),
            r_value=int(data["R"]),
            family=str(data["family"]),
        )

    def abstract_difference(self, other: "PlantedLatents") -> list[str]:
        diffs: list[str] = []
        if self.n1_index != other.n1_index:
            diffs.append("N1")
        if self.n2_index != other.n2_index:
            diffs.append("N2")
        if self.r_value != other.r_value:
            diffs.append("R")
        return diffs


@dataclass(frozen=True, slots=True)
class PlantedRun:
    latents: PlantedLatents
    activations: tuple[torch.Tensor, ...]
    output_index: int
    output_logits: torch.Tensor

    @property
    def output_token(self) -> str:
        return NAME_VOCAB[self.output_index]

    def site_activation(self, site: Site) -> torch.Tensor:
        return self.activations[site.layer_index][site.token_index].clone()


class S1PlantedModel:
    def __init__(
        self,
        *,
        plant_seed: int = S1_PLANT_SEED,
        dtype: torch.dtype = torch.float64,
        lambda_nuisance: float = 0.35,
    ) -> None:
        self.plant_seed = plant_seed
        self.dtype = dtype
        self.lambda_nuisance = lambda_nuisance
        self.name_vocab = NAME_VOCAB
        self.position_names = POSITION_NAMES
        self.sequence_length = SEQUENCE_LENGTH
        self.n_blocks = N_BLOCKS
        self.d_model = RESIDUAL_WIDTH
        self.site_universe = tuple(
            Site(layer_index=layer_index, token_index=token_index)
            for layer_index in range(self.n_blocks + 1)
            for token_index in range(self.sequence_length)
        )
        self.site_index = {
            (site.layer_index, site.token_index): idx
            for idx, site in enumerate(self.site_universe)
        }
        self.true_site_groups = S1_TRUE_SITE_GROUPS

        generator = torch.Generator().manual_seed(self.plant_seed)
        self._build_frozen_parameters(generator)

    def _build_frozen_parameters(self, generator: torch.Generator) -> None:
        n1_square = _sample_square_full_rank(size=8, generator=generator, dtype=self.dtype)
        n2_square = _sample_square_full_rank(size=8, generator=generator, dtype=self.dtype)
        r_square = _sample_square_full_rank(size=2, generator=generator, dtype=self.dtype)

        self.p_n1_1 = n1_square[:4, :]
        self.p_n1_2 = n1_square[4:, :]
        self.p_n2_1 = n2_square[:4, :]
        self.p_n2_2 = n2_square[4:, :]
        self.p_r_1 = r_square[:1, :]
        self.p_r_2 = r_square[1:, :]
        self.d_n1 = torch.linalg.inv(n1_square)
        self.d_n2 = torch.linalg.inv(n2_square)
        self.d_r = torch.linalg.inv(r_square)

        nuisance_scale = 0.2
        projection_scale = 0.25
        self.m_can_1 = _sample_rect(
            rows=6, cols=18, generator=generator, dtype=self.dtype, scale=nuisance_scale
        )
        self.m_can_2 = _sample_rect(
            rows=6, cols=18, generator=generator, dtype=self.dtype, scale=nuisance_scale
        )
        self.m_shift_1 = _sample_rect(
            rows=6, cols=18, generator=generator, dtype=self.dtype, scale=nuisance_scale
        )
        self.m_shift_2 = _sample_rect(
            rows=6, cols=18, generator=generator, dtype=self.dtype, scale=nuisance_scale
        )
        self.b_can = _sample_rect(
            rows=8, cols=12, generator=generator, dtype=self.dtype, scale=projection_scale
        )
        self.b_shift = _sample_rect(
            rows=8, cols=12, generator=generator, dtype=self.dtype, scale=projection_scale
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

    def one_hot_name(self, index: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.tensor(index), num_classes=len(self.name_vocab)
        ).to(dtype=self.dtype)

    def one_hot_r(self, value: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.tensor(value - 1), num_classes=2
        ).to(dtype=self.dtype)

    def one_hot_family(self, family: str) -> torch.Tensor:
        family_index = 0 if family == FAMILY_CANONICAL else 1
        return torch.nn.functional.one_hot(
            torch.tensor(family_index), num_classes=2
        ).to(dtype=self.dtype)

    def _base_embedding_state(self, latents: PlantedLatents) -> torch.Tensor:
        h0 = torch.zeros(
            (self.sequence_length, self.d_model),
            dtype=self.dtype,
        )
        h0[0, 0:8] = self.one_hot_name(latents.n1_index)
        h0[1, 0:8] = self.one_hot_name(latents.n2_index)
        h0[2, 0:2] = self.one_hot_r(latents.r_value)
        h0[3, 0:2] = self.one_hot_family(latents.family)
        return h0

    def _nuisance_inputs(self, state: torch.Tensor) -> torch.Tensor:
        return torch.cat([state[0, 0:8], state[1, 0:8], state[2, 0:2]], dim=0)

    def _apply_patches(
        self,
        state: torch.Tensor,
        *,
        layer_index: int,
        patch_sites_by_layer: dict[int, tuple[Site, ...]],
        source_run: PlantedRun | None,
    ) -> torch.Tensor:
        if source_run is None:
            return state
        patched = state.clone()
        for site in patch_sites_by_layer.get(layer_index, ()):
            patched[site.token_index] = source_run.site_activation(site)
        return patched

    def _decode_name_logits(
        self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor, family: str
    ) -> torch.Tensor:
        relay_n1 = torch.cat([h1[4, 0:4], h2[4, 4:8]], dim=0)
        relay_n2 = torch.cat([h1[5, 0:4], h2[5, 4:8]], dim=0)
        relay_r = torch.cat([h1[6, 0:1], h2[6, 1:2]], dim=0)

        n1_logits = self.d_n1 @ relay_n1
        n2_logits = self.d_n2 @ relay_n2
        r_logits = self.d_r @ relay_r
        predicted_r = int(torch.argmax(r_logits).item()) + 1

        true_logits = n2_logits if predicted_r == 1 else n1_logits
        if family == FAMILY_CANONICAL:
            nuisance_logits = self.lambda_nuisance * (self.b_can @ h3[7, 0:12])
        else:
            nuisance_logits = self.lambda_nuisance * (self.b_shift @ h3[7, 0:12])
        return true_logits + nuisance_logits

    def run_clean(self, latents: PlantedLatents) -> PlantedRun:
        return self._run(latents=latents, source_run=None, patch_sites=())

    def patch_and_run(
        self,
        *,
        base_latents: PlantedLatents,
        source_latents: PlantedLatents,
        intervention_type: str,
        patch_sites: Iterable[Site],
        allow_misaligned_source: bool = False,
    ) -> PlantedRun:
        if intervention_type not in INTERVENTION_VARS:
            raise ValueError(f"Unknown intervention_type: {intervention_type}")
        diffs = base_latents.abstract_difference(source_latents)
        if not allow_misaligned_source and diffs != [intervention_type]:
            raise ValueError(
                "base and source latents must differ in exactly the intervention variable"
            )
        if base_latents.family != source_latents.family:
            raise ValueError("base and source must share the same family")

        source_run = self.run_clean(source_latents)
        return self._run(
            latents=base_latents,
            source_run=source_run,
            patch_sites=tuple(sorted(set(patch_sites), key=_site_sort_key)),
        )

    def _run(
        self,
        *,
        latents: PlantedLatents,
        source_run: PlantedRun | None,
        patch_sites: tuple[Site, ...],
    ) -> PlantedRun:
        patch_sites_by_layer: dict[int, list[Site]] = {}
        for site in patch_sites:
            patch_sites_by_layer.setdefault(site.layer_index, []).append(site)
        normalized_patch_sites = {
            layer: tuple(sorted(sites, key=_site_sort_key))
            for layer, sites in patch_sites_by_layer.items()
        }

        h0 = self._apply_patches(
            self._base_embedding_state(latents),
            layer_index=0,
            patch_sites_by_layer=normalized_patch_sites,
            source_run=source_run,
        )

        h1 = h0.clone()
        h1[4, 0:4] += self.p_n1_1 @ h0[0, 0:8]
        h1[5, 0:4] += self.p_n2_1 @ h0[1, 0:8]
        h1[6, 0:1] += self.p_r_1 @ h0[2, 0:2]
        if latents.family == FAMILY_CANONICAL:
            h1[7, 0:6] += self.m_can_1 @ self._nuisance_inputs(h0)
        else:
            h1[7, 0:6] += self.m_shift_1 @ self._nuisance_inputs(h0)
        h1 = self._apply_patches(
            h1,
            layer_index=1,
            patch_sites_by_layer=normalized_patch_sites,
            source_run=source_run,
        )

        h2 = h1.clone()
        h2[4, 4:8] += self.p_n1_2 @ h1[0, 0:8]
        h2[5, 4:8] += self.p_n2_2 @ h1[1, 0:8]
        h2[6, 1:2] += self.p_r_2 @ h1[2, 0:2]
        if latents.family == FAMILY_CANONICAL:
            h2[7, 6:12] += self.m_can_2 @ self._nuisance_inputs(h1)
        else:
            h2[7, 6:12] += self.m_shift_2 @ self._nuisance_inputs(h1)
        h2 = self._apply_patches(
            h2,
            layer_index=2,
            patch_sites_by_layer=normalized_patch_sites,
            source_run=source_run,
        )

        h3 = self._apply_patches(
            h2.clone(),
            layer_index=3,
            patch_sites_by_layer=normalized_patch_sites,
            source_run=source_run,
        )

        final_logits = self._decode_name_logits(h1, h2, h3, latents.family)
        output_index = int(torch.argmax(final_logits).item())
        return PlantedRun(
            latents=latents,
            activations=(h0, h1, h2, h3),
            output_index=output_index,
            output_logits=final_logits.clone(),
        )


def stable_example_id(*parts: str) -> str:
    digest = sha256("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]

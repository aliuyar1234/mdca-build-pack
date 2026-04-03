from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class HighLevelModel:
    model_id: str
    description: str
    _predict_index: Callable[[int, int, int], int]

    def predict_index(self, n1_index: int, n2_index: int, r_value: int) -> int:
        return self._predict_index(n1_index, n2_index, r_value)


def _predict_true_other(n1_index: int, n2_index: int, r_value: int) -> int:
    return n2_index if r_value == 1 else n1_index


def _predict_first(n1_index: int, n2_index: int, r_value: int) -> int:
    del n2_index, r_value
    return n1_index


def _predict_second(n1_index: int, n2_index: int, r_value: int) -> int:
    del n1_index, r_value
    return n2_index


def _predict_rep(n1_index: int, n2_index: int, r_value: int) -> int:
    return n1_index if r_value == 1 else n2_index


HIGH_LEVEL_MODELS: dict[str, HighLevelModel] = {
    "H_true_other": HighLevelModel(
        model_id="H_true_other",
        description="Y = N2 if R=1 else N1",
        _predict_index=_predict_true_other,
    ),
    "H_first": HighLevelModel(
        model_id="H_first",
        description="Y = N1",
        _predict_index=_predict_first,
    ),
    "H_second": HighLevelModel(
        model_id="H_second",
        description="Y = N2",
        _predict_index=_predict_second,
    ),
    "H_rep": HighLevelModel(
        model_id="H_rep",
        description="Y = N1 if R=1 else N2",
        _predict_index=_predict_rep,
    ),
}

HIGH_LEVEL_MODEL_ORDER = tuple(HIGH_LEVEL_MODELS)

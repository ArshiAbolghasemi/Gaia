from __future__ import annotations

from typing import TYPE_CHECKING

from tigramite.independence_tests.parcorr import ParCorr

if TYPE_CHECKING:
    from src.config.model import DependenceConfig


def build_dependence_test(config: DependenceConfig) -> ParCorr:
    method = config.method.lower()
    if method != "parcorr":
        msg = f"Only 'parcorr' is supported right now, got '{config.method}'"
        raise ValueError(msg)
    return ParCorr(**config.params)

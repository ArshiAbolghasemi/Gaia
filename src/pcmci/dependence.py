from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING

from tigramite.independence_tests.parcorr import ParCorr

if TYPE_CHECKING:
    from collections.abc import Callable

    from tigramite.independence_tests.independence_tests_base import CondIndTest

    from src.config.model import DependenceConfig


def build_dependence_test(config: DependenceConfig) -> CondIndTest:
    method = config.method.lower()
    builders: dict[str, Callable[[dict], CondIndTest]] = {
        "parcorr": lambda params: ParCorr(**params),
        "gpdc_torch": _build_gpdc_torch,
    }

    try:
        builder = builders[method]
    except KeyError as exc:
        known_methods = ", ".join(sorted(builders))
        msg = (
            f"Unsupported dependence method '{config.method}'. "
            f"Known methods: {known_methods}"
        )
        raise ValueError(msg) from exc

    return builder(config.params)


def _build_gpdc_torch(params: dict) -> CondIndTest:
    try:
        module = import_module("tigramite.independence_tests.gpdc_torch")
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "an optional dependency"
        msg = (
            "GPDCTorch requires Tigramite optional dependencies. "
            f"Missing package: {missing_package}"
        )
        raise ModuleNotFoundError(msg) from exc

    _suppress_gpytorch_training_input_warning()
    gpdc_torch = module.GPDCtorch
    return gpdc_torch(**params)


def _suppress_gpytorch_training_input_warning() -> None:
    try:
        gpytorch_warnings = import_module("gpytorch.utils.warnings")
    except ModuleNotFoundError:
        return

    warnings.filterwarnings(
        "ignore",
        message=r"The input matches the stored training data\..*",
        category=gpytorch_warnings.GPInputWarning,
    )

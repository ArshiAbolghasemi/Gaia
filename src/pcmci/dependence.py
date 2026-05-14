from __future__ import annotations

import contextlib
import warnings
from importlib import import_module
from typing import TYPE_CHECKING

from tigramite.independence_tests.parcorr import ParCorr

from src.pcmci.cmi import FAISSCMI

if TYPE_CHECKING:
    from collections.abc import Callable

    from tigramite.independence_tests.independence_tests_base import CondIndTest

    from src.config.model import DependenceConfig


def build_dependence_test(config: DependenceConfig) -> CondIndTest:
    method = config.method.lower()
    builders: dict[str, Callable[[dict], CondIndTest]] = {
        "cmi": lambda params: FAISSCMI(**params),
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
    gpdc_params = dict(params)
    try:
        module = import_module("tigramite.independence_tests.gpdc_torch")
    except ModuleNotFoundError as exc:
        missing_package = exc.name or "an optional dependency"
        msg = (
            "GPDCTorch requires Tigramite optional dependencies. "
            f"Missing package: {missing_package}"
        )
        raise ModuleNotFoundError(msg) from exc

    _configure_gpdc_torch_runtime(gpdc_params)
    _suppress_gpytorch_training_input_warning()
    gpdc_torch = module.GPDCtorch
    return gpdc_torch(**gpdc_params)


def _configure_gpdc_torch_runtime(params: dict) -> None:
    torch_num_threads = params.pop("torch_num_threads", 1)
    torch_num_interop_threads = params.pop("torch_num_interop_threads", 1)
    with contextlib.suppress(ModuleNotFoundError):
        torch = import_module("torch")
        torch.set_num_threads(max(1, int(torch_num_threads)))
        with contextlib.suppress(RuntimeError):
            # PyTorch only allows changing inter-op threads before parallel work starts.
            torch.set_num_interop_threads(max(1, int(torch_num_interop_threads)))


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

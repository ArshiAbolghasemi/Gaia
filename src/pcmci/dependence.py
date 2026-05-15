from __future__ import annotations

import contextlib
import logging
import warnings
from importlib import import_module
from typing import TYPE_CHECKING

from tigramite.independence_tests.parcorr import ParCorr

from src.pcmci.cmi import FAISSCMI

if TYPE_CHECKING:
    from collections.abc import Callable

    from tigramite.independence_tests.independence_tests_base import CondIndTest

    from src.config.model import DependenceConfig

logger = logging.getLogger(__name__)


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
    _log_gpdc_torch_runtime(gpdc_params)
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


def _log_gpdc_torch_runtime(params: dict) -> None:
    with contextlib.suppress(ModuleNotFoundError):
        torch = import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        backend = "cuda" if cuda_available else "cpu"
        mps_available = False
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "is_available"):
            mps_available = bool(mps_backend.is_available())
        logger.info(
            "GPDCtorch runtime: backend=%s, cuda_available=%s, cuda_device_count=%d, "
            "mps_available=%s, torch_num_threads=%d, torch_num_interop_threads=%d, "
            "sig_samples=%s",
            backend,
            cuda_available,
            device_count,
            mps_available,
            torch.get_num_threads(),
            _safe_get_torch_interop_threads(torch),
            params.get("sig_samples"),
        )


def _safe_get_torch_interop_threads(torch: object) -> int | str:
    getter = getattr(torch, "get_num_interop_threads", None)
    if getter is None:
        return "n/a"
    with contextlib.suppress(RuntimeError):
        return int(getter())
    return "n/a"

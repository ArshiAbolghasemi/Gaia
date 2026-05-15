from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, cast

import faiss
import numpy as np
from scipy.special import digamma
from tigramite.independence_tests.independence_tests_base import CondIndTest

logger = logging.getLogger(__name__)


class FAISSCMI(CondIndTest):
    X_ID = 0
    Y_ID = 1
    Z_ID = 2

    @property
    def measure(self) -> Any:
        return self._measure

    def __init__(  # noqa: PLR0913
        self,
        k: int = 5,
        *,
        use_gpu: bool = True,
        gpu_device: int = 0,
        standardize: bool = True,
        jitter: float = 1e-6,
        significance: str = "shuffle_test",
        **kwargs: Any,
    ) -> None:
        if k < 1:
            msg = "CMI parameter 'k' must be at least 1."
            raise ValueError(msg)
        if significance == "analytic":
            msg = "FAISSCMI does not provide analytic significance; use 'shuffle_test'."
            raise ValueError(msg)

        self.k = int(k)
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.standardize = standardize
        self.jitter = jitter
        self._measure = "faiss_cmi"
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self._backend_logged = False
        self.res = self._build_gpu_resources() if use_gpu else None

        CondIndTest.__init__(self, significance=significance, **kwargs)
        self._log_runtime_configuration()

    def _build_gpu_resources(self) -> Any | None:
        standard_gpu_resources = getattr(faiss, "StandardGpuResources", None)
        if standard_gpu_resources is None:
            return None
        try:
            return cast("Any", standard_gpu_resources)()
        except Exception:
            return None

    def _build_index(self, dim: int) -> Any:
        cpu_index = faiss.IndexFlatL2(dim)
        index_cpu_to_gpu = getattr(faiss, "index_cpu_to_gpu", None)
        if self.res is None or index_cpu_to_gpu is None:
            self._log_backend_once(cpu_index)
            return cpu_index
        try:
            index = cast("Any", index_cpu_to_gpu)(self.res, self.gpu_device, cpu_index)
            self._log_backend_once(index)
        except Exception:
            self._log_backend_once(cpu_index)
            return cpu_index
        else:
            return index

    def _prepare_array(self, array: np.ndarray) -> np.ndarray:
        prepared = np.asarray(array, dtype=np.float64).copy()
        if self.standardize:
            prepared -= prepared.mean(axis=1, keepdims=True)
            std = prepared.std(axis=1, keepdims=True)
            std[std == 0.0] = 1.0
            prepared /= std

        if self.jitter > 0.0:
            scale = prepared.std(axis=1, keepdims=True)
            scale[scale == 0.0] = 1.0
            noise = self.random_state.random(prepared.shape)
            prepared += self.jitter * scale * noise

        return prepared

    def _knn_distances(self, data: np.ndarray) -> np.ndarray:
        samples = np.ascontiguousarray(data, dtype=np.float32)
        index = self._build_index(samples.shape[1])
        index.add(samples)
        distances, _ = index.search(samples, self.k + 1)
        return np.sqrt(distances[:, -1].astype(np.float64))

    def _count_neighbors(self, space: np.ndarray, eps: np.ndarray) -> np.ndarray:
        n_samples = space.shape[0]
        if space.shape[1] == 0:
            return np.full(n_samples, n_samples - 1, dtype=np.float64)

        counts = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            distances = np.linalg.norm(space - space[i], axis=1)
            counts[i] = np.sum(distances < eps[i]) - 1
        return counts

    def get_dependence_measure(
        self,
        array: np.ndarray,
        xyz: np.ndarray,
        data_type: np.ndarray | None = None,
    ) -> Any:
        del data_type

        prepared = self._prepare_array(array)
        x = prepared[xyz == self.X_ID].T
        y = prepared[xyz == self.Y_ID].T
        z = prepared[xyz == self.Z_ID].T

        xyz_space = np.column_stack([x, y, z])
        xz_space = np.column_stack([x, z])
        yz_space = np.column_stack([y, z])

        eps = self._knn_distances(xyz_space)
        nxz = self._count_neighbors(xz_space, eps)
        nyz = self._count_neighbors(yz_space, eps)
        nz = self._count_neighbors(z, eps)

        value = (
            digamma(self.k)
            + np.mean(digamma(nz + 1.0))
            - np.mean(digamma(nxz + 1.0) + digamma(nyz + 1.0))
        )
        return float(value)

    def get_shuffle_significance(
        self,
        array: np.ndarray,
        xyz: np.ndarray,
        value: float,
        data_type: np.ndarray | None = None,
        return_null_dist: bool = False,  # noqa: FBT002
    ) -> Any:
        del data_type

        try:
            null_dist = self._get_shuffle_dist(
                array,
                xyz,
                self.get_dependence_measure,
                sig_samples=self.sig_samples,
                sig_blocklength=self.sig_blocklength,
                verbosity=self.verbosity,
            )
        except (TypeError, ValueError):
            null_dist = self._get_shuffle_dist(
                array,
                xyz,
                self.get_dependence_measure,
                sig_samples=self.sig_samples,
                sig_blocklength=1,
                verbosity=self.verbosity,
            )
        pval = float(np.sum(null_dist >= value) + 1) / (self.sig_samples + 1)
        if return_null_dist:
            return pval, null_dist
        return pval

    def _log_runtime_configuration(self) -> None:
        gpu_count = _get_faiss_gpu_count()
        torch_runtime = _get_torch_runtime()
        selected_backend = "cuda" if self.res is not None else "cpu"
        logger.info(
            "FAISSCMI runtime: selected_backend=%s, use_gpu=%s, gpu_device=%d, "
            "gpu_resources=%s, faiss_gpu_count=%s, torch_cuda_available=%s, "
            "torch_mps_available=%s, standardize=%s, k=%d, sig_samples=%s",
            selected_backend,
            self.use_gpu,
            self.gpu_device,
            self.res is not None,
            gpu_count,
            torch_runtime["cuda_available"],
            torch_runtime["mps_available"],
            self.standardize,
            self.k,
            self.sig_samples,
        )
        if torch_runtime["mps_available"]:
            logger.info(
                "FAISSCMI detected MPS support, but FAISS does not use MPS; "
                "the backend remains %s.",
                selected_backend,
            )
        if self.use_gpu and self.res is None:
            logger.info(
                "FAISSCMI requested GPU but FAISS GPU resources are unavailable; "
                "falling back to CPU indexes."
            )

    def _log_backend_once(self, index: Any) -> None:
        if self._backend_logged:
            return
        self._backend_logged = True
        index_type = type(index).__name__
        using_gpu_backend = "gpu" in index_type.lower()
        logger.info(
            "FAISSCMI search backend: backend=%s, index_type=%s",
            "gpu" if using_gpu_backend else "cpu",
            index_type,
        )


def _get_faiss_gpu_count() -> int | str:
    get_num_gpus = getattr(faiss, "get_num_gpus", None)
    if get_num_gpus is None:
        return "n/a"
    try:
        return int(get_num_gpus())
    except Exception:
        return "n/a"


def _get_torch_runtime() -> dict[str, bool | str]:
    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return {
            "cuda_available": "n/a",
            "mps_available": "n/a",
        }

    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    mps_available = False
    if mps_backend is not None and hasattr(mps_backend, "is_available"):
        mps_available = bool(mps_backend.is_available())
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": mps_available,
    }

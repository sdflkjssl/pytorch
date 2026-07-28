from __future__ import annotations

import os
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import fresh_inductor_cache
from triton.testing import do_bench


PROVIDERS = ["eager_aten", "eager_inner_tree", "inductor_default"]
M_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
N_VALUES = [2**exponent for exponent in range(9, 18)]
MEMORY_BUDGET = 40 * (1 << 30)
COLORS = {
    "eager_aten": "C0",
    "inductor_default": "C1",
    "eager_inner_tree": "C2",
}
DTYPE = torch.float32
OUTPUT_PATH = "strict_numerics_row_reduction_gbps_vs_n.png"


def _gbps(m: int, n: int, milliseconds: float) -> float:
    return m * n * DTYPE.itemsize / (milliseconds * 1e-3) / 1e9


def _sum_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum(1)


def measure(m: int, n: int, provider: str) -> float:
    if m * n * DTYPE.itemsize > MEMORY_BUDGET:
        return float("nan")

    torch.manual_seed(0)
    tensor = torch.randn(m, n, device="cuda", dtype=DTYPE)
    inner_tree = provider == "eager_inner_tree"
    try:
        with patch.dict(
            os.environ,
            {"PYTORCH_SUM_INNER_TREE": "1" if inner_tree else "0"},
        ):
            if provider.startswith("eager"):
                _sum_rows(tensor)
                milliseconds = do_bench(lambda: _sum_rows(tensor))
            else:
                torch._dynamo.reset()
                with fresh_inductor_cache():
                    compiled = torch.compile(_sum_rows)
                    compiled(tensor)
                    milliseconds = do_bench(lambda: compiled(tensor))
        return _gbps(m, n, milliseconds)
    finally:
        del tensor
        torch.cuda.empty_cache()


def _assert_inner_tree_route() -> None:
    tensor = torch.randn(1024, 300, device="cuda", dtype=DTYPE)
    try:
        with patch.dict(os.environ, {"PYTORCH_SUM_INNER_TREE": "1"}):
            inner_tree = torch.sum(tensor, 1)
        with patch.dict(os.environ, {"PYTORCH_SUM_INNER_TREE": "0"}):
            aten = torch.sum(tensor, 1)
        assert not torch.equal(inner_tree, aten), (
            "eager_inner_tree did not route to CuTeDSL; refusing to label ATen "
            "results as inner-tree results"
        )
    finally:
        del tensor
        torch.cuda.empty_cache()


def _plot(data: dict[str, dict[int, list[float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    flat_axes = axes.flatten()
    for index, m in enumerate(M_VALUES):
        axis = flat_axes[index]
        for provider in PROVIDERS:
            axis.plot(
                N_VALUES,
                data[provider][m],
                marker="o",
                label=provider,
                color=COLORS[provider],
            )
        axis.set_xscale("log", base=2)
        axis.set_title(f"M={m}")
        axis.set_xlabel("N")
        axis.set_ylabel("GB/s")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
    figure.suptitle("Inner-tree SUM reduction GB/s vs N", y=1.0)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH)


def main() -> None:
    assert torch.cuda.is_available(), "needs CUDA"
    _assert_inner_tree_route()
    print("providers:", PROVIDERS)

    data = {provider: {m: [] for m in M_VALUES} for provider in PROVIDERS}
    with inductor_config.patch({"force_disable_caches": True}):
        for m in M_VALUES:
            for n in N_VALUES:
                for provider in PROVIDERS:
                    data[provider][m].append(measure(m, n, provider))
            row = "  ".join(
                f"{provider}={data[provider][m][-1]:.0f}"
                for provider in PROVIDERS
            )
            print(f"M={m:>6} (N={N_VALUES[-1]}): {row}")

    _plot(data)
    print(f"saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import torch


def reference_attention(q, k, v, *, compute_dtype=torch.float32):
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be torch.float32 or torch.float64")
    for tensor in (q, k, v):
        if tensor.device.type != "cpu" or tensor.ndim != 2 or not tensor.is_contiguous():
            raise ValueError("expected contiguous rank-2 CPU tensors")
        if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError("expected floating-point tensors")
        if not torch.isfinite(tensor).all().item():
            raise ValueError("expected finite inputs")
    m, d = q.shape
    if m <= 0 or d <= 0 or k.shape[0] <= 0 or k.shape[1] != d or v.shape != k.shape:
        raise ValueError("expected Q[M,d], K[N,d], V[N,d] with positive dimensions")
    scores = (q.to(compute_dtype) @ k.to(compute_dtype).T) / math.sqrt(d)
    return torch.softmax(scores, dim=-1) @ v.to(compute_dtype)


def self_check():
    generator = torch.Generator(device="cpu").manual_seed(20261029)
    for m, n, d in ((1, 1, 1), (3, 5, 7), (17, 33, 17), (64, 64, 32)):
        q = torch.randn((m, d), generator=generator).half()
        k = torch.randn((n, d), generator=generator).half()
        v = torch.randn((n, d), generator=generator).half()
        actual = reference_attention(q, k, v)
        expected = reference_attention(q, k, v, compute_dtype=torch.float64)
        torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=1e-4)
        error = (actual.double() - expected).abs()
        relative = error / expected.abs().clamp_min(1e-5)
        print(f"PASS CPU PyTorch M={m} N={n} d={d} atol=1e-5 rtol=1e-4 "
              f"max_abs_error={error.max().item():.9g} "
              f"max_rel_error={relative.max().item():.9g} "
              f"rms_error={error.square().mean().sqrt().item():.9g}")

    q = torch.zeros((3, 7), dtype=torch.float16)
    k = torch.ones((5, 7), dtype=torch.float16)
    v = torch.arange(35, dtype=torch.float16).reshape(5, 7)
    torch.testing.assert_close(reference_attention(q, k, v), v.float().mean(0).expand(3, 7))
    torch.testing.assert_close(reference_attention(q, k[:1], v[:1]), v[:1].float().expand(3, 7))
    q = torch.tensor([[1000.0]], dtype=torch.float16)
    k = torch.tensor([[1000.0], [-1000.0]], dtype=torch.float16)
    v = torch.tensor([[3.0], [-7.0]], dtype=torch.float16)
    torch.testing.assert_close(reference_attention(q, k, v), torch.tensor([[3.0]]))
    print("PASS CPU PyTorch uniform, single-key, and extreme-logit checks")


def compare_cuda(executable):
    """Run the native suite, then check its exact tensors against CPU PyTorch."""
    with tempfile.TemporaryDirectory(prefix="tilemix-reference-") as directory:
        fixtures = Path(directory) / "attention.jsonl"
        subprocess.run([str(Path(executable).resolve()), "--export-json", str(fixtures)], check=True)
        count = 0
        with fixtures.open() as source:
            for line in source:
                case = json.loads(line)
                m, n, d = case["m"], case["n"], case["d"]
                q = torch.tensor(case["q"], dtype=torch.float16).reshape(m, d)
                k = torch.tensor(case["k"], dtype=torch.float16).reshape(n, d)
                v = torch.tensor(case["v"], dtype=torch.float16).reshape(n, d)
                actual = torch.tensor(case["o"], dtype=torch.float32).reshape(m, d).double()
                expected = reference_attention(q, k, v, compute_dtype=torch.float64)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                error = (actual - expected).abs()
                relative = error / expected.abs().clamp_min(1e-5)
                print(f"PASS CUDA vs CPU PyTorch FP64 M={m} N={n} d={d} "
                      f"pattern={case['pattern']} atol=1e-5 rtol=1e-4 "
                      f"max_abs_error={error.max().item():.9g} "
                      f"max_rel_error={relative.max().item():.9g} "
                      f"rms_error={error.square().mean().sqrt().item():.9g}")
                count += 1
        if count == 0:
            raise RuntimeError("native CUDA suite exported no fixtures")
        print(f"PASS {count} CUDA fixtures agree with the independent CPU PyTorch oracle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-test", type=Path, help="path to built fused_attention_test")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        print(f"PyTorch={torch.__version__} compiled_CUDA={torch.version.cuda} oracle_device=CPU")
        self_check()
        if args.cuda_test:
            compare_cuda(args.cuda_test)

"""Deterministic subprocess settings for GPU parity, without changing the parent shell."""

import os


def configure_parity_environment(backend):
    if backend == "cuda":
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        os.environ["GGML_CUDA_CUBLAS_COMPUTE_TYPE"] = "f32"
        print("CUDA parity: NVIDIA_TF32_OVERRIDE=0, GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32", flush=True)

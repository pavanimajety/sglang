import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, Optional

import requests


WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _wait_for_server(port: int, timeout_s: int = 300) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    # simple backoff
    delay = 0.2
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(delay)
        delay = min(2.0, delay * 1.5)
    raise TimeoutError(f"Server on port {port} not ready after {timeout_s}s")


def _launch_server(
    cuda_devices: str,
    port: int,
    attention_backend: str,
    extra_args: Optional[list] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        "/tmp/DeepSeek-R1-FP4/snapshots/574fdb8a5347fdbc06b2c18488699c0c17d71e05",
        "--trust-remote-code",
        "--disable-radix-cache",
        "--quantization",
        "modelopt_fp4",
        "--attention-backend",
        attention_backend,
        "--tp-size",
        "4",
        "--disable-cuda-graph",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Ensure torch compile flag exists to avoid KeyError in imports
    env.setdefault("SGLANG_ENABLE_TORCH_COMPILE", "0")
    # Enable MLA debug dumping
    env.setdefault("SGLANG_MLA_DEBUG", "1")
    env.setdefault(
        "SGLANG_MLA_DEBUG_DIR", os.path.join(WORKSPACE_ROOT, "divergence_debug")
    )
    env.setdefault("SGLANG_MLA_DEBUG_LAYER_ID", "0")
    env.setdefault("SGLANG_MLA_DEBUG_STEPS", "10")
    if env_overrides:
        env.update(env_overrides)

    return subprocess.Popen(cmd, cwd=WORKSPACE_ROOT, env=env)


def _send_generate_request(port: int, prompt: str, max_new_tokens: int = 10) -> dict:
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()

def _send_batch_generate_request(port: int, prompts: list, max_new_tokens: int = 10) -> list:
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": prompts,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def _cleanup_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch flashinfer and trtllm MLA backends and dump per-step tensors"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is great because",
        help="Single prompt to generate from (default: use all test prompts)",
    )
    parser.add_argument(
        "--use-all-prompts",
        action="store_true",
        help="Explicitly use all test prompts (this is now the default behavior)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=10,
        help="Number of tokens to generate (and steps to dump)",
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        default=0,
        help="Layer id to dump q/k/out from (per decode)",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "divergence_debug"),
        help="Directory to write dump files",
    )
    parser.add_argument(
        "--flashinfer-gpus",
        type=str,
        default="4,5,6,7",
        help="CUDA visible devices for flashinfer server",
    )
    parser.add_argument(
        "--trtllm-gpus",
        type=str,
        default="0,1,2,3",
        help="CUDA visible devices for trtllm server",
    )
    parser.add_argument(
        "--flashinfer-port",
        type=int,
        default=30001,
        help="Port for flashinfer server",
    )
    parser.add_argument(
        "--trtllm-port",
        type=int,
        default=40000,
        help="Port for trtllm server",
    )
    args = parser.parse_args()

    base_dump_dir = os.path.abspath(args.dump_dir)
    _cleanup_dir(base_dump_dir)
    os.makedirs(os.path.join(base_dump_dir, "flashinfer"), exist_ok=True)
    os.makedirs(os.path.join(base_dump_dir, "trtllm"), exist_ok=True)

    # Define all test prompts
    all_prompts = [
        "The future of AI is",
        "The future of AI is",
        "Hello, my name is Simon. I work as a machine learning engineer. I love to ",
        "The president of the United States is",
        "The president of the Republic of India is",
        "The president of the New York City is",
        "The future of AI is mainly about how it can ",
        "The capital of France is great because"
    ]
    
    # Use all prompts by default, or single prompt if specified
    if args.prompt != "The capital of France is great because" or args.use_all_prompts:
        # User specified a custom prompt or explicitly requested all prompts
        if args.use_all_prompts:
            prompts_to_test = all_prompts
            print(f"Testing {len(prompts_to_test)} prompts")
        else:
            prompts_to_test = [args.prompt]
            print(f"Testing single prompt: {args.prompt}")
    else:
        # Default behavior: use all prompts
        prompts_to_test = all_prompts
        print(f"Testing {len(prompts_to_test)} prompts (default)")

    fi_env = {
        "SGLANG_MLA_DEBUG_LAYER_ID": str(args.layer_id),
        "SGLANG_MLA_DEBUG_STEPS": str(args.tokens),
        "SGLANG_MLA_DEBUG_VERBOSE": "1",
        "SGLANG_MLA_DEBUG_DIR": base_dump_dir,
    }
    trt_env = {
        "SGLANG_MLA_DEBUG_LAYER_ID": str(args.layer_id),
        "SGLANG_MLA_DEBUG_STEPS": str(args.tokens),
        "SGLANG_MLA_DEBUG_VERBOSE": "1",
        "SGLANG_MLA_DEBUG_DIR": base_dump_dir,
    }

    print("Launching flashinfer server...")
    fi_proc = _launch_server(
        cuda_devices=args.flashinfer_gpus,
        port=args.flashinfer_port,
        attention_backend="flashinfer",
        extra_args=None,
        env_overrides=fi_env,
    )

    print("Launching trtllm_mla server...")
    trt_proc = _launch_server(
        cuda_devices=args.trtllm_gpus,
        port=args.trtllm_port,
        attention_backend="trtllm_mla",
        extra_args=None,
        env_overrides=trt_env,
    )

    try:
        print("Waiting for servers to be ready...")
        _wait_for_server(args.flashinfer_port)
        _wait_for_server(args.trtllm_port)

        # Send batch requests to both servers
        print(f"\n=== Sending batch of {len(prompts_to_test)} prompts ===")
        print("Prompts:", prompts_to_test)
        
        print("Sending batch request to flashinfer...")
        fi_res = _send_batch_generate_request(args.flashinfer_port, prompts_to_test, args.tokens)
        print("Flashinfer responses:")
        for i, response in enumerate(fi_res):
            print(f"  {i+1}: {response.get('text', '<no text>')}")

        print("Sending batch request to trtllm_mla...")
        trt_res = _send_batch_generate_request(args.trtllm_port, prompts_to_test, args.tokens)
        print("TRTLLM responses:")
        for i, response in enumerate(trt_res):
            print(f"  {i+1}: {response.get('text', '<no text>')}")

        print(f"\nDone. Dumps are in:")
        print(os.path.join(base_dump_dir, "flashinfer"))
        print(os.path.join(base_dump_dir, "trtllm"))
    finally:
        for proc, name in ((fi_proc, "flashinfer"), (trt_proc, "trtllm")):
            if proc and proc.poll() is None:
                print(f"Terminating {name} server...")
                try:
                    proc.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        # give processes time to exit gracefully
        time.sleep(2)
        for proc in (fi_proc, trt_proc):
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()



import os
import fnmatch
import json
import pathlib
import threading
import random
import time
from warnings import warn

import torch
import datasets
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from eval.args import RunnerArguments, HFArguments, GenerationArguments
from eval.evaluator import HFEvaluator
from eval.tasks import ALL_TASKS

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()


def _start_keepalive():
    """
    Optional: keep the process 'active' to avoid idle timeouts on some remote shells.
    Enable by setting KEEPALIVE_INTERVAL (seconds) > 0.
    It prints a short token to stderr periodically.
    """
    try:
        requested = float(os.getenv("KEEPALIVE_INTERVAL", "0"))
    except ValueError:
        requested = 0.0

    # Default off unless explicitly enabled
    if requested <= 0:
        return

    token = os.getenv("KEEPALIVE_TOKEN", "[keepalive]")

    # Apply conservative minimums to avoid excessive noise
    try:
        min_interval = float(os.getenv("KEEPALIVE_MIN_INTERVAL", "60"))
    except ValueError:
        min_interval = 60.0

    try:
        non_tty_min = float(os.getenv("KEEPALIVE_NON_TTY_MIN", "300"))
    except ValueError:
        non_tty_min = 300.0

    try:
        jitter_pct = float(os.getenv("KEEPALIVE_JITTER_PCT", "0.1"))  # +/-10%
    except ValueError:
        jitter_pct = 0.1

    def _run():
        import sys

        # Use stricter minimum when stderr isn't a TTY (logs/CI)
        eff_min = max(min_interval, non_tty_min) if not sys.stderr.isatty() else min_interval
        base = max(requested, eff_min)

        while True:
            sys.stderr.write(f"{token}\n")
            sys.stderr.flush()
            # Add small jitter to avoid synchronized bursts
            jitter = 1.0 + random.uniform(-jitter_pct, jitter_pct)
            time.sleep(max(1.0, base * jitter))

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def main():
    # Optional heartbeat to stop 'idle pause' behaviour on some hosts
    _start_keepalive()

    args = HfArgumentParser(
        [RunnerArguments, HFArguments, GenerationArguments]
    ).parse_args()

    args.output_dir = pathlib.Path(__file__).parent / args.output_dir
    args.save_generations_raw_path = args.output_dir / args.save_generations_raw_path
    args.save_generations_prc_path = args.output_dir / args.save_generations_prc_path
    args.save_references_path = args.output_dir / args.save_references_path
    args.save_results_path = args.output_dir / args.save_results_path
    args.save_generations_raw_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_generations_prc_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_references_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_results_path.parent.mkdir(parents=True, exist_ok=True)

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = set()
        for pattern in args.tasks.split(","):
            for matching in fnmatch.filter(ALL_TASKS, pattern):
                task_names.add(matching)
        task_names = list(task_names)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    if args.generations_path:
        if accelerator.is_main_process:
            print("Evaluation only mode")
        evaluator = HFEvaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)
    else:
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(
                f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
            )
        print(f"Loading the model and tokenizer from HF (in {args.precision})")
        # Optional: small runtime tweaks that help with fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cuda.matmul.allow_tf32 = True

        # Switch between MP (device_map) and the default DP flow.
        if args.model_parallel:
            os.environ["LINC_MODEL_PARALLEL"] = "1"  # generation.py uses this to skip accelerator.prepare
            dm = args.device_map or "auto"
            # allow passing a JSON path
            if isinstance(dm, str) and dm.endswith(".json") and os.path.isfile(dm):
                with open(dm) as f:
                    dm = json.load(f)

            # Try requested device_map, fall back to sensible candidates if it fails.
            def _try_load(devmap):
                return AutoModelForCausalLM.from_pretrained(
                    args.model,
                    revision=args.revision,
                    torch_dtype=dict_precisions[args.precision],
                    trust_remote_code=args.trust_remote_code,
                    token=args.use_auth_token,
                    device_map=devmap,
                    low_cpu_mem_usage=True,
                )

            last_exc = None
            for candidate in ([dm] if dm is not None else []) + ["auto", "balanced", "balanced_low_0"]:
                try:
                    model = _try_load(candidate)
                    if candidate != dm:
                        warn(f"Requested device_map '{dm}' failed, falling back to '{candidate}'")
                    break
                except Exception as e:
                    last_exc = e
                    continue
            else:
                raise RuntimeError(f"Failed to load model with any device_map candidate") from last_exc
        else:
            # Data-parallel path: do not pass device_map so Accelerate can place the model.
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                torch_dtype=dict_precisions[args.precision],
                trust_remote_code=args.trust_remote_code,
                token=args.use_auth_token,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            token=args.use_auth_token,
            truncation_side="left",
        )
        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
        tokenizer.pad_token = tokenizer.eos_token
        evaluator = HFEvaluator(accelerator, model, tokenizer, args)

        for task in task_names:
            if args.generation_only:
                if accelerator.is_main_process:
                    print("Generation mode only")
                generations_prc, generations_raw, references = evaluator.generate_text(
                    task
                )
                if accelerator.is_main_process:
                    if args.save_generations_raw:
                        with open(args.save_generations_raw_path, "w") as fp:
                            json.dump(generations_raw, fp)
                            print("raw generations were saved")
                    if args.save_generations_prc:
                        with open(args.save_generations_prc_path, "w") as fp:
                            json.dump(generations_prc, fp)
                            print("processed generations were saved")
                    if args.save_references:
                        with open(args.save_references_path, "w") as fp:
                            json.dump(references, fp)
                            print("references were saved")
            else:
                results[task] = evaluator.evaluate(task)

    results["config"] = {"model": args.model}
    if not args.generation_only:
        dumped = json.dumps(results, indent=2, sort_keys=True)
        if accelerator.is_main_process:
            print(dumped)

        if args.save_results:
            with open(args.save_results_path, "w") as f:
                f.write(dumped)


if __name__ == "__main__":
    main()

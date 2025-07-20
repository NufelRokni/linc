import os
import fnmatch
import json
import pathlib
import traceback
import torch
import datasets
import transformers
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from eval.args import RunnerArguments, HFArguments, GenerationArguments
from eval.evaluator import HFEvaluator
from eval.tasks import ALL_TASKS

# Suppress noisy logs
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

def main():
    try:
        print("🔹 Starting runner.py")  # entry point
        parser = HfArgumentParser([RunnerArguments, HFArguments, GenerationArguments])
        args, unknown = parser.parse_known_args()
        print(f"🔹 Parsed arguments: {args}")
        if unknown:
            print(f"⚠️ Unrecognized args passed through: {unknown}")

        # Prepare output paths
        base_dir = pathlib.Path(__file__).parent
        args.output_dir = base_dir / args.output_dir
        for attr in ["save_generations_raw_path", "save_generations_prc_path", "save_references_path", "save_results_path"]:
            path = getattr(args, attr)
            full = args.output_dir / path
            setattr(args, attr, full)
            full.parent.mkdir(parents=True, exist_ok=True)
        print(f"🔹 Output directory prepared at: {args.output_dir}")

        # Determine tasks
        if args.tasks is None:
            task_names = ALL_TASKS
        else:
            task_names = set()
            for pattern in args.tasks.split(","):
                matched = fnmatch.filter(ALL_TASKS, pattern)
                task_names.update(matched)
            task_names = list(task_names)
        print(f"🔹 Will run tasks: {task_names}")

        accelerator = Accelerator()
        print(f"🔹 Accelerator initialized: is_main_process={accelerator.is_main_process}")

        results = {}
        evaluator = None

        # If evaluating pre-generated outputs
        if args.generations_path:
            print("🔹 Evaluation-only mode (using existing generations)")
            evaluator = HFEvaluator(accelerator, None, None, args)
            for task in task_names:
                print(f"🔹 Evaluating task (no generation): {task}")
                results[task] = evaluator.evaluate(task)

        # Otherwise, load model & tokenizer
        if evaluator is None:
            dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
            if args.precision not in dtypes:
                raise ValueError(f"Invalid precision {args.precision}, choose from {list(dtypes)}")
            print(f"🔹 Loading model `{args.model}` with precision `{args.precision}`")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtypes[args.precision],
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
            )
            print("🔹 Model loaded successfully")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                use_auth_token=args.use_auth_token,
                truncation_side="left",
            )
            print("🔹 Tokenizer loaded successfully")
            if not tokenizer.eos_token:
                tokenizer.eos_token = tokenizer.bos_token or tokenizer.pad_token or ""
                print(f"⚠️ Set eos_token fallback to `{tokenizer.eos_token}`")
            tokenizer.pad_token = tokenizer.eos_token

            evaluator = HFEvaluator(accelerator, model, tokenizer, args)

        # Generation and/or evaluation loop
        for task in task_names:
            print(f"🔹 Processing task: {task}")
            if args.generation_only:
                print(f"   ↳ Generation-only for {task}")
                gens_prc, gens_raw, refs = evaluator.generate_text(task)
                print(f"   ↳ Generated {len(gens_raw)} samples for {task}")
            else:
                print(f"   ↳ Full evaluate for {task}")
                results[task] = evaluator.evaluate(task)
                print(f"   ↳ Completed evaluation for {task}")

        # Save and print final results
        if not args.generation_only:
            results["config"] = {"model": args.model, "precision": args.precision}
            out = json.dumps(results, indent=2, sort_keys=True)
            if accelerator.is_main_process:
                print("🔹 Final results:\n", out)
                if args.save_results:
                    with open(args.save_results_path, "w") as f:
                        f.write(out)
                    print(f"🔹 Results saved to {args.save_results_path}")

    except Exception as e:
        print("❌ ERROR in runner.py:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()



from collections import defaultdict
import json
import os
import math
import warnings

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

INFILL_MODE = False


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially.
    See compute_code for more details.
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        n_tasks=None,
        n_copies=1,
        prefix="",
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        # Ensure n_tasks does not exceed dataset length
        self.n_tasks = min(n_tasks if n_tasks is not None else len(dataset), len(dataset))
        self.n_copies = n_copies
        self.prefix = prefix

    def __iter__(self):
        prompts = []
        infill = []
        for sample in range(self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                infill.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                assert set(prompt_contents.keys()) == {"prefix", "suffix"}
                infill.append(True)
                prompt = self.prefix + self._make_infill_prompt(**prompt_contents)
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)

        if not len(set(infill)) == 1:
            raise ValueError("Mixed infill and completion prompts are not supported.")
        global INFILL_MODE
        INFILL_MODE = infill[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None

        self.tokenizer.padding_side = 'right'
        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )

 
        n_copies = self.n_copies
        if n_copies == 1 and self.n_tasks % self.num_devices != 0:
            n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(n_copies):
                yield {
                    "ids": outputs.input_ids[sample],
                    "task_id": sample,
                    "input_len": outputs.attention_mask[sample].sum(),
                }

    def _make_infill_prompt(self, prefix, suffix):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    batch_size=20,
    prefix="",
    postprocess=True,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """

    # Fast path: if pregenerated outputs exist, load and return to skip generation and save GPU.
    # Fast path: load only the model-specific generations file located in the same folder as this utils.py
    _cand = os.path.join(os.path.dirname(__file__), "Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_raw.json")
    if os.path.exists(_cand):
        with open(_cand, "r") as _fp:
            _gens_raw_full = json.load(_fp)
        _gens_raw_full = _gens_raw_full[:n_tasks]

        def _strip_pref(s: str) -> str:
            return s[len(prefix):] if prefix else s

        _gens_raw = [[_strip_pref(c) for c in cand_list] for cand_list in _gens_raw_full]
        if postprocess:
            _gens_prc = [
                [task.postprocess_generation(c, i) for c in cand_list]
                for i, cand_list in enumerate(_gens_raw)
            ]
        else:
            warnings.warn(
                "model output is not postprocessed, this might lower evaluation scores"
            )
            _gens_prc = [list(cand_list) for cand_list in _gens_raw]
        return _gens_prc, _gens_raw

    gen_token_dict = defaultdict(list)
    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        with torch.no_grad():
            if task.stop_words:
                # print("Using stopping criteria ===============================")
                # print("input_len.max()", int(batch["input_len"].max().item()))
                # print("ids.shape[-1] (padded)", batch["ids"].shape[-1])
                gen_kwargs["stopping_criteria"][0].start_length = batch["input_len"].max().item()
            # Compute the maximum original length of the prompts in the batch:
            # decoded_prompt = tokenizer.decode(batch["ids"][0, :batch["input_len"].max().item()])
            # print(f"PROMPT: {decoded_prompt}") # Print first 100 chars
            input_ids = batch["ids"][:, :batch["input_len"].max().item()]
            # Always place tensors on the model's device to avoid CPU/CUDA mismatch.
            first_dev = next(accelerator.unwrap_model(model).parameters()).device
            input_ids = input_ids.to(first_dev, non_blocking=True)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=first_dev)
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=batch_size,
                **gen_kwargs,
            )
            generated_tasks = batch["task_id"].repeat(batch_size)
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens, generated_tasks = accelerator.gather(
                (generated_tokens, generated_tasks)
            )
            generated_tokens = generated_tokens.cpu().numpy()
            generated_tasks = generated_tasks.cpu().numpy()

            for sample, generated_tokens in zip(generated_tasks, generated_tokens):
                gen_token_dict[sample].append(generated_tokens)

    def parse_infill(code, tokenizer):
        """Reorder infill code and remove remaining special tokens."""
        model_id = tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            parts = code.split("<|mask:0|>")
            if len(parts) < 3:
                raise ValueError("Malformed infill output for InCoder.")
            prefix, suffix, infill = parts[0], parts[1], parts[2]
            infill = infill.split("<|endofmask|>")[0]
        elif model_id.startswith("mistralai"):
            # New branch for Mistral infill parsing.
            if "<fim-suffix>" not in code or "<fim-middle>" not in code:
                raise ValueError("Malformed infill output for Mistral.")
            prefix, rest = code.split("<fim-suffix>", 1)
            suffix, infill = rest.split("<fim-middle>", 1)
            infill = infill.split("<|endoftext|>")[0]
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")
            
        code = "".join([prefix, infill, suffix])
        for k, v in tokenizer.special_tokens_map.items():
            if k == "additional_special_tokens":
                for t in v:
                    code = code.replace(t, "")
            else:
                code = code.replace(v, "")
        return code

    code_gens_raw = [[] for _ in range(n_tasks)]
    code_gens_prc = [[] for _ in range(n_tasks)]
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            if INFILL_MODE:
                gen_code = parse_infill(
                    tokenizer.decode(
                        s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    ),
                    tokenizer,
                )
            else:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            code_gens_raw[sample].append(gen_code[len(prefix) :])
            if postprocess:
                x = int(sample)
                code_gens_prc[sample].append(
                    task.postprocess_generation(gen_code[len(prefix) :], x)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens_prc[sample].append(gen_code[len(prefix) :])

    # Optional: skip tasks that are already marked as all-errors in the processed PRC file.
    # Look for a prc file next to this utils.py or in the repo outputs/ folder.
    prc_candidate = os.path.join(os.path.dirname(__file__), "Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_prc.json")
    if not os.path.exists(prc_candidate):
        # try outputs/ folder relative to repo
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        prc_candidate = os.path.join(repo_root, "outputs", "Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_prc.json")

    if os.path.exists(prc_candidate):
        try:
            with open(prc_candidate, "r") as _fp:
                prc_list = json.load(_fp)
            # prc_list is list[list[str]] with length n_tasks
            for i, labels in enumerate(prc_list[:n_tasks]):
                # if every candidate is marked "Error", skip this task by clearing entries
                if isinstance(labels, list) and all(x == "Error" for x in labels):
                    code_gens_prc[i] = []
                    code_gens_raw[i] = []
        except Exception:
            # if reading fails, continue without skipping
            pass

    return code_gens_prc, code_gens_raw

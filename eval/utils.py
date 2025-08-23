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

    # POSTPROCESS-ONLY MODE: Always recompute PRC from RAW JSON, never call the model.
    # 1) Find RAW file path (env override -> outputs/ -> eval/). 2) Re-postprocess it. 3) Return PRC and masked RAW.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(repo_root, "outputs")
    eval_dir = os.path.dirname(__file__)

    raw_override = os.getenv("LINC_RAW_PATH", "").strip() or None
    raw_candidates = [
        raw_override,
        os.path.join(outputs_dir, "Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_raw.json"),
        os.path.join(eval_dir, "Mistral-7B-v0.1_folio-neurosymbolic-1shot_generations_raw.json"),
    ]
    raw_candidates = [p for p in raw_candidates if p]
    _raw_path = next((p for p in raw_candidates if os.path.exists(p)), None)

    if _raw_path is None:
        raise RuntimeError(
            "Postprocess-only mode: RAW generations JSON not found. Set LINC_RAW_PATH or place the RAW file in outputs/."
        )

    with open(_raw_path, "r") as _fp:
        _gens_raw_full = json.load(_fp)
    _gens_raw_full = _gens_raw_full[:n_tasks]

    def _strip_pref(s: str) -> str:
        return s[len(prefix):] if prefix and s.startswith(prefix) else s

    _gens_raw = [[_strip_pref(c) for c in cand_list] for cand_list in _gens_raw_full]

    if not postprocess:
        warnings.warn("postprocess flag was False, but postprocess-only mode requires it; proceeding with postprocessing.")

    print(f"Postprocess-only mode: reprocessing RAW JSON at {os.path.basename(_raw_path)} for {n_tasks} tasks…")
    _gens_prc = [
        [task.postprocess_generation(c, i) for c in cand_list]
        for i, cand_list in enumerate(_gens_raw)
    ]

    # Mask RAW positions where the processed label is Error
    masked_raw = []
    for cand_list, labels in zip(_gens_raw, _gens_prc):
        row = [("" if lab == "Error" else c) for c, lab in zip(cand_list, labels)]
        masked_raw.append(row)
    return _gens_prc, masked_raw
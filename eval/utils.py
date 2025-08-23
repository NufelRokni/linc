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

    with open("Mistral-7B-v0.1_folio-1shot_generations_raw.json", "r") as f:
        gens_raw_full = json.load(f)
    with open("Mistral-7B-v0.1_folio-1shot_references.json", "r") as f:
        gens_ref = json.load(f)
    with open("Mistral-7B-v0.1_folio-1shot_generations_prc.json", "r") as f:
        gens_prc = json.load(f)

    print(f"Loaded {len(gens_raw_full)} raw generations, {len(gens_ref)} references, and {len(gens_prc)} processed generations.")
    print(f"Sample raw generation: {gens_raw_full[0]}")
    print(f"Sample reference: {gens_ref[0]}")
    print(f"Sample processed generation: {gens_prc[0]}")
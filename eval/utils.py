from collections import defaultdict
import math
import warnings
import json
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm


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
        self.n_tasks = min(
            n_tasks if n_tasks is not None else len(dataset), len(dataset)
        )
        self.n_copies = n_copies
        self.prefix = prefix

    def __iter__(self):
        # Build plain completion prompts only.
        prompts = []
        for sample in range(self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if not isinstance(prompt_contents, str):
                raise ValueError(
                    "Infill-style (dict) prompts are no longer supported. Expected a string prompt."
                )
            prompts.append(self.prefix + prompt_contents)

        # Standard tokenization
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
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


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    batch_size=20,
    **gen_kwargs,
):
    """Generate multiple codes for each task in the dataset using multiple GPUs with accelerate.
    dataloader sends all the prompts from the evalution dataset to the model as the following:
    [p_0_0, p_0_1, ..., p_0_nc-1, p_1_0, ..., p_nt-1_nc-1] where nc is the number of copies of the prompt,
    and nt is the number of tasks. nc is such that num_samples(for each task)= nc * batch_size
    """
    debug = False
    
    gen_token_dict = defaultdict(list)
    for step, batch in tqdm(
        enumerate(dataloader),
        total=math.ceil(
            n_tasks * dataloader.dataset.n_copies / accelerator.num_processes
        ),
    ):
        if not debug:
            with torch.no_grad():
                if task.stop_words:
                    gen_kwargs["stopping_criteria"][0].start_length = (
                        batch["input_len"].max().item()
                    )
                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids=batch["ids"][:, : batch["input_len"].max().item()],
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
                
    if not debug:
        with open("gen_token_dict.json", "w") as f:
            # Convert both keys and numpy array values to JSON-serializable types
            serializable_dict = {int(k): [arr.tolist() for arr in v] for k, v in gen_token_dict.items()}
            json.dump(serializable_dict, f)
    
    else:        
        with open("gen_token_dict.json", "r") as f:
            gen_token_dict = json.load(f)

    code_gens_raw = [[] for _ in range(n_tasks)]
    code_gens_prc = [[] for _ in range(n_tasks)]
    for sample, generated_tokens in gen_token_dict.items():
        code_gens_raw[sample], code_gens_prc[sample] = \
            task.pre_postprocess_generation(tokenizer, sample, generated_tokens)
    return code_gens_prc, code_gens_raw
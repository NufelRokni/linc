from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import random
import json
import os
import warnings
from abc import abstractmethod, ABC

from eval import tasks
from eval.generation import parallel_generations

_WARNING = """
################################################################################
 !!!WARNING!!!
################################################################################
 The task you are about to use executes untrusted model-generated code. Although
 it is highly unlikely that model-generated code will do something overtly 
 malicious in response to this test suite, model-generated code may act 
 destructively due to a lack of model capability or alignment. Users are strongly
 encouraged to sandbox this evaluation suite so that it does not perform 
 destructive actions on their host or network. Once you have read this disclaimer
 and taken appropriate precautions, set the argument "allow_code_execution" to True.
################################################################################
"""

class Evaluator(ABC):
    def __init__(self, args):
        self.args = args
        self.allow_code_execution = args.allow_code_execution

    @abstractmethod
    def generate_text(self, task_name):
        pass

    def evaluate(self, task_name):
        task = tasks.get_task(task_name)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)
        generations_prc, generations_raw, references = self.generate_text(task_name)
        if len(generations_prc[0]) != self.args.n_samples:
            generations_prc = [l[: self.args.n_samples] for l in generations_prc]
            warnings.warn(
                "Number of tasks wasn't proportional to number of devices, we removed extra predictions"
            )
        if not hasattr(self, "accelerator") or self.accelerator.is_main_process:
            if not self.args.generations_path:
                if self.args.save_generations_raw:
                    with open(self.args.save_generations_raw_path, "w") as fp:
                        json.dump(generations_raw, fp)
                    print("raw generations were saved")
                if self.args.save_generations_prc:
                    with open(self.args.save_generations_prc_path, "w") as fp:
                        json.dump(generations_prc, fp)
                    print("processed generations were saved")
                if self.args.save_references:
                    with open(self.args.save_references_path, "w") as fp:
                        json.dump(references, fp)
                    print("references were saved")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        results = task.process_results(generations_prc, references)
        return results

class HFEvaluator(Evaluator):
    def __init__(self, accelerator, model, tokenizer, args):
        super().__init__(args)
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        generations_prc, generations_raw = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        return generations_prc, generations_raw, references


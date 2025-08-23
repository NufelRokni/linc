from functools import cache
from collections import Counter
from eval.tasks.utils import evaluate_fol_manually
from eval.task_base import Task
from eval.fol_utils import (
    reformat_fol_samples_train,
    add_conclusion_fols_train,
    add_cot_train,
)
from datasets import load_dataset


class OWAFOLTask(Task):
    """An OWA (Open World Assumption) FOL (First Order Logic) Task is a Task in which the goal
    is to generate True/False/Uncertain answers to First Order Logic questions.
    """

    TRAIN_DATASET_PATH = "minimario/FOLIO"
    ERROR_TOKEN = "Error"
    MAX_SHOTS = 16

    def __init__(self, mode, n):
        assert n <= self.MAX_SHOTS, f"supports up to {self.MAX_SHOTS}-shot"
        super().__init__(
            stop_words=["</EVALUATE>"],
            requires_execution=True,
        )
        self._mode = mode
        self._nshot = n
        self.train_dataset = load_dataset(self.TRAIN_DATASET_PATH)["train"]
        self._train_dataset = reformat_fol_samples_train(self.train_dataset)
        self._train_dataset = add_conclusion_fols_train(self._train_dataset)
        self._train_dataset = add_cot_train(self._train_dataset)
        self._train_dataset = self._train_dataset.map(
            lambda x: {"label": "Uncertain" if x["label"] == "Unknown" else x["label"]},
            remove_columns=["label"],
        )
        self._train_fewshot_indices_all = [
            125,
            23,
            60,
            275,
            148,
            261,
            263,
            683,
            299,
            684,
            850,
            853,
            886,
            892,
            930,
            980,
        ]
        # Labels:
        # 23 (True), 60 (False), 125 (Uncertain), 148 (False), 261 (True), 263 (True), 275 (Uncertain), 683 (Uncertain)
        # 299 (True), 684 (False), 850 (False), 853 (Uncertain), 886 (True), 892 (Uncertain), 930 (False), 980 (False)

        self._train_fewshot_indices = self._train_fewshot_indices_all[:n]
        self._train = self._train_dataset.select(self._train_fewshot_indices)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self._test

    def format_train_example(self, doc):
        example = self.format_test_example(doc)
        if self._mode == "baseline":
            example += f"{doc['label'].strip()}\n"
        elif self._mode == "cot":
            example += f"{doc['cot']}\n"
        else:
            for premise, fol in zip(doc["premises"], doc["premises-FOL"]):
                example += f"TEXT:\t{premise.strip()}\nFOL:\t{fol.strip()}\n"
            example += f"TEXT:\t{doc['conclusion'].strip()}\nFOL:\t{doc['conclusion-FOL'].strip()}\n"
            if self._mode == "scratchpad":
                example += f"ANSWER:\t{doc['label'].strip()}\n"
        return example + "</EVALUATE>\n"

    def format_test_example(self, doc):
        example = "<PREMISES>\n"
        for premise in doc["premises"]:
            example += f"{premise.strip()}\n"
        example += "</PREMISES>\n"
        example += f"<CONCLUSION>\n{doc['conclusion'].strip()}\n</CONCLUSION>\n"
        example += "<EVALUATE>\n"
        return example

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        instructions = self.get_instructions()
        train = self.fewshot_examples()
        test = self.format_test_example(doc)
        prompt = "\n".join([instructions, train, test])
        return prompt

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["label"]

    def postprocess_generation(self, generation, idx, completion_only=False):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        # """
        try:
            print(f"Length of generation: {len(generation)}")
            print(f"")
            gen = generation[len(prefix) :].strip()
            for stop_word in self.stop_words:
                gen = gen.split(stop_word)[0].strip()
            if self._mode == "baseline":
                resp = gen.strip()
            elif self._mode == "scratchpad":
                flag = "ANSWER:"
                resp = gen.split(flag)[-1].strip()
            elif self._mode == "neurosymbolic":
                flag = "FOL:"
                parses = [
                    line.replace(flag, "").strip()
                    for line in gen.split("\n")
                    if flag in line
                ]
                premises, conclusion = parses[:-1], parses[-1]
                print(f"Evaluating FOL with premises: {premises} and conclusion: {conclusion}")
                resp = evaluate_fol_manually(premises, conclusion)
            elif self._mode == "cot":
                flag = "ANSWER:"
                resp = gen.split(flag)[-1].strip()
            else:
                raise ValueError(f"Invalid mode: {self._mode}")
            assert resp in ["True", "False", "Uncertain"], f"Invalid generation: {resp}"
            return resp
        except Exception as e:
            # TODO: explore failure cases and improve postprocessing
            print(f"Error in parsing and/or evaluating LLM output: {e}")
            return self.ERROR_TOKEN

    @staticmethod
    def metric(generations, references, error_token):
        correct = 0
        for gens, ref in zip(generations, references):
            gens = [gen for gen in gens if gen != error_token]
            if len(gens) > 0:
                majority = Counter(gens).most_common(1)[0][0]
                if majority == ref:
                    correct += 1
        return {f"accuracy (pass@1 majority)": correct / len(references)}

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        return self.metric(generations, references, self.ERROR_TOKEN)

    @cache
    def fewshot_examples(self):
        """
        Returns a few-shot example for the task.
        :param n: int
            number of examples
        :param seed: int
            seed for random number generator
        :return: str
        """
        examples = []
        for doc in self._train.select(range(self._nshot)):
            examples.append(self.format_train_example(doc))
        return "\n".join(examples)

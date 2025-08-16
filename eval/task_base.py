from abc import ABC, abstractmethod
from datasets import load_dataset
from warnings import warn


class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except:
            warn(
                "This task will use a locally downloaded dataset, not from the HF hub."
            )
            
    def get_instruction(self):
        instructions = ""

        instructions += "You are a strict First-Order Logic (FOL) assistant.\n"
        instructions += "Premises are a set of FOL sentences; The conclusion is one FOL sentence.\n"

        if self._mode == "baseline":
            instructions += "Decide whether the CONCLUSION follows from the PREMISES.\n"
            instructions += "Labels are LIMITED to: True, False, Uncertain (case-sensitive; no quotes).\n"
            instructions += "Output exactly one Label as ANSWER: <label>\n"
            instructions += "add the finish end close </EVALUATE>\n\n"

        if self._mode == "cot":
            instructions += (
                "- Output a chain of thought leading to the conclusion.\n"
                "- Include all relevant premises in your reasoning.\n"
                "- Clearly indicate the final conclusion\n"
                "- At the end Output exactly one Label as ANSWER: <label>\n"
                "Labels are case-sensitive and LIMITED to: True, False, Uncertain.\n"
            )
        if self._mode == "scratchpad":
            instructions += (
                "Translate the premises and conclusion into FOL expressions.\n"
                "Use a scratchpad to work out the answer before responding.\n" # maybe you can call it FOL NLP reasoning
                "- At the end Output exactly one Label as ANSWER: <label>\n"
                "Labels are case-sensitive and LIMITED to: True, False, Uncertain.\n"
          if self._mode == "neurosymbolic":
            instructions += (
                "- Translate the premises and conclusion into FOL expressions.\n"
            )
            instructions += "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
            instructions += "Expressions should be adhere to the format of the Python NLTK package logic module."

    # def get_instructions(self):
    #     instructions = ""
    #     instructions += "The following is a first-order logic (FOL) problem.\n"
    #     instructions += "The problem is to determine whether the conclusion follows from the premises.\n"
    #     instructions += "The premises are given in the form of a set of first-order logic sentences.\n"
    #     instructions += "The conclusion is given in the form of a single first-order logic sentence.\n"
    #     if self._mode == "baseline":
    #         instructions += f"The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
    #     else:
    #         instructions += "The task is to translate each of the premises and conclusions into FOL expressions, "
    #         if self._mode == "scratchpad":
    #             instructions += f"and then to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises."
    #         elif self._mode == "neurosymbolic":
    #             instructions += "so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.\n"
    #             instructions += "Expressions should be adhere to the format of the Python NLTK package logic module."
    #     return instructions + "\n\n"

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass
    
    
    
import re
import nltk
from nltk.sem import logic
from nltk.sem import Expression

logic._counter._value = 0
read_expr = Expression.fromstring
prover = nltk.Prover9(10)


def convert_to_nltk_rep(logic_formula):
    translation_map = {
        "∀": "all ",
        "∃": "exists ",
        "→": "->",
        "¬": "-",
        "∧": "&",
        "∨": "|",
        "⟷": "<->",
        "↔": "<->",
        "0": "Zero",
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        ".": "Dot",
        "Ś": "S",
        "ą": "a",
        "’": "",
    }

    constant_pattern = r'\b([a-z]{2,})(?!\()'
    logic_formula = re.sub(constant_pattern, lambda match: match.group(1).capitalize(), logic_formula)

    for key, value in translation_map.items():
        logic_formula = logic_formula.replace(key, value)

    quant_pattern = r"(all\s|exists\s)([a-z])"
    def replace_quant(match):
        return match.group(1) + match.group(2) + "."
    logic_formula = re.sub(quant_pattern, replace_quant, logic_formula)

    dotted_param_pattern = r"([a-z])\.(?=[a-z])"
    def replace_dotted_param(match):
        return match.group(1)
    logic_formula = re.sub(dotted_param_pattern, replace_dotted_param, logic_formula)

    simple_xor_pattern = r"(\w+\([^()]*\)) ⊕ (\w+\([^()]*\))"
    def replace_simple_xor(match):
        return ("((" + match.group(1) + " & -" + match.group(2) + ") | (-" + match.group(1) + " & " + match.group(2) + "))")
    logic_formula = re.sub(simple_xor_pattern, replace_simple_xor, logic_formula)

    complex_xor_pattern = r"\((.*?)\)\) ⊕ \((.*?)\)\)"
    def replace_complex_xor(match):
        return ("(((" + match.group(1) + ")) & -(" + match.group(2) + "))) | (-(" + match.group(1) + ")) & (" + match.group(2) + "))))")
    logic_formula = re.sub(complex_xor_pattern, replace_complex_xor, logic_formula)

    special_xor_pattern = r"\(\(\((.*?)\)\)\) ⊕ (\w+\([^()]*\))"
    def replace_special_xor(match):
        return ("(((" + match.group(1) + ")) & -" + match.group(2) + ") | (-(" + match.group(1) + ")) & " + match.group(2) + ")")
    logic_formula = re.sub(special_xor_pattern, replace_special_xor, logic_formula)
    
    return logic_formula

def get_all_variables(text):
    pattern = r'\([^()]+\)'
    matches = re.findall(pattern, text)
    all_variables = []
    for m in matches:
        m = m[1:-1]
        m = m.split(",")
        all_variables += [i.strip() for i in m]
    return list(set(all_variables))

def reformat_fol(fol):
    translation_map = {
        "0": "Zero", 
        "1": "One",
        "2": "Two",
        "3": "Three",
        "4": "Four",
        "5": "Five",
        "6": "Six",
        "7": "Seven",
        "8": "Eight",
        "9": "Nine",
        ".": "Dot",
        "’": "",
        "-": "_",
        "'": "",
        # "\"": "",
        " ": "_"
        
    }
    # Handle equality specially: turn `a = b` into an atomic predicate `Equals(a,b)`
    # This keeps equality as an atomic formula during normalization and avoids
    # accidental token mangling when replacing characters.
    def _replace_equality(match):
        left = match.group(1).strip()
        right = match.group(2).strip()
        # Convert standalone digits to word constants within equality
        digit_word = {
            "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
            "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
        }
        if left in digit_word:
            left = digit_word[left]
        if right in digit_word:
            right = digit_word[right]
        return f"Equals({left},{right})"

    # Match terms that are either simple tokens or function applications like f(x)
    eq_pattern = r'([A-Za-z0-9_]+\([^()]*\)|[A-Za-z0-9_]+)\s*=\s*([A-Za-z0-9_]+\([^()]*\)|[A-Za-z0-9_]+)'
    fol = re.sub(eq_pattern, _replace_equality, fol)

    # Normalize variables/terms that appear inside parentheses (function args)
    all_variables = get_all_variables(fol)
    for variable in all_variables:
        variable_new = variable[:]
        for k, v in translation_map.items():
            variable_new = variable_new.replace(k, v)
        fol = fol.replace(variable, variable_new)

    return fol

def evaluate(premises, conclusion):
    premises = [reformat_fol(p) for p in premises]
    conclusion = reformat_fol(conclusion)

    
    c = read_expr(conclusion)
    p_list = []
    for p in premises:
        p_list.append(read_expr(p))
    truth_value = prover.prove(c, p_list)
    if truth_value:
        return "True"
    else:
        neg_c = -c
        negation_true = prover.prove(neg_c, p_list)
        if negation_true:
            return "False"
        else:
            return "Uncertain"
        
from .fol_normalization import normal_form
from typing import List
from nltk.sem import Expression
from nltk.sem import logic
from nltk.sem.logic import (
    Expression,
    AndExpression,
    OrExpression,
    ImpExpression,
    IffExpression,
    NegatedExpression,
    AllExpression,
    ExistsExpression,
    EqualityExpression,
    ApplicationExpression,
)
        
VERBOSE_EVAL = True
RECURSION_DEPTH = 0

def evaluate_fol_manually(premises: List[str], conclusion: str) -> str:
    # print(f"Evaluating FOL with premises: {premises} and conclusion: {conclusion}")
    premises_fmt = [reformat_fol(p) for p in premises]
    conclusion_fmt = reformat_fol(conclusion)
    
    premises_expr: List[Expression] = [read_expr(p) for p in premises_fmt]
    conclusion_expr: Expression = read_expr(conclusion_fmt)
    
    # if VERBOSE_EVAL:
    #     # for p_expr in premises_expr: print(f"Premise: {p_expr}")
    #     print(f"\nInitial Conclusion: {conclusion_expr}\n")

    # nf_conclusion could be produced by a normal_form() routine; for now keep the parsed expression
    nf_conclusion: Expression = normal_form(conclusion_expr)
    # if VERBOSE_EVAL:
    #     print(f"Normal Form Conclusion: {nf_conclusion}\n")

    # Optional sanity check: NF should be equivalent to original conclusion (up to prover power)
    if VERBOSE_EVAL:
        try:
            a_impl_b = (-conclusion_expr) | nf_conclusion
            b_impl_a = (-nf_conclusion) | conclusion_expr
            eq1 = prover.prove(a_impl_b, [])
            eq2 = prover.prove(b_impl_a, [])
            
            # print(f"Equivalence checks: orig -> NF: {'✓' if eq1 else '·'}, NF -> orig: {'✓' if eq2 else '·'}")
            if eq1 and eq2:
                return "valid equivalence"
            else:
                return "invalid equivalence"
        except Exception:
            print(f"Error occurred during equivalence checks")
            return "Error occurred"

    # return recursive_evaluate(premises_expr, nf_conclusion, depth=RECURSION_DEPTH)


    

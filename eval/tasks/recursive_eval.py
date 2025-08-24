import re
import nltk
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

logic._counter._value = 0
read_expr = Expression.fromstring
prover = nltk.Prover9(10)



def determine_truth(premises, conclusion):
    true_value = prover.prove(conclusion, premises)
    false_value = prover.prove(-conclusion, premises)
    if true_value and not false_value:
        return "True"
    elif false_value and not true_value:
        return "False"
    elif not false_value and not true_value:
        return "Uncertain"
    else:
        return "contradiction"

def recursive_evaluate(premises, conclusion, depth=0):
    if depth <= 0:
        return determine_truth(premises, conclusion)

    # Peel the quantifier prefix so we can reason about splitting on the quantifier-free matrix.
    prefix = []  # list of ("all"|"exists", variable)
    matrix = conclusion
    while isinstance(matrix, (AllExpression, ExistsExpression)):
        if isinstance(matrix, AllExpression):
            prefix.append(("all", matrix.variable))
        else:
            prefix.append(("exists", matrix.variable))
        matrix = matrix.term

    def wrap_prefix(expr):
        # Re-wrap the original prefix around a new matrix expr (outermost-first order in prefix)
        result = expr
        for kind, var in reversed(prefix):
            if kind == "all":
                result = AllExpression(var, result)
            else:
                result = ExistsExpression(var, result)
        return result

    only_universals = len(prefix) > 0 and all(k == "all" for k, _ in prefix)
    only_existentials = len(prefix) > 0 and all(k == "exists" for k, _ in prefix)
    no_quantifiers = len(prefix) == 0

    # Decide how to split based on connective and safe distribution across the prefix.
    if isinstance(matrix, OrExpression):
        # Safe equivalence for splitting OR holds if there are no universals in the prefix
        # i.e., prefix is empty or only-existentials.
        if no_quantifiers or only_existentials:
            list_of_sub = get_sub_conclusions_OR(matrix)
            results = []
            for sub in list_of_sub:
                sub_full = wrap_prefix(sub)
                result = recursive_evaluate(premises, sub_full, depth - 1)
                if result == "True":
                    return "True"
                results.append(result)
            if all(r == "False" for r in results):
                return "False"
            return "Uncertain"
        else:
            # Not safe to split OR across universal quantifiers — evaluate as a whole.
            return determine_truth(premises, conclusion)

    if isinstance(matrix, AndExpression):
        # Safe equivalence for splitting AND holds if there are no existentials in the prefix
        # i.e., prefix is empty or only-universals.
        if no_quantifiers or only_universals:
            list_of_sub = get_sub_conclusions_AND(matrix)
            results = []
            for sub in list_of_sub:
                sub_full = wrap_prefix(sub)
                result = recursive_evaluate(premises, sub_full, depth - 1)
                if result == "False":
                    return "False"
                results.append(result)
            if all(r == "True" for r in results):
                return "True"
            return "Uncertain"
        else:
            # Not safe to split AND across existential quantifiers — evaluate as a whole.
            return determine_truth(premises, conclusion)

    # Atomic or not safely splittable: evaluate with the prover.
    return determine_truth(premises, conclusion)
        
def get_sub_conclusions_AND(expr):
    """Flatten a conjunction into a list of conjuncts (always returns a list)."""
    if isinstance(expr, AndExpression):
        return get_sub_conclusions_AND(expr.first) + get_sub_conclusions_AND(expr.second)
    return [expr]

def get_sub_conclusions_OR(expr):
    """Flatten a disjunction into a list of disjuncts (always returns a list)."""
    if isinstance(expr, OrExpression):
        return get_sub_conclusions_OR(expr.first) + get_sub_conclusions_OR(expr.second)
    return [expr]



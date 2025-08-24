from typing import List, Tuple
from nltk.sem.logic import (
    Expression,
    AndExpression,
    OrExpression,
    AllExpression,
    ExistsExpression,
)

import nltk
from linc.eval.tasks.utils import read_expr
from linc.eval.tasks.fol_normalization import normal_form
from linc.eval.tasks.recursive_eval import get_sub_conclusions_AND, get_sub_conclusions_OR


def peel_prefix(expr: Expression) -> Tuple[List[Tuple[str, object]], Expression]:
    """Return (prefix, matrix) where prefix is a list of ("all"|"exists", var) in outermost-first order."""
    prefix = []
    matrix = expr
    while isinstance(matrix, (AllExpression, ExistsExpression)):
        if isinstance(matrix, AllExpression):
            prefix.append(("all", matrix.variable))
        else:
            prefix.append(("exists", matrix.variable))
        matrix = matrix.term
    return prefix, matrix


def wrap_prefix(prefix: List[Tuple[str, object]], matrix: Expression) -> Expression:
    """Wrap the prefix back around matrix."""
    result = matrix
    for kind, var in reversed(prefix):
        if kind == "all":
            result = AllExpression(var, result)
        else:
            result = ExistsExpression(var, result)
    return result


def build_split_equivalent(expr: Expression) -> Expression | None:
    """
    If splitting is equivalence-preserving for expr per our rules, return the split expression.
    Otherwise return None.
    """
    prefix, matrix = peel_prefix(expr)
    only_universals = len(prefix) > 0 and all(k == "all" for k, _ in prefix)
    only_existentials = len(prefix) > 0 and all(k == "exists" for k, _ in prefix)
    no_quantifiers = len(prefix) == 0

    if isinstance(matrix, OrExpression) and (no_quantifiers or only_existentials):
        parts = get_sub_conclusions_OR(matrix)
        # OR together the rewrapped parts
        result = wrap_prefix(prefix, parts[0])
        for p in parts[1:]:
            result = result | wrap_prefix(prefix, p)
        return result

    if isinstance(matrix, AndExpression) and (no_quantifiers or only_universals):
        parts = get_sub_conclusions_AND(matrix)
        # AND together the rewrapped parts
        result = wrap_prefix(prefix, parts[0])
        for p in parts[1:]:
            result = result & wrap_prefix(prefix, p)
        return result

    return None


def check_equiv(a: Expression, b: Expression) -> Tuple[bool, bool]:
    """Return (a->b, b->a) under empty premises using a fresh prover per check."""
    local_prover = nltk.Prover9(10)
    return local_prover.prove((-a) | b, []), local_prover.prove((-b) | a, [])


def run():
    cases = [
        # Safe OR splitting (existentials)
        "exists x.(P(x)) | exists y.(Q(y))",
        "exists x.(R(x) | S(x))",  # prefix only existentials (after nf), OR matrix
        # Safe AND splitting (universals)
        "all x.(P(x)) & all y.(Q(y))",
        "all x.(R(x) & S(x))",
        # Unsafe OR splitting (universals) – should not split
        "all x.(P(x) | Q(x))",
        # Unsafe AND splitting (existentials) – should not split
        "exists x.(P(x) & Q(x))",
        # Mixed prefix – should not split
        "all x.(exists y.(R(x,y) | S(x,y)))",
    ]

    passed = 0
    total = 0
    for s in cases:
        total += 1
        expr = read_expr(s)
        nf = normal_form(expr)
        split_expr = build_split_equivalent(nf)
        if split_expr is None:
            # Expect no split in unsafe cases; consider pass if matrix is unsafe under our rules
            prefix, matrix = peel_prefix(nf)
            only_universals = len(prefix) > 0 and all(k == "all" for k, _ in prefix)
            only_existentials = len(prefix) > 0 and all(k == "exists" for k, _ in prefix)
            no_q = len(prefix) == 0
            unsafe = (isinstance(matrix, OrExpression) and not (no_q or only_existentials)) or \
                     (isinstance(matrix, AndExpression) and not (no_q or only_universals))
            if unsafe:
                print(f"[PASS] no-split (unsafe): {s}")
                passed += 1
            else:
                print(f"[FAIL] expected split but got None: {s}  => NF: {nf}")
        else:
            a2b, b2a = check_equiv(nf, split_expr)
            if a2b and b2a:
                print(f"[PASS] split-equivalent: {s}")
                passed += 1
            else:
                print(f"[FAIL] split not equivalent: {s}\n  NF:   {nf}\n  SPLT: {split_expr}\n  (a->b={a2b}, b->a={b2a})")

    print(f"\nSummary: {passed}/{total} passed")


if __name__ == "__main__":
    run()

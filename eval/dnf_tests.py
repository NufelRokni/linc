"""Strict tests for DNF conversion (Step 5) in fol_normalization.

For each example we:
- parse and reformat the input
- eliminate implications, convert to NNF, standardise-apart
- extract prefix and matrix (quantifier-free)
- convert the matrix to DNF with a chosen clause cap
- re-wrap the prefix and prove equivalence both directions
- when using a large cap (default), assert the matrix is structurally in DNF
"""
from linc.eval.tasks.fol_normalization import (
    _eliminate_implications,
    _nnf,
    _standardise_apart,
    _free_variables,
    _to_prenex,
    _to_dnf,
    _wrap_prefix,
)
from linc.eval.tasks.utils import read_expr, reformat_fol, prover
from nltk.sem.logic import (
    Expression,
    AndExpression,
    OrExpression,
    NegatedExpression,
    ApplicationExpression,
    EqualityExpression,
)


def is_literal(e: Expression) -> bool:
    # A literal is an atom (predicate/equality) or its negation directly on an atom
    if isinstance(e, NegatedExpression):
        t = e.term
        return isinstance(t, (ApplicationExpression, EqualityExpression))
    return isinstance(e, (ApplicationExpression, EqualityExpression))


def flatten_or(e: Expression):
    if isinstance(e, OrExpression):
        return flatten_or(e.first) + flatten_or(e.second)
    return [e]


def flatten_and(e: Expression):
    if isinstance(e, AndExpression):
        return flatten_and(e.first) + flatten_and(e.second)
    return [e]


def is_dnf(e: Expression) -> bool:
    # DNF: OR of ANDs of literals (degenerate cases allowed: a single AND or a single literal)
    if isinstance(e, OrExpression):
        return all(is_dnf(d) for d in flatten_or(e))
    if isinstance(e, AndExpression):
        return all(is_literal(c) for c in flatten_and(e))
    return is_literal(e)


EXAMPLES = [
    # Simple distribution over one conjunction
    "(P(x) | Q(x)) & (R(x) | S(x))",
    # Three disjuncts times two
    "(A(x) | B(x) | C(x)) & (D(x) | E(x))",
    # Nested OR on one side with external OR
    "(A(x) & (B(x) | C(x))) | D(x)",
    # With quantifiers (DNF only applies to matrix)
    "all x.((P(x) | Q(x)) & (R(x) | S(x)))",
    # With negations and multiple groups
    "(P(x) | -Q(x) | R(x)) & (S(x) | -T(x))",
    # With equality and functions
    "(x = F(y) | P(x)) & (Q(y) | -R(F(y)))",
    # Larger conjunction of disjunctions
    "(P1(x) | P2(x)) & (Q1(x) | Q2(x)) & (R1(x) | R2(x))",
]


def run(cap: int = 256):
    total = len(EXAMPLES)
    passed = 0
    for i, s in enumerate(EXAMPLES, 1):
        print(f"Example {i}/{total}: {s} (cap={cap})")
        try:
            s_fmt = reformat_fol(s)
            orig = read_expr(s_fmt)
        except Exception as e:
            print("  PARSE ERROR:", e)
            print("  input:", s)
            continue

        # Steps up to matrix
        nnf_input = _eliminate_implications(orig)
        nnf_expr = _nnf(nnf_input)
        std_expr = _standardise_apart(nnf_expr, used=_free_variables(nnf_expr))
        prefix, matrix = _to_prenex(std_expr)

        # DNF
        dnf = _to_dnf(matrix, clause_cap=cap)
        rebuilt = _wrap_prefix(prefix, dnf)

        # Equivalence both directions
        ok1 = prover.prove((-orig) | rebuilt, [])
        ok2 = prover.prove((-rebuilt) | orig, [])

        # Structural DNF only asserted when cap is large enough
        structural_ok = True if cap >= 256 else True  # skip strict structural check under small caps
        if cap >= 256:
            structural_ok = is_dnf(dnf)

        if ok1 and ok2 and structural_ok:
            print("  [PASS] equiv both ways", end="")
            if cap >= 256:
                print("; matrix is DNF")
            else:
                print("; cap small, structural check skipped")
            passed += 1
        else:
            print("  [FAIL]")
            print("    orig   :", orig)
            print("    matrix :", matrix)
            print("    dnf    :", dnf)
            print("    orig->rebuilt:", ok1, ", rebuilt->orig:", ok2, ", structural DNF:", structural_ok)

    print(f"Summary: {passed}/{total} passed (cap={cap})")


if __name__ == "__main__":
    # First run with default large cap to enforce structural DNF
    run(cap=256)
    # Then run with a tiny cap to exercise fallback behavior while still preserving equivalence
    run(cap=3)

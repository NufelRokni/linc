from typing import List, Tuple
from nltk.sem.logic import Expression

import nltk
from linc.eval.tasks.utils import read_expr
from linc.eval.tasks.fol_normalization import normal_form
from linc.eval.tasks.recursive_eval import recursive_evaluate


def check_depth_consistency(expr: Expression, max_depth: int = 3) -> Tuple[bool, List[str]]:
    """Return (ok, results_per_depth). ok=True if all depths 0..max_depth agree."""
    results = []
    for d in range(max_depth + 1):
        # Use a fresh prover per evaluation by calling through recursive_evaluate, which uses the module-level prover;
        # to avoid symbol arity reuse across unrelated test cases, we keep premises empty and isolate per run.
        res = recursive_evaluate([], expr, depth=d)
        results.append(res)
    ok = all(r == results[0] for r in results)
    return ok, results


def run():
    cases: List[str] = [
        # Safe OR under existentials
        "exists x.(P(x)) | exists y.(Q(y))",
        "exists x.(R(x) | S(x)) | exists y.(T(y) & U(y))",
    # Use Hp for binary predicate and Hf for function to avoid arity/type conflicts
    "exists x.exists y.((P(F(x),G(y)) & Hp(x,y)) | (Q(x) & R(Hf(y))))",

        # Safe AND under universals
        "all x.(P(x)) & all y.(Q(y))",
        "all x.(R(x) & S(x)) & all y.(T(y))",
        "all x.all y.((P(x) & Q(y)) & (R(x) & S(y)))",

        # Unsafe OR under universals – ensure we don't degrade with depth
        "all x.(P(x) | Q(x))",
        "all x.((P(x) & R) | (Q(x) & S))",

        # Unsafe AND under existentials – ensure we don't degrade with depth
        "exists x.(P(x) & Q(x))",
        "exists x.((P(x) & R) & (Q(x) & S))",

        # Mixed prefix – should not split
        "all x.(exists y.(R(x,y) | S(x,y)))",
        "exists x.(all y.(T(y,x) & U(x))) & V(x)",

        # Equality in matrix
        "exists x.(x = F(x) | G(x))",
    "all x.(x = x) & all y.(-(y = y) | Hu(y))",

        # Deeply nested Or/And structure
    "exists x.((P(x) & (Q(x) | R(x))) | (S(x) & (T(x) | U(x))))",
        "all x.(((A(x) & B(x)) & (C(x) | D(x))) & (E(x) | (F(x) & G(x))))",

        # Combined functions and multiple vars
    "exists x.exists y.exists z.((M(F(x),y) & N(y,Hf(z))) | (O(x) & P(G(y),z)))",
        "all x.all y.((M(x,y) & (N(x) | O(y))) & (P(x) | (Q(y) & R(x))))",
    ]

    max_depth = 3
    total = len(cases)
    passed = 0

    for s in cases:
        raw_expr = read_expr(s)
        nf = normal_form(raw_expr)
        ok, results = check_depth_consistency(nf, max_depth)
        if ok:
            print(f"[PASS] depth-consistent: {s} -> {results}")
            passed += 1
        else:
            print(f"[FAIL] depth-inconsistent: {s} -> {results}\n  NF: {nf}")

    print(f"\nSummary: {passed}/{total} depth-consistency cases passed (max_depth={max_depth})")


if __name__ == "__main__":
    run()

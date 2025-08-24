"""Strict tests for prenex conversion in fol_normalization.

These tests exercise deep nesting, multiple-quantifier alternations, and shadowing.
Each example is parsed, standardized (normal_form), then _to_prenex is used to
extract a prefix and matrix. We then wrap the prefix back and check equivalence
in both directions via the prover, and check idempotence when applying prenex again.
"""
from linc.eval.tasks.fol_normalization import (
    _eliminate_implications,
    _nnf,
    _standardise_apart,
    _free_variables,
    _to_prenex,
    _wrap_prefix,
)
from linc.eval.tasks.utils import read_expr, prover, reformat_fol

EXAMPLES = [
    # simple single quantifier (lowercase and compact for parser)
    "all x.(P(x))",
    # nested quantifiers shifting from inner to outer
    "all x.(exists y.(P(x,y) -> exists z.(Q(z,y))))",
    # alternating quantifiers with nesting
    "exists x.(all y.(exists z.(R(x,y) & S(z,y))))",
    # shadowing (should be standardized before prenex)
    "all x.(exists x.(P(x)) & Q(x))",
    # deep nesting with negations and conjunctions
    "all x.(exists y.((P(x) & -Q(y)) -> (exists z.(R(z) | S(x,z,y)))))",
    # multiple alternations and binary connectives
    "(exists x.(P(x))) & (all y.(Q(y)))",
    # equality inside quantifiers and functions (function symbol capitalized)
    "all x.(exists y.((x = F(y)) -> P(F(y))))",
    # complex nested mixture
    "all x.((exists y.(P(x,y) & exists z.(Q(z) -> R(x,z)))) | exists w.(S(w)))",
    # deeply nested alternation
    "exists a.(all b.(exists c.(all d.((A(a,b,c,d)) -> exists e.(B(e,a))))))",
]


def free_individual_vars(expr):
    # reuse read_expr's expression helpers if needed; but we only need to compare provability
    return set()  # placeholder; prover ensures semantic equivalence


def run():
    total = len(EXAMPLES)
    passed = 0
    for i, s in enumerate(EXAMPLES, 1):
        print(f"Example {i}/{total}: {s}")
        try:
            s_fmt = reformat_fol(s)
            orig = read_expr(s_fmt)
        except Exception as e:
            print("  PARSE ERROR:", e)
            print(s)
            continue

        # perform the normalization steps up to standardise-apart (we need the quantified form)
        nnf_input = _eliminate_implications(orig)
        nnf_expr = _nnf(nnf_input)
        free_vars = _free_variables(nnf_expr)
        std_expr = _standardise_apart(nnf_expr, used=free_vars)

        # extract prenex prefix and matrix and rebuild
        prefix, matrix = _to_prenex(std_expr)
        prenexed = _wrap_prefix(prefix, matrix)

        # checks: equivalence both directions by proving implications as tautologies
        try:
            ok1 = prover.prove(( -orig ) | prenexed, [])
        except Exception:
            ok1 = False
        try:
            ok2 = prover.prove(( -prenexed ) | orig, [])
        except Exception:
            ok2 = False

        # idempotence: applying prenex again should give same prefix/matrix shape
        pref2, mat2 = _to_prenex(prenexed)
        prenexed2 = _wrap_prefix(pref2, mat2)
        idem = str(prenexed) == str(prenexed2)

        if ok1 and ok2 and idem:
            print("  [PASS] equivalence both directions and idempotence")
            passed += 1
        else:
            print("  [FAIL]")
            print("    orig    :", orig)
            print("    prenex  :", prenexed)
            print("    provable orig->prenex:", ok1)
            print("    provable prenex->orig:", ok2)
            print("    idempotent:", idem)

    print(f"Summary: {passed}/{total} passed")


if __name__ == '__main__':
    run()

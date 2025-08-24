"""
Run normalization tests on curated FOL examples and report free/bound variables.

Usage:
  python3 -m linc.eval.examples_normal_form
or
  python3 /app/linc/eval/examples_normal_form.py
"""
from __future__ import annotations

from typing import List, Dict, Any, Set

from linc.eval.tasks.fol_normalization import normal_form
from linc.eval.tasks.utils import reformat_fol, read_expr
import nltk
from nltk.sem.logic import (
    Expression,
    AllExpression,
    ExistsExpression,
    AndExpression,
    OrExpression,
    ImpExpression,
    IffExpression,
    NegatedExpression,
    EqualityExpression,
    ApplicationExpression,
)


Examples: List[Dict[str, Any]] = [
    {
        "name": "free-only",
        "premises": [],
        "conclusion": "P(x)",
    },
    {
        "name": "all-binds-x",
        "premises": [],
        "conclusion": "all x.(P(x))",
    },
    {
        "name": "x-bound-y-free",
        "premises": [],
        "conclusion": "all x.(P(x,y))",
    },
    {
        "name": "exists-left-x-free-right",
        "premises": [],
        "conclusion": "exists x.(P(x)) & Q(x)",
    },
    {
        "name": "equality-with-function-and-digit",
        "premises": [],
        "conclusion": "(8 = pm(x)) & Monday(x)",
    },
    {
        "name": "implication-with-nested-exists",
        "premises": [],
        "conclusion": "all x.(P(x) -> exists y.(R(x,y)))",
    },
    {
        "name": "shadowing-and-free",
        "premises": [],
        "conclusion": "exists x.(P(x) & all x.(Q(x))) & R(x)",
    },
    {
        "name": "multi-quant-and-free-u",
        "premises": [],
        "conclusion": "all x.(exists y.(R(x,y) & exists z.(S(y,z)))) | T(u)",
    },
    {
        "name": "negated-universal",
        "premises": [],
        "conclusion": "-all x.(Turtle(x) | Squirrel(x))",
    },
    {
        "name": "const-equality",
        "premises": [],
        "conclusion": "(Ted = _Ted)",
    },
    {
        "name": "bound-and-unbound-in-conclusion",
        "premises": ["all y.(Loves(y,Rock))"],
        "conclusion": "exists y.(Loves(y,x) & Cute(Rock)) & Skittish(y)",  # y bound inside left conj, unbound on right
    },
    {
        "name": "deep-shadowing-mixed-branches",
        "premises": [],
        "conclusion": "all x.( (P(x) & exists x.(Q(x) & R(x,y))) | (exists y.(S(y) & all y.(T(y,x)))) ) & U(x)",
    },
    {
        "name": "multi-shadow-different-sides",
        "premises": [],
        "conclusion": "exists x.(P(x) & all y.(Q(y,x) -> exists x.(R(x,y))) & exists y.(S(x,y) & all x.(T(x)))) & V(y)",
    },
    {
        "name": "negated-nested-quantifiers-same-name",
        "premises": [],
        "conclusion": "-exists x.(P(x) & -all x.(Q(x) | exists x.(R(x))))",
    },
    {
        "name": "double-shadow-both-vars",
        "premises": [],
        "conclusion": "all x.(exists y.(all x.(exists y.(F(x,y) & G(y))) & H(x,y)))",
    },
    {
        "name": "equality-mix-and-implications",
        "premises": [],
    "conclusion": "all x.(exists x.(x = F(x)) & (G(x) = y)) & -(y = y)",
    },
    {
        "name": "functions-and-implication",
        "premises": [],
    "conclusion": "all x.(P(F(x), G(y)) & exists y.(Q(H(y), x))) | exists x.(R(x) & exists y.(S(x,y) -> T(y)))",
    },
    {
        "name": "quantifiers-on-both-sides",
        "premises": [],
        "conclusion": "(all x.(A(x) | exists x.(B(x)))) & (exists x.(C(x) & all x.(D(x))))",
    },
]


def collect_bound(expr: Expression, acc: Set[str] | None = None) -> Set[str]:
    if acc is None:
        acc = set()
    if isinstance(expr, (AllExpression, ExistsExpression)):
        acc.add(str(expr.variable))
        return collect_bound(expr.term, acc)
    if isinstance(expr, NegatedExpression):
        return collect_bound(expr.term, acc)
    if isinstance(expr, (AndExpression, OrExpression, ImpExpression, IffExpression, EqualityExpression)):
        collect_bound(expr.first, acc)
        collect_bound(expr.second, acc)
        return acc
    if isinstance(expr, ApplicationExpression):
        collect_bound(expr.function, acc)
        # NLTK versions expose either .args (tuple) or a unary .argument
        args = []
        if hasattr(expr, 'args') and expr.args is not None:
            args = list(expr.args)
        elif hasattr(expr, 'argument') and expr.argument is not None:
            args = [expr.argument]
        for a in args:
            if isinstance(a, Expression):
                collect_bound(a, acc)
        return acc
    return acc


def has_imp_or_iff(expr: Expression) -> bool:
    if isinstance(expr, (ImpExpression, IffExpression)):
        return True
    if isinstance(expr, (AndExpression, OrExpression, EqualityExpression)):
        return has_imp_or_iff(expr.first) or has_imp_or_iff(expr.second)
    if isinstance(expr, NegatedExpression):
        return has_imp_or_iff(expr.term)
    if isinstance(expr, (AllExpression, ExistsExpression)):
        return has_imp_or_iff(expr.term)
    if isinstance(expr, ApplicationExpression):
        if has_imp_or_iff(expr.function):
            return True
        args = []
        if hasattr(expr, 'args') and expr.args is not None:
            args = list(expr.args)
        elif hasattr(expr, 'argument') and expr.argument is not None:
            args = [expr.argument]
        return any(has_imp_or_iff(a) for a in args if isinstance(a, Expression))
    return False


def negations_on_literals(expr: Expression) -> bool:
    """Return True if every negation is directly on an atom (NNF property)."""
    if isinstance(expr, NegatedExpression):
        t = expr.term
        if isinstance(t, (AndExpression, OrExpression, AllExpression, ExistsExpression, ImpExpression, IffExpression)):
            return False
        return negations_on_literals(t)
    if isinstance(expr, (AndExpression, OrExpression, EqualityExpression)):
        return negations_on_literals(expr.first) and negations_on_literals(expr.second)
    if isinstance(expr, (AllExpression, ExistsExpression)):
        return negations_on_literals(expr.term)
    if isinstance(expr, ApplicationExpression):
        if not negations_on_literals(expr.function):
            return False
        args = []
        if hasattr(expr, 'args') and expr.args is not None:
            args = list(expr.args)
        elif hasattr(expr, 'argument') and expr.argument is not None:
            args = [expr.argument]
        return all(negations_on_literals(a) for a in args if isinstance(a, Expression))
    return True


def bound_names(expr: Expression) -> List[str]:
    names: List[str] = []
    if isinstance(expr, (AllExpression, ExistsExpression)):
        names.append(str(expr.variable))
        names.extend(bound_names(expr.term))
    elif isinstance(expr, (AndExpression, OrExpression, EqualityExpression)):
        names.extend(bound_names(expr.first))
        names.extend(bound_names(expr.second))
    elif isinstance(expr, NegatedExpression):
        names.extend(bound_names(expr.term))
    return names


def free_individual_vars(expr: Expression) -> Set[str]:
    """Collect free individual variable names, ignoring predicate/function symbols."""
    from nltk.sem import logic as lg
    free: Set[str] = set()

    def visit(e: Expression, bound: Set[str]) -> None:
        if isinstance(e, (AllExpression, ExistsExpression)):
            v = str(e.variable)
            visit(e.term, bound | {v})
            return
        if isinstance(e, NegatedExpression):
            visit(e.term, bound)
            return
        if isinstance(e, (AndExpression, OrExpression, ImpExpression, IffExpression, EqualityExpression)):
            visit(e.first, bound)
            visit(e.second, bound)
            return
        if isinstance(e, ApplicationExpression):
            # function and arguments
            visit(e.function, bound)
            args = []
            if hasattr(e, 'args') and e.args is not None:
                args = list(e.args)
            elif hasattr(e, 'argument') and e.argument is not None:
                args = [e.argument]
            for a in args:
                if isinstance(a, Expression):
                    visit(a, bound)
            return
        # Leaf cases
        if isinstance(e, lg.IndividualVariableExpression):
            name = e.variable.name
            if name not in bound:
                free.add(name)

    visit(expr, set())
    return free


def run() -> None:
    total = 0
    passed = 0
    prover = nltk.Prover9(10)
    for ex in Examples:
        total += 1
        name = ex["name"]
        conc_raw = ex["conclusion"]
        conc_fmt = reformat_fol(conc_raw)
        try:
            conc_expr = read_expr(conc_fmt)
        except Exception as e:
            print(f"[FAIL] {name}: parse error (conclusion) -> {e}")
            continue

        orig_free = sorted(map(str, conc_expr.free()))
        orig_bound = sorted(collect_bound(conc_expr))

        # Normalize
        nf_expr = normal_form(conc_expr)
        # Idempotence
        nf2_expr = normal_form(nf_expr)
        idem_ok = (str(nf2_expr) == str(nf_expr))

        # Invariants
        inv_ok = True
        issues: List[str] = []
        if has_imp_or_iff(nf_expr):
            inv_ok = False
            issues.append("NF still contains -> or <->")
        if not negations_on_literals(nf_expr):
            inv_ok = False
            issues.append("Negations not only on literals (NNF violated)")
        # Standardize-apart safety: bound names in NF must not collide with original free vars
        nf_bound_names = set(bound_names(nf_expr))
        if nf_bound_names & set(orig_free):
            inv_ok = False
            issues.append("Bound names in NF collide with original free variables")
        # Free individual variables preserved
        fi_orig = sorted(free_individual_vars(conc_expr))
        fi_nf = sorted(free_individual_vars(nf_expr))
        if fi_orig != fi_nf:
            inv_ok = False
            issues.append(f"Free individual variables changed: {fi_orig} -> {fi_nf}")
        # Idempotence violation
        if not idem_ok:
            inv_ok = False
            issues.append("Normalization not idempotent (NF != NF(NF))")
        # Optional: check duplicate bound names in NF (should be rare after standardization)
        bn = bound_names(nf_expr)
        if len(bn) != len(set(bn)):
            issues.append("Duplicate bound variable names detected in NF (possible shadowing)")

        # Equivalence checks: prove (orig -> NF) and (NF -> orig)
        eq_ok = True
        eq_issues: List[str] = []
        try:
            a_impl_b = (-conc_expr) | nf_expr
            b_impl_a = (-nf_expr) | conc_expr
            if not prover.prove(a_impl_b, []):
                eq_ok = False
                eq_issues.append("Could not prove orig -> NF")
            if not prover.prove(b_impl_a, []):
                eq_ok = False
                eq_issues.append("Could not prove NF -> orig")
        except Exception as e:
            eq_ok = False
            eq_issues.append(f"Equivalence check error: {e}")

        status = "PASS" if (inv_ok and eq_ok) else "FAIL"
        if inv_ok and eq_ok:
            passed += 1
        print(f"[{status}] {name}")
        print(f"  conclusion raw : {conc_raw}")
        print(f"  conclusion fmt : {conc_fmt}")
        print(f"  free(orig)     : {orig_free}")
        print(f"  bound(orig)    : {orig_bound}")
        print(f"  NF             : {nf_expr}")
        if issues:
            for it in issues:
                print(f"    - {it}")
        if eq_issues:
            for it in eq_issues:
                print(f"    - {it}")
        else:
            print(f"    - Equivalence: orig -> NF ✓, NF -> orig ✓")
        print("-")

    print(f"Summary: {passed}/{total} passed")


if __name__ == "__main__":
    run()

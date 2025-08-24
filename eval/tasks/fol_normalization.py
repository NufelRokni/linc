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
def _eliminate_implications(expr: Expression) -> Expression:
    """Recursively eliminate → and ↔ from the expression, using only ¬, ∧, ∨."""
    if isinstance(expr, IffExpression):
        # (A ↔ B) ≡ [(¬A ∨ B) ∧ (¬B ∨ A)]
        A = _eliminate_implications(expr.first)
        B = _eliminate_implications(expr.second)
        return (_eliminate_implications(((-A) | B)) & _eliminate_implications(((-B) | A)))
    if isinstance(expr, ImpExpression):
        # (A → B) ≡ (¬A ∨ B)
        A = _eliminate_implications(expr.first)
        B = _eliminate_implications(expr.second)
        return _eliminate_implications(((-A) | B))
    if isinstance(expr, AndExpression):
        return _eliminate_implications(expr.first) & _eliminate_implications(expr.second)
    if isinstance(expr, OrExpression):
        return _eliminate_implications(expr.first) | _eliminate_implications(expr.second)
    if isinstance(expr, NegatedExpression):
        return -(_eliminate_implications(expr.term))
    if isinstance(expr, AllExpression):
        return AllExpression(expr.variable, _eliminate_implications(expr.term))
    if isinstance(expr, ExistsExpression):
        return ExistsExpression(expr.variable, _eliminate_implications(expr.term))
    return expr  # Atomic (no implications to eliminate)

def _nnf(expr: Expression) -> Expression:
    if isinstance(expr, NegatedExpression):
        t = expr.term
        if isinstance(t, NegatedExpression):
            # ¬¬A ≡ A
            return _nnf(t.term)
        if isinstance(t, AndExpression):
            # ¬(A ∧ B) ≡ (¬A ∨ ¬B)
            return _nnf(-t.first) | _nnf(-t.second)
        if isinstance(t, OrExpression):
            # ¬(A ∨ B) ≡ (¬A ∧ ¬B)
            return _nnf(-t.first) & _nnf(-t.second)
        if isinstance(t, AllExpression):
            # ¬∀x.P ≡ ∃x. ¬P
            return ExistsExpression(t.variable, _nnf(-t.term))
        if isinstance(t, ExistsExpression):
            # ¬∃x.P ≡ ∀x. ¬P
            return AllExpression(t.variable, _nnf(-t.term))
        # ¬ applied to an atomic formula (literal) – leave as is.
        return NegatedExpression(_nnf(t))
    if isinstance(expr, AndExpression):
        return _nnf(expr.first) & _nnf(expr.second)
    if isinstance(expr, OrExpression):
        return _nnf(expr.first) | _nnf(expr.second)
    if isinstance(expr, AllExpression):
        return AllExpression(expr.variable, _nnf(expr.term))
    if isinstance(expr, ExistsExpression):
        return ExistsExpression(expr.variable, _nnf(expr.term))
    return expr  # Atomic literal (no negation to push)

def _free_variables(expr: Expression, bound=None) -> set:
    """
    Gather the names of all free (unbound) individual variables in the expression.
    Robust across NLTK versions for ApplicationExpression (.args or .argument).
    """
    if bound is None:
        bound = set()

    # Quantifiers
    if isinstance(expr, (AllExpression, ExistsExpression)):
        v = expr.variable.name
        return _free_variables(expr.term, bound | {v})

    # Negation
    if isinstance(expr, NegatedExpression):
        return _free_variables(expr.term, bound)

    # Binary connectives (including equality)
    if isinstance(expr, (AndExpression, OrExpression, logic.EqualityExpression)):
        return _free_variables(expr.first, bound) | _free_variables(expr.second, bound)

    # ApplicationExpression: traverse functor and args
    if isinstance(expr, logic.ApplicationExpression):
        free = set()
        if isinstance(expr.function, logic.Expression):
            free |= _free_variables(expr.function, bound)
        args = []
        if hasattr(expr, 'args') and expr.args is not None:
            args = list(expr.args)
        elif hasattr(expr, 'argument') and expr.argument is not None:
            args = [expr.argument]
        for a in args:
            free |= _free_variables(a, bound)
        return free

    # Individual variable
    if isinstance(expr, logic.IndividualVariableExpression):
        name = expr.variable.name
        return {name} if name not in bound else set()

    # Other atoms/terms: no free individual variables contributed
    return set()

def _standardise_apart(expr: Expression, used: set = None) -> Expression:
    """
    α-convert bound variables so each quantifier has a unique name, distinct from all others 
    and from any free variables. This prevents unintended variable capture when pulling quantifiers out.
    `used` is the set of variable names already in use (bound in outer scopes or reserved as free).
    """
    if used is None:
        used = set()
    if isinstance(expr, AllExpression) or isinstance(expr, ExistsExpression):
        var = expr.variable  # an nltk.sem.logic.Variable
        vname = var.name
        if vname in used:
            # Choose a new variable name not used yet
            i = 0
            while f"{vname}_{i}" in used:
                i += 1
            new_name = f"{vname}_{i}"
            new_var = logic.Variable(new_name)
            # Replace occurrences of the old variable (bound by this quantifier) with the new variable in the body.
            # replace_bound=True confines replacement to the scope of this quantifier.
            body = expr.term.replace(var, logic.VariableExpression(new_var), 
                                     replace_bound=True, alpha_convert=False)
        else:
            # No conflict, reuse the same variable name
            new_var = var
            new_name = vname
            body = expr.term
        # Add the new variable name to the used set and recurse into the body
        used_next = used | {new_name}
        inner = _standardise_apart(body, used_next)
        return AllExpression(new_var, inner) if isinstance(expr, AllExpression) else ExistsExpression(new_var, inner)
    if isinstance(expr, AndExpression):
        # Standardize left, then right (carrying over any new bound names from left to avoid reuse in right)
        left_std = _standardise_apart(expr.first, used)
        used_after_left = _collect_bound_names(left_std) | used
        right_std = _standardise_apart(expr.second, used_after_left)
        return left_std & right_std
    if isinstance(expr, OrExpression):
        left_std = _standardise_apart(expr.first, used)
        used_after_left = _collect_bound_names(left_std) | used
        right_std = _standardise_apart(expr.second, used_after_left)
        return left_std | right_std
    if isinstance(expr, NegatedExpression):
        return NegatedExpression(_standardise_apart(expr.term, used))
    # Non-quantified atomic expression (predicate, function, or equality) – no change needed
    return expr

def _collect_bound_names(expr: Expression) -> set:
    """Helper for _standardise_apart: collect all variable names bound by quantifiers in expr."""
    names = set()
    if isinstance(expr, AllExpression) or isinstance(expr, ExistsExpression):
        names.add(expr.variable.name)
        names |= _collect_bound_names(expr.term)
    elif isinstance(expr, AndExpression) or isinstance(expr, OrExpression):
        names |= _collect_bound_names(expr.first)
        names |= _collect_bound_names(expr.second)
    elif isinstance(expr, NegatedExpression):
        names |= _collect_bound_names(expr.term)
    # Atomic: no bound variables to collect
    return names

def _to_prenex(expr: Expression):
    """
    Move all quantifiers to the front (prefix), returning (prefix_list, matrix).
    `prefix_list` is a list of tuples (type, Variable) with type "all" or "exists",
    in the **outermost-first** order.
    This step assumes bound variable names are unique (standardized apart).
    Returns (prefix_list, matrix_expr).
    """
    # Quantified formula: hoist the quantifier out and continue inside
    if isinstance(expr, AllExpression):
        pref, mat = _to_prenex(expr.term)
        return ([("all", expr.variable)] + pref, mat)
    if isinstance(expr, ExistsExpression):
        pref, mat = _to_prenex(expr.term)
        return ([("exists", expr.variable)] + pref, mat)
    # Binary connectives: get prenex form of both sides, then combine
    if isinstance(expr, AndExpression) or isinstance(expr, OrExpression):
        left_pref, left_mat = _to_prenex(expr.first)
        right_pref, right_mat = _to_prenex(expr.second)
        # Since all bound vars are distinct and not free in the other part (by standardization),
        # we can safely concatenate prefixes.
        prefix = left_pref + right_pref
        matrix = (left_mat & right_mat) if isinstance(expr, AndExpression) else (left_mat | right_mat)
        return (prefix, matrix)
    if isinstance(expr, NegatedExpression):
        # Negations in NNF only occur directly on atoms, so no quantifier to pull
        return ([], expr)
    # Atomic formula (or literal) with no quantifiers
    return ([], expr)

def _wrap_prefix(prefix: list, matrix: Expression) -> Expression:
    """Wrap the quantifier prefix around the matrix to rebuild a complete formula."""
    result = matrix
    # Note: prefix is in outermost-first order, so we apply it in reverse to rebuild inside-out.
    for quant, var in reversed(prefix):
        if quant == "all":
            result = AllExpression(var, result)
        else:  # "exists"
            result = ExistsExpression(var, result)
    return result

def log_difference(expr1, expr2, prefix=None):
    VERBOSE_NORMAL_FORM = True
    if VERBOSE_NORMAL_FORM and str(expr1) != str(expr2):
        print(f"Difference:\n  {expr1}\n  {expr2}")
        if prefix is not None:
            print(f"Prefix: {prefix}")

def normal_form(expr: Expression) -> Expression:
    # Step 1: eliminate -> and <-> 
    nnf_input = _eliminate_implications(expr)
    # log_difference(expr, nnf_input)
    
    #  Step 2: push negations to get NNF
    nnf_expr = _nnf(nnf_input)
    # log_difference(nnf_input, nnf_expr)

    # Step 3: standardize apart (rename) bound variables to avoid collisions
    #         (including with each other and with any free variables)
    free_vars = _free_variables(nnf_expr)
    std_expr = _standardise_apart(nnf_expr, used=free_vars)
    # log_difference(nnf_expr, std_expr)
    
    # Step 4: extract quantifier prefix and quantifier-free matrix (PNF) and rebuild
    prefix, matrix = _to_prenex(std_expr)
    # log_difference(std_expr, matrix, prefix)

    # Step 5: convert the matrix (quantifier-free) to DNF
    matrix_dnf = _to_dnf(matrix)
    # log_difference(matrix, matrix_dnf)
    
    # Recombine prefix and matrix
    return _wrap_prefix(prefix, matrix_dnf)









# ---- DNF on the quantifier-free matrix ----

def _flatten_or(expr: Expression) -> list:
    """Flatten nested disjunctions into a list of disjuncts."""
    if isinstance(expr, OrExpression):
        return _flatten_or(expr.first) + _flatten_or(expr.second)
    return [expr]


def _flatten_and(expr: Expression) -> list:
    """Flatten nested conjunctions into a list of conjuncts."""
    if isinstance(expr, AndExpression):
        return _flatten_and(expr.first) + _flatten_and(expr.second)
    return [expr]


def _join_or(expr_list: list) -> Expression:
    """Join a list of expressions into a single disjunction chain."""
    if not expr_list:
        return logic.TRUE   # edge case: no disjuncts (should not happen in practice)
    result = expr_list[0]
    for disj in expr_list[1:]:
        result = result | disj
    return result


def _to_dnf(expr: Expression, clause_cap: int = 256) -> Expression:
    """
    Distribute ∧ over ∨ in the quantifier-free expression to obtain DNF (OR-of-ANDs of literals).
    Note: `clause_cap` is accepted for API compatibility but is not enforced; the function fully distributes.
    """
    if isinstance(expr, AndExpression):
        # Convert both sides to DNF, then distribute
        A = _to_dnf(expr.first, clause_cap)
        B = _to_dnf(expr.second, clause_cap)
        return _distribute_and(A, B, clause_cap)
    if isinstance(expr, OrExpression):
        # Convert both sides to DNF and flatten
        left_dnf = _to_dnf(expr.first, clause_cap)
        right_dnf = _to_dnf(expr.second, clause_cap)
        # Flatten any nested ORs, then OR together all parts
        disjuncts = _flatten_or(left_dnf) + _flatten_or(right_dnf)
        return _join_or(disjuncts)
    # At this point, expr is either a negated atom or an atomic literal (cannot distribute further)
    return expr

def _distribute_and(A: Expression, B: Expression, clause_cap: int) -> Expression:
    """Distribute A ∧ B over any disjunctions within, to help form DNF."""
    # If one side is a disjunction, distribute conjunction across each disjunct
    if isinstance(A, OrExpression):
        disjuncts = _flatten_or(A)
        return _join_or([_to_dnf(disj & B, clause_cap) for disj in disjuncts])
    if isinstance(B, OrExpression):
        disjuncts = _flatten_or(B)
        return _join_or([_to_dnf(A & disj, clause_cap) for disj in disjuncts])
    # Neither A nor B is an OR-expression: just rebuild the conjunction (both are already in DNF or literal form)
    return A & B

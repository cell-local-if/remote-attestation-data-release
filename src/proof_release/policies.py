"""Versioned release-policy rule trees.

A rule is a small JSON tree. Leaf nodes locate a value with a top-level
claim name or an object path and carry exactly one comparison key::

    {"claim": "<top-level claim name>", "equals": <scalar>}
    {"path": ["<segment>", ...], "equals": <scalar>}
    {"path": ["items", {"wildcard": True}, "id"], "equals": <scalar>}
    {"claim": "...", "in": [<scalar>, ...]}        # 1..32 unique scalars
    {"claim": "...", "contains": <scalar>}          # array has an equal element
    {"claim": "...", "contains_any": [<scalar>, ...]}  # array hits >=1 of 1..32
    {"claim": "...", "contains_all": [<scalar>, ...]}  # array covers all of 1..32
    {"claim": "...", "lt"|"lte"|"gt"|"gte": <finite JSON number>}
    {"claim": "...", "exists": true}
    {"path": [...],
     "in"|"contains"|"contains_any"|"contains_all"|"lt"|"lte"|"gt"|"gte"|"exists": ...}
    {"all": [rule, ...]}
    {"any": [rule, ...]}
    {"not": rule}

A path segment is either a string naming one object field or the exact
object ``{"wildcard": True}`` (no other keys, and the value must be the
JSON boolean ``true``). A wildcard segment expands every element of the
JSON array at that position; several wildcards expand left to right into
the full set of candidate paths. A leaf is true when any complete
candidate satisfies its comparison.

Scalars are JSON scalars (string, number, boolean or null). Rules mention
only claim *names* (or object paths) and expected scalar values — they
never contain raw evidence, nonces or claim material — so persisting
their canonical serialization is compatible with the no-raw-evidence
guarantee.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "InvalidRule",
    "validate_rule",
    "evaluate_rule",
    "explain_rule",
    "rule_structure",
    "rules_equal",
    "diff_rules",
    "canonical_rule_json",
    "MAX_RULE_DEPTH",
    "MAX_RULE_NODES",
    "MAX_PATH_SEGMENTS",
    "MAX_PATH_SEGMENT_LENGTH",
    "MAX_IN_ITEMS",
]

#: Defensive bounds so a submitted rule cannot exhaust the stack or the
#: evaluator regardless of nesting.
MAX_RULE_DEPTH = 32
MAX_RULE_NODES = 256

#: Bounds on declared object paths so an unbounded path can never enter
#: persisted state.
MAX_PATH_SEGMENTS = 8
MAX_PATH_SEGMENT_LENGTH = 128

#: Bound on the candidate set of an ``in``, ``contains_any`` or
#: ``contains_all`` leaf so an unbounded set can never enter persisted
#: state.
MAX_IN_ITEMS = 32

#: The comparison keys a leaf may carry exactly one of, next to its
#: ``claim``/``path`` locator.
_COMPARISON_KEYS = frozenset(
    {"equals", "in", "contains", "contains_any", "contains_all",
     "lt", "lte", "gt", "gte", "exists"}
)

#: Comparison keys that order the actual value against one JSON number.
_ORDER_KEYS = frozenset({"lt", "lte", "gt", "gte"})

#: The locator keys a leaf may carry exactly one of.
_LOCATOR_KEYS = frozenset({"claim", "path"})


class InvalidRule(ValueError):
    """Raised when a submitted rule tree is not one of the valid forms."""


def _is_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        # Reject NaN/Infinity: they are not valid JSON values and would not
        # round-trip through the persisted canonical serialization.
        return math.isfinite(value)
    return False


def _is_number(value: Any) -> bool:
    """True for a finite JSON number — booleans are never numbers."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _is_wildcard_segment(segment: Any) -> bool:
    """True only for the exact wildcard object ``{"wildcard": True}``.

    The object must carry the single key ``wildcard`` whose value is the
    JSON boolean ``true``: any other key, any other value type (including
    ``1``) or an empty object is a malformed segment, not a wildcard.
    """
    return (
        isinstance(segment, dict)
        and len(segment) == 1
        and segment.get("wildcard") is True
    )


def _validate_path(path: Any) -> None:
    """Validate the ``path`` sibling of a path leaf.

    It must be a non-empty list of at most :data:`MAX_PATH_SEGMENTS`
    segments; each segment is either a non-blank string no longer than
    :data:`MAX_PATH_SEGMENT_LENGTH` characters, or the exact wildcard
    object ``{"wildcard": True}``. A string segment names exactly one
    object field — dots embedded in a segment are literal characters of
    that single name and never split it, and the literal string ``"*"``
    is an ordinary field name. A wildcard segment expands the elements of
    the JSON array found at that position.
    """
    if not isinstance(path, list) or not path:
        raise InvalidRule("path must be a non-empty list of segments")
    if len(path) > MAX_PATH_SEGMENTS:
        raise InvalidRule(f"path may name at most {MAX_PATH_SEGMENTS} segments")
    for segment in path:
        if _is_wildcard_segment(segment):
            continue
        if not isinstance(segment, str) or not segment or not segment.strip():
            raise InvalidRule(
                "path segments must be non-empty strings or "
                '{"wildcard": true} objects'
            )
        if len(segment) > MAX_PATH_SEGMENT_LENGTH:
            raise InvalidRule(
                f"path segments may be at most {MAX_PATH_SEGMENT_LENGTH} "
                "characters"
            )


def _scalar_equals(actual: Any, expected: Any) -> bool:
    """Type-strict JSON scalar equality.

    JSON distinguishes ``true`` from ``1`` even though Python does not, and
    ``null`` only equals ``null``; numbers of either width compare by value.
    """
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, bool) or isinstance(expected, bool):
        return (
            isinstance(actual, bool)
            and isinstance(expected, bool)
            and actual == expected
        )
    if isinstance(actual, str) or isinstance(expected, str):
        return (
            isinstance(actual, str)
            and isinstance(expected, str)
            and actual == expected
        )
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    return False


def _validate_scalar_candidates(key: str, value: Any) -> None:
    """Validate a 1..32 list of unique scalars.

    Used by ``in``, ``contains_any`` and ``contains_all``. Candidates
    must not repeat under the same type-strict equality the evaluator
    applies: 1 and 1.0 repeat, 1 and true do not.
    """
    if not isinstance(value, list) or not value:
        raise InvalidRule(f"{key} must be a non-empty list of scalars")
    if len(value) > MAX_IN_ITEMS:
        raise InvalidRule(f"{key} may name at most {MAX_IN_ITEMS} candidates")
    for item in value:
        if not _is_scalar(item):
            raise InvalidRule(f"{key} candidates must be scalars")
    for index in range(len(value)):
        for earlier in value[:index]:
            if _scalar_equals(value[index], earlier):
                raise InvalidRule(f"{key} candidates must not repeat")


def _validate_comparison(key: str, value: Any) -> None:
    """Validate the comparison value of a leaf against its comparison key."""
    if key == "equals":
        if not _is_scalar(value):
            raise InvalidRule("equals must compare against a scalar")
        return
    if key == "contains":
        # Only a scalar may be sought in an array: the membership test is
        # the same type-strict scalar equality ``equals`` applies.
        if not _is_scalar(value):
            raise InvalidRule("contains must compare against a scalar")
        return
    if key in ("in", "contains_any", "contains_all"):
        _validate_scalar_candidates(key, value)
        return
    if key in _ORDER_KEYS:
        if not _is_number(value):
            raise InvalidRule(f"{key} must compare against a finite JSON number")
        return
    if key == "exists":
        if value is not True:
            raise InvalidRule("exists must be true")
        return
    raise InvalidRule(f"unknown comparison: {key}")


def validate_rule(rule: Any) -> dict[str, Any]:
    """Validate a rule tree, returning it unchanged on success.

    Raises :class:`InvalidRule` for anything that is not exactly one of the
    documented forms: unknown node keys, multiple/unknown keys on a node,
    missing siblings, non-scalar comparisons, malformed ``in``,
    ``contains_any`` or ``contains_all`` sets (empty, oversized,
    non-scalar or repeating), a
    non-scalar ``contains`` target, boolean or non-finite
    ordinal bounds, an ``exists`` value other than ``true``, empty
    ``all``/``any`` lists, malformed ``path`` siblings (including an
    object segment that is not exactly ``{"wildcard": True}``), or
    structures past the defensive size bounds.
    """
    nodes = 0

    def visit(node: Any, depth: int) -> None:
        nonlocal nodes
        if depth > MAX_RULE_DEPTH:
            raise InvalidRule("rule nesting too deep")
        if not isinstance(node, dict):
            raise InvalidRule("rule node must be an object")
        keys = set(node.keys())
        if not keys:
            raise InvalidRule("rule node must not be empty")
        nodes += 1
        if nodes > MAX_RULE_NODES:
            raise InvalidRule("rule too large")
        locators = keys & _LOCATOR_KEYS
        comparisons = keys & _COMPARISON_KEYS
        if len(keys) == 2 and len(locators) == 1 and len(comparisons) == 1:
            if "claim" in locators:
                claim = node["claim"]
                if not isinstance(claim, str) or not claim:
                    raise InvalidRule("claim must name a top-level claim")
            else:
                _validate_path(node["path"])
            (comparison,) = comparisons
            _validate_comparison(comparison, node[comparison])
            return
        if len(keys) != 1:
            raise InvalidRule("rule node must have exactly one form")
        (key,) = keys
        if key in ("all", "any"):
            children = node[key]
            if not isinstance(children, list) or not children:
                raise InvalidRule(f"{key} must be a non-empty list of rules")
            for child in children:
                visit(child, depth + 1)
            return
        if key == "not":
            visit(node[key], depth + 1)
            return
        raise InvalidRule(f"unknown rule form: {key}")

    visit(rule, 0)
    return rule


def canonical_rule_json(rule: dict[str, Any]) -> str:
    """Canonical JSON for a validated rule (sorted keys, compact separators)."""
    return json.dumps(rule, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_path_candidates(claims: Any, segments: list[Any]) -> list[Any]:
    """Expand a declared path into its complete candidate values.

    String segments descend through one same-named JSON object field and
    never match array positions (the literal string ``"*"`` is still a
    field name). A wildcard segment expands only when the current value is
    a JSON array, producing one candidate per element; several wildcards
    expand left to right into the Cartesian product of the arrays met.

    An empty array, a non-array value at a wildcard, a missing object
    field, or a scalar/array/non-object met while further descent is still
    required all yield no candidates for that branch. The candidates of a
    fully walked path may be scalars, nulls, arrays or objects.
    """
    candidates: list[Any] = [claims]
    for segment in segments:
        if _is_wildcard_segment(segment):
            candidates = [
                element
                for current in candidates
                if isinstance(current, list)
                for element in current
            ]
        else:
            candidates = [
                current[segment]
                for current in candidates
                if isinstance(current, dict) and segment in current
            ]
        if not candidates:
            break
    return candidates


def _compare_order(actual: Any, key: str, bound: Any) -> bool:
    """Order the actual value against one numeric bound, type-strictly.

    The actual value must be a JSON number like the bound: booleans are
    not numbers, and any other scalar type fails the comparison.
    """
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    if key == "lt":
        return actual < bound
    if key == "lte":
        return actual <= bound
    if key == "gt":
        return actual > bound
    return actual >= bound


def _array_covers_all(actual: Any, candidates: list[Any]) -> bool:
    """True when ``actual`` is a JSON array covering every candidate.

    Each candidate scalar must find at least one type-strictly equal
    element in ``actual`` under :func:`_scalar_equals`. Element order,
    extra elements and repeated elements are irrelevant, and the
    candidates need not be adjacent; non-scalar elements (objects or
    arrays) simply never equal a candidate.
    """
    if not isinstance(actual, list):
        return False
    return all(
        any(_scalar_equals(element, candidate) for element in actual)
        for candidate in candidates
    )


def evaluate_rule(rule: dict[str, Any], claims: dict[str, Any]) -> bool:
    """Evaluate a validated rule against top-level verified claims.

    A missing claim or object path simply fails its comparison (an absent
    key is *not* equal to an explicit ``null``), and non-object values
    encountered while descending a path block further descent. Set,
    equality and ordering comparisons all require the actual value and the
    target to be the same scalar type — booleans are not numbers and
    ``null`` only equals ``null``. ``contains`` requires the located value
    itself to be a JSON array holding at least one element equal to the
    expected scalar under that same type-strict equality; a missing value,
    a non-array (including a bare scalar) or an empty/non-matching array
    all fail. ``contains_any`` generalizes that membership test to a
    candidate set: the located value must be a JSON array and at least one
    of its elements must equal at least one candidate scalar under the
    same type-strict equality, with the same missing/non-array/empty
    failures. ``contains_all`` requires the located value to be a JSON
    array that covers the *whole* candidate set: every candidate scalar
    must equal at least one of its elements under that same type-strict
    equality; element order, extra elements and repeated elements do not
    matter, and the candidates need not be adjacent. A missing value, a
    non-array or an array missing any candidate all fail.
    ``exists`` checks whether the claim key
    or the full object path is readable at all; a ``null`` value there
    still exists. A path leaf carrying wildcard segments is true when
    *any* of its complete candidates satisfies the leaf comparison
    (``exists`` is true when at least one complete candidate is
    readable); zero candidates — an empty array, a non-array at a
    wildcard, a missing intermediate field, or a scalar/non-object where
    further descent is required — fail the leaf. Compound nodes keep
    their short-circuit semantics.
    """
    keys = set(rule.keys())
    locators = keys & _LOCATOR_KEYS
    if len(keys) == 2 and len(locators) == 1:
        (comparison,) = keys - locators
        if "claim" in locators:
            if isinstance(claims, dict) and rule["claim"] in claims:
                actuals = [claims[rule["claim"]]]
            else:
                actuals = []
        else:
            actuals = _read_path_candidates(claims, rule["path"])
        if comparison == "exists":
            return bool(actuals)
        if comparison == "equals":
            return any(
                _scalar_equals(actual, rule["equals"]) for actual in actuals
            )
        if comparison == "in":
            return any(
                _scalar_equals(actual, candidate)
                for actual in actuals
                for candidate in rule["in"]
            )
        if comparison == "contains":
            # Only an actual JSON array can contain the expected scalar;
            # elements that are objects, arrays or otherwise non-scalar
            # simply never compare equal to it.
            return any(
                isinstance(actual, list)
                and any(
                    _scalar_equals(element, rule["contains"])
                    for element in actual
                )
                for actual in actuals
            )
        if comparison == "contains_any":
            # Like ``contains`` against a candidate set: only an actual
            # JSON array can hit, and only its scalar elements can equal
            # a candidate — objects and arrays never do.
            return any(
                isinstance(actual, list)
                and any(
                    _scalar_equals(element, candidate)
                    for element in actual
                    for candidate in rule["contains_any"]
                )
                for actual in actuals
            )
        if comparison == "contains_all":
            # The located value must itself be a JSON array covering the
            # whole candidate set: each candidate needs at least one
            # type-strictly equal scalar element. Order, extra elements
            # and repeats are irrelevant, and non-scalar elements simply
            # never match a candidate.
            return any(
                _array_covers_all(actual, rule["contains_all"])
                for actual in actuals
            )
        if comparison in _ORDER_KEYS:
            return any(
                _compare_order(actual, comparison, rule[comparison])
                for actual in actuals
            )
        raise InvalidRule(f"unknown comparison: {comparison}")
    (key,) = keys
    if key == "all":
        return all(evaluate_rule(child, claims) for child in rule["all"])
    if key == "any":
        return any(evaluate_rule(child, claims) for child in rule["any"])
    if key == "not":
        return not evaluate_rule(rule["not"], claims)
    raise InvalidRule(f"unknown rule form: {key}")


def explain_rule(
    rule: dict[str, Any], claims: dict[str, Any]
) -> list[dict[str, Any]]:
    """Explain a validated rule's evaluation as a depth-first node list.

    Every node of the tree appears exactly once, parents before their
    children (pre-order), with ``node_index`` running continuously from 0
    in that traversal order. Each node carries::

        {"node_index": int, "rule_path": [int, ...],
         "node_type": "leaf"|"all"|"any"|"not", "outcome": bool}

    ``rule_path`` is the integer child-index path from the root: the root
    is ``[]``, the first child of an ``all``/``any`` node is ``[0]`` (the
    second child ``[1]`` and so on), and the single child of ``not`` is
    ``[0]``. Every node's ``outcome`` is produced by
    :func:`evaluate_rule` against the same verified claims, so the root's
    outcome always equals the decision's overall verdict and every
    compound node's outcome agrees with its children's outcomes under the
    documented all/any/not semantics. Unlike :func:`evaluate_rule`'s
    short-circuit walk, every child is evaluated independently so the
    explanation always covers the complete tree.

    The explanation contains only node positions, structural types and
    booleans: it never records claim names, object paths, comparison
    operators, expected scalars, actual claim values or any evidence.
    """
    nodes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], path: list[int]) -> None:
        keys = set(node.keys())
        locators = keys & _LOCATOR_KEYS
        index = len(nodes)
        if len(keys) == 2 and len(locators) == 1:
            node_type = "leaf"
            children: list[tuple[Any, list[int]]] = []
        else:
            (key,) = keys
            node_type = key
            if key in ("all", "any"):
                children = [
                    (child, [*path, position])
                    for position, child in enumerate(node[key])
                ]
            else:
                children = [(node[key], [*path, 0])]
        nodes.append(
            {
                "node_index": index,
                "rule_path": list(path),
                "node_type": node_type,
                "outcome": evaluate_rule(node, claims),
            }
        )
        for child, child_path in children:
            visit(child, child_path)

    visit(rule, [])
    return nodes


def rule_structure(rule: dict[str, Any]) -> list[tuple[tuple[int, ...], str]]:
    """Return a validated rule tree's pre-order ``(path, node_type)`` shape.

    This is the explanation minus its outcomes: exactly one entry per node
    in the same depth-first order :func:`explain_rule` emits, with the
    same integer paths and ``leaf``/``all``/``any``/``not`` types. It
    contains no comparison data — only positions and structural types — so
    an explanation read back can be checked for completeness and shape
    against a policy version's immutable rule without touching claim names,
    paths-as-locators, expected scalars or actual values.
    """
    skeleton: list[tuple[tuple[int, ...], str]] = []

    def visit(node: dict[str, Any], path: list[int]) -> None:
        keys = set(node.keys())
        locators = keys & _LOCATOR_KEYS
        if len(keys) == 2 and len(locators) == 1:
            skeleton.append((tuple(path), "leaf"))
            return
        (key,) = keys
        skeleton.append((tuple(path), key))
        if key in ("all", "any"):
            for position, child in enumerate(node[key]):
                visit(child, [*path, position])
        else:
            visit(node[key], [*path, 0])

    visit(rule, [])
    return skeleton


def _json_values_equal(left: Any, right: Any) -> bool:
    """Structural JSON equality under the evaluator's scalar semantics.

    Object key order is insignificant and arrays compare element-wise by
    position. Scalars use the same type-strict equality
    :func:`_scalar_equals` defines: numbers of either width compare by
    value (``1`` equals ``1.0``) while booleans are never numbers and
    ``null`` only equals ``null``.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
        return False
    return _scalar_equals(left, right)


def rules_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when two validated rule trees are structurally equal.

    Only the rule trees participate: policy metadata (name, version,
    status, timestamps) never does. Equality is
    :func:`_json_values_equal` over the two trees, so object key order
    and JSON whitespace in the persisted serialization are insignificant,
    arrays compare by position, strings, booleans and null are strictly
    distinct, and numbers follow the evaluator's numeric equality.
    """
    return _json_values_equal(left, right)


def _node_form(node: dict[str, Any]) -> str:
    """The structural form of a validated rule node.

    One of ``leaf``, ``all``, ``any`` or ``not`` — the same classification
    :func:`explain_rule` and :func:`rule_structure` emit as ``node_type``.
    """
    keys = set(node.keys())
    if len(keys) == 2 and keys & _LOCATOR_KEYS:
        return "leaf"
    (key,) = keys
    return key


def diff_rules(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    """Diff two validated rule trees as a stable list of node changes.

    The result is empty exactly when :func:`rules_equal` holds. Every
    entry carries, in order::

        {"change": "added"|"removed"|"changed",
         "rule_path": [int, ...], "left": <node|null>, "right": <node|null>}

    ``rule_path`` is the integer child-index path :func:`explain_rule`
    defines. ``added``/``removed`` name a complete node only the
    right/left tree carries (the absent side is ``null``); ``changed``
    names two unequal nodes at the same path. Two nodes of different
    forms, or two unequal leaves, record a single ``changed`` at their
    path without descending. Two unequal ``all``/``any``/``not`` nodes
    with equal child counts descend into only the first differing child.
    An ``all``/``any`` pair whose child counts differ records one
    ``changed`` at its own path plus one ``added``/``removed`` per
    trailing child of the longer list. Entries are stably sorted by
    ``(rule_path, change)``.
    """
    changes: list[dict[str, Any]] = []

    def visit(
        left_node: dict[str, Any], right_node: dict[str, Any], path: list[int]
    ) -> None:
        if _json_values_equal(left_node, right_node):
            return
        left_form = _node_form(left_node)
        right_form = _node_form(right_node)
        if left_form != right_form or left_form == "leaf":
            changes.append(
                {
                    "change": "changed",
                    "rule_path": list(path),
                    "left": left_node,
                    "right": right_node,
                }
            )
            return
        if left_form == "not":
            visit(left_node["not"], right_node["not"], [*path, 0])
            return
        # all/any under the same keyword.
        key = left_form
        left_children = left_node[key]
        right_children = right_node[key]
        if len(left_children) == len(right_children):
            # Same form and child count but still unequal: record only
            # the first differing child.
            for position in range(len(left_children)):
                if not _json_values_equal(
                    left_children[position], right_children[position]
                ):
                    visit(
                        left_children[position],
                        right_children[position],
                        [*path, position],
                    )
                    break
            return
        # Child counts differ: one changed entry at this path, plus one
        # added/removed entry per trailing child of the longer list.
        changes.append(
            {
                "change": "changed",
                "rule_path": list(path),
                "left": left_node,
                "right": right_node,
            }
        )
        common = min(len(left_children), len(right_children))
        for position in range(common, len(right_children)):
            changes.append(
                {
                    "change": "added",
                    "rule_path": [*path, position],
                    "left": None,
                    "right": right_children[position],
                }
            )
        for position in range(common, len(left_children)):
            changes.append(
                {
                    "change": "removed",
                    "rule_path": [*path, position],
                    "left": left_children[position],
                    "right": None,
                }
            )

    visit(left, right, [])
    changes.sort(key=lambda entry: (entry["rule_path"], entry["change"]))
    return changes

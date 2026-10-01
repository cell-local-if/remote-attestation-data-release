"""Versioned release-policy rule trees.

A rule is a small JSON tree. Leaf nodes locate a value with a top-level
claim name or an object path and carry exactly one comparison key::

    {"claim": "<top-level claim name>", "equals": <scalar>}
    {"path": ["<segment>", ...], "equals": <scalar>}
    {"claim": "...", "in": [<scalar>, ...]}        # 1..32 unique scalars
    {"claim": "...", "lt"|"lte"|"gt"|"gte": <finite JSON number>}
    {"claim": "...", "exists": true}
    {"path": [...], "in"|"lt"|"lte"|"gt"|"gte"|"exists": ...}
    {"all": [rule, ...]}
    {"any": [rule, ...]}
    {"not": rule}

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
    "evaluate_rule_explained",
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

#: Bound on the candidate set of an ``in`` leaf so an unbounded set can
#: never enter persisted state.
MAX_IN_ITEMS = 32

#: The comparison keys a leaf may carry exactly one of, next to its
#: ``claim``/``path`` locator.
_COMPARISON_KEYS = frozenset({"equals", "in", "lt", "lte", "gt", "gte", "exists"})

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


def _validate_path(path: Any) -> None:
    """Validate the ``path`` sibling of a path leaf.

    It must be a non-empty list of at most :data:`MAX_PATH_SEGMENTS`
    segments; each segment is a non-blank string no longer than
    :data:`MAX_PATH_SEGMENT_LENGTH` characters. A segment names exactly
    one object field — dots embedded in a segment are literal characters
    of that single name and never split it.
    """
    if not isinstance(path, list) or not path:
        raise InvalidRule("path must be a non-empty list of segments")
    if len(path) > MAX_PATH_SEGMENTS:
        raise InvalidRule(f"path may name at most {MAX_PATH_SEGMENTS} segments")
    for segment in path:
        if not isinstance(segment, str) or not segment or not segment.strip():
            raise InvalidRule("path segments must be non-empty strings")
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


def _validate_comparison(key: str, value: Any) -> None:
    """Validate the comparison value of a leaf against its comparison key."""
    if key == "equals":
        if not _is_scalar(value):
            raise InvalidRule("equals must compare against a scalar")
        return
    if key == "in":
        if not isinstance(value, list) or not value:
            raise InvalidRule("in must be a non-empty list of scalars")
        if len(value) > MAX_IN_ITEMS:
            raise InvalidRule(f"in may name at most {MAX_IN_ITEMS} candidates")
        for item in value:
            if not _is_scalar(item):
                raise InvalidRule("in candidates must be scalars")
        # Candidates must not repeat under the same type-strict equality
        # the evaluator applies: 1 and 1.0 repeat, 1 and true do not.
        for index in range(len(value)):
            for earlier in value[:index]:
                if _scalar_equals(value[index], earlier):
                    raise InvalidRule("in candidates must not repeat")
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
    missing siblings, non-scalar comparisons, malformed ``in`` sets
    (empty, oversized, non-scalar or repeating), boolean or non-finite
    ordinal bounds, an ``exists`` value other than ``true``, empty
    ``all``/``any`` lists, malformed ``path`` siblings, or structures past
    the defensive size bounds.
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


_MISSING = object()


def _read_path(claims: Any, segments: list[str]) -> Any:
    """Read a declared path one object field at a time.

    Only JSON object fields are descended through: encountering an array
    or scalar before the last segment, or a missing field at any level,
    means the path is absent. Array elements never expand and indexing is
    not supported.
    """
    current = claims
    for segment in segments:
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


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


def _evaluate_leaf(rule: dict[str, Any], claims: dict[str, Any]) -> bool:
    """Evaluate one leaf node (locator plus exactly one comparison)."""
    (comparison,) = set(rule.keys()) - _LOCATOR_KEYS
    if "claim" in rule:
        if isinstance(claims, dict) and rule["claim"] in claims:
            actual = claims[rule["claim"]]
        else:
            actual = _MISSING
    else:
        actual = _read_path(claims, rule["path"])
    if comparison == "exists":
        return actual is not _MISSING
    if actual is _MISSING:
        return False
    if comparison == "equals":
        return _scalar_equals(actual, rule["equals"])
    if comparison == "in":
        return any(
            _scalar_equals(actual, candidate) for candidate in rule["in"]
        )
    if comparison in _ORDER_KEYS:
        return _compare_order(actual, comparison, rule[comparison])
    raise InvalidRule(f"unknown comparison: {comparison}")


def evaluate_rule(rule: dict[str, Any], claims: dict[str, Any]) -> bool:
    """Evaluate a validated rule against top-level verified claims.

    A missing claim or object path simply fails its comparison (an absent
    key is *not* equal to an explicit ``null``), and non-object values
    encountered while descending a path block further descent. Set,
    equality and ordering comparisons all require the actual value and the
    target to be the same scalar type — booleans are not numbers and
    ``null`` only equals ``null``. ``exists`` checks whether the claim key
    or the full object path is readable at all; a ``null`` value there
    still exists. Compound nodes keep their short-circuit semantics.
    """
    keys = set(rule.keys())
    locators = keys & _LOCATOR_KEYS
    if len(keys) == 2 and len(locators) == 1:
        return _evaluate_leaf(rule, claims)
    (key,) = keys
    if key == "all":
        return all(evaluate_rule(child, claims) for child in rule["all"])
    if key == "any":
        return any(evaluate_rule(child, claims) for child in rule["any"])
    if key == "not":
        return not evaluate_rule(rule["not"], claims)
    raise InvalidRule(f"unknown rule form: {key}")


def evaluate_rule_explained(
    rule: dict[str, Any], claims: dict[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate a validated rule and explain every node's outcome.

    Returns ``(root_outcome, nodes)`` where ``nodes`` lists every node of
    the rule tree in depth-first pre-order (parent before children,
    children in declared order). Each entry carries exactly::

        {"node_index": int, "rule_path": [int, ...],
         "node_type": "leaf"|"all"|"any"|"not", "outcome": bool}

    ``node_index`` numbers the nodes consecutively from 0 (the root);
    ``rule_path`` is the list of child positions from the root — ``[]``
    for the root, ``[0]`` for the first child of an ``all``/``any`` and
    for a ``not`` child's only subnode. The root outcome is the rule's
    boolean result, identical to :func:`evaluate_rule`.

    Unlike :func:`evaluate_rule`, compound nodes do not short-circuit:
    every child is evaluated so every node has a recorded outcome. The
    result is the same — leaf comparisons are total functions with no
    side effects, so skipped branches never change the root outcome.
    The explanation records only node positions, kinds and booleans —
    never claim names, comparison values or actual claim values.
    """
    nodes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], path: list[int]) -> bool:
        record: dict[str, Any] = {
            "node_index": len(nodes),
            "rule_path": list(path),
            "node_type": "",
            "outcome": False,
        }
        nodes.append(record)
        keys = set(node.keys())
        if len(keys) == 2 and keys & _LOCATOR_KEYS:
            record["node_type"] = "leaf"
            record["outcome"] = _evaluate_leaf(node, claims)
            return record["outcome"]
        (key,) = keys
        record["node_type"] = key
        if key in ("all", "any"):
            child_outcomes = [
                visit(child, path + [position])
                for position, child in enumerate(node[key])
            ]
            record["outcome"] = (
                all(child_outcomes) if key == "all" else any(child_outcomes)
            )
            return record["outcome"]
        if key == "not":
            record["outcome"] = not visit(node["not"], path + [0])
            return record["outcome"]
        raise InvalidRule(f"unknown rule form: {key}")

    root_outcome = visit(rule, [])
    return root_outcome, nodes

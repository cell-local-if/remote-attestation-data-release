"""Versioned release-policy rule trees.

A rule is a small JSON tree with exactly five node forms::

    {"claim": "<top-level claim name>", "equals": <scalar>}
    {"path": ["<segment>", ...], "equals": <scalar>}
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
    "canonical_rule_json",
    "MAX_RULE_DEPTH",
    "MAX_RULE_NODES",
    "MAX_PATH_SEGMENTS",
    "MAX_PATH_SEGMENT_LENGTH",
]

#: Defensive bounds so a submitted rule cannot exhaust the stack or the
#: evaluator regardless of nesting.
MAX_RULE_DEPTH = 32
MAX_RULE_NODES = 256

#: Bounds on declared object paths so an unbounded path can never enter
#: persisted state.
MAX_PATH_SEGMENTS = 8
MAX_PATH_SEGMENT_LENGTH = 128


class InvalidRule(ValueError):
    """Raised when a submitted rule tree is not one of the five forms."""


def _is_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        # Reject NaN/Infinity: they are not valid JSON values and would not
        # round-trip through the persisted canonical serialization.
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


def validate_rule(rule: Any) -> dict[str, Any]:
    """Validate a rule tree, returning it unchanged on success.

    Raises :class:`InvalidRule` for anything that is not exactly one of the
    five documented forms: unknown node keys, multiple/unknown keys on a
    node, missing siblings, non-scalar comparisons, empty ``all``/``any``
    lists, malformed ``path`` siblings, or structures past the defensive
    size bounds.
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
        if keys == {"claim", "equals"}:
            claim = node["claim"]
            if not isinstance(claim, str) or not claim:
                raise InvalidRule("claim must name a top-level claim")
            if not _is_scalar(node["equals"]):
                raise InvalidRule("equals must compare against a scalar")
            return
        if keys == {"path", "equals"}:
            _validate_path(node["path"])
            if not _is_scalar(node["equals"]):
                raise InvalidRule("equals must compare against a scalar")
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


def evaluate_rule(rule: dict[str, Any], claims: dict[str, Any]) -> bool:
    """Evaluate a validated rule against top-level verified claims.

    A missing claim or object path simply fails its comparison (an absent
    key is *not* equal to an explicit ``null``), and non-object values
    encountered while descending a path block further descent. Compound
    nodes keep their short-circuit semantics.
    """
    keys = set(rule.keys())
    if keys == {"claim", "equals"}:
        if not isinstance(claims, dict) or rule["claim"] not in claims:
            return False
        return _scalar_equals(claims[rule["claim"]], rule["equals"])
    if keys == {"path", "equals"}:
        actual = _read_path(claims, rule["path"])
        if actual is _MISSING:
            return False
        return _scalar_equals(actual, rule["equals"])
    (key,) = keys
    if key == "all":
        return all(evaluate_rule(child, claims) for child in rule["all"])
    if key == "any":
        return any(evaluate_rule(child, claims) for child in rule["any"])
    if key == "not":
        return not evaluate_rule(rule["not"], claims)
    raise InvalidRule(f"unknown rule form: {key}")

"""Versioned release-policy rule trees.

A rule is a small JSON tree with exactly five node forms::

    {"claim": "<top-level claim name>", "equals": <scalar>}
    {"path": ["<object field>", ...], "equals": <scalar>}
    {"all": [rule, ...]}
    {"any": [rule, ...]}
    {"not": rule}

Scalars are JSON scalars (string, number, boolean or null). Rules mention
only claim *names* and expected scalar values — they never contain raw
evidence, nonces or claim material — so persisting their canonical
serialization is compatible with the no-raw-evidence guarantee.

A ``path`` leaf descends the verified claims one object field per segment:
``["tenant", "region"]`` reads ``claims["tenant"]["region"]``. Segments are
literal field names — a dot inside a segment is part of that name, never a
separator — array elements are not expanded, and a missing intermediate
object (or any non-object value encountered while descending) is simply a
miss, never an error.
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

#: Bounds on declared path depth so an unbounded claim path can never be
#: persisted.
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


def validate_rule(rule: Any) -> dict[str, Any]:
    """Validate a rule tree, returning it unchanged on success.

    Raises :class:`InvalidRule` for anything that is not exactly one of the
    five documented forms: unknown node keys, multiple/unknown keys on a
    node, missing siblings, non-scalar comparisons, empty ``all``/``any``
    lists, malformed ``path`` declarations, or structures past the
    defensive size bounds.
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
            segments = node["path"]
            if not isinstance(segments, list):
                raise InvalidRule("path must be a non-empty list of segments")
            if not segments:
                raise InvalidRule("path must contain at least one segment")
            if len(segments) > MAX_PATH_SEGMENTS:
                raise InvalidRule(
                    f"path may contain at most {MAX_PATH_SEGMENTS} segments"
                )
            for segment in segments:
                if not isinstance(segment, str):
                    raise InvalidRule("path segments must be strings")
                if not segment or not segment.strip():
                    raise InvalidRule("path segments must be non-empty strings")
                if len(segment) > MAX_PATH_SEGMENT_LENGTH:
                    raise InvalidRule(
                        "path segments must be at most "
                        f"{MAX_PATH_SEGMENT_LENGTH} characters"
                    )
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


def _read_path(claims: Any, segments: list[str]) -> tuple[bool, Any]:
    """Read ``segments`` out of ``claims`` descending objects only.

    Returns ``(found, value)``. Each segment reads one JSON object field;
    a missing key or a current value that is not an object (an array, a
    scalar or null) means the path does not resolve — arrays are never
    indexed or expanded. Segment names are literal, so embedded dots are
    ordinary name characters.
    """
    current = claims
    for segment in segments:
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def evaluate_rule(rule: dict[str, Any], claims: dict[str, Any]) -> bool:
    """Evaluate a validated rule against top-level verified claims.

    A missing claim or path simply fails its comparison (an absent key is
    *not* equal to an explicit ``null``), and non-object claims never
    match. Descent beyond an array or scalar is likewise a miss rather
    than an error.
    """
    keys = set(rule.keys())
    if keys == {"claim", "equals"}:
        if not isinstance(claims, dict) or rule["claim"] not in claims:
            return False
        return _scalar_equals(claims[rule["claim"]], rule["equals"])
    if keys == {"path", "equals"}:
        found, value = _read_path(claims, rule["path"])
        if not found:
            return False
        return _scalar_equals(value, rule["equals"])
    (key,) = keys
    if key == "all":
        return all(evaluate_rule(child, claims) for child in rule["all"])
    if key == "any":
        return any(evaluate_rule(child, claims) for child in rule["any"])
    if key == "not":
        return not evaluate_rule(rule["not"], claims)
    raise InvalidRule(f"unknown rule form: {key}")

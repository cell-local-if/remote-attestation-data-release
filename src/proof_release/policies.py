"""Versioned release-policy rule trees.

A rule is a small JSON tree. A leaf node locates one value with either a
top-level ``claim`` name or an object ``path`` and carries exactly one
comparison key::

    {"claim": "<top-level claim name>", "equals": <scalar>}
    {"claim": "<top-level claim name>", "in": [<scalar>, ...]}
    {"claim": "<top-level claim name>", "lt": <number>}
    {"claim": "<top-level claim name>", "lte": <number>}
    {"claim": "<top-level claim name>", "gt": <number>}
    {"claim": "<top-level claim name>", "gte": <number>}
    {"claim": "<top-level claim name>", "exists": true}
    {"path": ["<segment>", ...], "equals": <scalar>}
    {"path": ["<segment>", ...], "in": [<scalar>, ...]}
    {"path": ["<segment>", ...], "lt"|"lte"|"gt"|"gte": <number>}
    {"path": ["<segment>", ...], "exists": true}

Compound nodes nest leaves and other compounds::

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
    "MAX_IN_CANDIDATES",
]

#: Defensive bounds so a submitted rule cannot exhaust the stack or the
#: evaluator regardless of nesting.
MAX_RULE_DEPTH = 32
MAX_RULE_NODES = 256

#: Bounds on declared object paths so an unbounded path can never enter
#: persisted state.
MAX_PATH_SEGMENTS = 8
MAX_PATH_SEGMENT_LENGTH = 128

#: Upper bound on the candidate set of an ``in`` leaf, so an unbounded
#: membership list can never enter persisted state or an evaluation.
MAX_IN_CANDIDATES = 32

#: Leaf locator keys: exactly one names where the actual value lives.
_LOCATOR_KEYS = frozenset({"claim", "path"})

#: Leaf comparison keys: exactly one says how the actual value is judged.
_COMPARISON_KEYS = frozenset(
    {"equals", "in", "lt", "lte", "gt", "gte", "exists"}
)
_ORDER_KEYS = frozenset({"lt", "lte", "gt", "gte"})


class InvalidRule(ValueError):
    """Raised when a submitted rule tree is not a documented form."""


def _is_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        # Reject NaN/Infinity: they are not valid JSON values and would not
        # round-trip through the persisted canonical serialization.
        return math.isfinite(value)
    return False


def _is_finite_number(value: Any) -> bool:
    """A JSON order operand: an int or finite float, never a boolean."""
    if isinstance(value, bool):
        # JSON booleans are their own scalar type; true never means 1.
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


def _validate_in_candidates(candidates: Any) -> None:
    """Validate the candidate set of an ``in`` leaf.

    It must be a non-empty list of at most :data:`MAX_IN_CANDIDATES`
    members; every member is a JSON scalar (string, finite number,
    boolean or null) and no two members may denote the same type-strict
    scalar value (``1`` and ``true`` differ, ``1`` and ``1.0`` do not).
    """
    if not isinstance(candidates, list) or not candidates:
        raise InvalidRule("in must compare against a non-empty list")
    if len(candidates) > MAX_IN_CANDIDATES:
        raise InvalidRule(
            f"in may list at most {MAX_IN_CANDIDATES} candidates"
        )
    for candidate in candidates:
        if not _is_scalar(candidate):
            raise InvalidRule(
                "in candidates must be strings, finite numbers, booleans "
                "or null"
            )
    # Type-strict duplicate detection using the same equality semantics
    # the evaluator uses, so a set that could never distinguish two
    # members is rejected as ambiguous rather than silently collapsed.
    for index, candidate in enumerate(candidates):
        for earlier in candidates[:index]:
            if _scalar_equals(candidate, earlier):
                raise InvalidRule("in candidates must not repeat")


def _validate_leaf(node: dict[str, Any], keys: set[str]) -> None:
    """Validate a leaf node: exactly one locator and one comparison key."""
    locators = keys & _LOCATOR_KEYS
    comparisons = keys & _COMPARISON_KEYS
    if len(locators) != 1 or len(comparisons) != 1 or len(keys) != 2:
        # Covers a missing locator/comparison, two locators, two
        # comparisons, a compound/leaf mix and any unknown sibling.
        raise InvalidRule("rule node must have exactly one form")
    (locator,) = locators
    if locator == "claim":
        claim = node["claim"]
        if not isinstance(claim, str) or not claim:
            raise InvalidRule("claim must name a top-level claim")
    else:
        _validate_path(node["path"])

    (comparison,) = comparisons
    value = node[comparison]
    if comparison == "equals":
        if not _is_scalar(value):
            raise InvalidRule("equals must compare against a scalar")
    elif comparison == "in":
        _validate_in_candidates(value)
    elif comparison in _ORDER_KEYS:
        if not _is_finite_number(value):
            raise InvalidRule(
                f"{comparison} must compare against a finite JSON number"
            )
    else:  # exists
        # The only accepted form is an explicit JSON true; false would be
        # a negated-existence spellable with `not` + `exists`, and any
        # other type is a malformed leaf.
        if value is not True:
            raise InvalidRule("exists must be true")


def validate_rule(rule: Any) -> dict[str, Any]:
    """Validate a rule tree, returning it unchanged on success.

    Raises :class:`InvalidRule` for anything that is not exactly one of
    the documented forms: unknown node keys, multiple/unknown keys on a
    node (including a missing or extra comparison key), non-scalar
    comparisons, an empty/over-long/duplicate-membered ``in`` set, an
    ordering bound that is a boolean or non-finite number, an ``exists``
    other than ``true``, empty ``all``/``any`` lists, malformed ``path``
    siblings, or structures past the defensive size bounds.
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
        if len(keys) == 1:
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
            if key in _LOCATOR_KEYS or key in _COMPARISON_KEYS:
                # A lone locator or comparison is a leaf with a missing
                # sibling; every other lone key is an unknown form.
                raise InvalidRule("rule node must have exactly one form")
            raise InvalidRule(f"unknown rule form: {key}")
        # Every multi-key node must be a well-formed leaf (one locator
        # plus one comparison). Compound nodes never carry siblings.
        _validate_leaf(node, keys)

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
    not supported. A complete path yields its terminal value even when
    that value is ``null``, an array or a scalar.
    """
    current = claims
    for segment in segments:
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def _locate(rule: dict[str, Any], claims: Any) -> Any:
    """Read the actual value a leaf refers to, or :data:`_MISSING`."""
    if "claim" in rule:
        if not isinstance(claims, dict) or rule["claim"] not in claims:
            return _MISSING
        return claims[rule["claim"]]
    if "path" in rule:
        return _read_path(claims, rule["path"])
    # Defensive: production rules always pass validate_rule first, so a
    # locator-less leaf can only come from a corrupted persisted tree.
    raise InvalidRule("rule leaf has no locator key")


def _finite_number(value: Any) -> Any:
    """Return ``value`` as a comparable number, else ``None``.

    Booleans are never numbers even though ``isinstance(True, int)`` in
    Python, and objects/arrays/strings/null do not order.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _evaluate_leaf(rule: dict[str, Any], claims: Any) -> bool:
    actual = _locate(rule, claims)
    if "equals" in rule:
        # An absent key is not equal to an explicit null, and a reached
        # object/array never equals a scalar.
        if actual is _MISSING:
            return False
        return _scalar_equals(actual, rule["equals"])
    if "in" in rule:
        # Membership is type-strict: the actual scalar must share the
        # candidate's scalar kind; a reached object/array never matches.
        if actual is _MISSING or not _is_scalar(actual):
            return False
        return any(_scalar_equals(actual, candidate) for candidate in rule["in"])
    if "exists" in rule:
        # Presence alone: a readable key or complete path exists, with a
        # null/array/scalar terminal value counting as present.
        return actual is not _MISSING
    for key in ("lt", "lte", "gt", "gte"):
        if key in rule:
            number = _finite_number(actual)
            if number is None:
                # Missing or wrong-kind actuals (including booleans and
                # explicit null) never satisfy an ordering bound.
                return False
            bound = rule[key]
            if key == "lt":
                return number < bound
            if key == "lte":
                return number <= bound
            if key == "gt":
                return number > bound
            return number >= bound
    raise InvalidRule("rule leaf has no comparison key")


def evaluate_rule(rule: dict[str, Any], claims: dict[str, Any]) -> bool:
    """Evaluate a validated rule against top-level verified claims.

    A missing claim or object path fails every comparison other than
    ``exists`` (an absent key is *not* equal to an explicit ``null`` and
    never falls inside a set or an ordering bound), and non-object values
    encountered while descending a path block further descent. Equality
    and set membership are type-strict (``true`` is not ``1``, numbers of
    either width compare by value, and ``null`` only matches ``null``);
    ordering comparisons require both sides to be JSON numbers. Compound
    nodes keep their short-circuit semantics.
    """
    keys = set(rule.keys())
    if len(keys) == 1:
        (key,) = keys
        if key == "all":
            return all(evaluate_rule(child, claims) for child in rule["all"])
        if key == "any":
            return any(evaluate_rule(child, claims) for child in rule["any"])
        if key == "not":
            return not evaluate_rule(rule["not"], claims)
    return _evaluate_leaf(rule, claims)

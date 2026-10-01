"""Unit tests for policy-rule evaluation explanations.

``explain_rule`` produces the depth-first node list persisted with each
decision and returned by GET /v1/decisions/{decision_id}/evaluation;
``rule_structure`` is its outcome-free skeleton used for the read-time
integrity check. These are pure-function tests against verified rule
trees: node ordering, integer paths, the four node types, complete (not
short-circuited) coverage and root-outcome agreement with
``evaluate_rule``.
"""

from __future__ import annotations

import pytest

from proof_release.policies import evaluate_rule, explain_rule, rule_structure


def test_single_leaf_explanation():
    rule = {"claim": "m", "equals": "x"}
    nodes = explain_rule(rule, {"m": "x"})
    assert nodes == [
        {"node_index": 0, "rule_path": [], "node_type": "leaf", "outcome": True}
    ]


def test_not_child_path_is_zero_and_outcome_is_inverted():
    rule = {"not": {"claim": "m", "exists": True}}
    assert explain_rule(rule, {"m": 1}) == [
        {"node_index": 0, "rule_path": [], "node_type": "not", "outcome": False},
        {"node_index": 1, "rule_path": [0], "node_type": "leaf", "outcome": True},
    ]
    assert explain_rule(rule, {}) == [
        {"node_index": 0, "rule_path": [], "node_type": "not", "outcome": True},
        {"node_index": 1, "rule_path": [0], "node_type": "leaf", "outcome": False},
    ]


def test_all_first_child_path_is_zero_and_order_is_depth_first():
    rule = {"all": [
        {"claim": "a", "equals": 1},
        {"claim": "b", "equals": 2},
    ]}
    nodes = explain_rule(rule, {"a": 1, "b": 2})
    assert [n["node_index"] for n in nodes] == [0, 1, 2]
    assert [n["rule_path"] for n in nodes] == [[], [0], [1]]
    assert [n["node_type"] for n in nodes] == ["all", "leaf", "leaf"]
    assert [n["outcome"] for n in nodes] == [True, True, True]


def test_explanation_covers_short_circuited_subtrees():
    # The first all-child is false, so evaluate_rule short-circuits, but
    # the explanation still evaluates the later subtree in full.
    rule = {"all": [
        {"claim": "a", "equals": 1},
        {"any": [
            {"claim": "b", "equals": 2},
            {"not": {"claim": "c", "equals": 3}},
        ]},
    ]}
    claims = {"a": 0, "b": 9, "c": 3}
    assert evaluate_rule(rule, claims) is False
    nodes = explain_rule(rule, claims)
    assert [n["rule_path"] for n in nodes] == [
        [], [0], [1], [1, 0], [1, 1], [1, 1, 0]
    ]
    assert [n["node_type"] for n in nodes] == [
        "all", "leaf", "any", "leaf", "not", "leaf"
    ]
    assert [n["outcome"] for n in nodes] == [
        False, False, False, False, False, True
    ]
    assert nodes[0]["outcome"] == evaluate_rule(rule, claims)


def test_node_indices_are_contiguous_from_zero():
    rule = {"not": {"all": [
        {"any": [{"claim": "a", "exists": True}, {"claim": "b", "exists": True}]},
        {"not": {"not": {"claim": "c", "exists": True}}},
    ]}}
    nodes = explain_rule(rule, {})
    assert [n["node_index"] for n in nodes] == list(range(len(nodes)))
    assert len(nodes) == 8


def test_structure_matches_explanation_minus_outcomes():
    rule = {"any": [
        {"all": [{"claim": "a", "equals": 1}, {"not": {"path": ["x"], "exists": True}}]},
        {"not": {"claim": "b", "in": [1, 2]}},
    ]}
    nodes = explain_rule(rule, {"a": 1, "x": None, "b": 2})
    skeleton = rule_structure(rule)
    assert len(skeleton) == len(nodes)
    for node, (path, node_type) in zip(nodes, skeleton):
        assert tuple(node["rule_path"]) == path
        assert node["node_type"] == node_type
    assert skeleton[0] == ((), "any")
    assert skeleton[1] == ((0,), "all")
    assert skeleton[-1] == ((1, 0), "leaf")


@pytest.mark.parametrize(
    "rule,claims",
    [
        ({"claim": "m", "equals": "x"}, {"m": "x"}),
        ({"claim": "m", "equals": "x"}, {"m": "y"}),
        ({"not": {"any": [
            {"claim": "x", "lt": 0}, {"claim": "x", "gt": 10},
        ]}}, {"x": 5}),
        ({"all": [
            {"any": [{"claim": "a", "equals": 1}, {"claim": "b", "equals": 2}]},
            {"not": {"claim": "c", "exists": True}},
        ]}, {"a": 1}),
        ({"not": {"not": {"not": {"claim": "z", "equals": None}}}}, {"z": None}),
    ],
)
def test_root_outcome_always_equals_evaluation(rule, claims):
    nodes = explain_rule(rule, claims)
    assert nodes[0]["rule_path"] == []
    assert nodes[0]["outcome"] == evaluate_rule(rule, claims)

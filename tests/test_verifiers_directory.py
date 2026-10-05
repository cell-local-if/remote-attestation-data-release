"""Tests for GET /v1/verifiers.

A read-only directory of the ``evidence_format`` names the verifier
registry currently accepts. The endpoint admits no query parameters and
no request body, never invokes a verifier, writes nothing, and returns
only the registered names — never plugin instances, class or module
names, configuration, secrets, certificates, evidence, or exception
text. A registry snapshot failure is a 500 with no partial list.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from proof_release.app import create_app
from proof_release.verifiers import (
    VerificationResult,
    Verifier,
    VerifierRegistry,
)

VERIFIERS_PATH = "/v1/verifiers"

BUILTIN_FORMATS = [
    "attested-nonce-json",
    "attested-nonce-json-v2",
    "x509-attested-nonce-json",
]


class _StubVerifier(Verifier):
    def __init__(self, format_name: str) -> None:
        self.format_name = format_name

    def verify(self, context) -> VerificationResult:  # pragma: no cover
        raise AssertionError("directory queries must never invoke verify")


@pytest.fixture()
def registry() -> VerifierRegistry:
    return VerifierRegistry()


@pytest.fixture()
def app(tmp_path, registry):
    return create_app(
        f"sqlite:///{tmp_path}/verifiers.db", verifier_registry=registry
    )


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)


def test_empty_registry_returns_empty_directory(client):
    response = client.get(VERIFIERS_PATH)
    assert response.status_code == 200
    assert response.text == '{"formats":[],"count":0}\n'
    assert response.headers["content-type"].startswith("application/json")


def test_default_registry_lists_builtin_formats(tmp_path):
    client = TestClient(create_app(f"sqlite:///{tmp_path}/default.db"))
    response = client.get(VERIFIERS_PATH)
    assert response.status_code == 200
    assert response.json() == {"formats": BUILTIN_FORMATS, "count": 3}


def test_field_order_is_formats_then_count(client, registry):
    registry.register(_StubVerifier("custom-format"))
    response = client.get(VERIFIERS_PATH)
    assert list(response.json()) == ["formats", "count"]
    assert response.text == '{"formats":["custom-format"],"count":1}\n'


def test_formats_sorted_by_unicode_code_point(client, registry):
    for name in ["zeta", "attested-nonce-json", "Éclair", "custom-Z", "custom-a"]:
        registry.register(_StubVerifier(name))
    response = client.get(VERIFIERS_PATH)
    formats = response.json()["formats"]
    assert formats == sorted(formats)
    assert formats == ["attested-nonce-json", "custom-Z", "custom-a", "zeta", "Éclair"]
    assert response.json()["count"] == len(formats)


def test_registration_replacement_and_removal_reflected_immediately(
    client, registry
):
    registry.register(_StubVerifier("alpha"))
    registry.register(_StubVerifier("beta"))
    assert client.get(VERIFIERS_PATH).json() == {
        "formats": ["alpha", "beta"],
        "count": 2,
    }

    # A same-name replacement appears exactly once.
    registry.register(_StubVerifier("alpha"))
    assert client.get(VERIFIERS_PATH).json() == {
        "formats": ["alpha", "beta"],
        "count": 2,
    }

    # An unregistered name disappears from the very next response.
    registry.unregister("alpha")
    assert client.get(VERIFIERS_PATH).json() == {
        "formats": ["beta"],
        "count": 1,
    }


def test_formats_have_no_duplicates(client, registry):
    for _ in range(3):
        registry.register(_StubVerifier("dup"))
    formats = client.get(VERIFIERS_PATH).json()["formats"]
    assert formats == ["dup"]


@pytest.mark.parametrize(
    "query",
    ["?format=x", "?format=x&format=y", "?=", "?format=", "?count=1"],
)
def test_any_query_parameter_is_rejected(client, registry, query):
    registry.register(_StubVerifier("secret-format"))
    response = client.get(VERIFIERS_PATH + query)
    assert response.status_code == 422
    # The failure precedes the registry read: no format existence leaks.
    assert "secret-format" not in response.text


def test_non_empty_body_is_rejected(client):
    response = client.request("GET", VERIFIERS_PATH, content=b"{}")
    assert response.status_code == 422
    response = client.request("GET", VERIFIERS_PATH, content=b" ")
    assert response.status_code == 422


def test_empty_body_is_accepted(client):
    response = client.request("GET", VERIFIERS_PATH, content=b"")
    assert response.status_code == 200


def test_verify_is_never_invoked(client, registry):
    registry.register(_StubVerifier("attested-nonce-json"))
    response = client.get(VERIFIERS_PATH)
    assert response.status_code == 200


def test_registry_failure_is_500_without_partial_list(tmp_path):
    class FailingRegistry(VerifierRegistry):
        def format_names(self):
            raise RuntimeError("registry internals must not leak")

    app = create_app(
        f"sqlite:///{tmp_path}/failing.db", verifier_registry=FailingRegistry()
    )
    response = TestClient(app).get(VERIFIERS_PATH)
    assert response.status_code == 500
    assert response.json() == {"detail": "verifier registry unavailable"}
    assert "registry internals" not in response.text

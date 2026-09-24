"""Envelope encryption for protected payloads.

Each payload is encrypted under a freshly generated 32-byte data key with
AES-256-GCM (random 96-bit IV, 128-bit tag); the data key is then wrapped
with the configured 32-byte master key using AES Key Wrap (RFC 3394). The
plaintext payload and the plaintext data key exist only on the stack of the
encrypting call and are never persisted, logged, or returned.

Master keys are versioned: a keyring configuration names the current
version used to wrap new envelopes and retains older versions so existing
envelopes can be unwrapped and re-wrapped (rotated) in place.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap, aes_key_wrap

#: Environment variable carrying the master key: unpadded base64url of
#: exactly 32 bytes (AES-256). Consulted only when KEYRING_ENV is unset.
MASTER_KEY_ENV = "PROOF_RELEASE_MASTER_KEY"

#: Environment variable carrying the master keyring as a JSON object:
#: ``{"current_version": <positive int>, "keys": {"<decimal positive int>":
#: "<unpadded base64url of exactly 32 bytes>", ...}}`` with
#: ``current_version`` present in ``keys``. When set it is the sole source
#: of master keys; MASTER_KEY_ENV is never read.
KEYRING_ENV = "PROOF_RELEASE_KEYRING"

MASTER_KEY_BYTES = 32
DATA_KEY_BYTES = 32
IV_BYTES = 12
TAG_BYTES = 16

#: Version of the master key / wrapping scheme used when only the legacy
#: single-key variable is configured. Stored on every envelope so key
#: rotation can select the right unwrapping key.
KEY_VERSION = 1

_B64URL_UNPADDED_RE = re.compile(r"^[A-Za-z0-9_-]*$")
_DECIMAL_RE = re.compile(r"^[0-9]+$")


class MasterKeyError(ValueError):
    """The configured master key is missing or malformed."""


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(value: str) -> bytes:
    """Decode strict unpadded base64url."""
    if not isinstance(value, str) or not _B64URL_UNPADDED_RE.fullmatch(value):
        raise ValueError("value is not unpadded base64url")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def load_master_key() -> bytes:
    """Return the configured 32-byte master key.

    Raises MasterKeyError if the environment variable is missing, is not
    strict unpadded base64url, or decodes to anything other than exactly
    32 bytes.
    """
    raw = os.environ.get(MASTER_KEY_ENV)
    if not raw:
        raise MasterKeyError("master key is not configured")
    try:
        key = b64url_decode(raw)
    except (ValueError, TypeError) as exc:
        raise MasterKeyError("master key is not unpadded base64url") from exc
    if len(key) != MASTER_KEY_BYTES:
        raise MasterKeyError("master key must decode to exactly 32 bytes")
    return key


@dataclass(frozen=True)
class KeyRing:
    """The configured master keys and the version used for new envelopes.

    ``keys`` maps each configured version to its 32-byte master key;
    ``current_version`` is guaranteed to be present. The mapping is
    immutable; key material is never logged or returned by the service.
    """

    current_version: int
    keys: Mapping[int, bytes]

    def current_key(self) -> bytes:
        return self.keys[self.current_version]

    def key_for(self, version: int) -> bytes:
        """Return the master key for ``version`` (any configured version)."""
        try:
            return self.keys[version]
        except KeyError:
            raise MasterKeyError(
                f"master key version {version} is not configured"
            ) from None


def _parse_keyring(raw: str) -> KeyRing:
    """Parse and validate the KEYRING_ENV JSON document.

    Raises MasterKeyError on any malformed shape; error messages describe
    only the failure kind, never the configured values.
    """
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MasterKeyError("keyring is not valid JSON") from exc
    if not isinstance(document, dict):
        raise MasterKeyError("keyring must be a JSON object")
    current = document.get("current_version")
    if isinstance(current, bool) or not isinstance(current, int) or current < 1:
        raise MasterKeyError("keyring current_version must be a positive integer")
    raw_keys = document.get("keys")
    if not isinstance(raw_keys, dict) or not raw_keys:
        raise MasterKeyError("keyring keys must be a non-empty object")
    keys: dict[int, bytes] = {}
    for name, value in raw_keys.items():
        if (
            not isinstance(name, str)
            or not _DECIMAL_RE.fullmatch(name)
            or int(name) < 1
            or str(int(name)) != name
        ):
            raise MasterKeyError(
                "keyring versions must be decimal positive integers"
            )
        version = int(name)
        if not isinstance(value, str):
            raise MasterKeyError("keyring keys must be unpadded base64url")
        try:
            key = b64url_decode(value)
        except (ValueError, TypeError) as exc:
            raise MasterKeyError("keyring key is not unpadded base64url") from exc
        if len(key) != MASTER_KEY_BYTES:
            raise MasterKeyError("keyring keys must decode to exactly 32 bytes")
        keys[version] = key
    if current not in keys:
        raise MasterKeyError("keyring current_version has no configured key")
    return KeyRing(current_version=current, keys=MappingProxyType(keys))


def load_keyring() -> KeyRing:
    """Return the configured master keyring.

    When PROOF_RELEASE_KEYRING is set it is the sole source of master keys
    and PROOF_RELEASE_MASTER_KEY is not read. When it is unset, the legacy
    PROOF_RELEASE_MASTER_KEY acts as the only key, at version 1. Raises
    MasterKeyError if the active configuration is missing or malformed.
    """
    raw = os.environ.get(KEYRING_ENV)
    if raw is not None:
        return _parse_keyring(raw)
    return KeyRing(
        current_version=KEY_VERSION, keys=MappingProxyType({KEY_VERSION: load_master_key()})
    )


@dataclass(frozen=True)
class EncryptedEnvelope:
    #: GCM ciphertext without the trailing tag.
    ciphertext: bytes
    #: 12-byte GCM nonce.
    iv: bytes
    #: 16-byte GCM authentication tag.
    tag: bytes
    #: AES-KW wrapped data key (40 bytes for a 32-byte data key).
    wrapped_key: bytes


def encrypt_payload(master_key: bytes, plaintext: bytes) -> EncryptedEnvelope:
    """Encrypt one payload under a fresh data key and wrap that key.

    Before returning, the produced material is self-checked by unwrapping
    the data key and authenticated-decrypting the ciphertext with the given
    master key, guaranteeing the persisted envelope is recoverable. The
    plaintext never leaves this call.
    """
    data_key = secrets.token_bytes(DATA_KEY_BYTES)
    iv = secrets.token_bytes(IV_BYTES)
    sealed = AESGCM(data_key).encrypt(iv, plaintext, None)
    ciphertext, tag = sealed[:-TAG_BYTES], sealed[-TAG_BYTES:]
    wrapped_key = aes_key_wrap(master_key, data_key)

    # Recoverability self-check: the stored material must unwrap with this
    # master key version and authenticate under the recovered data key.
    recovered_key = aes_key_unwrap(master_key, wrapped_key)
    AESGCM(recovered_key).decrypt(iv, sealed, None)

    return EncryptedEnvelope(
        ciphertext=ciphertext, iv=iv, tag=tag, wrapped_key=wrapped_key
    )


def decrypt_payload(
    master_key: bytes,
    wrapped_key: bytes,
    iv: bytes,
    ciphertext: bytes,
    tag: bytes,
) -> bytes:
    """Unwrap the data key with ``master_key`` and authenticated-decrypt.

    Used on the release path to recover a sealed payload. Both AES Key
    Wrap and AES-256-GCM are authenticated: a wrong master key, damaged
    wrapping material, or any tampering with the IV, tag or ciphertext
    raises before any plaintext is returned. The unwrapped data key
    exists only on this stack frame.
    """
    data_key = aes_key_unwrap(master_key, wrapped_key)
    return AESGCM(data_key).decrypt(iv, ciphertext + tag, None)


def rewrap_data_key(
    unwrapping_key: bytes, wrapping_key: bytes, wrapped_key: bytes
) -> bytes:
    """Re-wrap an AES-KW-wrapped data key from one master key to another.

    Only the wrapped form changes: the data key, and therefore the payload
    ciphertext, IV and tag, are untouched. The plaintext data key exists
    only on this stack frame. The result is self-checked by unwrapping
    under the new master key before returning, guaranteeing the rotated
    material is recoverable.
    """
    data_key = aes_key_unwrap(unwrapping_key, wrapped_key)
    rewrapped = aes_key_wrap(wrapping_key, data_key)
    if aes_key_unwrap(wrapping_key, rewrapped) != data_key:
        raise RuntimeError("rewrap self-check failed")
    return rewrapped

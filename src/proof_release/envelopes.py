"""Envelope encryption for protected payloads.

Each payload is encrypted under a freshly generated 32-byte data key with
AES-256-GCM (random 96-bit IV, 128-bit tag); the data key is then wrapped
with the configured 32-byte master key using AES Key Wrap (RFC 3394). The
plaintext payload and the plaintext data key exist only on the stack of the
encrypting call and are never persisted, logged, or returned.

Master keys are versioned. With only ``PROOF_RELEASE_MASTER_KEY`` set, that
single key is version 1. With ``PROOF_RELEASE_KEYRING`` set, a JSON object
``{"current_version": <positive int>, "keys": {"<decimal positive int>":
"<unpadded base64url 32-byte key>", ...}}`` supplies every known version and
selects the one used for new envelopes and rewraps; older versions remain
available so historical envelopes can still be unwrapped and rotated.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap, aes_key_wrap

#: Environment variable carrying the master key: unpadded base64url of
#: exactly 32 bytes (AES-256). Consulted only when PROOF_RELEASE_KEYRING is
#: not set; the single key then acts as version 1.
MASTER_KEY_ENV = "PROOF_RELEASE_MASTER_KEY"

#: Environment variable carrying the versioned master keyring as a JSON
#: object. When set, PROOF_RELEASE_MASTER_KEY is never read.
KEYRING_ENV = "PROOF_RELEASE_KEYRING"

MASTER_KEY_BYTES = 32
DATA_KEY_BYTES = 32
IV_BYTES = 12
TAG_BYTES = 16

#: Version of the master key / wrapping scheme used when only the legacy
#: single-key variable is configured. Stored on every envelope so key
#: rotation can select the right key.
KEY_VERSION = 1

_B64URL_UNPADDED_RE = re.compile(r"^[A-Za-z0-9_-]*$")

#: Canonical decimal rendering of a positive integer (no sign, no leading
#: zeros), as required for keyring version names.
_KEY_VERSION_RE = re.compile(r"^[1-9][0-9]*$")


class MasterKeyError(ValueError):
    """The configured master key or keyring is missing or malformed."""


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
class MasterKeyring:
    """Every known master key version and the version used for new wraps.

    ``keys`` maps each version to its 32-byte master key; historical
    versions stay available so envelopes written under them can still be
    unwrapped and rewrapped onto ``current_version``.
    """

    current_version: int
    keys: dict[int, bytes]

    def current_key(self) -> bytes:
        return self.keys[self.current_version]


def _decode_keyring_key(value: object) -> bytes:
    if not isinstance(value, str):
        raise MasterKeyError("keyring keys must be unpadded base64url strings")
    try:
        key = b64url_decode(value)
    except (ValueError, TypeError) as exc:
        raise MasterKeyError("keyring key is not unpadded base64url") from exc
    if len(key) != MASTER_KEY_BYTES:
        raise MasterKeyError("keyring keys must decode to exactly 32 bytes")
    return key


def load_keyring() -> MasterKeyring:
    """Return the configured master keyring.

    When PROOF_RELEASE_KEYRING is not set, the legacy single-key variable
    PROOF_RELEASE_MASTER_KEY supplies the sole key as version 1. When it is
    set, it must be a JSON object
    ``{"current_version": <positive int>, "keys": {...}}`` whose
    ``current_version`` is present in ``keys``; the legacy variable is then
    never read. Any malformed configuration raises MasterKeyError.
    """
    raw = os.environ.get(KEYRING_ENV)
    if raw is None:
        return MasterKeyring(
            current_version=KEY_VERSION, keys={KEY_VERSION: load_master_key()}
        )
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MasterKeyError("keyring is not valid JSON") from exc
    if not isinstance(document, dict):
        raise MasterKeyError("keyring must be a JSON object")
    current = document.get("current_version")
    if not isinstance(current, int) or isinstance(current, bool) or current < 1:
        raise MasterKeyError("keyring current_version must be a positive integer")
    raw_keys = document.get("keys")
    if not isinstance(raw_keys, dict):
        raise MasterKeyError("keyring keys must be a JSON object")
    keys: dict[int, bytes] = {}
    for name, value in raw_keys.items():
        if not isinstance(name, str) or not _KEY_VERSION_RE.fullmatch(name):
            raise MasterKeyError(
                "keyring key versions must be decimal positive integers"
            )
        keys[int(name)] = _decode_keyring_key(value)
    if current not in keys:
        raise MasterKeyError("keyring current_version is not present in keys")
    return MasterKeyring(current_version=current, keys=keys)


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


def rewrap_payload(
    old_master_key: bytes, new_master_key: bytes, wrapped_key: bytes
) -> bytes:
    """Re-wrap a data key from one master key version onto another.

    The wrapped data key is unwrapped with ``old_master_key`` (AES-KW
    authenticates the unwrap, so a wrong key fails here) and immediately
    re-wrapped with ``new_master_key``. The plaintext data key exists only
    on this stack frame and is never persisted, logged, or returned. The
    result is self-checked by unwrapping it with the new master key, so a
    failure raises before any durable state is touched.
    """
    data_key = aes_key_unwrap(old_master_key, wrapped_key)
    rewrapped = aes_key_wrap(new_master_key, data_key)

    # Recoverability self-check: the produced material must unwrap with the
    # new master key version back to the same data key.
    if aes_key_unwrap(new_master_key, rewrapped) != data_key:
        raise RuntimeError("rewrap self-check failed")
    return rewrapped

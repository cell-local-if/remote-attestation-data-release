"""Envelope encryption for protected payloads.

Each payload is encrypted under a freshly generated 32-byte data key with
AES-256-GCM (random 96-bit IV, 128-bit tag); the data key is then wrapped
with the configured 32-byte master key using AES Key Wrap (RFC 3394). The
plaintext payload and the plaintext data key exist only on the stack of the
encrypting call and are never persisted, logged, or returned.
"""

from __future__ import annotations

import base64
import os
import re
import secrets
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.keywrap import aes_key_unwrap, aes_key_wrap

#: Environment variable carrying the master key: unpadded base64url of
#: exactly 32 bytes (AES-256).
MASTER_KEY_ENV = "PROOF_RELEASE_MASTER_KEY"

MASTER_KEY_BYTES = 32
DATA_KEY_BYTES = 32
IV_BYTES = 12
TAG_BYTES = 16

#: Version of the master key / wrapping scheme currently in use. Stored on
#: every envelope so future key rotation can select the right key.
KEY_VERSION = 1

_B64URL_UNPADDED_RE = re.compile(r"^[A-Za-z0-9_-]*$")


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

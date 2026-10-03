"""Phase 15.1 — password hashing.

Stdlib-only (`hashlib.pbkdf2_hmac`), so a multi-user password store needs
no new dependency (`passlib` / `bcrypt`) beyond what the rest of this
codebase already uses. 200,000 iterations of SHA-256 follows OWASP's
current PBKDF2-HMAC-SHA256 minimum recommendation; a random 16-byte salt
per password means two users with the same password never produce the
same hash.
"""

from __future__ import annotations

import hashlib
import secrets

_PBKDF2_ITERATIONS = 200_000
_ALGORITHM = "sha256"


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return `(password_hash, salt)`, both hex-encoded. Generates a new
    random salt when `salt` is not given (registration); pass the stored
    salt back in to verify a login attempt."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        _ALGORITHM, password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return digest.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Constant-time comparison against a stored `(password_hash, salt)` pair."""
    candidate, _ = hash_password(password, salt)
    return secrets.compare_digest(candidate, password_hash)


__all__ = ["hash_password", "verify_password"]

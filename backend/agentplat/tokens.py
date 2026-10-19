"""Small HMAC-signed tokens (a minimal JWT-like format with one algorithm and no header).

Two uses:
  * session tokens: issued by the gateway's dev login, sent by the browser on the WebSocket URL;
  * caller context: minted by the gateway for each run and sent to the MCP server in the `_meta`
    of every MCP request. The MCP server trusts only the identity inside a valid signature, never
    anything the LLM wrote, so a model that hallucinates or is prompt-injected cannot act as
    another user.

The `aud` claim keeps one kind of token from being replayed as the other.
"""
import base64
import hashlib
import hmac
import json
import time

SESSION_AUD = "agentplat-session"
CALLER_AUD = "agentplat-mcp"


class TokenError(ValueError):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign(claims: dict, secret: str) -> str:
    body = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
    mac = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64(mac)}"


def verify(token: str, secret: str, *, aud: str, now: float | None = None) -> dict:
    if not isinstance(token, str) or token.count(".") != 1:
        raise TokenError("malformed token")
    body, mac = token.split(".")
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
    try:
        given = _unb64(mac)
    except ValueError as exc:
        raise TokenError("malformed signature") from exc
    if not hmac.compare_digest(expected, given):
        raise TokenError("bad signature")
    claims = json.loads(_unb64(body))
    if claims.get("aud") != aud:
        raise TokenError("wrong audience")
    if claims.get("exp", 0) < (time.time() if now is None else now):
        raise TokenError("expired")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise TokenError("missing subject")
    return claims


def session_token(user: str, secret: str, ttl_s: int = 12 * 3600) -> str:
    return sign({"sub": user, "aud": SESSION_AUD, "exp": int(time.time()) + ttl_s}, secret)


def caller_token(user: str, workspace: str, thread_id: str, secret: str, ttl_s: int = 900) -> str:
    return sign({"sub": user, "ws": workspace, "thread": thread_id, "aud": CALLER_AUD,
                 "exp": int(time.time()) + ttl_s}, secret)

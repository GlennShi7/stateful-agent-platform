import time

import pytest

from agentplat import tokens


def test_roundtrip_and_audience():
    t = tokens.caller_token("bob", "default", "th1", "s")
    claims = tokens.verify(t, "s", aud=tokens.CALLER_AUD)
    assert claims["sub"] == "bob" and claims["ws"] == "default" and claims["thread"] == "th1"
    with pytest.raises(tokens.TokenError, match="audience"):
        tokens.verify(t, "s", aud=tokens.SESSION_AUD)  # a caller token is not a session token


def test_tampering_and_wrong_secret_are_rejected():
    t = tokens.session_token("carol", "s")
    body, mac = t.split(".")
    forged = tokens.sign({"sub": "alice", "aud": tokens.SESSION_AUD, "exp": int(time.time()) + 60}, "other")
    with pytest.raises(tokens.TokenError):
        tokens.verify(forged, "s", aud=tokens.SESSION_AUD)
    with pytest.raises(tokens.TokenError):
        tokens.verify(body[:-2] + "xx." + mac, "s", aud=tokens.SESSION_AUD)
    with pytest.raises(tokens.TokenError):
        tokens.verify("garbage", "s", aud=tokens.SESSION_AUD)


def test_expiry():
    t = tokens.session_token("carol", "s", ttl_s=10)
    tokens.verify(t, "s", aud=tokens.SESSION_AUD)
    with pytest.raises(tokens.TokenError, match="expired"):
        tokens.verify(t, "s", aud=tokens.SESSION_AUD, now=time.time() + 11)

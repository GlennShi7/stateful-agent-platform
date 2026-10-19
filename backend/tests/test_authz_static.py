from agentplat.authz import PERMISSIONS, TOOL_PERMISSIONS, StaticAuthorizer


async def test_role_matrix():
    a = StaticAuthorizer()
    expect = {
        "alice": set(PERMISSIONS),
        "bob": {"view", "annotate", "manage_alerts"},
        "carol": {"view"},
        "dave": {"view", "manage_keys"},  # viewer plus a single extra capability
        "mallory": set(),
    }
    for user, perms in expect.items():
        got = await a.check_many(user, "default", list(PERMISSIONS))
        assert {p for p, ok in got.items() if ok} == perms, user


def test_every_tool_needs_a_known_permission():
    assert len(TOOL_PERMISSIONS) >= 15
    assert set(TOOL_PERMISSIONS.values()) <= set(PERMISSIONS)


async def test_grant_and_revoke():
    a = StaticAuthorizer()
    await a.grant("default", "erin", "operator")
    assert (await a.check_many("erin", "default", ["manage_alerts"]))["manage_alerts"]
    await a.revoke("default", "erin", "operator")
    assert not (await a.check_many("erin", "default", ["manage_alerts"]))["manage_alerts"]

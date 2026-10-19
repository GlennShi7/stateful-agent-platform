"""Settings, read once from environment variables."""
import os
from dataclasses import dataclass, field

# Demo accounts for the dev login. Their roles live in SpiceDB (see authz.DEMO_RELATIONSHIPS);
# this list only says who may log in.
DEV_USERS = ("alice", "bob", "carol", "dave")


def _int(name, default):
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    app_database_url: str = "postgresql://agent:agent@localhost:5433/agent"
    analytics_database_url: str = "postgresql://agent:agent@localhost:5433/agent"
    spicedb_endpoint: str = "localhost:50051"
    spicedb_key: str = "devkey"
    workspace: str = "default"
    # Shared between the gateway (signs caller context) and the MCP server (verifies it).
    caller_secret: str = "dev-caller-secret-change-me"
    # Signs browser session tokens issued by the dev login.
    session_secret: str = "dev-session-secret-change-me"
    mcp_url: str = "http://localhost:8001/mcp"
    llm_provider: str = "deepseek"  # "deepseek" or "fake"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    # deepseek-flash thinks by default. With tools enabled the API then requires every earlier
    # reasoning_content to be sent back, and thinking adds latency before the first visible token,
    # so it is off unless asked for.
    deepseek_thinking: str = "disabled"  # "disabled" or "enabled"
    deepseek_reasoning_effort: str = "low"
    max_iterations: int = 10
    dashboard_events_url: str = ""  # e.g. http://host.docker.internal:5000/events
    fake_first_token_ms: int = 30
    fake_token_delay_ms: int = 15
    fake_answer_tokens: int = 60
    dev_users: tuple = field(default=DEV_USERS)


def get_settings() -> Settings:
    e = os.environ.get
    app_db = e("APP_DATABASE_URL", Settings.app_database_url)
    return Settings(
        app_database_url=app_db,
        analytics_database_url=e("ANALYTICS_DATABASE_URL", app_db),
        spicedb_endpoint=e("SPICEDB_ENDPOINT", Settings.spicedb_endpoint),
        spicedb_key=e("SPICEDB_KEY", Settings.spicedb_key),
        workspace=e("WORKSPACE", Settings.workspace),
        caller_secret=e("CALLER_SECRET", Settings.caller_secret),
        session_secret=e("SESSION_SECRET", Settings.session_secret),
        mcp_url=e("MCP_URL", Settings.mcp_url),
        llm_provider=e("LLM_PROVIDER", Settings.llm_provider),
        deepseek_api_key=e("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=e("DEEPSEEK_BASE_URL", Settings.deepseek_base_url),
        deepseek_model=e("DEEPSEEK_MODEL", Settings.deepseek_model),
        deepseek_thinking=e("DEEPSEEK_THINKING", Settings.deepseek_thinking),
        deepseek_reasoning_effort=e("DEEPSEEK_REASONING_EFFORT", Settings.deepseek_reasoning_effort),
        max_iterations=_int("MAX_ITERATIONS", Settings.max_iterations),
        dashboard_events_url=e("DASHBOARD_EVENTS_URL", ""),
        fake_first_token_ms=_int("FAKE_FIRST_TOKEN_MS", Settings.fake_first_token_ms),
        fake_token_delay_ms=_int("FAKE_TOKEN_DELAY_MS", Settings.fake_token_delay_ms),
        fake_answer_tokens=_int("FAKE_ANSWER_TOKENS", Settings.fake_answer_tokens),
    )

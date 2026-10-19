"""LLM clients.

`DeepSeekLLM` speaks the OpenAI-compatible chat completions API over SSE with plain httpx, so the
streaming, tool-call reassembly and reasoning_content round trip are all visible here instead of
hidden in an SDK. `ScriptedLLM` is a deterministic stand-in with configurable timing, used by the
tests and by the load benchmark (which measures the platform, not the model provider).

Both return a `Turn`; message dicts use the OpenAI format throughout.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

OnToken = Callable[[str], Awaitable[None] | None]


@dataclass
class Turn:
    content: str = ""
    tool_calls: list = field(default_factory=list)  # [{"id", "type": "function", "function": {"name", "arguments"}}]
    reasoning: str | None = None
    ttft_ms: float | None = None
    usage: dict | None = None

    def as_message(self) -> dict:
        msg = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        return msg


class LLMError(RuntimeError):
    pass


async def _emit(on_token, text):
    if on_token is not None and text:
        r = on_token(text)
        if asyncio.iscoroutine(r):
            await r


class DeepSeekLLM:
    def __init__(self, api_key, base_url="https://api.deepseek.com", model="deepseek-flash",
                 thinking="disabled", reasoning_effort="low", timeout_s=120, transport=None, max_retries=2):
        if not api_key:
            raise LLMError("DEEPSEEK_API_KEY is not set")
        self.model, self.thinking, self.effort, self.max_retries = model, thinking, reasoning_effort, max_retries
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport,
                                       headers={"Authorization": f"Bearer {api_key}"})

    def payload(self, messages, tools):
        body = {"model": self.model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}, "thinking": {"type": self.thinking}}
        if self.thinking == "enabled":
            body["reasoning_effort"] = self.effort
        if tools:
            body["tools"] = tools
        return body

    async def complete(self, messages, tools, on_token: OnToken | None = None) -> Turn:
        for attempt in range(self.max_retries + 1):
            try:
                return await self._complete_once(messages, tools, on_token)
            except LLMError as exc:
                retryable = getattr(exc, "retryable", False)
                if not retryable or attempt == self.max_retries:
                    raise
                await asyncio.sleep(0.5 * 2 ** attempt)

    async def _complete_once(self, messages, tools, on_token):
        t0 = time.perf_counter()
        turn, calls = Turn(), {}
        content, reasoning = [], []
        async with self._http.stream("POST", "/chat/completions", json=self.payload(messages, tools)) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:500]
                err = LLMError(f"DeepSeek HTTP {resp.status_code}: {body}")
                err.retryable = resp.status_code in (429, 500, 502, 503, 504)
                raise err
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    turn.usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if delta.get("content"):
                        if turn.ttft_ms is None:
                            turn.ttft_ms = (time.perf_counter() - t0) * 1000
                        content.append(delta["content"])
                        await _emit(on_token, delta["content"])
                    # Tool calls arrive in fragments keyed by index: the first carries id and name,
                    # later ones append to the JSON arguments string.
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
        turn.content = "".join(content)
        turn.reasoning = "".join(reasoning) or None
        turn.tool_calls = [{"id": c["id"] or f"call_{i}", "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                           for i, c in sorted(calls.items())]
        if turn.ttft_ms is None:
            turn.ttft_ms = (time.perf_counter() - t0) * 1000
        return turn

    async def aclose(self):
        await self._http.aclose()


class ScriptedLLM:
    """Deterministic LLM. `script` is either a list of Turns returned in order, or a function
    (messages, tools) -> Turn. Content is streamed word by word with the configured delays."""

    def __init__(self, script, first_token_ms=0, token_delay_ms=0):
        self.script, self.first_token_ms, self.token_delay_ms = script, first_token_ms, token_delay_ms
        self.calls = 0
        self.seen = []

    async def complete(self, messages, tools, on_token: OnToken | None = None) -> Turn:
        self.calls += 1
        self.seen.append({"messages": list(messages), "tools": [t["function"]["name"] for t in tools or []]})
        if callable(self.script):
            planned = self.script(messages, tools)
        else:
            planned = self.script[min(self.calls - 1, len(self.script) - 1)]
        t0 = time.perf_counter()
        if self.first_token_ms:
            await asyncio.sleep(self.first_token_ms / 1000)
        words = planned.content.split(" ") if planned.content else []
        for i, w in enumerate(words):
            if i and self.token_delay_ms:
                await asyncio.sleep(self.token_delay_ms / 1000)
            await _emit(on_token, w if i == 0 else " " + w)
        return Turn(content=planned.content, tool_calls=list(planned.tool_calls), reasoning=planned.reasoning,
                    ttft_ms=(time.perf_counter() - t0) * 1000)


def tool_call(name, args, call_id=None):
    return {"id": call_id or f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def benchmark_script(answer_tokens=60):
    """Policy for the load benchmark: one real tool call through MCP, then a streamed answer."""
    answer = " ".join(f"tok{i}" for i in range(answer_tokens))

    def policy(messages, tools):
        names = {t["function"]["name"] for t in tools or []}
        if messages[-1]["role"] == "user" and "get_error_rate" in names:
            return Turn(tool_calls=[tool_call("get_error_rate", {"window_minutes": 60},
                                              call_id=f"call_{len(messages)}")])
        return Turn(content=answer)
    return policy


class MissingKeyLLM:
    """Lets the stack start without a key and explains the problem in the chat instead of crashing."""

    async def complete(self, messages, tools, on_token=None):
        raise LLMError("DEEPSEEK_API_KEY is not set. Put it in .env and restart (make up), "
                       "or run the scripted model with make up-fake.")


def make_llm(settings):
    if settings.llm_provider == "fake":
        return ScriptedLLM(benchmark_script(settings.fake_answer_tokens),
                           first_token_ms=settings.fake_first_token_ms, token_delay_ms=settings.fake_token_delay_ms)
    if not settings.deepseek_api_key:
        log.warning("DEEPSEEK_API_KEY not set; runs will fail with an explanation")
        return MissingKeyLLM()
    return DeepSeekLLM(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model,
                       settings.deepseek_thinking, settings.deepseek_reasoning_effort)

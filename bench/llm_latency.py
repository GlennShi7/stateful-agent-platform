"""Time-to-first-token and total latency of the real model (DeepSeek), measured directly, without
the platform. Compare with ws_load.py: that one measures what the platform adds.

Usage: DEEPSEEK_API_KEY=... python bench/llm_latency.py --n 10
"""
import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from agentplat.config import get_settings  # noqa: E402
from agentplat.llm import DeepSeekLLM  # noqa: E402


def pct(v, p):
    s = sorted(v)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    s = get_settings()
    llm = DeepSeekLLM(s.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", ""), s.deepseek_base_url,
                      s.deepseek_model, s.deepseek_thinking, s.deepseek_reasoning_effort)
    ttft, total = [], []
    for i in range(a.n):
        t0 = time.perf_counter()
        turn = await llm.complete([{"role": "user", "content": f"In one sentence, what is an error rate? ({i})"}], None)
        total.append((time.perf_counter() - t0) * 1000)
        ttft.append(turn.ttft_ms)
    await llm.aclose()
    out = {"model": s.deepseek_model, "thinking": s.deepseek_thinking, "n": a.n,
           "ttft_ms": {"p50": pct(ttft, 50), "p95": pct(ttft, 95)},
           "total_ms": {"p50": pct(total, 50), "p95": pct(total, 95)}}
    print(json.dumps(out, indent=2))
    path = pathlib.Path(__file__).parent / "results" / f"llm_latency_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

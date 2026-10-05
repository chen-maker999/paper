"""阿里云百炼（DashScope）DeepSeek V4 Flash 调用封装。

走百炼的 OpenAI 兼容接口，只用标准库，不引入额外依赖。
API key 依次读取环境变量 DASHSCOPE_API_KEY、OPENAI_API_KEY。

用法示例：
  python -m experiments.llm_client "你好"
  python -m experiments.llm_client --model deepseek-v4-flash --max-tokens 64 "回复 OK"
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MODEL = "deepseek-v4-flash"
# 限流和服务端错误时重试，其余错误（如 401、400）直接抛出
RETRY_STATUS = {429, 500, 502, 503, 504}


def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    return key


def chat_completion(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 120.0,
    retries: int = 3,
    **extra,
) -> dict:
    """调用 /chat/completions，返回完整的响应 JSON。"""
    body = {"model": model, "messages": messages, **extra}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code not in RETRY_STATUS or attempt == retries:
                raise RuntimeError(f"DashScope 返回 HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def chat(prompt: str, system: str | None = None, **kwargs) -> str:
    """单轮对话，只返回回复文本。"""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    resp = chat_completion(messages, **kwargs)
    return resp["choices"][0]["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("prompt")
    p.add_argument("--system")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)
    a = p.parse_args()
    resp = chat_completion(
        ([{"role": "system", "content": a.system}] if a.system else [])
        + [{"role": "user", "content": a.prompt}],
        model=a.model,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
    )
    print(resp["choices"][0]["message"]["content"])
    if "usage" in resp:
        print(f"[usage] {resp['usage']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from experiments import llm_client


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_chat_builds_request_and_parses_reply(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        reply = {"choices": [{"message": {"content": "OK"}}]}
        return _Resp(json.dumps(reply).encode())

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    assert llm_client.chat("hi", system="sys", max_tokens=5) == "OK"
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "deepseek-v4-flash"
    assert seen["body"]["max_tokens"] == 5
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


def test_non_retryable_http_error_raises(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="401"):
        llm_client.chat("hi")
    assert len(calls) == 1

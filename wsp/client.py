"""wsp/client.py: a streaming client for an OpenAI-compatible chat endpoint, standard library only.

It folds the server-sent events of one reply (text, reasoning, tool calls, usage) into a Reply, waits out dropped
connections and busy or throttled answers, and stops for good on a spent quota or a refused key."""
import http.client
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DROP_WAITS = (15, 30, 60, 120, 120)            # seconds to wait after a dropped connection, before an attempt counts
BUSY_WAITS = (20, 30, 45, 60, 60, 90, 120)      # seconds to wait while every slot of the account is busy
READ_TIMEOUT, DEADLINE = 180, 900               # one read on the socket, and one whole call


class QuotaExhausted(RuntimeError):
    """The account has no quota left for this model, so every later call would fail the same way."""


class AccessDenied(RuntimeError):
    """The key is wrong or not allowed to use this model."""


@dataclass
class Reply:
    model: str = ""
    text: str = ""
    reasoning: str = ""
    status: str = "ok"                         # ok, truncated, refusal or infra
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0                     # includes the reasoning tokens
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    attempts: int = 0
    error: str = ""
    tool_calls_by_index: dict = field(default_factory=dict)

    @property
    def tool_calls(self):
        return [self.tool_calls_by_index[k] for k in sorted(self.tool_calls_by_index)]


def fold_event(line, reply, t0):
    """Add one server-sent event line of a streamed reply to the Reply being built."""
    line = line.strip()
    if not line.startswith("data:") or line == "data: [DONE]":
        return
    chunk = json.loads(line[5:])
    reply.model = chunk.get("model") or reply.model
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        reply.reasoning += delta.get("reasoning_content") or ""
        reply.text += delta.get("content") or ""
        for tc in delta.get("tool_calls") or []:
            slot = reply.tool_calls_by_index.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
            slot["id"] = tc.get("id") or slot["id"]
            slot["name"] = (tc.get("function") or {}).get("name") or slot["name"]
            slot["arguments"] += (tc.get("function") or {}).get("arguments") or ""
        reply.stop_reason = choice.get("finish_reason") or reply.stop_reason
    usage = chunk.get("usage") or {}
    if usage:
        reply.input_tokens = usage.get("prompt_tokens", reply.input_tokens)
        reply.output_tokens = usage.get("completion_tokens", reply.output_tokens)
        reply.reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        reply.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def classify(status, body):
    """What to do after an HTTP error: retry, throttled, busy, quota, denied or rejected."""
    low = body.lower()
    if status == 429 and any(w in low for w in ("insufficient", "balance", "quota exhausted", "quota exceeded")):
        return "quota"
    if status == 429:
        return "busy" if "concurrency" in low else "throttled"
    if status in (401, 403):
        return "denied"
    if status in (400, 404, 413, 422):
        return "rejected"
    return "retry"


class ChatClient:
    """chat() streams one request and returns a Reply, retrying what is worth retrying."""

    def __init__(self, endpoint, key, model, max_attempts=4):
        self.endpoint, self._key, self.model, self.max_attempts = endpoint, key, model, max_attempts

    def _once(self, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data,
                                     headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        t0, reply = time.perf_counter(), Reply()
        with urllib.request.urlopen(req, timeout=max(READ_TIMEOUT, len(data) / 5000)) as resp:
            opened, pending = time.perf_counter(), b""
            while True:
                piece = resp.read1(65536)
                if not piece:
                    break
                if time.perf_counter() - opened > DEADLINE:
                    raise TimeoutError(f"call ran past its {DEADLINE}s deadline")
                *done, pending = (pending + piece).split(b"\n")
                for d in done:
                    fold_event(d.decode("utf-8"), reply, t0)
            if pending.strip():
                fold_event(pending.decode("utf-8"), reply, t0)
        if not reply.stop_reason:                   # a cut stream is never treated as the model's answer
            raise ConnectionError("stream ended without a finish_reason")
        reply.seconds = time.perf_counter() - t0
        reply.status = {"length": "truncated", "content_filter": "refusal"}.get(reply.stop_reason, "ok")
        return reply

    def chat(self, messages, effort=None, max_completion_tokens=None, extra=None):
        body = {"model": self.model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
        if effort:
            body["reasoning_effort"] = effort
        if max_completion_tokens:
            body["max_completion_tokens"] = max_completion_tokens
        body.update(extra or {})
        attempt, drops, busy_waits, last = 0, 0, 0, ""
        while attempt < self.max_attempts:
            attempt += 1
            dropped = busy = throttled = False
            try:
                reply = self._once(body)
                reply.attempts = attempt + drops
                return reply
            except urllib.error.HTTPError as e:
                text = e.read()[:600].decode("utf-8", "replace")
                action = classify(e.code, text)
                if action == "quota":
                    raise QuotaExhausted(text)
                if action == "denied":
                    raise AccessDenied(f"HTTP {e.code}: {text}")
                if action == "rejected":
                    return Reply(model=self.model, status="infra", attempts=attempt + drops, error=f"HTTP {e.code}: {text}")
                last, busy, throttled = f"HTTP {e.code}: {text[:200]}", action == "busy", action == "throttled"
            except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as e:
                last = f"{type(e).__name__}: {e}"
            except (urllib.error.URLError, ConnectionError, http.client.HTTPException, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                dropped = not isinstance(getattr(e, "reason", None), TimeoutError)
            if dropped and drops < len(DROP_WAITS):         # a dropped network uses up no attempt
                time.sleep(DROP_WAITS[drops] * (0.5 + random.random() / 2))
                drops, attempt = drops + 1, attempt - 1
            elif busy and busy_waits < len(BUSY_WAITS):     # neither does a busy account
                time.sleep(BUSY_WAITS[busy_waits] * (0.5 + random.random() / 2))
                busy_waits, attempt = busy_waits + 1, attempt - 1
            elif attempt < self.max_attempts:
                time.sleep(min(90, (10 if throttled else 2) * 2 ** (attempt - 1)) * (0.5 + random.random() / 2))
        return Reply(model=self.model, status="infra", attempts=attempt + drops, error=last or "stopped")

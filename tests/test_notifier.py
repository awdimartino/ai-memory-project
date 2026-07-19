"""Offline coverage for the phone-push notifier (self-hosted Bark).

Uses httpx's MockTransport to capture the request without any server, so we verify the
Bark payload shape, the disabled no-op, the optional tap-to-open url, and that a failing
push is swallowed (never breaks a reach-out). No network / LM Studio.

Run:  python tests/test_notifier.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from infrastructure.notifier import PhonePush


def _capturing():
    """Return (transport, captured_list); every request is recorded and answered 200."""
    captured = []
    def handler(request):
        captured.append(request)
        return httpx.Response(200)
    return httpx.MockTransport(handler), captured


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
async def noop_when_url_empty():
    transport, captured = _capturing()
    p = PhonePush("", "Mari", transport=transport)
    assert not p.enabled()
    await p.push("hi")
    assert captured == [], "disabled notifier must not POST"


@case
async def posts_bark_payload_with_url():
    transport, captured = _capturing()
    p = PhonePush("http://alex-pi:8090/devkey", "Mari",
                  ui_url="https://pc.tail.ts.net", transport=transport)
    assert p.enabled()
    await p.push("hey, no rush, was thinking about you")
    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "http://alex-pi:8090/devkey"
    assert req.method == "POST"
    body = json.loads(req.content)
    assert body["title"] == "Mari"
    assert body["body"] == "hey, no rush, was thinking about you"
    assert body["group"] == "Mari"
    assert body["url"] == "https://pc.tail.ts.net"


@case
async def omits_url_when_ui_unset():
    transport, captured = _capturing()
    p = PhonePush("http://alex-pi:8090/devkey", "Mari", transport=transport)
    await p.push("hi")
    body = json.loads(captured[0].content)
    assert "url" not in body, "no tap-to-open link when NOTIFY_UI_URL is unset"


@case
async def swallows_push_errors():
    def handler(request):
        raise httpx.ConnectError("pi unreachable")
    p = PhonePush("http://alex-pi:8090/devkey", "Mari", transport=httpx.MockTransport(handler))
    await p.push("hi")   # must NOT raise — a dead server can't break a reach-out
    assert True


async def main() -> int:
    failed = 0
    for fn in CASES:
        try:
            await fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

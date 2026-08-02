"""Offline coverage for the phone-push notifier (self-hosted Bark).

Uses httpx's MockTransport to capture the request without any server, so we verify the
Bark payload shape, the disabled no-op, the optional tap-to-open url, and that a failing
push is swallowed (never breaks a reach-out). No network / LM Studio.

Run:  python tests/test_notifier.py
"""
import json

from _harness import case, run  # also puts the repo root on sys.path

import httpx

from infrastructure.notifier import PhonePush


def _capturing():
    """Return (transport, captured_list); every request is recorded and answered 200."""
    captured = []
    def handler(request):
        captured.append(request)
        return httpx.Response(200)
    return httpx.MockTransport(handler), captured


@case
async def noop_when_url_empty():
    transport, captured = _capturing()
    p = PhonePush("", "the companion", transport=transport)
    assert not p.enabled()
    await p.push("hi")
    assert captured == [], "disabled notifier must not POST"


@case
async def posts_bark_payload_with_url():
    transport, captured = _capturing()
    p = PhonePush("http://bark-host:8090/devkey", "the companion",
                  ui_url="https://pc.tail.ts.net", transport=transport)
    assert p.enabled()
    await p.push("hey, no rush, was thinking about you")
    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "http://bark-host:8090/devkey"
    assert req.method == "POST"
    body = json.loads(req.content)
    assert body["title"] == "the companion"
    assert body["body"] == "hey, no rush, was thinking about you"
    assert body["group"] == "the companion"
    assert body["url"] == "https://pc.tail.ts.net"


@case
async def strips_trailing_slash():
    transport, captured = _capturing()
    p = PhonePush("http://bark-host:8090/devkey/", "the companion", transport=transport)
    await p.push("hi")
    assert str(captured[0].url) == "http://bark-host:8090/devkey", "trailing slash must be trimmed"


@case
async def omits_url_when_ui_unset():
    transport, captured = _capturing()
    p = PhonePush("http://bark-host:8090/devkey", "the companion", transport=transport)
    await p.push("hi")
    body = json.loads(captured[0].content)
    assert "url" not in body, "no tap-to-open link when NOTIFY_UI_URL is unset"
    assert "icon" not in body, "no icon when NOTIFY_ICON is unset"


@case
async def includes_icon_when_set():
    transport, captured = _capturing()
    p = PhonePush("http://bark-host:8090/devkey", "the companion",
                  icon="https://pc.tail.ts.net/static/mari.png", transport=transport)
    await p.push("hi")
    body = json.loads(captured[0].content)
    assert body["icon"] == "https://pc.tail.ts.net/static/mari.png"


@case
async def swallows_push_errors():
    def handler(request):
        raise httpx.ConnectError("pi unreachable")
    p = PhonePush("http://bark-host:8090/devkey", "the companion", transport=httpx.MockTransport(handler))
    # The assertion is that this returns at all — a dead Pi must not break a reach-out.
    delivered, detail = await p.push("hi")
    assert delivered is False, "a transport error is not a delivery"
    assert "ConnectError" in detail


# --- delivery is REPORTED, not assumed ---------------------------------------------
# httpx does not raise on 4xx/5xx. Without an explicit status check a wrong Bark device
# key (404) looked identical to a successful push, and /admin/test_notify said ok:True.

@case
async def reports_success_on_2xx():
    transport, _ = _capturing()
    p = PhonePush("http://bark-host:8090/devkey", "the companion", transport=transport)
    assert await p.push("hi") == (True, "ok")


@case
async def reports_failure_on_http_error():
    def handler(request):
        return httpx.Response(404, text="failed to get device token")
    p = PhonePush("http://bark-host:8090/badkey", "the companion", transport=httpx.MockTransport(handler))
    delivered, detail = await p.push("hi")
    assert delivered is False, "a 404 from Bark is a FAILED push, not a successful one"
    assert "404" in detail and "device token" in detail, detail


@case
async def disabled_push_is_not_a_delivery():
    p = PhonePush("", "the companion")
    assert await p.push("hi") == (False, "disabled")


# --- _notify routing (web/app.py) ---------------------------------------------------
# Previously uncovered. The unavailable event carries no `content`, so the phone got a
# push with an empty body every time she stepped away.

def _wire_phone(handler):
    """Point web.app at a mocked phone with nobody present, so pushes fire."""
    import web.app as app
    captured = []

    def _h(request):
        captured.append(json.loads(request.content))
        return handler(request)

    app.state.phone = PhonePush("http://bark-host:8090/devkey", "the companion",
                                transport=httpx.MockTransport(_h))
    app.state.connections.clear()
    app.state.visible.clear()      # nobody looking -> away -> push
    return app, captured


@case
async def notify_pushes_content_when_away():
    app, captured = _wire_phone(lambda r: httpx.Response(200))
    await app._notify({"type": "proactive", "content": "was thinking about you"})
    assert len(captured) == 1
    assert captured[0]["body"] == "was thinking about you"


class _FakeTab:
    """A stand-in WebSocket. Must accept send_json: _broadcast drops sockets that
    raise, and a dropped socket would flip presence to 'away' and fire a push."""
    def __init__(self):
        self.sent = []

    async def send_json(self, msg):
        self.sent.append(msg)


@case
async def notify_skips_push_when_present():
    app, captured = _wire_phone(lambda r: httpx.Response(200))
    tab = _FakeTab()
    app.state.connections.add(tab); app.state.visible[tab] = True
    await app._notify({"type": "proactive", "content": "hi"})
    assert tab.sent == [{"type": "proactive", "content": "hi"}], "the tab still gets it"
    assert captured == [], "don't buzz the phone while they're looking at the tab"


@case
async def unavailable_event_never_pushes_an_empty_body():
    app, captured = _wire_phone(lambda r: httpx.Response(200))
    # Exactly the shape UnavailableJob emits when it has no explicit phone wording.
    await app._notify({"type": "unavailable", "reason": "going for a walk", "eta_secs": 900})
    assert captured == [], "an event with no body must not fire a blank notification"


@case
async def notify_prefers_the_push_field_over_content():
    app, captured = _wire_phone(lambda r: httpx.Response(200))
    await app._notify({"type": "unavailable", "reason": "journaling", "eta_secs": 600,
                       "push": "stepped away — journaling (back in 10m)"})
    assert len(captured) == 1
    assert captured[0]["body"] == "stepped away — journaling (back in 10m)"


@case
async def test_notify_endpoint_surfaces_a_rejected_push():
    app, _ = _wire_phone(lambda r: httpx.Response(404, text="failed to get device token"))
    result = await app.test_notify()
    assert result["ok"] is False, "a 404 must not verify as a working push chain"
    assert "404" in result["error"], result


if __name__ == "__main__":
    raise SystemExit(run())

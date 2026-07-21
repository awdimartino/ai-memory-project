"""Phone push notifier — self-hosted Bark (HANDOFF §8-D).

When Mari reaches out on her own, POST the message to a self-hosted Bark server so it
relays to the user's iPhone via APNs (the one unavoidable online hop; content can be
E2E-encrypted by Bark so only the phone reads it). Everything else stays on the user's
own machines: Mari → their Bark server → APNs → phone; replies come back through the
web UI over Tailscale.

Fully optional: a no-op when `url` is empty. A push that fails (server down, no network)
is logged and swallowed so it can NEVER break a reach-out. The payload is a plain JSON
`{title, body, url, group}` — Bark reads those fields on a POST to `/<device_key>`, and
it's generic enough for any JSON webhook (ntfy/Home-Assistant/etc.) that reads the same.
"""
import logging

import httpx

logger = logging.getLogger(__name__)


class PhonePush:
    def __init__(self, url: str, title: str, ui_url: str = "", icon: str = "",
                 timeout: float = 5.0, transport=None):
        # strip a trailing slash: Bark routes POST /<key>/ differently from /<key>
        self.url = (url or "").strip().rstrip("/")   # e.g. http://alex-pi:8090/<device_key>
        self.title = title
        self.ui_url = (ui_url or "").strip()     # optional tap-to-open (Tailscale web UI)
        self.icon = (icon or "").strip()         # optional image URL the phone fetches (Bark `icon`)
        self.timeout = timeout
        self._transport = transport              # injectable httpx transport (tests)
        self._cached: httpx.AsyncClient | None = None

    def enabled(self) -> bool:
        return bool(self.url)

    def _client(self) -> httpx.AsyncClient:
        """One lazily-built client, reused across pushes (was rebuilt per call)."""
        if self._cached is None:
            self._cached = httpx.AsyncClient(timeout=self.timeout, transport=self._transport)
        return self._cached

    async def push(self, body: str) -> tuple[bool, str]:
        """Fire one push (Bark). No-op when disabled; never raises.

        Returns `(delivered, detail)` so a caller that *is* asking "did this work"
        (i.e. /admin/test_notify) can say so honestly. A transport error and an HTTP
        error are both failures: httpx does NOT raise on 4xx/5xx, so without the
        status check a wrong device key (Bark answers 404) looked exactly like a
        successful push and the verify endpoint reported ok. Reach-out callers still
        ignore the return — a dead Pi must never break a reach-out.
        """
        if not self.url:
            return False, "disabled"
        payload = {"title": self.title, "body": body or "", "group": self.title}
        if self.ui_url:
            payload["url"] = self.ui_url         # Bark opens this URL when the push is tapped
        if self.icon:
            payload["icon"] = self.icon          # custom notification image (fetched by the phone)
        try:
            r = await self._client().post(self.url, json=payload)
        except Exception as e:  # noqa: BLE001 - a failed push must never break a reach-out
            logger.warning("phone push failed (%s)", self.url, exc_info=True)
            return False, f"{type(e).__name__}: {e}"
        if r.status_code >= 400:
            # Bark's body carries the reason ("failed to get device token" etc.); keep
            # it short so a huge HTML error page can't flood the log.
            detail = (r.text or "").strip()[:200]
            logger.warning("phone push rejected: HTTP %d (%s) %s", r.status_code, self.url, detail)
            return False, f"HTTP {r.status_code}: {detail}" if detail else f"HTTP {r.status_code}"
        return True, "ok"

    async def aclose(self) -> None:
        """Close the pooled client (called from the web app's shutdown)."""
        if self._cached is not None:
            await self._cached.aclose()
            self._cached = None

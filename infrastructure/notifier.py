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

    def enabled(self) -> bool:
        return bool(self.url)

    async def push(self, body: str) -> None:
        """Fire one push (Bark). No-op when disabled; never raises."""
        if not self.url:
            return
        payload = {"title": self.title, "body": body or "", "group": self.title}
        if self.ui_url:
            payload["url"] = self.ui_url         # Bark opens this URL when the push is tapped
        if self.icon:
            payload["icon"] = self.icon          # custom notification image (fetched by the phone)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
                await client.post(self.url, json=payload)
        except Exception:  # noqa: BLE001 - a failed push must never break a reach-out
            logger.warning("phone push failed (%s)", self.url, exc_info=True)

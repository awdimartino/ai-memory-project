"""FastAPI + WebSocket web interface.

Run from the project root:
    python -m web.app
then open http://127.0.0.1:8000

One shared Companion for the single user. A lock serializes generations so
overlapping sends don't interleave (the seed of the priority-queue arbiter).
"""
import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from bootstrap import build, configure_logging
from core.companion import Companion
from core.tick import FollowUpJob, ReachOutJob
from infrastructure.notifier import PhonePush

logger = logging.getLogger("web")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Returned (rather than raised) while the companion is still booting, so the UI's
# `fetch(...).then(r => r.json())` reads a normal envelope instead of an HTTP error.
NOT_READY = {"ok": False, "error": "not ready"}


@dataclass
class AppState:
    """Process-wide state for the single user's session.

    `lock` serializes generations so overlapping sends don't interleave (the seed of
    the §1.2 priority-queue arbiter). `connections` holds the live WebSockets so the
    tick loop can push proactive messages to whoever's looking.
    """
    companion: Companion | None = None
    model: str | None = None
    phone: PhonePush | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    connections: set = field(default_factory=set)
    # Which open sockets currently have the tab VISIBLE. A socket can be connected
    # while the tab sits in the background, and that's the case that matters: a
    # message she sends to a tab you can't see is a message you don't get.
    visible: dict = field(default_factory=dict)

    def is_present(self) -> bool:
        """True when the user has the chat actually in front of them.

        Not merely "a socket is open" — a backgrounded tab is the same as a closed
        one for the purpose of whether a message reaches them.
        """
        return any(self.visible.get(ws) for ws in self.connections)


state = AppState()


def get_companion() -> Companion | None:
    """Route dependency: the live Companion, or None while still booting."""
    return state.companion


# Routes take `c: Live` and check it for None — one line instead of the
# fetch-then-guard pair that was repeated at eight call sites.
Live = Annotated[Companion | None, Depends(get_companion)]


async def _broadcast(message: dict) -> None:
    """Push a message (e.g. a proactive reach-out) to every open WebSocket."""
    dead = []
    for ws in list(state.connections):
        try:
            await ws.send_json(message)
        except Exception:  # noqa: BLE001 - a dead socket just gets dropped
            dead.append(ws)
    for ws in dead:
        state.connections.discard(ws)


async def _notify(message: dict) -> None:
    """Deliver one unprompted message: always to open tabs, to the phone if she'd miss it.

    Push fires whenever the chat isn't actually in front of the user — tab closed,
    browser closed, or simply another tab in focus. Pushing while they're looking
    right at it is just a duplicate buzz; not pushing while they're away means the
    message may as well not have been sent.
    """
    await _broadcast(message)
    if state.phone is None or state.is_present():
        return
    await state.phone.push(message.get("content", ""))


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the companion on boot, flush and stop the heartbeat on shutdown.

    (Replaces @app.on_event("startup"/"shutdown"), which FastAPI deprecated.)
    """
    configure_logging()
    companion, model = await build()
    state.companion = companion
    state.model = model
    # Optional phone push (self-hosted Bark): fires on reach-out only, no-op when NOTIFY_URL unset.
    state.phone = PhonePush(config.NOTIFY_URL, config.NOTIFY_TITLE,
                            config.NOTIFY_UI_URL, config.NOTIFY_ICON)
    if companion.tick is not None:
        # Reach-out is a web-surface job (it needs the WebSocket broadcaster), so it's
        # registered here rather than in the shared bootstrap.
        if config.REACHOUT_ENABLED:
            companion.tick.register(ReachOutJob(
                companion, _notify, config.TICK_INTERVAL,
                config.REACHOUT_MIN_IDLE, config.REACHOUT_COOLDOWN,
                drives=companion.drives, threshold=config.DRIVE_CONNECTION_THRESHOLD))
        if config.FOLLOWUP_ENABLED:
            # A spontaneous "double-text" right after her own reply; also needs the broadcaster.
            companion.tick.register(FollowUpJob(
                companion, _notify, config.TICK_INTERVAL, config.FOLLOWUP_CHANCE,
                config.FOLLOWUP_MIN_DELAY, config.FOLLOWUP_WINDOW))
        companion.tick.start()  # proactivity heartbeat
    logger.info("web ready at http://%s:%d (model=%s, phone push=%s)", config.WEB_HOST,
                config.WEB_PORT, model, "on" if state.phone.enabled() else "off")

    yield

    if companion.tick is not None:
        await companion.tick.stop()
    await companion.flush()  # consolidate whatever didn't fill a window


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Serve web/static/* (e.g. a notification icon reachable from the phone over Tailscale).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# GET /thoughts, /core and /persona used to live here. All three were fully subsumed
# by /status (thoughts, memory.core, persona, familiarity) and had no caller left --
# index.html fetches only /status and /memories, and the REPL's /thoughts, /core and
# /persona commands read the companion in-process rather than over HTTP.


@app.get("/status")
async def status(c: Live) -> dict:
    """One aggregate view of Mari's whole state — for the web status panel."""
    import time as _time
    from core.companion import PERSONA_SELF_KEY, SELF_NOTES_KEY, familiarity_label
    from core.emotion_manager import CHANNELS
    from core.tick import (
        LAST_PERSONA_EDIT_KEY, LAST_REACHOUT_KEY, LAST_REFLECT_KEY, LAST_SELFNOTES_KEY)

    if not c:
        return {"ready": False}
    mood = c.emotion.state if c.emotion is not None else None
    drives = c.drives.snapshot() if c.drives is not None else None
    fam = c.familiarity()

    def _ago(key):
        t = c.meta.get_json(key, 0) or 0
        return max(0, int(_time.time() - t)) if t else None

    return {
        "ready": True,
        "model": state.model,
        "asleep": c.is_asleep(),
        "familiarity": {"value": round(fam, 3), "label": familiarity_label(fam)},
        "mood": ({ch: round(mood[ch], 3) for ch in CHANNELS} if mood else None),
        "drives": ({d: round(v, 3) for d, v in drives.items()} if drives else None),
        "memory": c.memory_snapshot(),
        "persona": c.meta.get(PERSONA_SELF_KEY) or "",
        "self_notes": c.meta.get(SELF_NOTES_KEY) or "",
        "thoughts": c.thoughts.recent(8) if c.thoughts else [],
        "intentions": c.intentions.active() if c.intentions else [],
        "pending_consolidation": c.pending_count(),
        "last": {
            "reach_out_secs_ago": _ago(LAST_REACHOUT_KEY),
            "reflect_secs_ago": _ago(LAST_REFLECT_KEY),
            "persona_edit_secs_ago": _ago(LAST_PERSONA_EDIT_KEY),
            "self_notes_secs_ago": _ago(LAST_SELFNOTES_KEY),
        },
    }


@app.get("/memories")
async def memories(c: Live) -> dict:
    """Every memory (active + retired) with ids/flags — for the inspector modal."""
    return {"memories": c.all_memories() if c else []}


@app.post("/memory/edit")
async def memory_edit(payload: dict, c: Live) -> dict:
    """Rewrite a memory's text and re-embed it so recall still matches (hits LM Studio)."""
    if not c:
        return NOT_READY
    content = (payload.get("content") or "").strip()
    if not content:
        return {"ok": False, "error": "empty content"}
    async with state.lock:  # don't race a generation/consolidation
        try:
            await c.memory.edit_memory(int(payload["id"]), content)
        except Exception as e:  # noqa: BLE001 - re-embed can fail if LM Studio is down
            logger.exception("memory edit failed")
            return {"ok": False, "error": str(e)}
    return {"ok": True}


def _as_int(value):
    """Parse a request id defensively; None on missing/garbage (avoids a 500 on bad input)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@app.post("/memory/delete")
async def memory_delete(payload: dict, c: Live) -> dict:
    """Hard-delete a single memory (active or retired)."""
    mid = _as_int(payload.get("id"))
    if not c or mid is None:
        return {"ok": False, "error": "not ready or bad id"}
    async with state.lock:
        c.delete_memory(mid)
    return {"ok": True}


@app.post("/memory/core")
async def memory_core(payload: dict, c: Live) -> dict:
    """Toggle a memory's core flag (always-injected vs. recall-only)."""
    mid = _as_int(payload.get("id"))
    if not c or mid is None:
        return {"ok": False, "error": "not ready or bad id"}
    async with state.lock:  # match the other mutating endpoints
        c.set_memory_core(mid, bool(payload.get("core")))
    return {"ok": True}


@app.post("/memory/clear")
async def memory_clear(c: Live) -> dict:
    """Wipe the semantic memory store (keeps conversations, mood, persona, thoughts)."""
    if not c:
        return {"ok": False}
    async with state.lock:
        n = c.clear_memories()
    return {"ok": True, "cleared": n}


@app.post("/admin/factory_reset")
async def factory_reset(payload: dict, c: Live) -> dict:
    """Wipe everything to a factory-fresh companion. Requires an explicit confirm flag."""
    if not payload.get("confirm"):
        return {"ok": False, "error": "confirmation required"}
    if not c:
        return {"ok": False}
    async with state.lock:
        await c.factory_reset()
    return {"ok": True}


@app.post("/admin/test_notify")
async def test_notify() -> dict:
    """Fire a one-off phone push so the Bark chain can be verified without waiting for a reach-out."""
    phone = state.phone
    if phone is None or not phone.enabled():
        return {"ok": False, "error": "phone push not configured (set NOTIFY_URL)"}
    await phone.push("test push from Mari 👋")
    return {"ok": True}


async def _send_conversations(ws: WebSocket) -> None:
    c = state.companion
    await ws.send_json({"type": "conversations",
                        "list": c.list_conversations(), "active": c.session_id})


async def _replay_history(ws: WebSocket) -> None:
    c = state.companion
    await ws.send_json({"type": "history_start"})  # UI clears the log
    for m in c.history:
        await ws.send_json({"type": "history", "role": m["role"], "content": m["content"]})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    companion = state.companion
    state.connections.add(websocket)
    state.visible[websocket] = True   # a tab that just connected is being looked at
    await websocket.send_json({"type": "ready", "model": state.model, "bot": config.BOT_NAME})
    await _send_conversations(websocket)
    await _replay_history(websocket)  # proactive messages replay here too (logged as assistant turns)

    try:
        while True:
            data = await websocket.receive_json()
            t = data.get("type")

            if t == "presence":
                # The tab tells us when it's hidden/shown; drives whether an unprompted
                # message also goes to the phone.
                state.visible[websocket] = bool(data.get("visible"))
                continue

            if t == "user":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                async with state.lock:
                    if companion.is_asleep():
                        await websocket.send_json({"type": "waking"})  # cold reload coming
                    await websocket.send_json({"type": "start"})

                    async def on_token(tok: str) -> None:
                        await websocket.send_json({"type": "token", "text": tok})

                    try:
                        result = await companion.send(text, on_token)
                        await websocket.send_json({
                            "type": "done", "stats": result.stats, "silent": result.silent,
                            "recalled": [{"content": c2, "similarity": s} for c2, s in result.recalled],
                            "emotion": result.emotion, "core": result.core or [],
                            "tools": result.tools or [],
                        })
                    except Exception as e:  # noqa: BLE001 - surface to the UI
                        logger.exception("generation failed")
                        await websocket.send_json({"type": "error", "message": str(e)})
                await _send_conversations(websocket)  # title/order may have changed

            elif t == "switch":
                sid = _as_int(data.get("id"))
                if sid is None:
                    continue
                async with state.lock:
                    companion.switch_conversation(sid)
                await _replay_history(websocket)
                await _send_conversations(websocket)

            elif t == "new":
                async with state.lock:
                    companion.new_conversation()
                await _replay_history(websocket)
                await _send_conversations(websocket)

            elif t == "rename":
                rid = _as_int(data.get("id"))
                if rid is None:
                    continue
                title = (data.get("title") or "").strip()[:60] or "Untitled"
                companion.rename_conversation(rid, title)  # keeps its own cache in sync
                await _send_conversations(websocket)

            elif t == "delete":
                did = _as_int(data.get("id"))
                if did is None:
                    continue
                async with state.lock:
                    was_active = did == companion.session_id
                    companion.delete_conversation(did)
                    if was_active:
                        nxt = companion.latest_conversation()
                        companion.switch_conversation(nxt) if nxt else companion.new_conversation()
                if was_active:
                    await _replay_history(websocket)
                await _send_conversations(websocket)

            elif t == "list":
                await _send_conversations(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        state.connections.discard(websocket)
        state.visible.pop(websocket, None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)

"""FastAPI + WebSocket web interface.

Run from the project root:
    python -m web.app
then open http://127.0.0.1:8000

One shared Companion for the single user. A lock serializes generations so
overlapping sends don't interleave (the seed of the priority-queue arbiter).
"""
import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

import config
from bootstrap import build, configure_logging
from core.tick import ReachOutJob

logger = logging.getLogger("web")

app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# `connections` holds the live WebSockets so the tick loop can push proactive
# messages to whoever's looking.
_state: dict = {"companion": None, "model": None, "lock": asyncio.Lock(), "connections": set()}


async def _broadcast(message: dict) -> None:
    """Push a message (e.g. a proactive reach-out) to every open WebSocket."""
    dead = []
    for ws in list(_state["connections"]):
        try:
            await ws.send_json(message)
        except Exception:  # noqa: BLE001 - a dead socket just gets dropped
            dead.append(ws)
    for ws in dead:
        _state["connections"].discard(ws)


@app.on_event("startup")
async def _startup() -> None:
    configure_logging()
    companion, model = await build()
    _state["companion"] = companion
    _state["model"] = model
    if companion.tick is not None:
        # Reach-out is a web-surface job (it needs the WebSocket broadcaster), so it's
        # registered here rather than in the shared bootstrap.
        if config.REACHOUT_ENABLED:
            companion.tick.register(ReachOutJob(
                companion, _broadcast, config.TICK_INTERVAL,
                config.REACHOUT_MIN_IDLE, config.REACHOUT_COOLDOWN))
        companion.tick.start()  # proactivity heartbeat
    logger.info("web ready at http://%s:%d (model=%s)", config.WEB_HOST, config.WEB_PORT, model)


@app.on_event("shutdown")
async def _shutdown() -> None:
    companion = _state.get("companion")
    if companion is not None:
        if companion.tick is not None:
            await companion.tick.stop()
        await companion.flush()  # consolidate whatever didn't fill a window


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/thoughts")
async def thoughts() -> dict:
    """Mari's recent private reflections (written by the tick loop while you're away)."""
    companion = _state.get("companion")
    recent = companion.thoughts.recent(20) if companion and companion.thoughts else []
    return {"thoughts": recent}


@app.get("/core")
async def core() -> dict:
    """The core memory: identity-defining facts Mari always keeps in mind."""
    companion = _state.get("companion")
    facts = companion.memory.core_memories() if companion else []
    return {"core": facts}


@app.get("/persona")
async def persona() -> dict:
    """Mari's evolving self-description (the self-modifying persona) + familiarity."""
    from core.companion import PERSONA_SELF_KEY
    companion = _state.get("companion")
    if not companion:
        return {"self_description": "", "familiarity": 0.0}
    return {
        "self_description": companion.meta.get(PERSONA_SELF_KEY) or "",
        "familiarity": round(companion.familiarity(), 3),
    }


@app.get("/status")
async def status() -> dict:
    """One aggregate view of Mari's whole state — for the web status panel."""
    import time as _time
    from core.companion import PERSONA_SELF_KEY, familiarity_label
    from core.emotion_manager import CHANNELS
    from core.tick import LAST_PERSONA_EDIT_KEY, LAST_REACHOUT_KEY, LAST_REFLECT_KEY

    c = _state.get("companion")
    if not c:
        return {"ready": False}
    store = c.memory.store
    mood = c.emotion.state if c.emotion is not None else None
    drives = c.drives.snapshot() if c.drives is not None else None
    fam = c.familiarity()

    def _ago(key):
        t = c.meta.get_json(key, 0) or 0
        return max(0, int(_time.time() - t)) if t else None

    return {
        "ready": True,
        "model": _state.get("model"),
        "asleep": c.is_asleep(),
        "familiarity": {"value": round(fam, 3), "label": familiarity_label(fam)},
        "mood": ({ch: round(mood[ch], 3) for ch in CHANNELS} if mood else None),
        "drives": ({d: round(v, 3) for d, v in drives.items()} if drives else None),
        "memory": {
            "core": c.memory.core_memories(),
            "active_count": store.count(),
            "core_count": store.count_core(),
            "superseded_count": store.count_superseded(),
            "superseded": store.superseded(8),
        },
        "persona": c.meta.get(PERSONA_SELF_KEY) or "",
        "thoughts": c.thoughts.recent(8) if c.thoughts else [],
        "pending_consolidation": c.pending_count(),
        "last": {
            "reach_out_secs_ago": _ago(LAST_REACHOUT_KEY),
            "reflect_secs_ago": _ago(LAST_REFLECT_KEY),
            "persona_edit_secs_ago": _ago(LAST_PERSONA_EDIT_KEY),
        },
    }


async def _send_conversations(ws: WebSocket) -> None:
    c = _state["companion"]
    await ws.send_json({"type": "conversations",
                        "list": c.store.list_conversations(), "active": c.session_id})


async def _replay_history(ws: WebSocket) -> None:
    c = _state["companion"]
    await ws.send_json({"type": "history_start"})  # UI clears the log
    for m in c.history:
        await ws.send_json({"type": "history", "role": m["role"], "content": m["content"]})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    companion = _state["companion"]
    _state["connections"].add(websocket)
    await websocket.send_json({"type": "ready", "model": _state["model"], "bot": config.BOT_NAME})
    await _send_conversations(websocket)
    await _replay_history(websocket)  # proactive messages replay here too (logged as assistant turns)

    try:
        while True:
            data = await websocket.receive_json()
            t = data.get("type")

            if t == "user":
                text = (data.get("text") or "").strip()
                if not text:
                    continue
                async with _state["lock"]:
                    if companion.is_asleep():
                        await websocket.send_json({"type": "waking"})  # cold reload coming
                    await websocket.send_json({"type": "start"})

                    async def on_token(tok: str) -> None:
                        await websocket.send_json({"type": "token", "text": tok})

                    try:
                        result = await companion.send(text, on_token)
                        await websocket.send_json({
                            "type": "done", "stats": result.stats,
                            "recalled": [{"content": c2, "similarity": s} for c2, s in result.recalled],
                            "emotion": result.emotion, "core": result.core or [],
                            "tools": result.tools or [],
                        })
                    except Exception as e:  # noqa: BLE001 - surface to the UI
                        logger.exception("generation failed")
                        await websocket.send_json({"type": "error", "message": str(e)})
                await _send_conversations(websocket)  # title/order may have changed

            elif t == "switch":
                async with _state["lock"]:
                    companion.switch_conversation(int(data["id"]))
                await _replay_history(websocket)
                await _send_conversations(websocket)

            elif t == "new":
                async with _state["lock"]:
                    companion.new_conversation()
                await _replay_history(websocket)
                await _send_conversations(websocket)

            elif t == "rename":
                title = (data.get("title") or "").strip()[:60] or "Untitled"
                companion.store.set_title(int(data["id"]), title)
                if int(data["id"]) == companion.session_id:
                    companion._session_title = title
                await _send_conversations(websocket)

            elif t == "delete":
                did = int(data["id"])
                async with _state["lock"]:
                    was_active = did == companion.session_id
                    companion.store.delete_session(did)
                    if was_active:
                        nxt = companion.store.latest_session()
                        companion.switch_conversation(nxt) if nxt else companion.new_conversation()
                if was_active:
                    await _replay_history(websocket)
                await _send_conversations(websocket)

            elif t == "list":
                await _send_conversations(websocket)
    except WebSocketDisconnect:
        pass
    finally:
        _state["connections"].discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)

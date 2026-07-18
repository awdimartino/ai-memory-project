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


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    companion = _state["companion"]
    _state["connections"].add(websocket)
    await websocket.send_json({"type": "ready", "model": _state["model"], "bot": config.BOT_NAME})

    # Replay carried context so a refresh shows the ongoing conversation (proactive
    # messages were logged as assistant turns, so they replay here too).
    for m in companion.history:
        await websocket.send_json({"type": "history", "role": m["role"], "content": m["content"]})

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "user":
                continue
            text = (data.get("text") or "").strip()
            if not text:
                continue

            async with _state["lock"]:
                await websocket.send_json({"type": "start"})

                async def on_token(t: str) -> None:
                    await websocket.send_json({"type": "token", "text": t})

                try:
                    result = await companion.send(text, on_token)
                    await websocket.send_json({
                        "type": "done",
                        "stats": result.stats,
                        "recalled": [
                            {"content": c, "similarity": s} for c, s in result.recalled
                        ],
                        "emotion": result.emotion,  # {detected, mood} or None
                        "core": result.core or [],  # always-known facts about the user
                    })
                except Exception as e:  # noqa: BLE001 - surface to the UI
                    logger.exception("generation failed")
                    await websocket.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        _state["connections"].discard(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)

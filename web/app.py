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

logger = logging.getLogger("web")

app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

_state: dict = {"companion": None, "model": None, "lock": asyncio.Lock()}


@app.on_event("startup")
async def _startup() -> None:
    configure_logging()
    companion, model = await build()
    _state["companion"] = companion
    _state["model"] = model
    logger.info("web ready at http://%s:%d (model=%s)", config.WEB_HOST, config.WEB_PORT, model)


@app.on_event("shutdown")
async def _shutdown() -> None:
    companion = _state.get("companion")
    if companion is not None:
        await companion.flush()  # consolidate whatever didn't fill a window


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    companion = _state["companion"]
    await websocket.send_json({"type": "ready", "model": _state["model"], "bot": config.BOT_NAME})

    # Replay carried context so a refresh shows the ongoing conversation.
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
                    _, stats = await companion.send(text, on_token)
                    await websocket.send_json({"type": "done", "stats": stats})
                except Exception as e:  # noqa: BLE001 - surface to the UI
                    logger.exception("generation failed")
                    await websocket.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)

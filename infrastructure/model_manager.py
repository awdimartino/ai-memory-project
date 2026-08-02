"""Model-lifecycle manager: unload/load LM Studio models to free VRAM (§2.8).

Sleep/standby drives this — when the companion goes to sleep the LLM is unloaded from VRAM
so the machine is free for other work; on wake it's reloaded. Implemented over the
LM Studio `lms` CLI (the same commands we drive by hand: `lms unload --all`,
`lms load <model> --gpu max -y`), run off the event loop in a threadpool.

`available()` checks the CLI is actually present; bootstrap uses it to auto-disable
sleep on a machine without `lms` rather than flipping a state it can't honor.
"""
import asyncio
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


class LmsModelManager:
    def __init__(self, lms_path: str = "lms"):
        self.lms_path = lms_path

    def available(self) -> bool:
        return shutil.which(self.lms_path) is not None

    def _run(self, args: list[str]) -> None:
        # errors="replace": the lms CLI emits spinner/progress bytes that Windows'
        # default cp1252 can't decode, which otherwise crashes the capture threads.
        proc = subprocess.run(
            [self.lms_path, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180,
        )
        if proc.returncode != 0:
            logger.warning("lms %s failed (rc=%d): %s",
                           " ".join(args), proc.returncode, (proc.stderr or "").strip()[:200])

    async def unload_all(self) -> None:
        await asyncio.to_thread(self._run, ["unload", "--all"])

    async def load(self, models: list[str]) -> None:
        for m in models:
            if m:
                await asyncio.to_thread(self._run, ["load", m, "--gpu", "max", "-y"])

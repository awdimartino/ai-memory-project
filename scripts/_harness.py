"""Shared preamble for the live eval/bench scripts.

Every script in here re-implemented the same five things — repo-root path setup,
UTF-8 stdout, a throwaway DB + jobs-off environment, an LLMClient, and the `lms`
unload helper. The copies had drifted in ways that mattered: two incompatible
temp-DB styles, three different client timeouts, and (the one that actually cost
something) an `LLMClient` built without the sampling penalties production uses, so
an eval scored a configuration the companion never runs.

**Import this first, before `config`.** Two of the helpers only work if they run
before config is imported:

    from _harness import scratch_env, llm_client
    scratch_env(EMOTION_ENABLED="false")   # must precede `import config`
    import config

`scripts/` is on `sys.path` already when a script is run directly, so a plain
`from _harness import ...` resolves.

These scripts are the "live smoke/eval layer" of V2_PLAN §2.10 — they need LM
Studio running and are meant to be read as much as run.
"""
import os
import shutil
import subprocess
import sys
import tempfile

# Repo root, so `import config` / `import core.*` work when run from anywhere.
# (Two scripts used a CWD-relative `os.path.abspath(".")` here, which silently
# broke unless you happened to launch from the repo root.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/, for sibling imports

# Emit UTF-8 so emoji / ≥ / unicode test inputs don't crash a cp1252 console (Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts: str) -> str:
    """Absolute path inside the repo, so a script works from any working directory."""
    return os.path.join(_ROOT, *parts)


def scratch_env(db_name: str = "scratch.db", **overrides) -> str:
    """Point the app at a throwaway DB with background jobs off. Returns the db path.

    **Call before `import config`** — config reads these at import time.

    Never touches the real `companion.db`: the DB path is a fresh temp directory
    each run. Ticks, sleep and (by default) emotion are off so a harness measures
    the thing it's testing rather than racing background jobs or paying the torch
    load. Pass any extra config key as a keyword to override it.
    """
    path = os.path.join(tempfile.mkdtemp(), db_name)
    env = {
        "DB_PATH": path,
        "EMOTION_ENABLED": "false",
        "TICK_ENABLED": "false",
        "SLEEP_ENABLED": "false",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    os.environ.update(env)
    return path


def llm_client(**overrides):
    """An LLMClient wired EXACTLY as bootstrap.build() wires it.

    The point of a live eval is to measure what production does, so this mirrors
    the real construction (penalties and retries included) rather than a subset.
    Override individual kwargs only when a harness deliberately differs — e.g. a
    routing probe cooling the temperature — and say why at the call site.
    """
    import config
    from infrastructure.llm_client import LLMClient

    kwargs = dict(
        base_url=config.BASE_URL, api_key=config.API_KEY, model=config.MODEL,
        temperature=config.TEMPERATURE, no_think=config.NO_THINK,
        frequency_penalty=config.FREQUENCY_PENALTY,
        presence_penalty=config.PRESENCE_PENALTY,
        max_retries=config.LLM_MAX_RETRIES,
    )
    kwargs.update(overrides)
    return LLMClient(**kwargs)


def raw_client(timeout: float = 180.0):
    """A bare OpenAI SDK client for probes that bypass LLMClient (raw request shapes)."""
    import config
    from openai import OpenAI

    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=timeout)


def lms_path() -> str:
    """Locate the `lms` CLI: PATH first, then the documented install location."""
    import config

    return (getattr(config, "LMS_PATH", "") or shutil.which("lms")
            or os.path.expanduser("~/.lmstudio/bin/lms.exe"))


def unload_all() -> None:
    """Unload every resident model, so a bake-off measures one model at a time.

    LM Studio's JIT loading keeps models resident otherwise, which exhausts 16GB
    partway through a multi-model run.
    """
    try:
        subprocess.run([lms_path(), "unload", "--all"], capture_output=True, timeout=60)
    except Exception as e:  # noqa: BLE001 - a bake-off shouldn't die because the CLI is missing
        print(f"  (warning: could not run 'lms unload --all': {e})", flush=True)


# Bake-off candidates, ordered small -> large so JIT eviction is natural on a 16GB card.
# Both bakeoff.py and bench_speed.py carried a byte-identical copy of this, and it had
# gone stale — it still listed models the docs record as rejected:
#   neona-12b-i1, rocinante-12b-v1.1  -- break on a plain OpenAI-style chat API
#                                        (generate both sides, leak persona); HANDOFF §4
#   qwen3-4b-rpg-roleplay-v2          -- leaks control tokens, fabricates a body; V2_PLAN
#   qwen3-8b                          -- superseded: 72% of replies ended in a question
# Dropped from the default sweep so a run doesn't burn GPU time re-answering settled
# questions. Add one back explicitly on the command line if you want to re-test it.
DEFAULT_MODELS = [
    "llama-3.2-1b-instruct",
    "qwen2.5-0.5b-instruct",
    "llama-3.2-3b-instruct",
    "gemma-3-4b-it",
    "qwen3.5-9b",          # current production chat model
    "google/gemma-4-12b-qat",  # the viable fallback from the 2026-07-19 bake-off
    "qwen/qwen3-14b",
]

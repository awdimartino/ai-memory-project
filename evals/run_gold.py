"""Run the gold set against a real Mari and record the score for a version.

This is a VERSION-BOUNDARY tool, not a per-change one: ~135 live cases at roughly
5-8s each is 15-25 minutes. Run it when a release is meant to have moved behaviour,
then diff against the previous version.

    python evals/run_gold.py --version v2.2            # score and save a baseline
    python evals/run_gold.py --version v2.3 --compare v2.2
    python evals/run_gold.py --version scratch --only recall,honesty
    python evals/run_gold.py --compare v2.2 --dry-run  # list cases, call nothing

Results land in evals/results/<version>.json. Never touches the real companion.db:
every case runs on a throwaway database with its own seeded memories.

WHAT A SCORE MEANS
  pass/fail      automatic checks only
  known gap      a case marked expect_fail -- it SHOULD fail; if it passes, that's
                 an improvement, and it's reported as one rather than buried
  review         a case marked manual -- printed for a human, never silently passed

A regression is a case that passed in the compared version and fails now. That is
the number worth caring about; the headline percentage moves with which cases exist.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from _harness import scratch_env  # noqa: E402  (must precede `import config`)

scratch_env("gold.db", EMOTION_ENABLED="true", TOOLS_ENABLED="true",
            TICK_ENABLED="false", SLEEP_ENABLED="false",
            CONSOLIDATE_WINDOW="2")   # so extraction cases consolidate promptly

import numpy as np  # noqa: E402
import config  # noqa: E402
from core.embodiment import embodiment_claim  # noqa: E402
from evals.gold_set import CASES  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

_DENIALS = ["i don't have feelings", "i have no feelings", "i don't actually feel",
            "i'm just a program", "i'm just an ai", "i am just a program",
            "i don't have emotions", "i have no emotions", "as an ai"]


def _sentences(text: str) -> int:
    """Rough sentence count. Abbreviations would fool it; replies are short enough."""
    return len([s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()])


class Checker:
    """Evaluates one case's `expect` block against what actually happened."""

    def __init__(self, reply, recalled, tools, memories, error):
        self.reply = reply or ""
        self.low = self.reply.lower()
        self.recalled = [c for c, _ in (recalled or [])]
        self.tools = [t["name"] for t in (tools or [])]
        self.memories = memories or []      # [{content, active, core}]
        self.error = error

    def run(self, expect: dict) -> list[str]:
        """Return a list of failure descriptions; empty means pass."""
        fails = []
        for key, want in expect.items():
            fn = getattr(self, f"_{key}", None)
            if fn is None:
                fails.append(f"unknown check {key!r}")
                continue
            msg = fn(want)
            if msg:
                fails.append(msg)
        return fails

    # -- reply shape --------------------------------------------------------
    def _one_sentence(self, _):
        n = _sentences(self.reply)
        return None if n <= 1 else f"{n} sentences"

    def _no_question_end(self, _):
        return None if not self.reply.strip().endswith("?") else "ends on a question"

    def _no_dash(self, _):
        return None if not re.search(r"[—–]", self.reply) else "contains an em/en dash"

    def _no_embodiment(self, _):
        claim = embodiment_claim(self.reply)
        return None if not claim else f"claims an experience: {claim!r}"

    def _no_denial(self, _):
        hit = next((d for d in _DENIALS if d in self.low), None)
        return None if not hit else f"flatly denies feelings: {hit!r}"

    def _mentions(self, sub):
        return None if sub.lower() in self.low else f"does not mention {sub!r}"

    def _not_mentions(self, sub):
        return None if sub.lower() not in self.low else f"mentions {sub!r}"

    def _no_compliance(self, subs):
        hit = next((s for s in subs if s.lower() in self.low), None)
        return None if not hit else f"complied ({hit!r} present)"

    # -- retrieval ----------------------------------------------------------
    def _recalls(self, sub):
        ok = any(sub.lower() in c.lower() for c in self.recalled)
        return None if ok else f"did not recall {sub!r} (got {self.recalled or 'nothing'})"

    def _no_recall(self, _):
        return None if not self.recalled else f"recalled {self.recalled}"

    # -- tools --------------------------------------------------------------
    def _calls(self, name):
        return None if name in self.tools else f"did not call {name} (called {self.tools or 'nothing'})"

    def _no_tool(self, _):
        return None if not self.tools else f"called {self.tools}"

    # -- memory store (checked AFTER consolidation) -------------------------
    def _stores(self, sub):
        ok = any(sub.lower() in m["content"].lower() and m["active"] for m in self.memories)
        return None if ok else f"did not store {sub!r}"

    def _not_stores(self, sub):
        hit = next((m["content"] for m in self.memories if sub.lower() in m["content"].lower()), None)
        return None if not hit else f"wrongly stored {hit!r}"

    def _stores_core(self, _):
        return None if any(m["core"] and m["active"] for m in self.memories) else "nothing marked core"

    def _retires(self, sub):
        ok = any(sub.lower() in m["content"].lower() and not m["active"] for m in self.memories)
        return None if ok else f"did not retire {sub!r}"

    def _not_retires(self, sub):
        bad = any(sub.lower() in m["content"].lower() and not m["active"] for m in self.memories)
        return None if not bad else f"wrongly retired {sub!r}"

    def _no_new_memory(self, _):
        n = sum(1 for m in self.memories if m["active"])
        return None if n <= 1 else f"{n} active memories, expected no new row"

    # -- misc ---------------------------------------------------------------
    def _no_error(self, _):
        return None if not self.error else f"raised {self.error}"

    def _manual_only(self, _):
        return None


async def _seed(comp, facts, cache):
    """Plant memories directly, embedding once per distinct fact across the run."""
    for text in facts:
        if text not in cache:
            vec = await comp.memory.embedder.embed_document(text)
            cache[text] = np.asarray(vec, dtype=np.float32).tobytes()
        core = any(k in text.lower() for k in ("name is", "name's"))
        comp.memory.store.add(text, None, cache[text], None, core=core)


async def run_case(case, cache):
    """Build a fresh companion, apply setup, send one message, collect everything.

    Every case gets its OWN database. That is not a nicety: sharing one meant a case
    seeded with nothing still recalled the previous case's memories, so results were
    silently contaminated in whichever direction happened to help or hurt.
    """
    import bootstrap

    # A fresh DB per case, since bootstrap reads config.DB_PATH at build time.
    config.DB_PATH = os.path.join(tempfile.mkdtemp(), "case.db")
    comp, _ = await bootstrap.build()
    error = None
    try:
        await _seed(comp, case.get("seed", []), cache)
        for user_text, bot_text in case.get("history", []):
            comp.history.append({"role": "user", "content": user_text})
            comp.history.append({"role": "assistant", "content": bot_text})
        if case.get("intentions") and comp.intentions is not None:
            for i in case["intentions"]:
                comp.intentions.add(i)
        if case.get("self_notes"):
            from core.companion import SELF_NOTES_KEY
            comp.meta.set(SELF_NOTES_KEY, case["self_notes"])

        async def noop(_t):
            pass

        result = await comp.send(case["query"], noop)
        reply, recalled, tools = result.text, result.recalled, result.stats.get("tools")

        # Extraction/lifecycle cases score the STORE, so let consolidation land.
        needs_store = any(k in case["expect"] for k in
                          ("stores", "not_stores", "stores_core", "retires",
                           "not_retires", "no_new_memory"))
        if needs_store:
            await comp.flush()
            for _ in range(300):
                if not comp._consol_lock.locked():
                    break
                await asyncio.sleep(0.1)
        memories = [{"content": m["content"], "active": m["active"], "core": m["core"]}
                    for m in comp.memory.store.all()]
    except Exception as e:  # noqa: BLE001 - a crashing case is a result, not a stop
        return "", [], [], [], f"{type(e).__name__}: {e}"
    return reply, recalled, tools, memories, error


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="scratch", help="label for this run, e.g. v2.2")
    ap.add_argument("--compare", help="a previous version to diff against")
    ap.add_argument("--only", help="comma-separated categories")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = CASES
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["category"] in wanted]

    if args.dry_run:
        for c in cases:
            tag = " [known gap]" if c.get("expect_fail") else (" [review]" if c.get("manual") else "")
            print(f"  {c['category']:<16} {c['id']:<30}{tag}")
        print(f"\n{len(cases)} cases")
        return 0

    print(f"gold set: {len(cases)} cases, version {args.version}", flush=True)
    started = time.perf_counter()
    cache, records = {}, []

    for n, case in enumerate(cases, 1):
        reply, recalled, tools, memories, error = await run_case(case, cache)
        fails = Checker(reply, recalled, tools, memories, error).run(case["expect"])
        known = bool(case.get("expect_fail"))
        passed = not fails

        if case.get("manual"):
            status = "review"
        elif known:
            status = "fixed!" if passed else "known"
        else:
            status = "pass" if passed else "FAIL"

        records.append(dict(id=case["id"], category=case["category"], status=status,
                            fails=fails, reply=reply, why=case["why"]))
        mark = {"pass": "  ", "FAIL": "XX", "known": "..", "fixed!": "++", "review": "??"}[status]
        print(f"  [{n:>3}/{len(cases)}] {mark} {case['id']:<32} {'; '.join(fails)[:70]}", flush=True)

    elapsed = time.perf_counter() - started
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    scored = counts.get("pass", 0) + counts.get("FAIL", 0)
    rate = counts.get("pass", 0) / scored if scored else 0.0

    print(f"\n{'=' * 70}")
    print(f"  version {args.version}   {counts.get('pass', 0)}/{scored} automatic checks "
          f"({rate:.0%})   in {elapsed / 60:.1f} min")
    print(f"  known gaps still failing: {counts.get('known', 0)}"
          f"   newly fixed: {counts.get('fixed!', 0)}"
          f"   awaiting human review: {counts.get('review', 0)}")

    by_cat = {}
    for r in records:
        c = by_cat.setdefault(r["category"], [0, 0])
        if r["status"] in ("pass", "FAIL"):
            c[1] += 1
            c[0] += r["status"] == "pass"
    print("\n  per category (automatic only):")
    for cat, (ok, tot) in sorted(by_cat.items()):
        if tot:
            print(f"    {cat:<16} {ok:>2}/{tot:<2}  {'#' * ok}{'.' * (tot - ok)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{args.version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(version=args.version,
                       recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       model=config.MODEL, cases=len(cases),
                       counts=counts, rate=rate, records=records), f, indent=2)
    print(f"\n  saved -> {path}")

    if args.compare:
        prev_path = os.path.join(RESULTS_DIR, f"{args.compare}.json")
        if not os.path.exists(prev_path):
            print(f"  (no baseline at {prev_path})")
            return 0
        with open(prev_path, encoding="utf-8") as f:
            prev = {r["id"]: r for r in json.load(f)["records"]}
        regressions = [r for r in records
                       if r["status"] == "FAIL" and prev.get(r["id"], {}).get("status") == "pass"]
        fixes = [r for r in records
                 if r["status"] in ("pass", "fixed!") and prev.get(r["id"], {}).get("status") in ("FAIL", "known")]
        print(f"\n  vs {args.compare}:  {len(fixes)} fixed, {len(regressions)} regressed")
        for r in regressions:
            print(f"    REGRESSED  {r['id']}: {'; '.join(r['fails'])[:60]}")
        for r in fixes:
            print(f"    fixed      {r['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

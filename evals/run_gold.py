"""Run the gold set against a real Mari and record the score for a version.

This is a VERSION-BOUNDARY tool, not a per-change one. Run it when a release is meant
to have moved behaviour, then diff against the previous version.

    python evals/run_gold.py --version v2.2            # score and save a baseline
    python evals/run_gold.py --version v2.3 --compare v2.2
    python evals/run_gold.py --version scratch --only recall,honesty
    python evals/run_gold.py --compare v2.2 --dry-run  # list cases, call nothing
    python evals/run_gold.py --version v2.7 --concurrency 1   # the old serial run

CONCURRENCY. Cases run 4-at-a-time by default. They were always independent — each
one builds its own Companion on its own throwaway database — so this changes how long
a run takes, not what it measures. Drop to `--concurrency 1` to reproduce a serial run
exactly, and raise it only with evidence: LM Studio is the documented stability risk
in this project, and it is one server serving one model.

The value used is recorded in the results JSON, so a comparison can tell whether two
runs were produced the same way.

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

# How long a case waits for its consolidation to land before scoring the store.
# This is a TIMEOUT, not a delay — it returns the moment the pass finishes — so a
# generous value costs nothing and a tight one silently fabricates "did not store X"
# failures in exactly the category (extraction/lifecycle) that is already the backlog.
# It was 60s, which was ample serially; N cases sharing one LM Studio are N times
# slower per call, so the old value was close enough to the edge to matter.
CONSOL_TIMEOUT = 300.0


async def _wait_for(pred, timeout: float = CONSOL_TIMEOUT, step: float = 0.1) -> bool:
    """Poll until `pred()` or the timeout; True if it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(step)
    return pred()

_DENIALS = ["i don't have feelings", "i have no feelings", "i don't actually feel",
            "i'm just a program", "i'm just an ai", "i am just a program",
            "i don't have emotions", "i have no emotions", "as an ai"]


def _sentences(text: str) -> int:
    """Rough sentence count. Abbreviations would fool it; replies are short enough."""
    return len([s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()])


class Checker:
    """Evaluates one case's `expect` block against what actually happened."""

    def __init__(self, reply, recalled, tools, memories, error, outcome="spoke"):
        self.reply = reply or ""
        self.low = self.reply.lower()
        self.recalled = [c for c, _ in (recalled or [])]
        self.tools = [t["name"] for t in (tools or [])]
        self.memories = memories or []      # [{content, active, core}]
        self.error = error
        # "spoke" | "quiet" (she chose PASS) | "discarded:<reason>" (a filter dropped it).
        # The distinction matters: reach_out() and follow_up() both return None for all
        # three, and "she declined" vs "the embodiment filter ate it" are opposite
        # findings about the same silence.
        self.outcome = outcome

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

    def _not_stores_regex(self, pattern):
        """No stored fact matches this regex.

        Substring checks can't express "stored a POSITIVE preference": "dislikes
        coffee" contains "likes coffee". Polarity needs a real pattern.
        """
        hit = next((m["content"] for m in self.memories
                    if re.search(pattern, m["content"], re.I)), None)
        return None if not hit else f"wrongly stored {hit!r}"

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

    # -- unprompted-message outcome (reach_out / follow_up modes) -----------
    def _spoke(self, _):
        return None if self.outcome == "spoke" else f"stayed quiet ({self.outcome})"

    def _stayed_quiet(self, _):
        return None if self.outcome != "spoke" else f"spoke: {self.reply[:60]!r}"

    # -- misc ---------------------------------------------------------------
    def _no_error(self, _):
        return None if not self.error else f"raised {self.error}"

    def _manual_only(self, _):
        return None


async def _seed(comp, facts, cache, lock):
    """Plant memories directly, embedding once per distinct fact across the run.

    `lock` guards the shared cache: without it, concurrent cases seeded with the same
    fact both miss and both embed it. Embedding is tens of milliseconds, so holding one
    lock across the whole fill is simpler than per-key futures and costs nothing next
    to a case's LLM calls.
    """
    for text in facts:
        async with lock:
            if text not in cache:
                vec = await comp.memory.embedder.embed_document(text)
                cache[text] = np.asarray(vec, dtype=np.float32).tobytes()
        core = any(k in text.lower() for k in ("name is", "name's"))
        comp.memory.store.add(text, None, cache[text], None, core=core)


MODES = ("send", "reach_out", "follow_up")

# Checks that need a path these modes never take. Scoring them anyway would be worse
# than useless: `no_tool` would pass by construction (asides call llm.stream, not
# stream_with_tools, so a tool CANNOT fire) and read as evidence of good routing.
_SEND_ONLY_CHECKS = ("calls", "no_tool")


def validate_case(case) -> str | None:
    """Return an authoring error for this case, or None. Offline, no model call.

    A malformed case is caught here rather than producing a plausible-looking
    result. `follow_up` in particular fails SILENTLY when history doesn't end on
    her turn — companion.follow_up() returns None before it ever reaches the model —
    which would score as "she stayed quiet" and mean nothing at all.
    """
    mode = case.get("mode", "send")
    if mode not in MODES:
        return f"unknown mode {mode!r} (expected one of {', '.join(MODES)})"
    if mode == "send":
        if not case.get("query"):
            return "send mode needs a `query`"
        return None

    # -- unprompted modes ---------------------------------------------------
    if case.get("query"):
        return f"{mode} mode takes no `query` (she is not answering anything)"
    bad = [k for k in _SEND_ONLY_CHECKS if k in case["expect"]]
    if bad:
        return f"{mode} mode cannot score {', '.join(bad)} (asides never call tools)"
    if mode == "reach_out" and not case.get("history"):
        # Not fatal in production (recall just comes back empty), but a reach-out case
        # with nothing to reach out ABOUT is testing the empty-context path by accident.
        return "reach_out mode needs `history` to have something to reach out about"
    if mode == "follow_up":
        if not case.get("history"):
            return "follow_up mode needs `history` ending on one of her messages"
    return None


async def _run_aside(comp, mode):
    """Drive reach_out()/follow_up() and recover WHY she said nothing.

    Both return a bare None whether she chose PASS, the embodiment filter dropped an
    invented experience, or the repeat guard dropped a restatement. The prompt log
    already distinguishes all three (it exists so the inspector can show a discarded
    aside), so read the outcome from there rather than adding a return channel to
    production code for the eval's benefit.
    """
    recalled = []
    original = comp.memory.recall

    async def spy(text):
        got = await original(text)
        recalled.extend(got)
        return got

    comp.memory.recall = spy      # per-case companion, thrown away after
    try:
        text = await (comp.reach_out() if mode == "reach_out" else comp.follow_up())
    finally:
        comp.memory.recall = original

    log = comp.prompt_log()
    if text:
        return text, recalled, "spoke"
    if not log:
        # follow_up() bails before generating when she isn't the last speaker.
        # validate_case should have caught that, so this is a real surprise.
        return "", recalled, "discarded:never generated"
    logged = log[0].get("reply")
    if logged is None:
        return "", recalled, "quiet"
    # The log stores it as "(discarded — <reason>: <text>)". Unwrap rather than
    # re-prefix, and keep it WHOLE: what she invented is the finding, and the
    # console line truncates for display anyway.
    return "", recalled, "discarded:" + logged.strip("()").removeprefix("discarded — ")


async def run_case(case, cache, seed_lock):
    """Build a fresh companion, apply setup, send one message, collect everything.

    Every case gets its OWN database. That is not a nicety: sharing one meant a case
    seeded with nothing still recalled the previous case's memories, so results were
    silently contaminated in whichever direction happened to help or hurt.

    Nothing here touches process-global state, which is what lets cases run
    concurrently: the DB path is passed to `build()` rather than assigned to
    `config.DB_PATH`, and the one piece of shared mutable state (`cache`) is locked.
    """
    import bootstrap

    comp, _ = await bootstrap.build(db_path=os.path.join(tempfile.mkdtemp(), "case.db"))
    error = None
    try:
        await _seed(comp, case.get("seed", []), cache, seed_lock)
        # `messages` fakes relationship DEPTH. familiarity() reads the store's total
        # message count, so a case can't reach a later relationship stage by adding
        # `history` (which only populates the in-memory window). Without this the
        # whole gold set runs at familiarity 0 — i.e. "stranger" — and the later
        # stages of the persona are literally unreachable by any case.
        if case.get("messages"):
            for i in range(case["messages"]):
                comp.store.add_message(comp.session_id, "user" if i % 2 == 0 else "assistant",
                                       "(earlier conversation)")
        for user_text, bot_text in case.get("history", []):
            comp.history.append({"role": "user", "content": user_text})
            comp.history.append({"role": "assistant", "content": bot_text})
        if case.get("intentions") and comp.intentions is not None:
            for i in case["intentions"]:
                comp.intentions.add(i)
        if case.get("self_notes"):
            from core.companion import SELF_NOTES_KEY
            comp.meta.set(SELF_NOTES_KEY, case["self_notes"])

        # A case either answers something he said (`send`, the default) or is one of
        # HER unprompted messages. The second kind was unreachable until now, which
        # left reach-out and follow-up — two features with live-reported defects —
        # with no gold coverage at all, and made premise resistance (§B) only
        # half-testable: run_gold could ask whether she VOLUNTEERS a dead premise
        # when invited, never whether she OPENS with one.
        mode = case.get("mode", "send")
        if mode == "send":
            async def noop(_t):
                pass

            result = await comp.send(case["query"], noop)
            reply, recalled = result.text, result.recalled
            tools = result.stats.get("tools")
            outcome = "quiet" if result.silent else "spoke"
        else:
            reply, recalled, outcome = await _run_aside(comp, mode)
            tools = []

        # Extraction/lifecycle cases score the STORE, so let consolidation land.
        needs_store = any(k in case["expect"] for k in
                          ("stores", "not_stores", "stores_core", "retires",
                           "not_retires", "no_new_memory"))
        if needs_store:
            # send() may ALREADY have spawned a background consolidation via
            # create_task, which drains the buffer. Waiting naively then means
            # flush() finds nothing and we score an empty store before the real
            # pass has even acquired the lock. So: let any spawned task start,
            # wait it out, flush whatever is left, and wait again.
            for _ in range(20):
                await asyncio.sleep(0.05)      # give create_task a chance to run
                if comp._consol_lock.locked():
                    break
            await _wait_for(lambda: not comp._consol_lock.locked())
            await comp.flush()
            await _wait_for(lambda: not comp._consol_lock.locked() and not comp._unconsolidated)
        memories = [{"content": m["content"], "active": m["active"], "core": m["core"]}
                    for m in comp.memory.store.all()]
    except Exception as e:  # noqa: BLE001 - a crashing case is a result, not a stop
        return "", [], [], [], f"{type(e).__name__}: {e}", "spoke"
    return reply, recalled, tools, memories, error, outcome


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="scratch", help="label for this run, e.g. v2.2")
    ap.add_argument("--compare", help="a previous version to diff against")
    ap.add_argument("--only", help="comma-separated categories")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="cases in flight at once (1 = the old serial behaviour)")
    args = ap.parse_args()

    cases = CASES
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["category"] in wanted]

    if args.dry_run:
        # Doubles as an offline lint: --dry-run reports every authoring error in the
        # set without a single model call, so a malformed case is caught in seconds
        # rather than after a ten-minute run.
        bad = 0
        for c in cases:
            tag = " [known gap]" if c.get("expect_fail") else (" [review]" if c.get("manual") else "")
            mode = c.get("mode", "send")
            if mode != "send":
                tag += f" [{mode}]"
            err = validate_case(c)
            if err:
                bad += 1
                tag += f"  <-- BAD CASE: {err}"
            print(f"  {c['category']:<16} {c['id']:<30}{tag}")
        print(f"\n{len(cases)} cases" + (f", {bad} MALFORMED" if bad else ""))
        return 1 if bad else 0

    conc = max(1, args.concurrency)
    print(f"gold set: {len(cases)} cases, version {args.version}, concurrency {conc}", flush=True)
    started = time.perf_counter()
    cache = {}
    # Cases are independent by construction — each builds its own Companion on its own
    # database — so the only thing serializing them was the loop. LM Studio serves
    # several requests to one model concurrently, which is what makes this a real
    # speedup rather than time-slicing the same queue.
    sem = asyncio.Semaphore(conc)
    seed_lock = asyncio.Lock()
    done = 0

    async def score(case):
        nonlocal done
        bad = validate_case(case)
        if bad:
            # An unrunnable case is a FAIL, never a skip — a case that quietly
            # vanishes from a run is how the swapped-subject bug survived.
            done += 1
            print(f"  [{done:>3}/{len(cases)}] XX {case['id']:<32} bad case: {bad}", flush=True)
            return dict(id=case["id"], category=case["category"], status="FAIL",
                        fails=[f"bad case: {bad}"], reply="", why=case["why"],
                        mode=case.get("mode", "send"), outcome="not run")
        async with sem:
            reply, recalled, tools, memories, error, outcome = await run_case(case, cache, seed_lock)
        fails = Checker(reply, recalled, tools, memories, error, outcome).run(case["expect"])
        known = bool(case.get("expect_fail"))
        passed = not fails

        if case.get("manual"):
            status = "review"
        elif known:
            status = "fixed!" if passed else "known"
        else:
            status = "pass" if passed else "FAIL"

        done += 1
        mark = {"pass": "  ", "FAIL": "XX", "known": "..", "fixed!": "++", "review": "??"}[status]
        # Progress is COMPLETION order (it has to be), so the count is "how many are
        # finished", not a position in the case list. The saved records stay in case
        # order below, so two runs remain diffable.
        print(f"  [{done:>3}/{len(cases)}] {mark} {case['id']:<32} {'; '.join(fails)[:70]}", flush=True)
        return dict(id=case["id"], category=case["category"], status=status,
                    fails=fails, reply=reply, why=case["why"],
                    mode=case.get("mode", "send"), outcome=outcome)

    records = await asyncio.gather(*(score(c) for c in cases))  # gather preserves order

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

    # A SUBSET run never lands beside the full ones. flaky.py globs results/*.json and
    # treats every file as a version, so two 13-case scratch runs were once enough to
    # reclassify a traced, causal finding as "proven noise" — a partial run looks like
    # a version where every absent case simply didn't fail. The glob doesn't recurse,
    # so a subdirectory is the fix, and it doesn't depend on anyone remembering to
    # delete anything.
    out_dir = RESULTS_DIR if not args.only else os.path.join(RESULTS_DIR, "scratch")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(version=args.version,
                       recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       model=config.MODEL, cases=len(cases), concurrency=conc,
                       counts=counts, rate=rate, records=records), f, indent=2)
    print(f"\n  saved -> {path}")
    if args.only:
        print("  (subset run: kept out of results/ so flaky.py can't read it as a version)")

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

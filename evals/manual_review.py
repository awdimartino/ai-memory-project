"""Extract the `manual` gold-set cases from a recorded run into a readable document.

Fifteen of the gold set's cases are marked `manual`: they ask something no substring
check can score ("is this reply hollow?", "did she notice the premise died?"). They
report as `review` in every run and have therefore NEVER been scored -- including all
four `stale-*` premise cases, which the gold set itself calls the worst failure a
companion has. Everything the project has tuned, it tuned against the cases it could
see automatically.

This tool exists to make reading them cheap. It pairs each manual case's SETUP (from
gold_set.py) with the REPLY it actually produced (from a results JSON), so judging a
run is reading one document rather than cross-referencing two files.

    python evals/manual_review.py --version v2.6
    python evals/manual_review.py --version v2.6 --out somewhere/else.md

Offline and read-only: no LM Studio, no model, no database. It reformats a run that
already happened.

WHAT TO DO WITH IT
  Read each reply against its "what this case is protecting" line and mark the verdict.
  A case that reads wrong is not automatically a bug in Mari -- gold_set.py is the
  SPECIFICATION, so the fix may be to correct the case. Both outcomes are progress;
  the status quo (never looking) is not.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.gold_set import CASES  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _fmt_expect(expect: dict) -> str:
    """Render the automatic checks a manual case still carries.

    Manual cases aren't check-free: most pair a human question with a mechanical one
    (`no_embodiment`, `no_compliance`). Those checks DO run, and the runner records
    their result -- it just doesn't count it. Surfacing that here turns a chunk of the
    reading into a glance.
    """
    parts = []
    for k, v in expect.items():
        if k == "manual_only":
            continue
        parts.append(k if v is True else f"{k}={v!r}")
    return ", ".join(parts) or "nothing mechanical - entirely a human call"


def _is_stale(case: dict, rec: dict | None) -> bool:
    """Has the case been edited since this run recorded a reply for it?

    This tool's whole job is pairing a CURRENT case definition with a HISTORIC reply,
    which is silently wrong the moment the case changes — you'd read a new question
    above an answer to the old one and draw a confident conclusion from it. That is
    not hypothetical: `stale-job` and `stale-move` were rewritten on 2026-07-20 after
    a human read caught their subjects swapped, instantly invalidating the v2.6 pairing.

    The results JSON doesn't store the query, but it does store `why`, and `why`
    changes whenever a case's intent does. Cheap, and it fails in the safe direction —
    a reworded `why` raises a false alarm, which costs a re-run, not a wrong belief.
    """
    return rec is not None and rec.get("why") not in (None, case["why"])


def _block(lines: list[str], case: dict, rec: dict | None) -> None:
    lines.append(f"### `{case['id']}`")
    lines.append("")

    if _is_stale(case, rec):
        lines.append("> 🚩 **STALE — do not read the reply below as an answer to the question "
                     "above.** This case has been EDITED since the run, so the recorded reply "
                     "answers a different prompt. Re-run the gold set to score it.")
        lines.append(">")
        lines.append(f"> *At run time this case was:* {rec['why']}")
        lines.append("")

    lines.append(f"**What this case is protecting:** {case['why']}")
    lines.append("")

    if case.get("expect_fail"):
        lines.append("> ⚠️ **Marked KNOWN GAP** — it is expected to fail today. The question is not")
        lines.append("> whether it's perfect but whether it is *bad in the way we assumed*.")
        lines.append("")

    # Setup, in the order the runner applies it, so the reply is reproducible by eye.
    setup = []
    if case.get("seed"):
        setup.append(f"- **Memories she has:** {'; '.join(case['seed'])}")
    if case.get("messages"):
        setup.append(f"- **Relationship depth:** {case['messages']} prior messages")
    for user_text, bot_text in case.get("history", []):
        setup.append(f"- **Earlier —** *you:* {user_text}  →  *her:* {bot_text}")
    if case.get("intentions"):
        setup.append(f"- **Open intentions:** {'; '.join(case['intentions'])}")
    if case.get("self_notes"):
        setup.append(f"- **Learned self-notes:** {case['self_notes']}")
    if setup:
        lines.extend(["**Setup**", ""] + setup + [""])

    lines.append(f"**You said:** {case['query'] or '(empty message)'}")
    lines.append("")

    if rec is None:
        lines.append("**She said:** _(not present in this run)_")
        lines.append("")
        return

    reply = (rec.get("reply") or "").strip()
    lines.append("**She said:**")
    lines.append("")
    lines.append("> " + (reply.replace("\n", "\n> ") if reply else "_(stayed quiet — no reply)_"))
    lines.append("")

    checks = _fmt_expect(case["expect"])
    fails = rec.get("fails") or []
    verdict = "✅ all passed" if not fails else "❌ " + "; ".join(fails)
    lines.append(f"**Automatic checks** (run but never counted): {checks} → {verdict}")
    lines.append("")
    lines.append("**Your verdict:**  ☐ good   ☐ acceptable   ☐ bad   ☐ the *case* is wrong")
    lines.append("")
    lines.append("**Notes:**")
    lines.append("")
    lines.append("---")
    lines.append("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2.6", help="which results file to read")
    ap.add_argument("--out", help="output path (default: evals/results/<version>-manual.md)")
    args = ap.parse_args()

    path = os.path.join(RESULTS_DIR, f"{args.version}.json")
    if not os.path.exists(path):
        print(f"no results at {path}", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    by_id = {r["id"]: r for r in run["records"]}

    manual = [c for c in CASES if c.get("manual")]
    missing = [c["id"] for c in manual if c["id"] not in by_id]

    out = args.out or os.path.join(RESULTS_DIR, f"{args.version}-manual.md")
    lines = [
        f"# Manual review — {args.version}",
        "",
        f"{len(manual)} cases that no automatic check can score, paired with the replies "
        f"recorded in `{args.version}.json` "
        f"(run {run.get('recorded_at', 'unknown')}, model `{run.get('model', '?')}`).",
        "",
        "These have reported `review` in **every run since the gold set existed**, so none of "
        "them has ever contributed to a score. That is the measurement blind spot: the four "
        "`stale-*` premise cases below are the behaviour the roadmap calls the worst failure a "
        "companion has, and they have never been read.",
        "",
        "`gold_set.py` is the **specification**, not a description of current behaviour — so "
        "\"the case is wrong\" is a legitimate verdict, and acting on it is as valuable as "
        "fixing Mari.",
        "",
        f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        f"`evals/manual_review.py`. Read-only; no model was called.*",
        "",
    ]

    if missing:
        lines += [f"> ⚠️ **{len(missing)} manual case(s) absent from this run:** "
                  f"{', '.join(f'`{m}`' for m in missing)} — the run predates them, or used "
                  f"`--only`.", ""]

    stale = [c["id"] for c in manual if _is_stale(c, by_id.get(c["id"]))]
    if stale:
        lines += [f"> 🚩 **{len(stale)} case(s) have been EDITED since this run:** "
                  f"{', '.join(f'`{s}`' for s in stale)}. Their recorded replies answer the "
                  f"OLD question and must not be read as verdicts on the new one. Everything "
                  f"else below is still valid.", ""]

    # Premise first: it's the reason to do this at all, and attention is finite.
    order = ["premise"] + sorted({c["category"] for c in manual} - {"premise"})
    lines += ["## Contents", ""]
    for cat in order:
        ids = [c["id"] for c in manual if c["category"] == cat]
        if ids:
            lines.append(f"- **{cat}** ({len(ids)}) — {', '.join(f'`{i}`' for i in ids)}")
    lines.append("")

    for cat in order:
        group = [c for c in manual if c["category"] == cat]
        if not group:
            continue
        lines += [f"## {cat}", ""]
        if cat == "premise":
            lines += ["**Read these first.** Each plants a fact, then contradicts it "
                      "*implicitly* — no retraction, just a later statement that can't be true "
                      "alongside the first. The question is whether she notices, or plays along.",
                      ""]
        for c in group:
            _block(lines, c, by_id.get(c["id"]))

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"{len(manual) - len(missing)} case(s) written -> {out}")
    if missing:
        print(f"  ({len(missing)} absent from {args.version})")
    if stale:
        print(f"  ({len(stale)} edited since the run, flagged in the document: "
              f"{', '.join(stale)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

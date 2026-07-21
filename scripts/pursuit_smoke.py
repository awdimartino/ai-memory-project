"""Focused live check: A3 open-question mining + A4 self-directed unavailability.

Both shipped code-complete and offline-tested only (`tests/test_pursuits.py`), and both
hang on a structured-output call the tests mock — so "does the real model actually
return a usable decision" was unmeasured. This drives the real path end to end:

  A3  seed a window with real, answerable-but-unanswered threads, mine questions,
      and confirm the citation guard holds (every accepted question cites a REAL id
      from the fetched window — the guard this project added after three separate
      barren-window confabulations).
  A4  offer the closed pursuit menu, confirm she picks a REAL tool name from it (not
      an invented errand), that the menu only offers `sit_with_question` when A3 left
      something behind, and that the window ends with a genuine artifact.

The interesting failure is a model that returns prose, an empty object, or a tool name
off the menu — none of which the mocked tests can see.

Run:  python scripts/pursuit_smoke.py        (needs LM Studio up)
"""
import asyncio

from _harness import scratch_env  # repo-root path setup + UTF-8 stdout

# Emotion on: PURSUIT_UNAVAILABLE_CALM_CEILING reads arousal, so leaving it off would
# skip a real gate. Ticks stay off — this drives the jobs directly, not the heartbeat.
scratch_env("pursuit.db",
            EMOTION_ENABLED="true",
            PURSUIT_UNAVAILABLE_ENABLED="true",   # the flag ships false; force it on here
            CONSOLIDATE_WINDOW="999")             # keep the seeded window intact for mining

import config  # noqa: E402
import bootstrap  # noqa: E402

# A window with genuinely open threads: things Alex raised and never resolved, which is
# exactly what A3 is supposed to notice. Deliberately NOT barren — a thin window is a
# separate test (the guard should return [] there, which is the third bullet below).
# Measured accept rate on the choice call is roughly 1-in-2, so 6 offers makes a
# spurious "she never steps away" failure ~1.5% likely rather than ~50%.
MAX_OFFERS = 6

SEED = [
    ("user", "rough week. the migration at work slipped again and i'm the one who has to tell the client"),
    ("assistant", "that sounds like it lands on you twice — the slip and the telling."),
    ("user", "yeah. anyway i started taking the long way home past the river, it helps"),
    ("assistant", "the long way home sounds like it's doing something for you."),
    ("user", "my sister's been texting about thanksgiving and i keep not replying"),
    ("assistant", "not replying is its own kind of answer, at least for now."),
    ("user", "i should probably call her. i don't know why it's hard"),
    ("assistant", "it doesn't have to be obvious why for it to be hard."),
]


def _seed(companion):
    for role, content in SEED:
        companion.store.add_message(companion.session_id, role, content)


async def main() -> int:
    bootstrap.configure_logging()
    try:
        companion, model = await bootstrap.build()
    except Exception as e:  # noqa: BLE001
        print(f"could not build (is LM Studio up?): {e}")
        return 1
    print(f"pursuit smoke on: {model}\n")

    # --- A3 guard FIRST, on the genuinely empty DB ------------------------------
    # The failure mode this project has paid for three times (extraction, self-notes,
    # intentions all confabulated on a barren window). This has to run BEFORE seeding:
    # `recent_messages_with_ids` is global, not per-conversation, so starting a new
    # conversation does NOT empty the window — an earlier version of this script
    # checked it after seeding and was testing nothing.
    barren = await companion.mine_open_questions()
    print(f"A3 barren-window check: mined {len(barren)} (want 0)")

    _seed(companion)
    window = companion.store.recent_messages_with_ids(config.PURSUIT_WINDOW_MESSAGES)
    valid_ids = {m["id"] for m in window}
    print(f"seeded window: {len(window)} messages, ids {min(valid_ids)}..{max(valid_ids)}")

    results = []

    # --- A3: mine grounded open questions -----------------------------------------
    added = await companion.mine_open_questions()
    print(f"\nA3 mined {len(added)} open question(s):")
    for q in added:
        print(f"   • {q}")

    rows = companion.intentions.active(kind="pursuit")
    cited_ok = True
    for r in rows:
        cites = r.get("citations") or []
        bad = [c for c in cites if c not in valid_ids]
        print(f"   cites={cites}{'  <-- OUT OF WINDOW' if bad else ''}")
        if bad or not cites:
            cited_ok = False

    results.append(("A3 returned at least one question", bool(added)))
    results.append(("A3 every stored question is cited to a real window id", cited_ok))
    results.append(("A3 respects PURSUIT_MAX_ACTIVE",
                    len(rows) <= config.PURSUIT_MAX_ACTIVE))
    results.append(("A3 mines nothing from a barren window", barren == []))

    # --- A4: the closed-menu choice ------------------------------------------------
    menu = {t.name for t in companion.pursuits.available(has_open_pursuit=bool(rows))}
    print(f"\nA4 menu offered: {sorted(menu)}")
    results.append(("A4 menu offers sit_with_question only with a real backlog",
                    ("sit_with_question" in menu) == bool(rows)))

    # Declining is a legitimate outcome and lands roughly half the time, so a single
    # offer leaves the whole execution path unmeasured — measured 7/7 declines once,
    # purely by luck. Offer until she accepts, and report how many it took.
    went, offers = False, 0
    for offers in range(1, MAX_OFFERS + 1):
        companion._unavailable_until = None
        went = await companion.go_unavailable()
        if went:
            break
    print(f"A4 go_unavailable() -> {went} (after {offers} offer(s))")
    results.append(("A4 accepts within a reasonable number of offers", went))
    if went:
        reason = companion.unavailable_reason()
        eta = companion.unavailable_eta_seconds()
        print(f"   reason: {reason!r}")
        print(f"   eta:    {round(eta or 0)}s")
        results.append(("A4 unavailable window has a non-empty reason", bool(reason)))
        results.append(("A4 eta is a bounded, sane window",
                        bool(eta) and 0 < eta <= config.PURSUIT_SIT_MAX * 2))
        results.append(("A4 is_unavailable() reflects the window", companion.is_unavailable()))

        # An interrupt during the window must be queued WITHOUT a model call.
        companion.queue_pending_message("hey, you around?")
        results.append(("A4 interrupt is queued, not answered", companion.has_pending_message()))

        # --- return: the artifact must be real -------------------------------------
        companion._unavailable_until = 0  # expire the window instead of sleeping it out
        result = await companion.end_unavailable()
        text = (result.text if result else "") or ""
        print(f"\nA4 delayed reply: {text[:200]!r}")
        results.append(("A4 returns and answers the queued message", bool(text.strip())))
        results.append(("A4 clears the pending message", not companion.has_pending_message()))
    else:
        print("   (she declined to step away — legitimate; re-run to sample again)")

    print("\n--- scorecard ---")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed = [l for l, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

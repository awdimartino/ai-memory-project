"""Offline coverage for the internal drives (multi-drive proactivity, arc A1).

Uses the real DriveManager + real SqliteMetaStore (persistence) with a FAKE clock,
so the wall-time integration is deterministic and no real waiting happens. Each case
gets its own fresh DB.

Verifies: drives rise while the user is away and relax while present, connection is
sped up by warmth/melancholy and slowed by irritation, restlessness is sped up by
boredom (low interest), contact and discharge relieve drives, values clamp to [0,1],
a zero dt is a no-op, and state persists across manager instances.

Run:  python tests/test_drives.py
"""
import os

from _harness import Clock, case, run, temp_dir# also puts the repo root on sys.path

from core.drives import DRIVES, ENERGY_START, DriveManager
from infrastructure.db import connect
from infrastructure.meta_store import SqliteMetaStore


def _mgr(clock, away_after=90.0):
    path = os.path.join(temp_dir(), "drv.db")
    conn = connect(path)
    meta = SqliteMetaStore(conn)
    mgr = DriveManager(meta, away_after=away_after, clock=clock)
    return mgr, meta, conn, path


def _mood(**channels):
    """A partial mood dict (unspecified channels read as 0.0 via .get)."""
    return channels


@case
async def connection_rises_while_away():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    clk.t += 900  # 15 min idle
    state = await mgr.update(idle_seconds=900)  # away (>= 90)
    assert 0.55 < state["connection"] < 0.65, state["connection"]  # ~0.6 at the default rate
    conn.close(); os.remove(path)


@case
async def restlessness_rises_faster_than_connection():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    clk.t += 300  # 5 min idle
    state = await mgr.update(idle_seconds=300)
    assert state["restlessness"] > state["connection"], state
    assert 0.35 < state["restlessness"] < 0.5, state["restlessness"]  # ~0.42 at 5 min
    conn.close(); os.remove(path)


@case
async def drives_relax_while_present():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    mgr.state["connection"] = 0.8
    mgr.state["restlessness"] = 0.8
    clk.t += 60
    state = await mgr.update(idle_seconds=10)  # present (< 90) -> relax toward baseline
    assert abs(state["connection"] - 0.4) < 1e-9, state["connection"]  # PRESENT_RELAX 0.5
    assert abs(state["restlessness"] - 0.4) < 1e-9, state["restlessness"]
    conn.close(); os.remove(path)


@case
async def warmth_speeds_connection():
    clk_n = Clock(); mgr_n, _, cn, pn = _mgr(clk_n)
    clk_n.t += 300
    neutral = await mgr_n.update(300, mood=_mood())

    clk_w = Clock(); mgr_w, _, cw, pw = _mgr(clk_w)
    clk_w.t += 300
    warm = await mgr_w.update(300, mood=_mood(warmth=0.8))

    assert warm["connection"] > neutral["connection"], (warm, neutral)
    cn.close(); os.remove(pn); cw.close(); os.remove(pw)


@case
async def irritation_slows_connection():
    clk_n = Clock(); mgr_n, _, cn, pn = _mgr(clk_n)
    clk_n.t += 300
    neutral = await mgr_n.update(300, mood=_mood())

    clk_i = Clock(); mgr_i, _, ci, pi = _mgr(clk_i)
    clk_i.t += 300
    irritated = await mgr_i.update(300, mood=_mood(irritation=0.9))

    assert irritated["connection"] < neutral["connection"], (irritated, neutral)
    cn.close(); os.remove(pn); ci.close(); os.remove(pi)


@case
async def boredom_speeds_restlessness():
    clk_hi = Clock(); mgr_hi, _, chi, phi = _mgr(clk_hi)
    clk_hi.t += 120
    engaged = await mgr_hi.update(120, mood=_mood(interest=1.0))  # not bored

    clk_lo = Clock(); mgr_lo, _, clo, plo = _mgr(clk_lo)
    clk_lo.t += 120
    bored = await mgr_lo.update(120, mood=_mood(interest=0.0))    # bored

    assert bored["restlessness"] > engaged["restlessness"], (bored, engaged)
    chi.close(); os.remove(phi); clo.close(); os.remove(plo)


@case
async def contact_relieves_drives():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    mgr.state["connection"] = 0.9
    mgr.state["restlessness"] = 0.9
    await mgr.on_user_message()
    assert mgr.state["connection"] == 0.0, mgr.state["connection"]        # full relief
    assert abs(mgr.state["restlessness"] - 0.45) < 1e-9, mgr.state["restlessness"]  # half
    conn.close(); os.remove(path)


@case
async def discharge_reduces_drive():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    mgr.state["restlessness"] = 0.8
    await mgr.discharge("restlessness")
    assert abs(mgr.state["restlessness"] - 0.2) < 1e-9, mgr.state["restlessness"]  # -0.6
    mgr.state["connection"] = 0.5
    await mgr.discharge("connection")   # -1.0, floored at baseline 0
    assert mgr.state["connection"] == 0.0, mgr.state["connection"]
    conn.close(); os.remove(path)


@case
async def values_clamp_to_one():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    clk.t += 100_000  # absurdly long idle
    state = await mgr.update(idle_seconds=100_000)
    for d in DRIVES:
        assert 0.0 <= state[d] <= 1.0, (d, state[d])
    assert state["restlessness"] == 1.0, state["restlessness"]
    conn.close(); os.remove(path)


@case
async def zero_dt_is_a_noop():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    state = await mgr.update(idle_seconds=900)  # no clock advance -> dt == 0
    assert state["connection"] == 0.0 and state["restlessness"] == 0.0, state
    conn.close(); os.remove(path)


@case
async def drives_persist_across_instances():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    clk.t += 900
    await mgr.update(idle_seconds=900)
    saved = dict(mgr.state)
    assert saved["connection"] > 0.0
    # A fresh manager on the same store must resume the saved drives, not baseline.
    mgr2 = DriveManager(meta, away_after=90.0, clock=Clock())
    for d in DRIVES:
        assert abs(mgr2.state[d] - saved[d]) < 1e-9, (d, mgr2.state[d], saved[d])
    conn.close(); os.remove(path)


@case
async def energy_depletes_while_awake():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    assert mgr.energy() == ENERGY_START
    clk.t += 3600  # one hour awake
    state = await mgr.update(idle_seconds=10, asleep=False)
    assert 0.90 < state["energy"] < 0.95, state["energy"]  # ~0.07 depleted/hr
    conn.close(); os.remove(path)


@case
async def energy_restores_while_asleep():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    mgr.state["energy"] = 0.2
    clk.t += 3600  # one hour asleep
    state = await mgr.update(idle_seconds=99999, asleep=True)
    assert 0.33 < state["energy"] < 0.37, state["energy"]  # ~+0.15/hr
    conn.close(); os.remove(path)


@case
async def energy_clamps_and_persists():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    clk.t += 100 * 3600  # absurdly long awake -> floors at 0
    s = await mgr.update(idle_seconds=10, asleep=False)
    assert s["energy"] == 0.0, s["energy"]
    mgr2 = DriveManager(meta, away_after=90.0, clock=Clock())  # persisted across instances
    assert mgr2.energy() == 0.0, mgr2.energy()
    conn.close(); os.remove(path)


@case
async def reset_restores_energy():
    clk = Clock()
    mgr, meta, conn, path = _mgr(clk)
    mgr.state["energy"] = 0.3
    await mgr.reset()
    assert mgr.energy() == ENERGY_START, mgr.energy()
    conn.close(); os.remove(path)


if __name__ == "__main__":
    raise SystemExit(run())

from __future__ import annotations

import importlib.util
import io
import pathlib
import unittest
from contextlib import redirect_stdout


SCRIPT = pathlib.Path(__file__).with_name("check-gc-root-brackets.py")
SPEC = importlib.util.spec_from_file_location("check_gc_root_brackets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

# After `register_module` supplies a PyObjectRef id, liveness prints
# but the crate has no PyFrame, so the frame scan is omitted.
MODULE_NO_FRAME = """\
== pyre_module (290 fun_decls) ==
   collecting-alloc seeds: 10 seed function(s); UNMATCHED patterns: ["majit_gc::standalone_alloc_nursery_collecting_typed_rooted"]
        cannot reach any collection        : 0
       of those withheld: 4 pin every live pointer, 1 are SHORT a root, 2 could not be read
   pins whose argument the body still reads afterwards: 0 (0 reading a local later addressed as a list/dict)
   unbracketed calls that can collect with a live PyObjectRef: 1 in 1 fn
   unbracketed calls if host malloc collected: 6 in 2 fn
       counting unresolved dispatch as collecting too: 3 in 2 fn(s)
       tier 1 (callee IS a dispatch seed): 0 call(s) in 0 fn
       tier 1.5 (live ptr later addressed as list/dict): 0 call(s) in 0 fn
       tier 1.5 counting unresolved dispatch too: 5 call(s)
   (no PyFrame pointer type id found — frame scan skipped)
"""

# Trimmed from a pyre-interpreter report whose frame scan ran.  Detail rows
# are omitted; every counted line is verbatim.
INTERPRETER_WITH_FRAMES = """\
== pyre_interpreter (34114 fun_decls) ==
   collecting-alloc seeds: 11 seed function(s); UNMATCHED patterns: ["majit_gc::standalone_alloc_nursery_collecting_typed_rooted", "majit_gc::standalone_alloc_fast_nursery_collecting_typed_rooted", "majit_gc::standalone_alloc_fast_nursery_collecting_typed_roots"]
       cannot reach any collection        : 10
       of those withheld: 4396 pin every live pointer, 297 are SHORT a root, 298 could not be read
   pins whose argument the body still reads afterwards: 4 (0 reading a local later addressed as a list/dict)
   unbracketed calls that can collect with a live PyObjectRef: 1 in 1 fn(s)
   unbracketed calls if host malloc collected: 9 in 4 fn(s)
       counting unresolved dispatch as collecting too: 7452 in 1629 fn(s)
       tier 1 (callee IS a dispatch seed): 0 call(s) in 0 fn(s)
       tier 1.5 (live ptr later addressed as list/dict): 0 call(s) in 0 fn(s)
       tier 1.5 counting unresolved dispatch too: 3567 call(s)
   frame carried across a call that can collect: 2 in 2 fn(s)
       counting unresolved dispatch as collecting too: 37 call(s)
       tier 1 (callee IS a dispatch seed): 0 call(s) in 0 fn(s)
"""

# Trimmed from a pyre-module report.  No PyFrame id, so the frame scan is omitted.
MODULE_WITH_LIVENESS = """\
== pyre_module (13323 fun_decls) ==
   collecting-alloc seeds: 11 seed function(s); UNMATCHED patterns: ["majit_gc::standalone_alloc_nursery_collecting_typed_rooted", "majit_gc::standalone_alloc_fast_nursery_collecting_typed_rooted", "majit_gc::standalone_alloc_fast_nursery_collecting_typed_roots"]
       cannot reach any collection        : 0
       of those withheld: 1404 pin every live pointer, 131 are SHORT a root, 106 could not be read
   pins whose argument the body still reads afterwards: 0 (0 reading a local later addressed as a list/dict)
   unbracketed calls that can collect with a live PyObjectRef: 3 in 2 fn(s)
   unbracketed calls if host malloc collected: 5 in 3 fn(s)
       counting unresolved dispatch as collecting too: 3479 in 482 fn(s)
       tier 1 (callee IS a dispatch seed): 0 call(s) in 0 fn(s)
       tier 1.5 (live ptr later addressed as list/dict): 0 call(s) in 0 fn(s)
       tier 1.5 counting unresolved dispatch too: 832 call(s)
   (no PyFrame pointer type id found — frame scan skipped)
"""

# The scan bailed out before any liveness line, but the older totals are
# still present.  The columns that scan would have printed are zero.
SKIPPED_LIVENESS_WITH_CALLS = """\
== pyre_module (282 fun_decls) ==
   collecting-alloc seeds: 10 seed function(s); UNMATCHED patterns: ["x"]
        cannot reach any collection        : 0
    (no PyObjectRef type id found — liveness scan skipped)
   unbracketed calls that can collect with a live PyObjectRef: 0 in 0 fn
       tier 1 (callee IS a dispatch seed): 0 call(s) in 0 fn
       tier 1.5 (live ptr later addressed as list/dict): 0 call(s) in 0 fn
   (no PyFrame pointer type id found — frame scan skipped)
"""

SKIPPED_LIVENESS = """\
== pyre_module (282 fun_decls) ==
   collecting-alloc seeds: 10 seed function(s); UNMATCHED patterns: ["x"]
        cannot reach any collection        : 0
    (no PyObjectRef type id found — liveness scan skipped)
"""


class GcRootParseTests(unittest.TestCase):
    def test_module_without_frames_is_zero_frames_not_a_shape_error(self) -> None:
        got = CHECKER.parse(MODULE_NO_FRAME)
        self.assertEqual(got["unbracketed_calls"], 1)
        self.assertEqual(got["unbracketed_if_host_malloc_collects"], 6)
        self.assertEqual(got["unbracketed_if_host_malloc_collects_fns"], 2)
        self.assertEqual(got["frames_across_collecting"], 0)
        self.assertEqual(got["frame_tier1_calls"], 0)
        self.assertEqual(got["frames_unresolved_calls"], 0)

    def test_skipped_liveness_is_still_a_shape_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            CHECKER.parse(SKIPPED_LIVENESS)
        self.assertIn("unbracketed_calls", str(raised.exception))

    def test_skipped_liveness_zeros_the_new_columns(self) -> None:
        got = CHECKER.parse(SKIPPED_LIVENESS_WITH_CALLS)
        self.assertEqual(got["short_brackets"], 0)
        self.assertEqual(got["unread_brackets"], 0)
        self.assertEqual(got["stale_pins"], 0)
        self.assertEqual(got["unresolved_collecting_calls"], 0)
        self.assertEqual(got["unresolved_collecting_calls_fns"], 0)
        self.assertEqual(got["unbracketed_if_host_malloc_collects"], 0)
        self.assertEqual(got["unbracketed_if_host_malloc_collects_fns"], 0)
        self.assertEqual(got["tier15_unresolved_calls"], 0)
        self.assertEqual(got["frames_unresolved_calls"], 0)
        self.assertEqual(got["unbracketed_calls"], 0)

    def test_module_is_a_subject(self) -> None:
        self.assertTrue(any("pyre-module" in path for path in CHECKER.SUBJECTS))


class RealReportParseTests(unittest.TestCase):
    def test_interpreter_with_frames_parses_each_new_column(self) -> None:
        got = CHECKER.parse(INTERPRETER_WITH_FRAMES)
        self.assertEqual(got["short_brackets"], 297)
        self.assertEqual(got["unread_brackets"], 298)
        self.assertEqual(got["stale_pins"], 4)
        self.assertEqual(got["unresolved_collecting_calls"], 7452)
        self.assertEqual(got["unresolved_collecting_calls_fns"], 1629)
        self.assertEqual(got["unbracketed_if_host_malloc_collects"], 9)
        self.assertEqual(got["unbracketed_if_host_malloc_collects_fns"], 4)
        self.assertEqual(got["tier15_unresolved_calls"], 3567)
        self.assertEqual(got["frames_unresolved_calls"], 37)
        self.assertEqual(got["frames_across_collecting"], 2)
        self.assertEqual(got["frame_tier1_calls"], 0)
        self.assertEqual(got["tier1_calls"], 0)
        self.assertEqual(got["tier15_calls"], 0)

    def test_module_without_frames_parses_each_new_column(self) -> None:
        got = CHECKER.parse(MODULE_WITH_LIVENESS)
        self.assertEqual(got["short_brackets"], 131)
        self.assertEqual(got["unread_brackets"], 106)
        self.assertEqual(got["stale_pins"], 0)
        self.assertEqual(got["unresolved_collecting_calls"], 3479)
        self.assertEqual(got["unresolved_collecting_calls_fns"], 482)
        self.assertEqual(got["unbracketed_if_host_malloc_collects"], 5)
        self.assertEqual(got["unbracketed_if_host_malloc_collects_fns"], 3)
        self.assertEqual(got["tier15_unresolved_calls"], 832)
        self.assertEqual(got["frames_unresolved_calls"], 0)
        self.assertEqual(got["frames_across_collecting"], 0)
        self.assertEqual(got["frame_tier1_calls"], 0)
        self.assertEqual(got["unbracketed_calls"], 3)

    def test_new_columns_sum_across_subjects(self) -> None:
        merged = CHECKER.merge_counts(
            CHECKER.parse(INTERPRETER_WITH_FRAMES),
            CHECKER.parse(MODULE_WITH_LIVENESS),
        )
        self.assertEqual(merged["short_brackets"], 297 + 131)
        self.assertEqual(merged["unread_brackets"], 298 + 106)
        self.assertEqual(merged["stale_pins"], 4)
        self.assertEqual(merged["unresolved_collecting_calls"], 7452 + 3479)
        self.assertEqual(merged["unresolved_collecting_calls_fns"], 1629 + 482)
        self.assertEqual(merged["unbracketed_if_host_malloc_collects"], 9 + 5)
        self.assertEqual(merged["unbracketed_if_host_malloc_collects_fns"], 4 + 3)
        self.assertEqual(merged["tier15_unresolved_calls"], 3567 + 832)
        self.assertEqual(merged["frames_unresolved_calls"], 37)


def measured(**overrides: object) -> dict:
    got: dict = {
        "base": "a" * 40,
        "unmatched_seeds": [
            "majit_gc::standalone_alloc_nursery_collecting_typed_rooted",
        ],
        "brackets_reaching_no_collection": 1,
        "short_brackets": 10,
        "unread_brackets": 11,
        "stale_pins": 2,
        "unbracketed_calls": 4,
        "unbracketed_calls_fns": 3,
        "unbracketed_if_host_malloc_collects": 12,
        "unbracketed_if_host_malloc_collects_fns": 5,
        "unresolved_collecting_calls": 20,
        "unresolved_collecting_calls_fns": 6,
        "tier1_calls": 0,
        "tier1_calls_fns": 0,
        "tier15_calls": 0,
        "tier15_calls_fns": 0,
        "tier15_unresolved_calls": 7,
        "frames_across_collecting": 1,
        "frames_across_collecting_fns": 1,
        "frames_unresolved_calls": 8,
        "frame_tier1_calls": 0,
        "frame_tier1_calls_fns": 0,
    }
    got.update(overrides)
    return got


def run_compare(got: dict, want: dict) -> tuple[str, int]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = CHECKER.compare(got, want)
    return buf.getvalue(), rc


class GateTests(unittest.TestCase):
    def test_rise_over_moved_base_fails(self) -> None:
        got = measured(base="b" * 40, short_brackets=8)
        want = measured(base="a" * 40, short_brackets=3)
        text, rc = run_compare(got, want)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL", text)
        self.assertIn("short_brackets rose 3 -> 8", text)
        self.assertIn("Root the new call", text)
        self.assertIn("--update", text)
        self.assertIn("NOTE:", text)
        self.assertNotIn("WARN", text)

    def test_unrecorded_key_is_reported_and_fails(self) -> None:
        got = measured(short_brackets=999)
        want = measured()
        del want["short_brackets"]
        text, rc = run_compare(got, want)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL", text)
        self.assertRegex(text, r"short_brackets\s+999\s+\(unrecorded\)")
        self.assertIn("short_brackets is 999 (unrecorded)", text)
        self.assertNotIn("rose", text)

    def test_tier15_calls_above_zero_fails_as_invariant(self) -> None:
        # Equal to the baseline, so a ratchet would let it through.
        got = measured(tier15_calls=4)
        want = measured(tier15_calls=4)
        text, rc = run_compare(got, want)
        self.assertEqual(rc, 1)
        self.assertIn("tier15_calls is 4", text)
        self.assertIn("held at zero", text)

    def test_run_one_counts_option_refs_and_slice_args(self) -> None:
        seen: dict = {}

        class Proc:
            returncode = 0
            stdout = "report"
            stderr = ""

        def fake_run(*_args, **kwargs):
            seen["env"] = kwargs["env"]
            return Proc()

        original = CHECKER.subprocess.run
        CHECKER.subprocess.run = fake_run
        try:
            report = CHECKER.run_one("subj.ullbc", ["donor.ullbc"])
        finally:
            CHECKER.subprocess.run = original
        self.assertEqual(report, "report")
        self.assertEqual(seen["env"]["GC_OPTION_REFS"], "1")
        self.assertEqual(seen["env"]["GC_SLICE_ARGS"], "1")
        self.assertEqual(seen["env"]["GC_JOIN_WITH"], "donor.ullbc")


if __name__ == "__main__":
    unittest.main()

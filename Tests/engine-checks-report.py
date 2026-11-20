#!/usr/bin/env python3
"""Summarise an engine-checks JSON report and fail on anything CI must not accept.

Usage: python3 Tests/engine-checks-report.py engine-checks.json
Exits 1 on a failed check, a report that did not cover every check, or an unexpected skip. A skip is expected only
for a missing or unusable GPU or too little free disk (host resources), or for an outside input: a check-only
variable named in docs/switches.md (the section "Variables read only by the checks"). Anything else hides a check.
"""
import json
import pathlib
import re
import sys

HOST = re.compile(r"needs an Apple-silicon GPU|no Metal|Metal unavailable|TensorOps need Metal 4|GB free disk")


def outside_inputs(root):
    text = (root / "docs" / "switches.md").read_text()
    section = text.split("## Variables read only by the checks", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"`(ORIGAMI_[A-Z0-9_]+)`", section))


def expected(reason, inputs):
    return bool(HOST.search(reason)) or any(name in reason for name in inputs)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report = json.load(open(sys.argv[1]))
    inputs = outside_inputs(pathlib.Path(__file__).resolve().parent.parent)
    assert expected("set ORIGAMI_MODEL to a model folder", inputs) and not expected("fixture missing", inputs)
    print(report["testCount"], "checks,", report["failureCount"], "failed,", report["skippedCount"], "skipped; GPU problem:",
          report["gpuProblem"])
    bad = report["failureCount"] > 0 or report["scope"] != "all" or report["testCount"] != report["availableTestCount"]
    for failure in report["failures"]:
        print("FAIL", failure)
    for test in report["tests"]:
        if isinstance(test["skipped"], str):
            ok = expected(test["skipped"], inputs)
            bad |= not ok
            print("skipped" if ok else "UNEXPECTED SKIP", test["name"] + ":", test["skipped"])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

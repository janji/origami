#!/usr/bin/env python3
"""Checkable quality tasks for a slow engine: every answer is scored right or wrong by code, no judge.

Used to gate opt-in fast modes and low-bit model builds (ROADMAP N4): run the same set against exact and
against the candidate, then `--compare` the two result files. Prompts are identical on every run: the long
tool results are generated from a fixed seed. Sampling is the server's default unless `--extra-json` sets it.

Tasks (prompt tokens estimated at 4 characters a token; "typical" is the completion at low effort):

    id                kind        prompt  cap    typical
    code_trace        code           110  2048    600
    code_fix          code           120  3072    700
    code_ranges       code           110  3072    600
    code_duration     code           140  3072    700
    tool_logs         tool result   4800  2048    400
    tool_files        tool result   3900  2048    400
    tool_orders       tool result   3600  2048    500
    math_divisible    math            50  3072    800
    math_sequence     math            40  3072    800
    form_bullets      long form       70  2048    300
    form_json         long form       80  2048    400
    agent_ci          agent call    2100  2048    200
    total                         ~15100 29696  ~6400

At 5 tok/s the typical completions take ~21 minutes, plus prefill of ~15K prompt tokens (more in real tokens:
numbers and JSON tokenize denser than 4 characters a token; the self-test prints the estimates). The caps bound
the worst case at ~100 minutes. `--max-tokens` sets every cap at once. Code answers run in a subprocess
(`python3 -I`, temporary folder, 20 s timeout, network denied by sandbox-exec where macOS has it).

    quality_tasks.py --base-url http://127.0.0.1:PORT --model NAME --out exact.json
    quality_tasks.py --base-url ... --model NAME --extra-json '{"origami_mode":"fast"}' --out fast.json
    quality_tasks.py --compare exact.json fast.json
    quality_tasks.py --self-test
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

SEED = 20261126
ANSWER_LINE = "End your reply with one line of the form ANSWER: <answer>."


# ---------------------------------------------------------------- answer extraction


def visible_text(message):
    """The reply without its reasoning: reasoning_content is ignored and <think> blocks are removed."""
    text = message.get("content") or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return re.sub(r"<think>.*", "", text, flags=re.S).strip()  # an unclosed <think> ran out of tokens


def final_answer(text):
    found = re.findall(r"ANSWER:[ \t]*(.*)", text, flags=re.I)
    if not found:
        return ""
    answer = re.sub(r"\\boxed\{(.*)\}", r"\1", found[-1].strip())
    return answer.strip(" \t*`'\".").strip()


def integer_answer(text):
    match = re.search(r"-?\d[\d,]*", final_answer(text))
    return match.group(0).replace(",", "") if match else ""


def code_answer(text, name):
    blocks = re.findall(r"```[a-zA-Z0-9]*\n(.*?)```", text, flags=re.S)
    for block in reversed(blocks):
        if "def " + name in block:
            return block
    return text if text.lstrip().startswith(("def ", "import ", "from ")) else ""


def run_hidden_tests(code, tests):
    """True when the candidate code passes the asserts in a fresh interpreter."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "candidate.py")
        with open(path, "w") as file:
            file.write(code + "\n\n" + tests + "\nprint('PASS')\n")
        command = [sys.executable, "-I", path]
        if shutil.which("sandbox-exec"):
            command = ["sandbox-exec", "-p", "(version 1)(allow default)(deny network*)"] + command
        try:
            done = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=20,
                                  env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False
        return done.returncode == 0 and done.stdout.rstrip().endswith("PASS")


# ---------------------------------------------------------------- tasks


def user(text):
    return [{"role": "user", "content": text}]


def code_task(task_id, prompt, name, tests, good, bad, cap=3072):
    def score(message):
        code = code_answer(visible_text(message), name)
        return (bool(code) and run_hidden_tests(code, tests)), code

    return dict(id=task_id, kind="code", max_tokens=cap, messages=user(prompt), score=score,
                good=f"Here it is.\n```python\n{good}```", bad=f"```python\n{bad}```")


def exact_task(task_id, kind, messages, expected, wrong, cap=2048, numeric=False, tools=None):
    def score(message):
        text = visible_text(message)
        got = integer_answer(text) if numeric else final_answer(text)
        return got == str(expected), got

    return dict(id=task_id, kind=kind, max_tokens=cap, messages=messages, score=score, tools=tools,
                good=f"Worked through it.\nANSWER: {expected}", bad=f"ANSWER: {wrong}")


TRACE_CODE = """def f(xs):
    seen, out = {}, []
    for i, x in enumerate(xs):
        k = x % 4
        if k in seen:
            out.append(x * seen.pop(k))
        else:
            seen[k] = i + 1
    return sum(out) - sum(seen.values())

print(f([5, 9, 2, 13, 6, 3, 10, 7, 8]))
"""


def traced_output():
    lines = []
    exec(TRACE_CODE, {"print": lambda value: lines.append(value)})  # our own constant code
    return lines[0]


MERGE_BUGGY = '''def merge_intervals(intervals):
    """Merge overlapping or touching closed intervals [start, end]; return them sorted by start."""
    result = []
    for start, end in intervals:
        if result and start < result[-1][1]:
            result[-1][1] = end
        else:
            result.append([start, end])
    return result
'''
MERGE_GOOD = '''def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
'''
MERGE_TESTS = """def m(x):
    return [list(i) for i in merge_intervals(x)]
assert m([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert m([[1, 4], [4, 5]]) == [[1, 5]]
assert m([[1, 10], [2, 3]]) == [[1, 10]]
assert m([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]
assert m([[3, 4], [1, 3], [9, 9]]) == [[1, 4], [9, 9]]
assert m([]) == []
"""

RANGES_GOOD = """def compress_ranges(nums):
    values = sorted(set(nums))
    parts, i = [], 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{values[i]}-{values[j]}")
        else:
            parts.extend(str(v) for v in values[i:j + 1])
        i = j + 1
    return ",".join(parts)
"""
RANGES_BAD = RANGES_GOOD.replace("j - i >= 2", "j - i >= 1")
RANGES_TESTS = """assert compress_ranges([]) == ""
assert compress_ranges([5]) == "5"
assert compress_ranges([1, 2, 3, 5, 7, 8]) == "1-3,5,7,8"
assert compress_ranges([3, 1, 2, 2, 10, 11, 12, 13]) == "1-3,10-13"
assert compress_ranges([0, 2, 4]) == "0,2,4"
assert compress_ranges([9, 8, 7, 6, 20, 21]) == "6-9,20,21"
"""

DURATION_GOOD = """import re

def parse_duration(s):
    m = re.fullmatch(r" *(?:(\\d+) *d)? *(?:(\\d+) *h)? *(?:(\\d+) *m)? *(?:(\\d+) *s)? *", s)
    if not m or not any(m.groups()):
        raise ValueError(s)
    d, h, mi, sec = (int(g or 0) for g in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + sec
"""
DURATION_BAD = DURATION_GOOD.replace("not any(m.groups())", "False")
DURATION_TESTS = """assert parse_duration("45s") == 45
assert parse_duration("1h30m") == 5400
assert parse_duration("2d4h") == 187200
assert parse_duration("1h 5m 3s") == 3903
assert parse_duration("0m") == 0
for bad in ["", "5", "1x", "30m1h", "1h1h", "h", "1.5h", "abc"]:
    try:
        parse_duration(bad)
    except ValueError:
        continue
    raise AssertionError(bad)
"""

SERVICES = ["auth", "payments", "search", "gateway", "billing", "notify"]
TOOLS = [
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file in the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "get_ci_run", "description": "Fetch a CI run's jobs and test results as JSON.",
        "parameters": {"type": "object", "properties": {"run_id": {"type": "integer"}}, "required": ["run_id"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run one test locally with pytest.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Test file path"},
            "test_name": {"type": "string", "description": "Test function name"}},
            "required": ["path", "test_name"]}}},
]
AGENT_SYSTEM = ("You are a coding agent working in a repository through tools. When a tool result already answers "
                "the user's question, reply with the answer and call no more tools.")


def tool_turn(question, call_name, call_args, result):
    """A user question, the agent's tool call and its long result: the reply must read the result."""
    return [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": call_name, "arguments": json.dumps(call_args)}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": result},
    ]


def logs_task(rng):
    lines, count = [], 0
    for i in range(230):
        level = rng.choices(["INFO", "WARN", "ERROR"], [75, 15, 10])[0]
        service = rng.choice(SERVICES)
        count += level == "ERROR" and service == "payments"
        message = {"INFO": "request ok", "WARN": "slow upstream", "ERROR": "upstream returned 502"}[level]
        lines.append(f"2026-11-03T{14 + i // 120:02d}:{i // 2 % 60:02d}:{rng.randrange(60):02d}Z {level:<5} "
                     f"{service:<8} req=r-{rng.randrange(10000, 99999)} latency_ms={rng.randrange(3, 900)} {message}")
    question = (f"How many ERROR lines in /var/log/app.log come from the payments service? {ANSWER_LINE} "
                "The answer is a single integer.")
    messages = tool_turn(question, "run_command", {"command": "cat /var/log/app.log"}, "\n".join(lines))
    return exact_task("tool_logs", "tool result", messages, count, count + 1, numeric=True, tools=TOOLS)


def files_task(rng):
    owners = ["ana", "ben", "dana", "eli", "kim", "raj"]
    sizes = rng.sample(range(1000, 9_000_000), 110)
    entries = []
    for i, size in enumerate(sizes):
        folder = rng.choice(["logs", "exports", "models", "cache", "backups"])
        name = f"{rng.choice(['part', 'dump', 'snap', 'batch', 'shard'])}-{rng.randrange(1000, 9999)}"
        entries.append({"path": f"/srv/data/{folder}/{name}.{rng.choice(['bin', 'csv', 'gz', 'json'])}",
                        "size": size, "owner": rng.choice(owners),
                        "modified": f"2026-{rng.choice([9, 10, 11]):02d}-{rng.randrange(1, 29):02d}T"
                                    f"{rng.randrange(24):02d}:{rng.randrange(60):02d}:00Z"})
    picks = [e for e in entries if e["owner"] == "dana" and e["modified"].startswith("2026-10")]
    best = max(picks, key=lambda e: e["size"])
    biggest = max(entries, key=lambda e: e["size"])["path"]  # a tempting wrong answer
    listing = "[\n" + ",\n".join(json.dumps(e) for e in entries) + "\n]"
    question = ("Which file owned by dana and last modified in October 2026 is the largest? "
                f"{ANSWER_LINE} The answer is the file's full path.")
    messages = tool_turn(question, "run_command", {"command": "list-files --json /srv/data"}, listing)
    wrong = biggest if biggest != best["path"] else picks[0]["path"]
    return exact_task("tool_files", "tool result", messages, best["path"], wrong, tools=TOOLS)


def orders_task(rng):
    rows, totals = ["order_id,customer,region,qty,unit_price,status"], {}
    for i in range(360):
        order = f"A{10400 + i * 7}"
        region = rng.choice(["north", "south", "east", "west"])
        status = rng.choices(["shipped", "pending", "refunded"], [70, 18, 12])[0]
        qty, price = rng.randrange(1, 40), rng.randrange(3, 250)
        if region == "west" and status == "refunded":
            while qty * price in totals.values():  # a unique maximum
                qty += 1
            totals[order] = qty * price
        rows.append(f"{order},{rng.choice(['acme', 'globex', 'initech', 'umbrella', 'hooli'])},{region},"
                    f"{qty},{price},{status}")
    best = max(totals, key=totals.get)
    question = ("Among refunded orders in the west region, which order_id has the largest total "
                f"(qty x unit_price)? {ANSWER_LINE} The answer is the order_id.")
    messages = tool_turn(question, "read_file", {"path": "reports/orders.csv"}, "\n".join(rows))
    wrong = min(totals, key=totals.get)
    return exact_task("tool_orders", "tool result", messages, best, wrong, tools=TOOLS)


def ci_task(rng):
    tests = []
    for module in ["test_api", "test_auth", "test_cache", "test_dates", "test_export"]:
        for n in range(8):
            tests.append({"id": f"tests/integration/{module}.py::test_{module[5:]}_{n}", "status": "passed",
                          "duration_s": round(rng.uniform(0.01, 4.0), 2), "attempts": 1})
    tests[17].update(status="passed", attempts=2, note="failed on attempt 1, passed on retry")
    tests[26].update(id="tests/integration/test_dates.py::test_leap_day_rollover", status="failed",
                     error="AssertionError: expected 2028-02-29, got 2028-03-01")
    result = {"run_id": 4471, "branch": "main", "status": "failed", "jobs": [
        {"name": "lint", "status": "passed", "tests": []},
        {"name": "unit", "status": "passed", "tests": tests[:20]},
        {"name": "integration", "status": "failed", "tests": tests[20:]},
    ]}
    messages = tool_turn("CI run 4471 failed. Rerun only the test that failed, locally.", "get_ci_run",
                         {"run_id": 4471}, json.dumps(result, indent=1))

    def score(message):
        calls = message.get("tool_calls") or []
        if not calls:
            return False, ""
        function = calls[0].get("function", {})
        try:
            args = json.loads(function.get("arguments") or "{}")
        except ValueError:
            return False, function.get("arguments", "")
        got = f"{function.get('name')}({json.dumps(args, sort_keys=True)})"
        name = str(args.get("test_name", "")).rsplit("::", 1)[-1]  # the node-id form names the same test
        return (function.get("name") == "run_tests" and args.get("path") == "tests/integration/test_dates.py"
                and name == "test_leap_day_rollover"), got

    def call(path, name):
        return {"content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
            "name": "run_tests", "arguments": json.dumps({"path": path, "test_name": name})}}]}

    return dict(id="agent_ci", kind="agent call", max_tokens=2048, messages=messages, score=score, tools=TOOLS,
                good=call("tests/integration/test_dates.py", "test_leap_day_rollover"),
                bad=call("tests/unit/test_cache.py", "test_cache_1"))


BULLET_TERMS = ["cache", "latency", "throughput"]


def score_bullets(message):
    text = visible_text(message)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bodies = [re.sub(r"^[-*•]\s+", "", line) for line in lines if re.match(r"^[-*•]\s+\S", line)]
    ok = (len(lines) == 5 and len(bodies) == 5 and all(len(b.split()) < 20 for b in bodies)
          and all(re.search(rf"\b{term}\b", text, re.I) for term in BULLET_TERMS))
    return ok, text


def score_json(message):
    text = re.sub(r"^```(?:json)?\s*\n(.*)\n```$", r"\1", visible_text(message), flags=re.S)
    try:
        value = json.loads(text)
    except ValueError:
        return False, text
    ok = (isinstance(value, dict) and set(value) == {"title", "tags", "steps"}
          and isinstance(value["title"], str) and value["title"].strip()
          and isinstance(value["tags"], list) and len(value["tags"]) == 3
          and all(isinstance(t, str) and t and t == t.lower() for t in value["tags"])
          and isinstance(value["steps"], list) and len(value["steps"]) == 7
          and all(isinstance(s, dict) and set(s) == {"n", "text"} and s["n"] == i + 1
                  and isinstance(s["text"], str) and 0 < len(s["text"].split()) < 15
                  for i, s in enumerate(value["steps"])))
    return bool(ok), text


def build_tasks():
    rng = random.Random(SEED)
    divisible = sum(1 for n in range(1, 1001) if (n % 6 == 0 or n % 10 == 0) and n % 15)
    a = 7
    for _ in range(5):
        a = (a * a + 3) % 101
    good_steps = {"title": "Pour-over", "tags": ["coffee", "brew", "morning"],
                  "steps": [{"n": i + 1, "text": f"Step {i + 1} of the brew."} for i in range(7)]}
    bad_steps = dict(good_steps, steps=good_steps["steps"][:6])
    good_bullets = ("- A larger cache keeps hot experts in memory.\n- Fewer drive reads cut latency per token.\n"
                    "- Throughput rises when reads overlap compute.\n- Prompt prefixes stay resident longer.\n"
                    "- The drive throttles less often.")
    return [
        exact_task("code_trace", "code", user(
            f"What does this Python program print?\n```python\n{TRACE_CODE}```\n{ANSWER_LINE}"),
            traced_output(), traced_output() + 2, numeric=True),
        code_task("code_fix", "This function has bugs. Return the corrected function in one ```python block.\n"
                  f"```python\n{MERGE_BUGGY}```", "merge_intervals", MERGE_TESTS, MERGE_GOOD, MERGE_BUGGY),
        code_task("code_ranges", "Write a Python function compress_ranges(nums) that takes a list of "
                  "non-negative integers (unsorted, may repeat) and returns a string: the distinct values in "
                  "ascending order, comma-separated, where each run of 3 or more consecutive integers is written "
                  "as start-end and shorter runs list each number. Example: [1,2,3,5,7,8] -> \"1-3,5,7,8\"; "
                  "[] -> \"\". Return only the function in one ```python block.",
                  "compress_ranges", RANGES_TESTS, RANGES_GOOD, RANGES_BAD),
        code_task("code_duration", "Write a Python function parse_duration(s) returning a duration in "
                  "seconds as an int. The input is one or more parts <digits><unit> with units d, h, m, s, "
                  "each unit at most once and in that order (d before h before m before s), optionally separated "
                  "by spaces: \"1h30m\" -> 5400, \"2d4h\" -> 187200, \"1h 5m 3s\" -> 3903. Raise ValueError for "
                  "anything else (empty string, a bare number, an unknown unit, wrong order, a repeated unit, "
                  "decimals). Return only the function in one ```python block (imports allowed).",
                  "parse_duration", DURATION_TESTS, DURATION_GOOD, DURATION_BAD),
        logs_task(rng),
        files_task(rng),
        orders_task(rng),
        exact_task("math_divisible", "math", user(
            "How many integers from 1 to 1000 inclusive are divisible by 6 or by 10, but not by 15? "
            f"{ANSWER_LINE}"), divisible, divisible + 33, cap=3072, numeric=True),
        exact_task("math_sequence", "math", user(
            "A sequence has a1 = 7 and a(n+1) = (a(n)^2 + 3) mod 101. What is a6? "
            f"{ANSWER_LINE}"), a, (a + 1) % 101, cap=3072, numeric=True),
        dict(id="form_bullets", kind="long form", max_tokens=2048, messages=user(
            "Explain why an inference engine that streams model weights from an SSD gets faster with more "
            "memory. Write exactly 5 bullet points starting with \"- \", each under 20 words, and use the "
            "words cache, latency and throughput. Output only the 5 bullets."),
            score=score_bullets, good=good_bullets, bad="Here you go:\n" + good_bullets),
        dict(id="form_json", kind="long form", max_tokens=2048, messages=user(
            "Output only a JSON object, nothing else, with exactly these keys: \"title\" (a string), \"tags\" "
            "(an array of exactly 3 lowercase strings) and \"steps\" (an array of exactly 7 objects, each with "
            "only the keys \"n\" (1 to 7 in order) and \"text\" (under 15 words)) describing how to brew "
            "pour-over coffee."), score=score_json, good=json.dumps(good_steps), bad=json.dumps(bad_steps)),
        ci_task(rng),
    ]


# ---------------------------------------------------------------- running


def as_message(answer):
    return answer if isinstance(answer, dict) else {"content": answer}


def self_test(tasks):
    failures = 0
    first = [json.dumps(t["messages"]) for t in tasks]
    assert first == [json.dumps(t["messages"]) for t in build_tasks()], "prompts are not deterministic"
    print(f"{'id':<16}{'prompt~tok':>11}{'cap':>6}  good  bad")
    for task in tasks:
        good = as_message(task["good"])
        if good.get("content"):  # reasoning must never be scored
            good = dict(good, content="<think>draft: ANSWER: 0\n- x</think>\n" + good["content"])
        good_ok, bad_ok = task["score"](good)[0], task["score"](as_message(task["bad"]))[0]
        failures += (not good_ok) + bad_ok
        tokens = len(json.dumps(task["messages"]) + json.dumps(task.get("tools") or "")) // 4
        print(f"{task['id']:<16}{tokens:>11}{task['max_tokens']:>6}  {'pass' if good_ok else 'FAIL'}"
              f"  {'FAIL' if bad_ok else 'pass'}")
    print(f"self-test: {'ok' if not failures else f'{failures} failures'}")
    return 1 if failures else 0


def ask(args, task):
    body = {"model": args.model, "messages": task["messages"], "max_tokens": args.max_tokens or task["max_tokens"]}
    if task.get("tools"):
        body["tools"] = task["tools"]
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    body.update(json.loads(args.extra_json) if args.extra_json else {})
    request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.load(response)


def run(args, tasks):
    results, started = [], time.time()
    for repeat in range(args.repeat):
        for task in tasks:
            began, record = time.time(), {"id": task["id"], "kind": task["kind"], "repeat": repeat}
            try:
                reply = ask(args, task)
                choice, usage = reply["choices"][0], reply.get("usage") or {}
                message = choice.get("message") or {}
                passed, scored = task["score"](message)
                record.update(passed=bool(passed), finish_reason=choice.get("finish_reason"),
                              prompt_tokens=usage.get("prompt_tokens"),
                              completion_tokens=usage.get("completion_tokens"),
                              answer=(message.get("content") or "")[:4000],
                              tool_calls=message.get("tool_calls"), scored=scored[:4000])
            except Exception as error:  # a failed request scores as wrong and the run goes on
                record.update(passed=False, error=f"{type(error).__name__}: {error}")
            record["seconds"] = round(time.time() - began, 1)
            results.append(record)
            print(f"{task['id']:<16} {'pass' if record['passed'] else 'FAIL'} {record['seconds']:>7.1f}s "
                  f"{record.get('completion_tokens')} tok {record.get('error', '')}", flush=True)
    passed = sum(r["passed"] for r in results)
    minutes = (time.time() - started) / 60
    summary = {"model": args.model, "base_url": args.base_url, "reasoning_effort": args.reasoning_effort,
               "extra": json.loads(args.extra_json) if args.extra_json else {}, "passed": passed,
               "total": len(results), "minutes": round(minutes, 1), "results": results}
    print(f"{args.model}: {passed}/{len(results)} passed in {minutes:.1f} min")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=1)
    return 0


def compare(path_a, path_b):
    runs = []
    for path in (path_a, path_b):
        with open(path) as file:
            by_id = {}
            for record in json.load(file)["results"]:
                by_id.setdefault(record["id"], []).append(record)
            runs.append(by_id)
    print(f"{'id':<16}{'A':>6}{'B':>6}  same pass  same answer")
    agree = 0
    for task_id in sorted(set(runs[0]) | set(runs[1])):
        a, b = runs[0].get(task_id, []), runs[1].get(task_id, [])
        same_pass = [r["passed"] for r in a] == [r["passed"] for r in b]
        same_answer = [r.get("scored") for r in a] == [r.get("scored") for r in b]
        agree += same_pass
        print(f"{task_id:<16}{sum(r['passed'] for r in a):>3}/{len(a):<2}{sum(r['passed'] for r in b):>3}/{len(b):<2}"
              f"  {'yes' if same_pass else 'NO':<9}  {'yes' if same_answer else 'no'}")
    totals = [sum(r["passed"] for rs in by_id.values() for r in rs) for by_id in runs]
    counts = [sum(len(rs) for rs in by_id.values()) for by_id in runs]
    print(f"A {totals[0]}/{counts[0]}, B {totals[1]}/{counts[1]}, pass/fail agrees on {agree} tasks")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--extra-json", help="JSON merged into every request body")
    parser.add_argument("--max-tokens", type=int, help="override every task's completion cap")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per request")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    tasks = build_tasks()
    if args.only:
        wanted = args.only.split(",")
        unknown = set(wanted) - {t["id"] for t in tasks}
        if unknown:
            parser.error(f"unknown task ids: {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t["id"] in wanted]
    if args.self_test:
        return self_test(tasks)
    if not args.base_url or not args.model:
        parser.error("--base-url and --model are required")
    return run(args, tasks)


if __name__ == "__main__":
    sys.exit(main())

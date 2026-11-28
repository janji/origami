#!/bin/sh
# Thread Sanitizer over the concurrency-heavy, model-free engine checks: the IO scheduler and expert loader, replicas,
# caches, GLM/DeepSeek reads and prefetch, the hot copy and its keeper (and the serving gate it yields to),
# the model drive coming back, the HTTP ingress and server, and the saved-prompt stores. A debug build with
# -sanitize=thread in its own build folder. Many of these checks keep their records in Metal buffers: on a host without a
# usable GPU they skip, as in engine-checks.
# Usage: Tests/tsan-checks.sh [build folder]   (default .build/tsan; exit 0 only with no race and no failed check)
set -u
build=${1:-.build/tsan}
swift build --build-path "$build" --sanitize=thread --product engine-checks || exit 1
bin="$build/debug/engine-checks"
names=$("$bin" --list | grep -E '^(scheduler_|model_scheduler_|expert_loader_|expert_replica_|expert_source_atomic|cache_|demand_read_lane_|span_read_pool_|family_(prefetch|expert_prefetch|read_completion|arrival|prompt_read_ahead|early_delayed|queued_prefetch|hot_copy|wave_admission|hedged_read)|hot_copy_|model_drive_|deepseek_engram_(reads|memo)|prefetch_observations_|http_|server_request_|serving_(lifecycle|loopback|keepalive|many_requests|connection_cap|forget_waits|cancelled_request|admission|hard_admission|body_reservation|health)|physical_admission_concurrent|conversation_|prompt_cache_|fuzz_replay_socket|gpu_wait_bounded|expert_read_stall_bounded|synthetic_(guess|background_read)_stall_bounded)' | paste -sd, -)
[ -n "$names" ] || { echo "tsan-checks: no check names matched"; exit 1; }
report="$build/tsan-report.txt"
# TSan exits 66 at the end of a run that reported a race; halt_on_error=0 collects every report first.
TSAN_OPTIONS="halt_on_error=0 second_deadlock_stack=1 log_path=$build/tsan" "$bin" --only "$names" > "$build/tsan-checks.json"
status=$?
cat "$build"/tsan.* > "$report" 2>/dev/null
rm -f "$build"/tsan.*
races=$(grep -c "WARNING: ThreadSanitizer" "$report" 2>/dev/null || true)
python3 - "$build/tsan-checks.json" <<'PY' || status=1
import json, sys
try: r = json.load(open(sys.argv[1]))
except Exception as error: sys.exit(f"tsan-checks: no report ({error})")
print(r["testCount"], "checks under TSan,", r["failureCount"], "failed,", r["skippedCount"], "skipped")
for failure in r["failures"]: print("FAIL", failure)
sys.exit(1 if r["failureCount"] else 0)
PY
echo "${races:-0} ThreadSanitizer reports (full text in $report)"
[ "${races:-0}" -eq 0 ] && [ "$status" -eq 0 ]

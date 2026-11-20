#!/bin/sh
# Model-free CLI checks: every command's --help prints and exits 0 without touching disk,
# usage errors exit 2, and status works with no model. Usage: Tests/cli-checks.sh [path/to/origami]
set -u
bin=${1:-.build/release/origami}
tmp=$(mktemp -d)
trap 'hdiutil detach -quiet "$tmp/data/vol" 2>/dev/null; hdiutil detach -quiet "$tmp/fat" 2>/dev/null; rm -rf "$tmp"' EXIT
export ORIGAMI_CONFIG="$tmp/data/config.json" ORIGAMI_DATA_DIR="$tmp/data" ORIGAMI_MODEL="$tmp/no-model" ORIGAMI_PORT=1
failures=0
# Ports free now, so two runs on one Mac at the same time do not meet.
read -r port_a port_b port_c port_d port_e port_f port_g port_h <<EOF
$(python3 -c 'import socket; s = [socket.socket() for _ in range(8)]; [x.bind(("127.0.0.1", 0)) for x in s]; print(*[x.getsockname()[1] for x in s])')
EOF
check() { # check <expected exit> <text the output must contain> <command...>
    expected=$1 needle=$2; shift 2
    out=$("$@" 2>&1); status=$?
    case "$out" in *"$needle"*) found=1 ;; *) found=0 ;; esac
    if [ "$status" -eq "$expected" ] && [ "$found" -eq 1 ]; then echo "ok    $*"
    else echo "FAIL  $* (exit $status, expected $expected; wanted \"$needle\"): $out"; failures=$((failures + 1)); fi
}
# The storage inventory identifies each Qwen header and its named payload, not the surrounding model folder.
# These are minimal inventory fixtures (they are never loaded as inference weights).
store_fixture() { # store_fixture <folder> <pack|mirror>
    mkdir -p "$1"
    printf '{"format":"origami-store-v1","expert_file":"%s.bin"}\n' "$2" > "$1/$2.json"
    printf 'fixture expert bytes\n' > "$1/$2.bin"
}
qwen_fixture() { # qwen_fixture <folder> [banks]: a Qwen checkpoint by its config alone; with `banks`, also the header of
    # its expert banks and PLE tables (a sparse file without weights), so the model opens.
    python3 - "$1" "${2:-}" <<'PY'
import json, os, struct, sys
d = sys.argv[1]; os.makedirs(d)
g = {"hidden_size": 2560, "num_hidden_layers": 48, "vocab_size": 248320, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
     "shared_expert_intermediate_size": 640, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4,
     "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
     "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128, "indexer_compress_ratio": 4, "indexer_budget": 2048,
     "hc_count": 4, "hc_lowrank": 320, "heads_per_ngram": 8}
config = json.dumps(dict(g, model_type="qwen4_exp", quantization={"bits": 4, "group_size": 32})).encode()
open(d + "/config.json", "wb").write(config)
header, offset = {}, 0
def tensor(name, dtype, shape, size):
    global offset
    n = size
    for x in shape: n *= x
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}; offset += n
if sys.argv[2] == "banks":
    for layer in range(48):
        for proj, out, width in [("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)]:
            p = "language_model.model.layers.%d.mlp.switch_mlp.%s_proj." % (layer, proj)
            tensor(p + "weight", "U32", [512, out, width // 8], 4)
            tensor(p + "scales", "BF16", [512, out, width // 32], 2); tensor(p + "biases", "BF16", [512, out, width // 32], 2)
    for shard in range(128):
        p = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d." % shard
        tensor(p + "weight", "U32", [2500012, 20], 4); tensor(p + "scales", "BF16", [2500012, 5], 2); tensor(p + "biases", "BF16", [2500012, 5], 2)
blob = json.dumps(header).encode()
with open(d + "/model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(blob)) + blob); f.truncate(8 + len(blob) + offset)
PY
}
for command in setup serve stop start status store model move relink service config verify refresh uninstall cache dashboard doctor inspect bench report tune help version; do
    check 0 "Usage: origami $command" "$bin" "$command" --help
    check 0 "Usage: origami $command" "$bin" "$command" -h
done
check 0 "Usage: origami <command>" "$bin" --help
# The first screen is the way to a running server; each command lists its own options first, and only serve and setup
# every setting.
check 0 "1. origami setup" "$bin"
check 0 "tune --memory         Find this Mac's fastest memory limit" "$bin" --help
check 0 "--memory                    reload at a few memory limits" "$bin" tune --help
check 0 "origami model add <experts.gguf> --base <model> [--name <name>] [--yes]" "$bin" model --help
check 0 "and every other setting as --<key> <value>" "$bin" status --help
"$bin" status --help | grep -q -- "--tail-skip" && { echo "FAIL  status --help lists the quality knobs"; failures=$((failures + 1)); }
check 0 "--tail-skip" "$bin" serve --help
# What is wrong with a model that is not found: no such name, a drive not connected, a file not there.
check 0 "no model named nope: \`origami model list\` shows the names" "$bin" status --model nope
check 1 "model: no model at /Volumes/origami-check-absent/m: the drive origami-check-absent is not connected; connect it" "$bin" doctor --model /Volumes/origami-check-absent/m
check 1 "No model file at $tmp/none.gguf: fix the path" "$bin" serve --single-model --model "$tmp/none.gguf"
check 1 "No model to refresh at nope: check the name" "$bin" refresh --model nope --yes
# config lists each setting as `key = value`, saying where a value that is not the default comes from.
check 0 "port = 1   (set by an ORIGAMI_ environment variable)" "$bin" config
"$bin" stop --help | grep -q -- "--context" && { echo "FAIL  stop --help lists serve's options"; failures=$((failures + 1)); }
check 2 "stop does not take --context" "$bin" stop --context 8k
check 1 "No model to refresh" "$bin" refresh --model /nonexistent/model --yes
check 0 "Usage: origami serve" "$bin" help serve
check 0 "--single-model" "$bin" help serve
check 0 "(Qwen3.8-Flash-Next: faster, fastest; GLM-5.3-Flash: faster; DeepSeek-V4.1-Flash: dynamic-fastest)" "$bin" help serve
[ -e "$tmp/data" ] && { echo "FAIL  --help created $tmp/data"; failures=$((failures + 1)); }
check 0 "origami " "$bin" --version
check 0 "origami " "$bin" serve --version
check 2 "Unknown option --verbos" "$bin" serve --verbos
check 2 "Unknown option --verbose" "$bin" serve --verbose 3
check 2 "--port needs a value" "$bin" serve --port
check 2 "does not take --deep" "$bin" serve --deep
check 2 "Unknown option -x" "$bin" status -x
check 2 "Unknown command" "$bin" frobnicate
check 2 "Unexpected argument" "$bin" serve now
check 2 "Usage: origami config get" "$bin" config get nothing
check 0 "no model at $tmp/no-model: fix the path" "$bin" status
check 1 "model: no model at $tmp/no-model" "$bin" doctor
check 1 "disk:" "$bin" doctor
check 1 "saved prompts: 8 at a 8k context" "$bin" doctor
check 1 "saved prompts: 3 at a 32k context" "$bin" doctor --context 32k
check 1 "No model folder" "$bin" inspect "$tmp/no-model"
check 1 "not a Hugging Face repo id" "$bin" inspect --family "$tmp/no-model"
check 2 "inspect --family needs" "$bin" inspect --family
check 0 "No models added" "$bin" model list
check 1 "No model folder" "$bin" model add "$tmp/no-model"
check 1 "No model named nothing" "$bin" model default nothing
check 2 "Usage: origami model list" "$bin" model frobnicate
check 2 "A model name is" "$bin" model add "$tmp/no-model" --name "../x"
check 1 "owner/../x (it needs config.json): fix the path" "$bin" model add "owner/../x" --yes
printf '{' > "$tmp/models.json"; mkdir -p "$tmp/data"; cp "$tmp/models.json" "$tmp/data/models.json"
check 1 "is not valid JSON" "$bin" model list
rm "$tmp/data/models.json"
# The index of a model folder is untrusted: a shard name may not leave the folder.
mkdir -p "$tmp/evil"; echo '{}' > "$tmp/evil/config.json"
echo '{"weight_map": {"a": "../other/model.safetensors"}}' > "$tmp/evil/model.safetensors.index.json"
check 1 "names a shard outside the folder" "$bin" inspect "$tmp/evil"
check 1 "No ready Origami server" "$bin" bench
check 1 "No ready Origami server" "$bin" tune
check 2 "--minutes needs a value" "$bin" tune --minutes
check 2 "--minutes is a number of minutes" "$bin" tune --minutes abc
check 2 "--minutes is 3 to 30" "$bin" tune --minutes 99
check 2 "--review is true or false" "$bin" tune --review maybe
check 2 "--tradeoffs is true or false" "$bin" tune --tradeoffs sure
# tune measures the server as it runs: a setting given to it would only mislabel the result.
check 2 "--context would only mislabel the result" "$bin" tune --context 32k
check 2 "--memory-limit would only mislabel the result" "$bin" tune --memory --memory-limit 20GB
check 0 '"memory_learned"' "$bin" status --json
# The memory rule in force: automatic and not measured; a measured limit for this model, Mac and context (`origami tune
# --memory`) is what automatic mode serves with; one for another model, or measured with another memory size, is not.
check 0 "Memory    automatic (not measured: run \`origami tune --memory\`)" "$bin" status
memory_result() { # memory_result <model_setting> <memory_bytes> <date>
    printf '{"entries":{"memory@x@8192":{"date":"%s","identity":{"mac":"%s","memory_bytes":%s},"model_setting":"%s","context":8192,"automatic_bytes":24100000000,"trials":[{"limit":22500000000,"first":9,"warm":9,"firstToken":3,"cache":1,"cuts":0,"swapped":0},{"limit":24100000000,"excluded":"cut 2 times","first":9,"warm":9,"firstToken":3,"cache":1,"cuts":2,"swapped":0}]}}}' \
        "$3" "$(sysctl -n hw.model)" "$2" "$1" > "$tmp/data/tuning.json"
}
now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
memory_result "$tmp/no-model" "$(sysctl -n hw.memsize)" "$now"
check 0 "Memory    automatic, using the measured 22.5 GB" "$bin" status
check 0 '"applied" : true' "$bin" status --json
check 0 "Memory    your limit 24.0 GB (ceiling)" "$bin" status --memory-limit 24GB
memory_result "$tmp/other-model" "$(sysctl -n hw.memsize)" "$now"
check 0 "Memory    automatic (not measured" "$bin" status
memory_result "$tmp/no-model" 1234567 "$now"
check 0 "measured (another memory size): run \`origami tune --memory\`" "$bin" status
memory_result "$tmp/no-model" "$(sysctl -n hw.memsize)" 2020-01-01T00:00:00Z
check 0 "(over 90 days old)" "$bin" status
# Cuts under the measured limit since it was measured make it stale only on three calendar days or more: 300 in one
# evening do not. A memory_limit's advice follows the same rule (a limit always starts at itself; no server here).
memory_cuts() { # memory_cuts <seconds ago>...
    { printf '{"identity":"%s %s","buckets":[],"trims":[' "$(sysctl -n hw.model)" "$(sysctl -n hw.memsize)"
      sep=; for ago in "$@"; do printf '%s{"t":%s,"to":20000000000,"limit":22500000000,"model":"%s","context":8192}' "$sep" "$(( $(date +%s) - ago ))" "$tmp/no-model"; sep=,; done
      printf ']}'; } > "$tmp/data/memory-history.json"
}
memory_result "$tmp/no-model" "$(sysctl -n hw.memsize)" "$(date -u -v-5d +%Y-%m-%dT%H:%M:%SZ)"
memory_cuts $(seq 3600 10 6590)
check 0 "Memory    automatic, using the measured 22.5 GB" "$bin" status
memory_cuts 3600 172800 345600
check 0 "(cut on 3 days since)" "$bin" status
rm -f "$tmp/data/tuning.json"
check 0 "could not be held on this Mac (cut 3 times)" "$bin" status --json --memory-limit 22.5GB
memory_cuts $(seq 3600 10 6590)
check 0 '"limit_advice" : null' "$bin" status --json --memory-limit 22.5GB
rm -f "$tmp/data/memory-history.json"
# What memory_auto_measure last did, for this model and context only.
printf '{"entries":{"memory-auto@'"$tmp/no-model"'@8192":{"date":"%s","line":"memory: automatic measurement stopped: a request from an app arrived; will try again tomorrow","model_setting":"%s","context":8192}}}' "$now" "$tmp/no-model" > "$tmp/data/tuning.json"
check 0 "Memory    automatic measurement stopped: a request from an app arrived; will try again tomorrow" "$bin" status
check 0 '"auto_measure" : "memory: automatic measurement stopped' "$bin" status --json
check 0 '"auto_measure" : null' "$bin" status --json --context 32k
# Why it has not run, kept with its time; another model's record is its own.
printf '{"entries":{"memory-auto@'"$tmp/no-model"'@8192":{"waiting":"the Mac is on battery","waiting_since":"%s","model_setting":"%s","context":8192},"memory-auto@other@8192":{"date":"%s","line":"memory: measured automatically while idle: 20.0 GB"}}}' "$now" "$tmp/no-model" "$now" > "$tmp/data/tuning.json"
check 0 "Memory    not measured yet: the Mac is on battery since" "$bin" status
check 0 '"auto_measure" : "memory: measured automatically while idle: 20.0 GB"' "$bin" status --json --model other
rm -f "$tmp/data/tuning.json"
check 0 '"tuning"' "$bin" status --json
check 0 "Speed not measured" "$bin" report
# Without a default model, status names the added ones (when no download of the model is found either).
printf '{"models":[{"name":"m-a","path":"%s/no-a"},{"name":"m-b","path":"%s/no-b"}]}' "$tmp" "$tmp" > "$tmp/data/models.json"
out=$(env -u ORIGAMI_MODEL HF_HUB_CACHE="$tmp/no-hub" "$bin" status 2>&1)
case "$out" in
    *"no default model; added: m-a, m-b (pass --model <name>, or \`origami model default <name>\`)"*) echo "ok    status names the added models" ;;
    *"Model     none found"* | *"no default model"*) echo "FAIL  status without a default model: $out"; failures=$((failures + 1)) ;;
    *"Model     "*) echo "skip  status without a default model (this Mac has a download of the model)" ;;
    *) echo "FAIL  status without a default model: $out"; failures=$((failures + 1)) ;;
esac
rm "$tmp/data/models.json"
# A store row says what the folder holds on the drive now, the Qwen hot copy in its `hot` subfolder included, in the
# hot copy line's words (it said "empty" beside "Hot copy 2889 experts").
if command -v python3 > /dev/null; then
    qwen_fixture "$tmp/hq" banks
    hot=$("$bin" status --model "$tmp/hq" --json | python3 -c 'import json, sys; r = json.load(sys.stdin); print(r["hot_copy"]["path"], "qwen %s %s" % (r["model"]["fingerprint"], r["model"]["quantization"]))')
    mkdir -p "${hot%% *}"; head -c 32768 /dev/zero > "${hot%% *}/experts.bin"
    printf '{"identity":"%s","keys":[3,9],"offsets":[0,16384]}' "${hot#* }" > "${hot%% *}/index.json"
    check 0 "free: hot copy 2 experts (0.0 GB)" "$bin" status --model "$tmp/hq"
    check 0 "Hot copy  2 experts (0.0 GB)" "$bin" status --model "$tmp/hq"
    check 0 "hot copy 2 experts (0.0 GB)" env ORIGAMI_MODEL="$tmp/hq" "$bin" store list
    # Saving the drives with `drive` on a place the same save removes (the data folder's store): that folder is now the
    # first place's, so delete_removed keeps its copy.
    ORIGAMI_STOPPED=true "$bin" serve --port $port_g --model "$tmp/hq" > "$tmp/serve-hq.log" 2>&1 & hqfront=$!
    for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_g/health > /dev/null 2>&1 && break; sleep 0.5; done
    hqsave() { curl -s -m 10 -X POST 127.0.0.1:$port_g/v1/hot-copy -H "Content-Type: application/json" -d "$1"; }
    check 0 '"saved":["hot_copy_more"]' hqsave "{\"places\":[{\"folder\":\"$tmp/data/store\"}]}"
    check 0 '"saved":["hot_copy_dir","hot_copy_more"]' hqsave "{\"drive\":\"$tmp\",\"places\":[],\"delete_removed\":true}"
    case "$hot" in "$tmp/data/store/"*) [ -f "${hot%% *}/index.json" ] ;; *) false ;; esac \
        && echo "ok    a removed place that becomes the first keeps its copy" \
        || { echo "FAIL  delete_removed deleted the first place's copy (${hot%% *})"; failures=$((failures + 1)); }
    kill "$hqfront"; wait "$hqfront" 2>/dev/null
    "$bin" config unset hot_copy_dir > /dev/null; "$bin" config unset hot_copy_more > /dev/null
    # With routing counts saved, the share of expert uses a copy of each size would hold (`HotCopyCurve`).
    fp=${hot#* qwen }; fp=${fp%% *}
    python3 -c 'import base64, json, struct, sys
counts = [0] * (48 * 512); counts[3] = 3; counts[9] = 1
raw = lambda v: base64.b64encode(struct.pack("<%dI" % len(v), *v)).decode()
json.dump({"windows": 48, "counts": raw(counts), "missed": raw([0] * len(counts))}, open(sys.argv[1], "w"))' "$tmp/data/routing-${fp:0:12}.json"
    check 0 "20.0 GB would cover 100%" "$bin" status --model "$tmp/hq"
    rm -f "$tmp/data/routing-${fp:0:12}.json"
    rm -rf "$tmp/hq" "$(dirname "${hot%% *}")"
else echo "skip  a store row with a Qwen hot copy (no python3)"; fi
# Low-bit experts on an exact model (`model add <experts.gguf> --base <model>`): a sparse 4-bit group-64 base and a sparse
# GGUF transformer part of the same checkpoint (zero routers, the same two tokens) pair; a GGUF whose layer 0 router
# differs, or whose header is cut short, is refused; removing the variant leaves the base's files and the GGUF.
if command -v python3 > /dev/null; then
    python3 - "$tmp" <<'PY'
import json, os, struct, sys
t = sys.argv[1]
g = {"hidden_size": 2560, "num_hidden_layers": 48, "vocab_size": 248320, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
     "shared_expert_intermediate_size": 640, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4,
     "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
     "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128, "indexer_compress_ratio": 4, "indexer_budget": 2048,
     "hc_count": 4, "hc_lowrank": 320, "heads_per_ngram": 8}
ple = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d"
q = dict({"bits": 4, "group_size": 64}, **{ple % k: {"bits": 4, "group_size": 32} for k in range(128)})
os.makedirs(t + "/vb")
json.dump(dict(g, model_type="qwen4_exp", quantization=q, quantization_config=q), open(t + "/vb/config.json", "w"))
json.dump({"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": []}, open(t + "/vb/tokenizer.json", "w"))
header, offset = {}, 0
def tensor(name, dtype, shape, size):
    global offset
    n = size
    for x in shape: n *= x
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}; offset += n
for layer in range(48):
    for proj, out, width in [("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)]:
        p = "language_model.model.layers.%d.mlp.switch_mlp.%s_proj." % (layer, proj)
        tensor(p + "weight", "U32", [512, out, width // 8], 4)
        tensor(p + "scales", "BF16", [512, out, width // 64], 2); tensor(p + "biases", "BF16", [512, out, width // 64], 2)
for layer in [0, 11, 23, 35, 47]:
    tensor("language_model.model.layers.%d.mlp.gate.weight" % layer, "BF16", [512, 2560], 2)
    tensor("language_model.model.layers.%d.mlp.shared_expert_gate.weight" % layer, "BF16", [1, 2560], 2)
for shard in range(128):
    tensor(ple % shard + ".weight", "U32", [2500012, 20], 4); tensor(ple % shard + ".scales", "BF16", [2500012, 5], 2)
    tensor(ple % shard + ".biases", "BF16", [2500012, 5], 2)
blob = json.dumps(header).encode()
with open(t + "/vb/model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(blob)) + blob); f.truncate(8 + len(blob) + offset)
def gguf(path, wrong):
    os.makedirs(os.path.dirname(path))
    s = lambda x: struct.pack("<Q", len(x.encode())) + x.encode()
    u = lambda k, v: s(k) + struct.pack("<II", 4, v)
    keys = [s("general.architecture") + struct.pack("<I", 8) + s("qwen4exp"), u("split.count", 2), u("qwen4exp.block_count", 48),
            u("qwen4exp.embedding_length", 2560), u("qwen4exp.expert_count", 512), u("qwen4exp.expert_used_count", 10),
            u("qwen4exp.expert_feed_forward_length", 640), u("qwen4exp.expert_shared_feed_forward_length", 640),
            u("qwen4exp.attention.head_count", 24), u("qwen4exp.attention.head_count_kv", 2),
            s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + s("a") + s("b")]
    infos, at = [], 0
    def info(name, dims, typ, nbytes):
        nonlocal at
        infos.append(s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in reversed(dims)) + struct.pack("<IQ", typ, at))
        at += (nbytes + 31) // 32 * 32
    for layer in [0, 11, 23, 35, 47]:
        info("blk.%d.ffn_gate_inp.weight" % layer, [512, 2560], 30, 512 * 2560 * 2)
        info("blk.%d.ffn_gate_inp_shexp.weight" % layer, [2560], 30, 2560 * 2)
    for layer in range(48):
        for proj, shape in [("gate", [512, 640, 2560]), ("up", [512, 640, 2560]), ("down", [512, 2560, 640])]:
            info("blk.%d.ffn_%s_exps.weight" % (layer, proj), shape, 42, shape[0] * shape[1] * shape[2] // 64 * 18)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(infos), len(keys)) + b"".join(keys) + b"".join(infos)
    head += b"\0" * ((32 - len(head) % 32) % 32)
    with open(path, "wb") as f:
        f.write(head + (b"\x40" if wrong else b"")); f.truncate(len(head) + at)
gguf(t + "/vx/x-00001-of-00002.gguf", False)
gguf(t + "/vw/w-00001-of-00002.gguf", True)
PY
    mkdir -p "$tmp/vt"; head -c 200 "$tmp/vx/x-00001-of-00002.gguf" > "$tmp/vt/t-00001-of-00002.gguf"
    check 0 "Added vbase" "$bin" model add "$tmp/vb" --name vbase
    qfp() { "$bin" status --model "$1" --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["model"]["fingerprint"][:12])'; }
    echo '{"windows": 1, "counts": "", "missed": ""}' > "$tmp/data/routing-$(qfp vbase).json"
    check 0 "with low-bit experts from x-00001-of-00002.gguf (Q2_0 ×144; 2.25 bits per weight)" "$bin" model add "$tmp/vx/x-00001-of-00002.gguf" --base vbase --name vq
    cmp -s "$tmp/data/routing-$(qfp vbase).json" "$tmp/data/routing-$(qfp vq).json" \
        && [ "$(stat -f %i "$tmp/data/routing-$(qfp vbase).json")" != "$(stat -f %i "$tmp/data/routing-$(qfp vq).json")" ] \
        && echo "ok    a new Qwen variant's routing counts are copied from its base's" \
        || { echo "FAIL  a new Qwen variant's routing counts"; failures=$((failures + 1)); }
    rm -f "$tmp/data"/routing-*.json
    [ -L "$tmp/data/variants/vq/model.safetensors" ] && grep -q '"expert_format" : "gguf"' "$tmp/data/variants/vq/config.json" \
        && python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["files"][0]["path"] != sys.argv[2])' \
            "$tmp/data/variants/vq/gguf-experts.json" "$tmp/vx/x-00001-of-00002.gguf" \
        && echo "ok    the variant's folder links the base and points into the GGUF" \
        || { echo "FAIL  the variant's folder"; failures=$((failures + 1)); }
    check 0 "vbase with low-bit experts" "$bin" model list
    check 0 "experts file $tmp/vx/x-00001-of-00002.gguf on " "$bin" model list
    check 0 "low-bit experts file)" "$bin" status --model vq
    check 0 "experts file: $tmp/vx/x-00001-of-00002.gguf on " "$bin" status --model vq
    check 0 "low-bit experts from x-00001-of-00002.gguf" "$bin" status --model vq
    check 0 "\"quantization\" : \"affine-4bit-g64\"" "$bin" status --model vbase --json
    check 1 "layer 0's router differs from the base's" "$bin" model add "$tmp/vw/w-00001-of-00002.gguf" --base vbase --name vw
    check 1 "Nothing was added: these experts need the exact model" "$bin" model add "$tmp/vw/w-00001-of-00002.gguf" --base vbase --name vw
    check 1 "Cannot read t-00001-of-00002.gguf as a GGUF" "$bin" model add "$tmp/vt/t-00001-of-00002.gguf" --base vbase --name vt
    check 1 "vbase already names a model" "$bin" model add "$tmp/vx/x-00001-of-00002.gguf" --base vbase --name vbase
    check 0 "untouched" "$bin" model remove vq
    [ ! -e "$tmp/data/variants/vq" ] && [ -f "$tmp/vb/model.safetensors" ] && [ -f "$tmp/vx/x-00001-of-00002.gguf" ] \
        && echo "ok    removing the variant keeps the base and the GGUF" \
        || { echo "FAIL  removing the variant"; failures=$((failures + 1)); }
    "$bin" model remove vbase > /dev/null; rm -rf "$tmp/vb" "$tmp/vx" "$tmp/vw" "$tmp/vt"
else echo "skip  low-bit experts on an exact model (no python3)"; fi
check 0 "\"version\"" "$bin" status --json
check 0 "\"source\"" "$bin" config --json
# Memory and drive settings, as `config set` and the dashboard's Apply take them.
check 0 "memory_auto_measure = true" "$bin" config set memory_auto_measure true
check 1 "is true or false" "$bin" config set memory_auto_measure sometimes
check 0 "memory_auto_measure = false" "$bin" config set memory_auto_measure false
check 1 "idle_unload_minutes is a number of minutes up to 10080 (7 days)" "$bin" config set idle_unload_minutes 1e19
# Writers in other processes never lose each other's changes (a lock around each read-modify-write): eight concurrent
# `config set` on different keys keep all eight, 20 times; two concurrent `model add` keep both entries.
race=0
for trial in $(seq 20); do
    rm -f "$tmp/race-config.json"
    for kv in "memory_auto_measure true" "hot_copy_keep true" "expert_memory 12GB" "temperature 0.5" "hot_copy off" \
        "quality fastest" "context_dir $tmp/race-context" "sampling custom"; do
        ORIGAMI_CONFIG="$tmp/race-config.json" "$bin" config set $kv > /dev/null 2>&1 &
    done
    wait
    python3 -c 'import json, sys; c = json.load(open(sys.argv[1])); sys.exit(len({"memory_auto_measure", "hot_copy_keep", "expert_memory", "temperature", "hot_copy", "quality", "context_dir", "sampling"} - set(c)))' \
        "$tmp/race-config.json" || race=$((race + 1))
done
[ "$race" -eq 0 ] && echo "ok    eight concurrent config set keep every key (20 trials)" \
    || { echo "FAIL  concurrent config set lost keys in $race of 20 trials"; failures=$((failures + 1)); }
qwen_fixture "$tmp/race-m1" banks; qwen_fixture "$tmp/race-m2" banks
race=0
for trial in 1 2 3; do
    for m in m1 m2; do ORIGAMI_DATA_DIR="$tmp/race-data$trial" ORIGAMI_CONFIG="$tmp/race-data$trial/config.json" "$bin" model add "$tmp/race-$m" --name "$m" > /dev/null 2>&1 & done
    wait
    python3 -c 'import json, sys; sys.exit(sorted(m["name"] for m in json.load(open(sys.argv[1]))["models"]) != ["m1", "m2"])' \
        "$tmp/race-data$trial/models.json" 2>/dev/null || race=$((race + 1))
done
[ "$race" -eq 0 ] && echo "ok    two concurrent model add keep both entries (3 trials)" \
    || { echo "FAIL  concurrent model add lost an entry in $race of 3 trials"; failures=$((failures + 1)); }
rm -rf "$tmp/race-config.json" "$tmp/.race-config.json.lock" "$tmp/race-context" "$tmp/race-m1" "$tmp/race-m2" "$tmp"/race-data*
# `config set` says when a change applies, and what overrides it.
check 0 "Applies now." "$bin" config set memory_auto_measure false
check 0 "Applies when the model next loads" "$bin" config set reasoning_effort low
check 0 'Applies when `origami serve` starts again' "$bin" config set listen local
check 0 "--reasoning-effort given here overrides it" "$bin" config set reasoning_effort low --reasoning-effort high
"$bin" config unset reasoning_effort > /dev/null; "$bin" config unset listen > /dev/null
# expert_bits is retired: refused when set, told (never applied) while a settings file still has it, and removable.
check 1 "expert_bits is no longer an option" "$bin" config set expert_bits 3
check 1 "origami model add <experts.gguf> --base <model>" "$bin" config set expert_bits 2
python3 -c 'import json, os, sys; p = sys.argv[1]; c = json.load(open(p)) if os.path.exists(p) else {}; c["expert_bits"] = 3; json.dump(c, open(p, "w"))' "$tmp/data/config.json"
check 0 "Settings  expert_bits 3 (settings file) is no longer an option" "$bin" status
check 1 "warn  expert_bits 3 (settings file) is no longer an option" "$bin" doctor
check 0 "expert_bits 3 (ORIGAMI_EXPERT_BITS) is no longer an option" env ORIGAMI_EXPERT_BITS=3 "$bin" status
check 0 "expert_bits reset to its default" "$bin" config unset expert_bits
"$bin" status 2>&1 | grep -q "expert_bits" && { echo "FAIL  status still names expert_bits once unset"; failures=$((failures + 1)); }
check 2 "serve does not take --expert-bits" "$bin" serve --expert-bits 3
qwen_fixture "$tmp/qbits"
check 1 "--expert-bits is for converting a GLM-5.3-Flash checkpoint" "$bin" model add "$tmp/qbits" --expert-bits 3
rm -rf "$tmp/qbits"
# 3- and 2-bit packs and converted GLM experts an earlier version made: listed as unused with their size and the command
# that deletes them, never read and never deleted here.
mkdir -p "$tmp/old/m-0123456789ab" "$tmp/old/family-0123456789abcdef/pack-Q4_K" "$tmp/old/family-0123456789abcdef/hot-12"
printf '{"format":"origami-store-v1","expert_file":"experts-3bit.bin"}' > "$tmp/old/m-0123456789ab/pack-3bit.json"
head -c 4096 /dev/zero > "$tmp/old/m-0123456789ab/experts-3bit.bin"
echo '{}' > "$tmp/old/family-0123456789abcdef/pack-Q4_K/pack.json"
check 0 "/old/m-0123456789ab/pack-3bit.json (0.0 GB): smaller experts an earlier Origami made" env ORIGAMI_STORES="$tmp/old" "$bin" status
check 0 "/old/m-0123456789ab/experts-3bit.bin'   (or in the dashboard's Models view)" env ORIGAMI_STORES="$tmp/old" "$bin" status
check 0 "delete: rm -r '$tmp/old/family-0123456789abcdef/pack-Q4_K'" env ORIGAMI_STORES="$tmp/old" "$bin" status
check 0 "delete: rm -r '$tmp/old/family-0123456789abcdef/hot-12'" env ORIGAMI_STORES="$tmp/old" "$bin" status
check 1 "warn  unused: $tmp/old/family-0123456789abcdef/pack-Q4_K" env ORIGAMI_STORES="$tmp/old" "$bin" doctor
check 0 '"unused_expert_files"' env ORIGAMI_STORES="$tmp/old" "$bin" status --json
[ -e "$tmp/old/m-0123456789ab/experts-3bit.bin" ] && [ -e "$tmp/old/family-0123456789abcdef/pack-Q4_K/pack.json" ] \
    && echo "ok    unused smaller experts are listed, not deleted" || { echo "FAIL  unused smaller experts were deleted"; failures=$((failures + 1)); }
rm -rf "$tmp/old"
check 0 "hot_copy = off" "$bin" config set hot_copy off
check 0 "off" "$bin" config get hot_copy
# More hot copy places (`hot_copy_more`): folder=size pairs, stored as a JSON list in bytes; one place per drive.
check 0 "hot_copy_more = $tmp/place=40GB" "$bin" config set hot_copy_more "$tmp/place=40GB"
check 0 "$tmp/place=40.0 GB" "$bin" config get hot_copy_more
python3 -c 'import json, sys; c = json.load(open(sys.argv[1])); assert json.loads(c["hot_copy_more"]) == [{"folder": sys.argv[2], "size": 40000000000}], c' \
    "$tmp/data/config.json" "$tmp/place" && echo "ok    hot_copy_more stored as a list in bytes" || { echo "FAIL  hot_copy_more stored"; failures=$((failures + 1)); }
check 0 "$tmp/place=auto" env ORIGAMI_HOT_COPY_MORE="$tmp/place" "$bin" config get hot_copy_more
check 1 "on the same drive" "$bin" config set hot_copy_more "$tmp/a=40GB,$tmp/b=auto"
check 1 "is a size such as 200GB" "$bin" config set hot_copy_more "$tmp/a=lots"
check 1 "needs full paths" "$bin" config set hot_copy_more "relative=40GB"
check 0 "$tmp/place=40.0 GB" "$bin" config get hot_copy_more
# Keep this size when idle (`hot_copy_keep`): a true/false setting, false by default.
check 0 "false" "$bin" config get hot_copy_keep
check 0 "hot_copy_keep = true" "$bin" config set hot_copy_keep true
check 1 "is true or false" "$bin" config set hot_copy_keep maybe
check 0 "off" "$bin" config get hot_copy
check 0 "hot_copy_more reset to its default" "$bin" config unset hot_copy_more
check 0 "(none)" "$bin" config get hot_copy_more
check 0 "expert_memory = 12GB" "$bin" config set expert_memory 12GB
check 0 "12" "$bin" config get expert_memory
check 1 "expert_memory is a size" "$bin" config set expert_memory lots
check 0 "context_dir = $tmp/context" "$bin" config set context_dir "$tmp/context"
check 0 "$tmp/context" "$bin" config get context_dir
# The quality dial: a level, knobs on top of it (custom), and setting a level again resets the knobs.
check 0 "quality = fastest" "$bin" config set quality fastest
check 0 "fastest" "$bin" config get quality
check 0 "exact" env ORIGAMI_ALLOW_FAST=false "$bin" config get quality  # a level allow_fast false refuses serves exact
check 0 "experts_per_token = 6" "$bin" config set experts_per_token 6
check 0 "custom (k6" "$bin" config get quality
# Sampling: the model's recommendation, or custom values over it, and whether they beat an app's own.
check 0 "sampling = custom" "$bin" config set sampling custom
check 0 "temperature = 0.5" "$bin" config set temperature 0.5
check 0 "0.5" "$bin" config get temperature
check 0 "(the model's)" "$bin" config get top_p
check 1 'temperature "3000" is not 0 to 100' "$bin" config set temperature 3000
check 1 'top_k "2.5" is not a whole number' "$bin" config set top_k 2.5
check 1 "sampling is recommended (the model's own) or custom" "$bin" config set sampling maybe
check 0 "sampling_override = true" "$bin" config set sampling_override true
check 1 "sampling_override is true or false" "$bin" config set sampling_override maybe
check 0 "exact" env ORIGAMI_QUALITY=exact "$bin" config get quality  # a file knob does not override a stronger level
check 1 "is not 1 to 10" "$bin" config set experts_per_token 11
check 1 "late_miss" "$bin" config set late_miss later
check 1 "quality is exact or a fast level the model offers (Qwen3.8-Flash-Next: faster, fastest;" "$bin" config set quality turbo
check 1 "is a probability" "$bin" config set confident_above 1.5
check 0 "quality = dynamic" "$bin" config set quality dynamic
check 0 "0.9" "$bin" config get confident_above
check 0 "confident_above = 0.8" "$bin" config set confident_above 0.8
check 0 "custom (t0.1m-c0.8" "$bin" config get quality
check 0 "quality = dynamic-fastest" "$bin" config set quality dynamic-fastest  # stored as named; each model runs its nearest
check 0 "skip" "$bin" config get late_miss
check 0 "quality = exact" "$bin" config set quality exact
check 0 "all" "$bin" config get experts_per_token
# A knob on a named model is kept in its entry and served over the entry's own level; a level unset there leaves a null
# that outranks nothing, so the file's knob (the global default) applies.
printf '{"models":[{"name":"knobbed","path":"%s/no-a","quality":"faster"}]}' "$tmp" > "$tmp/data/models.json"
check 0 "experts_per_token = 6" "$bin" config set experts_per_token 6 --model knobbed
check 0 "custom (k6" "$bin" config get quality --model knobbed
check 0 "all" "$bin" config get experts_per_token
check 0 "quality reset to its default" "$bin" config unset quality --model knobbed
grep -q '"quality" : null' "$tmp/data/models.json" || { echo "FAIL  config unset left no null in the model entry"; failures=$((failures + 1)); }
printf '{"models":[{"name":"knobbed","path":"%s/no-a","quality":null}]}' "$tmp" > "$tmp/data/models.json"
check 0 "experts_per_token = 5" "$bin" config set experts_per_token 5
check 0 "custom (k5" "$bin" config get quality --model knobbed
check 0 "experts_per_token reset to its default" "$bin" config unset experts_per_token
rm "$tmp/data/models.json"
# A bad value in the settings file: the error names the file, and `config unset` still removes it without a full load.
setkey() { python3 -c 'import json, sys; p = sys.argv[1]; d = json.load(open(p)); d[sys.argv[2]] = sys.argv[3]; json.dump(d, open(p, "w"))' "$tmp/data/config.json" "$1" "$2"; }
setkey quality turbo
check 1 "(quality in $tmp/data/config.json)" "$bin" config get context
check 0 "quality removed from $tmp/data/config.json" "$bin" config unset quality
check 0 "8k" "$bin" config get context
# A key no setting reads is named by `config` and `doctor`, and `config unset` removes it.
setkey contxt 32k
check 0 "holds contxt, which no setting reads" "$bin" config
"$bin" doctor 2>&1 | grep -q "warn  settings: .*holds contxt" && echo "ok    doctor warns about an unknown settings key" || { echo "FAIL  doctor does not warn about an unknown settings key"; failures=$((failures + 1)); }
check 0 "contxt reset to its default" "$bin" config unset contxt
"$bin" config 2>&1 | grep -q "which no setting reads" && { echo "FAIL  config warns about a key that is gone"; failures=$((failures + 1)); }
# A new data folder while the old one holds models.json: refused without --yes, then what stays behind is named.
mkdir -p "$tmp/olddata"; printf '{"models":[]}' > "$tmp/olddata/models.json"
check 1 "holds models.json" env ORIGAMI_DATA_DIR="$tmp/olddata" "$bin" config set data_dir "$tmp/newdata"
check 0 "Left in $tmp/olddata: models.json" env ORIGAMI_DATA_DIR="$tmp/olddata" "$bin" config set data_dir "$tmp/newdata" --yes
check 0 "data_dir reset to its default" "$bin" config unset data_dir
# The front starts its model in a child; a model that cannot start leaves `serve` up, saying why (the dashboard fixes it),
# and the front itself answers the page, the model search and the jobs.
store_fixture "$tmp/data/store/p-0123456789ab" pack
"$bin" serve --port $port_a --model "$tmp/no-model" > "$tmp/serve.log" 2>&1 &
front=$!
for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_a/health 2>/dev/null | grep -q '"failed"' && break; sleep 0.5; done
check 0 "No model at" curl -s -m 5 127.0.0.1:$port_a/health
# A running server keeps commands that rewrite its files out, saying how to quit it (`origami stop` is not enough).
check 1 "Run \`origami stop\` (no model loads until \`origami start\`)" "$bin" cache prune --port $port_a
"$bin" store add "$tmp/sr" > /dev/null 2>&1
check 1 "Quit it (Ctrl-C where \`origami serve\` runs" "$bin" store remove "$tmp/sr" --port $port_a
"$bin" config unset stores > /dev/null
# A second server on a port in use says so and names the way out.
check 1 "Port $port_a is already in use" "$bin" serve --port $port_a --model "$tmp/no-model"
# Until the front's first search for models on this Mac ends, a pack of a model not registered may be that model's:
# its removal is refused (the first ask starts that search, without waiting for it).
out=$(curl -s -m 20 -w ' %{http_code}' -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/store/p-0123456789ab/pack.json\"}")
case "$out" in *"may be on this Mac: the search for models on this Mac has not finished"*" 409") echo "ok    a removal before the model search ends is refused" ;;
    *) echo "FAIL  a removal before the model search ends: $out"; failures=$((failures + 1)) ;; esac
[ -f "$tmp/data/store/p-0123456789ab/pack.json" ] || { echo "FAIL  a pack was removed before the model search ended"; failures=$((failures + 1)); }
rm -rf "$tmp/data/store"
check 0 '<title>Origami</title>' curl -s -m 5 127.0.0.1:$port_a/dashboard
check 0 "image/svg+xml" curl -s -m 5 -D - -o /dev/null 127.0.0.1:$port_a/favicon.ico
# Added to a home screen: the manifest and the icons, without the key, cached; the PNGs opaque at their sizes.
check 0 '"start_url": "./dashboard"' curl -s -m 5 127.0.0.1:$port_a/manifest.webmanifest
check 0 "Content-Type: application/manifest+json" curl -s -m 5 -D - -o /dev/null 127.0.0.1:$port_a/manifest.webmanifest
check 0 "Cache-Control: max-age=86400" curl -s -m 5 -D - -o /dev/null 127.0.0.1:$port_a/icon-192.png
check 0 "manifest-src 'self'" curl -s -m 5 -D - -o /dev/null 127.0.0.1:$port_a/dashboard
check 0 'rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png"' curl -s -m 5 127.0.0.1:$port_a/dashboard
for icon in apple-touch-icon:180 apple-touch-icon-precomposed:180 icon-192:192 icon-512:512; do
    curl -s -m 5 -o "$tmp/icon.png" "127.0.0.1:$port_a/${icon%%:*}.png"
    check 0 "pixelWidth: ${icon##*:}" sips -g pixelWidth "$tmp/icon.png"
    check 0 "hasAlpha: no" sips -g hasAlpha "$tmp/icon.png"
done
check 0 '"job":null' curl -s -m 5 127.0.0.1:$port_a/v1/jobs
# Tuning needs a loaded model before any mode can run; the learned memory budget can be forgotten without one.
check 0 'No model is running' curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d '{"kind":"tune","mode":"fast"}'
check 0 'No model is running' curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d '{"kind":"tune","mode":"tune"}'
check 1 'No ready Origami server: ' "$bin" tune --port $port_a
check 1 'see `origami status`' "$bin" tune --port $port_a  # the state it is in (here a model that cannot start)
check 0 '"reset":true' curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/memory/learned -H "Content-Type: application/json" -d '{}'
check 0 '"status":403' curl -s -m 5 -w '"status":%{http_code}' -X POST 127.0.0.1:$port_a/v1/memory/learned -H "Content-Type: application/json" -H "Origin: http://evil.example" -d '{}'
# What harnesses read to offer a thinking-effort menu: each model's distinct efforts (Qwen's here), parameters, context and capabilities.
check 0 '"reasoning_efforts":["none","low","medium","high"]' curl -s -m 5 127.0.0.1:$port_a/v1/models
check 0 '"capabilities":["completion","tools","thinking","vision"],"context_length":8192' curl -s -m 5 127.0.0.1:$port_a/v1/models
check 0 '"supported_parameters":["reasoning","reasoning_effort","include_reasoning","tools"' curl -s -m 5 127.0.0.1:$port_a/v1/models
check 0 '"id":"origami"' curl -s -m 5 127.0.0.1:$port_a/v1/models/origami
# Each model's sampling defaults per mode in the front's listing: a folder's generation_config.json (thinking alone, as
# Qwen's), the custom temperature over every mode, and the override. Its config.json is empty, so no model loads.
mkdir -p "$tmp/gc"; echo '{}' > "$tmp/gc/config.json"; echo '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}' > "$tmp/gc/generation_config.json"
"$bin" serve --port $port_b --model "$tmp/gc" > "$tmp/serve-gc.log" 2>&1 & gc=$!
for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_b/v1/models 2>/dev/null | grep -q '"sampling"' && break; sleep 0.5; done
check 0 '"recommended":{"non_thinking":{"min_p":0,"presence_penalty":1.5,"temperature":0.7,"top_k":20,"top_p":0.8},"source":"generation_config.json","thinking":{"min_p":0,"presence_penalty":0,"temperature":0.6' curl -s -m 5 127.0.0.1:$port_b/v1/models
check 0 '"defaults":{"non_thinking":{"min_p":0,"presence_penalty":1.5,"temperature":0.5,"top_k":20,"top_p":0.8},"source":"custom","thinking":{"min_p":0,"presence_penalty":0,"temperature":0.5' curl -s -m 5 127.0.0.1:$port_b/v1/models
check 0 '"override":true' curl -s -m 5 127.0.0.1:$port_b/v1/models
kill "$gc"; wait "$gc" 2>/dev/null
for key in sampling temperature sampling_override; do "$bin" config unset "$key" > /dev/null; done
check 0 '"model_not_found"' curl -s -m 5 127.0.0.1:$port_a/v1/models/nothing
check 0 '"capabilities":["completion","tools","thinking","vision"]' curl -s -m 5 127.0.0.1:$port_a/api/tags
check 0 '"unreadable"' curl -s -m 30 127.0.0.1:$port_a/v1/models/found
check 0 '"refused"' curl -s -m 10 -G 127.0.0.1:$port_a/v1/models/check --data-urlencode "source=$tmp/nothing-here"
# A conversion is judged by what it writes and 10 GB to spare, by the page's check as by `model add`: a GLM checkpoint
# (headers only, its weights a sparse file) whose GGUF is 9.3 TB fits on no drive, and both name that size.
if command -v python3 > /dev/null; then
    python3 - "$tmp/glm-big" <<'PY'
import json, os, struct, sys
d = sys.argv[1]; os.makedirs(d)
json.dump({"architectures": ["Glm5NextForCausalLM"], "model_type": "glm5_next", "num_hidden_layers": 1, "hidden_size": 32,
           "vocab_size": 1 << 24, "n_routed_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 32,
           "layer_types": ["linear_attention"], "mlp_layer_types": ["sparse"]}, open(d + "/config.json", "w"))
open(d + "/chat_template.jinja", "w").write("{{ messages }}")
p = "model.layers.0."
small = ["model.norm.weight"] + [p + n for n in ["input_layernorm.weight", "post_attention_layernorm.weight", "hc_attn_fn", "hc_attn_base",
    "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
    + ["self_attn." + n for n in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
        "f_a_proj.weight", "f_b_proj.weight", "dt_bias", "A_log", "b_proj.weight", "g_a_proj.weight", "g_b_proj.weight", "o_norm.weight", "o_proj.weight"]]
    + ["mlp.shared_experts.%s_proj.weight" % m for m in ["gate", "up", "down"]]]
tensors = [("model.embed_tokens.weight", "BF16", [1 << 24, 1 << 18], 2)] + [(n, "F32", [32, 32], 4) for n in small] \
    + [(p + "mlp.experts.%d.%s_proj.weight" % (e, m), "F32", [32, 256], 4) for e in range(2) for m in ["gate", "up", "down"]]
header, offset = {}, 0
for name, dtype, shape, size in tensors:
    n = shape[0] * shape[1] * size; header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}; offset += n
blob = json.dumps(header).encode()
with open(d + "/model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(blob)) + blob); f.truncate(8 + len(blob) + offset)
PY
    check 0 '"needs_bytes":9355848930816' curl -s -m 20 -G 127.0.0.1:$port_a/v1/models/check --data-urlencode "source=$tmp/glm-big"
    check 0 '"fits":false' curl -s -m 20 -G 127.0.0.1:$port_a/v1/models/check --data-urlencode "source=$tmp/glm-big"
    check 1 "Converting needs 9355.8 GB free there" "$bin" model add "$tmp/glm-big"
else echo "skip  the GLM conversion's room (no python3)"; fi
check 0 "own page" curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/jobs -H "Origin: http://example.com" -H "Content-Type: application/json" -d '{"kind":"add","source":"a/b"}'
check 0 "source must be" curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d '{"kind":"add","source":"--yes"}'
# A job runs `model add` and reports how it ended.
curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d "{\"kind\":\"add\",\"source\":\"$tmp/no-model\"}" > /dev/null
for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_a/v1/jobs | grep -q '"running"' || break; sleep 0.5; done
check 0 '"failed"' curl -s -m 5 127.0.0.1:$port_a/v1/jobs
check 0 '"started"' curl -s -m 5 127.0.0.1:$port_a/v1/jobs
# The finished job stays, with when it ended, and the server log has one line for its end.
check 0 '"ended"' curl -s -m 5 127.0.0.1:$port_a/v1/jobs
check 0 "origami: adding no-model: failed: " cat "$tmp/serve.log"
# The folder browser: folders and, for a model, .gguf files (a split one by its first part) and checkpoint folders under
# Origami's places; hidden entries and links leading out are left out, as is anything outside them. A new folder is made
# only from the page, inside a place, with a plain name.
mkdir -p "$tmp/data/tree/sub" "$tmp/data/tree/.hidden" "$tmp/data/tree/ckpt" "$tmp/outside"
echo '{}' > "$tmp/data/tree/ckpt/config.json"; touch "$tmp/data/tree/ckpt/model.safetensors" "$tmp/data/tree/m.gguf" "$tmp/data/tree/m-00002-of-00002.gguf"
ln -s "$tmp/outside" "$tmp/data/tree/away"
lacks() { # lacks <text the output must not contain> <command...>
    needle=$1; shift
    out=$("$@" 2>&1)
    case "$out" in *"$needle"*) echo "FAIL  $* shows \"$needle\": $out"; failures=$((failures + 1)) ;; *) echo "ok    $* (without $needle)" ;; esac
}
fs() { curl -s -m 10 -G 127.0.0.1:$port_a/v1/fs "$@"; }
mkdir_() { curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/fs/mkdir -H "Content-Type: application/json" "$@"; }
check 0 '"Origami data"' fs
check 0 '"name":"sub"' fs --data-urlencode "path=$tmp/data/tree"
lacks '.hidden' fs --data-urlencode "path=$tmp/data/tree"
lacks '"name":"away"' fs --data-urlencode "path=$tmp/data/tree"
lacks 'm.gguf' fs --data-urlencode "path=$tmp/data/tree"
check 0 '"name":"m.gguf"' fs --data-urlencode "path=$tmp/data/tree" --data-urlencode models=
check 0 '"model":true,"name":"ckpt"' fs --data-urlencode "path=$tmp/data/tree" --data-urlencode models=
lacks 'm-00002' fs --data-urlencode "path=$tmp/data/tree" --data-urlencode models=
check 0 "Origami's folders only" fs --data-urlencode "path=$tmp/outside"
check 0 "Origami's folders only" fs --data-urlencode "path=$tmp/data/tree/away"
check 0 "Origami's folders only" fs --data-urlencode "path=$tmp/data/tree/../../outside"
check 0 "No folder at" fs --data-urlencode "path=$tmp/data/nothing"
# A hidden folder the listing leaves out cannot be opened or written by name either; a file is not a folder (not a 500).
check 0 "Origami's folders only" fs --data-urlencode "path=$tmp/data/tree/.hidden"
check 0 "is not a folder" fs --data-urlencode "path=$tmp/data/tree/m.gguf"
check 0 "inside a drive" mkdir_ -d "{\"path\":\"$tmp/data/tree/.hidden/x\"}"
check 0 '"name":"new folder"' mkdir_ -d "{\"path\":\"$tmp/data/tree/new folder\"}"
[ -d "$tmp/data/tree/new folder" ] || { echo "FAIL  mkdir made no folder"; failures=$((failures + 1)); }
# The page encodes a space in a query as +, as web forms do ("Application Support").
check 0 '"name":"new folder"' curl -s -m 10 "127.0.0.1:$port_a/v1/fs?path=$(printf %s "$tmp/data/tree/new folder" | sed 's/ /+/g')"
check 0 "already exists" mkdir_ -d "{\"path\":\"$tmp/data/tree/new folder\"}"
check 0 "not starting with a dot" mkdir_ -d "{\"path\":\"$tmp/data/tree/.secret\"}"
check 0 "not starting with a dot" mkdir_ -d "{\"path\":\"$tmp/data/tree/a:b\"}"
check 0 "full path" mkdir_ -d "{\"path\":\"$tmp/data/tree/sub/../x\"}"
check 0 "full path" mkdir_ -d '{"path":"relative/x"}'
check 0 "inside a drive" mkdir_ -d "{\"path\":\"$tmp/outside/x\"}"
check 0 "inside a drive" mkdir_ -d "{\"path\":\"$tmp/data/tree/away/x\"}"
check 0 "own page" mkdir_ -H "Origin: http://example.com" -d "{\"path\":\"$tmp/data/tree/x\"}"
check 0 "own page" curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/fs/mkdir -d "{\"path\":\"$tmp/data/tree/x\"}"
[ -e "$tmp/data/tree/x" ] || [ -e "$tmp/outside/x" ] || [ -e "$tmp/data/tree/.hidden/x" ] && { echo "FAIL  a refused mkdir made a folder"; failures=$((failures + 1)); }
# The folders the page sets go through the config writer; a job's destination is a full path.
check 0 '"saved":["models_dirs"]' curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d "{\"models_dirs\":\"$tmp/data/tree\"}"
check 0 '"folders":["' curl -s -m 30 "127.0.0.1:$port_a/v1/models/found?fresh"
check 0 '"Model search"' fs
check 0 "needs full paths" curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d '{"stores":"relative/store"}'
check 0 '"destination"' curl -s -m 10 -G 127.0.0.1:$port_a/v1/models/check --data-urlencode "source=$tmp/nothing-here" --data-urlencode "to=$tmp/data"
check 0 "to must be a full folder path" curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d '{"kind":"add","source":"a/b","to":"rel"}'
"$bin" config unset models_dirs > /dev/null
# A GLM GGUF of experts only (Q8_0, headers and zeros): a hot copy is never put on its drive; a retired expert_bits is
# refused from the dashboard and only ever removed there; a named model keeps its own settings.
if command -v python3 > /dev/null; then
    python3 - "$tmp/fam.gguf" <<'PY'
import struct, sys
out = bytearray(b"GGUF"); u32 = lambda v: out.extend(struct.pack("<I", v)); u64 = lambda v: out.extend(struct.pack("<Q", v))
def s(t): u64(len(t)); out.extend(t.encode())
meta = [("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256), ("glm5-next.vocab_size", 512), ("glm5-next.expert_count", 4),
        ("glm5-next.expert_used_count", 1), ("glm5-next.expert_feed_forward_length", 256), ("glm5-next.leading_dense_block_count", 1),
        ("glm5-next.block_count", 3), ("glm5-next.attention.head_count_kv", [0, 64, 64]), ("glm5-next.hyper_connection.count", 4)]
tensors = ["blk.%d.ffn_%s_exps.weight" % (l, p) for l in (1, 2) for p in ("gate", "up", "down")]
size = 4 * 256 * 256 // 32 * 34
u32(3); u64(len(tensors)); u64(len(meta))
for k, v in meta:
    s(k)
    if isinstance(v, str): u32(8); s(v)
    elif isinstance(v, list): u32(9); u32(5); u64(len(v)); [out.extend(struct.pack("<i", x)) for x in v]
    else: u32(4); u32(v)
for i, name in enumerate(tensors): s(name); u32(3); [u64(d) for d in (256, 256, 4)]; u32(8); u64(i * size)
out.extend(bytes((32 - len(out) % 32) % 32)); open(sys.argv[1], "wb").write(bytes(out) + bytes(size * len(tensors)))
PY
    check 0 "expert_bits is no longer an option" curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d "{\"model\":\"$tmp/fam.gguf\",\"expert_bits\":\"3\"}"
    check 0 "a hot copy on the model's own drive is disabled by default" curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d "{\"model\":\"$tmp/fam.gguf\",\"hot_copy_dir\":\"$tmp/hcx\"}"
    echo "{\"models\": [{\"name\": \"fam\", \"family\": \"glm\", \"path\": \"$tmp/fam.gguf\"}]}" > "$tmp/data/models.json"
    check 0 '"saved":["expert_bits","model"]' curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d '{"model":"fam","expert_bits":""}'
    model_setting() { # one model's setting and the independent global default, compared as JSON values
        python3 - "$tmp/data/models.json" "$tmp/data/config.json" "$1" "$2" "$3" <<'PY'
import json, sys
models, config = (json.load(open(p)) for p in sys.argv[1:3])
entry = next(m for m in models["models"] if m["name"] == "fam")
key = sys.argv[3]
assert key in entry and entry[key] == json.loads(sys.argv[4]), (key, entry)
assert config[key] == json.loads(sys.argv[5]), (key, config)
print("model override and global default preserved")
PY
    }
    # A named model's entry that still holds expert_bits (written by an earlier version): told, and unset removes it there.
    python3 -c 'import json, sys; p = sys.argv[1]; m = json.load(open(p)); m["models"][0]["expert_bits"] = 3; json.dump(m, open(p, "w"))' "$tmp/data/models.json"
    check 0 "expert_bits 3 (model entry) is no longer an option" env ORIGAMI_MODEL=fam "$bin" status
    check 0 "reset to its default" env ORIGAMI_MODEL=fam "$bin" config unset expert_bits
    grep -q expert_bits "$tmp/data/models.json" && { echo "FAIL  config unset left expert_bits in the model entry"; failures=$((failures + 1)); }
    # An explicit null resets this model to its own recommendation; global defaults remain available to others.
    "$bin" config set temperature 0.9 > /dev/null
    check 0 '"saved":["model","sampling","sampling_override","temperature"]' curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d '{"model":"fam","sampling":"custom","temperature":"0.4","sampling_override":"true"}'
    check 0 "model override and global default preserved" model_setting temperature 0.4 0.9
    check 0 "custom" env ORIGAMI_MODEL=fam "$bin" config get sampling
    check 0 "true" env ORIGAMI_MODEL=fam "$bin" config get sampling_override
    check 0 "0.4" env ORIGAMI_MODEL=fam "$bin" config get temperature
    check 0 '"saved":["model","temperature"]' curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d '{"model":"fam","temperature":""}'
    check 0 "(the model's)" env ORIGAMI_MODEL=fam "$bin" config get temperature
    "$bin" config unset model > /dev/null; rm -f "$tmp/data/models.json"
fi
# doctor and status for a registered GLM GGUF that serving accepts (headers and zeros): no problem, its rows; the hot copy's
# line once a copy is there; a part missing is a problem. Lines from a server show only when it serves the model asked about.
if command -v python3 > /dev/null; then
    fam_gguf() { # fam_gguf <file> [parts]: Q8_0 experts and the tensors serving checks for; with parts, the first of that many
        python3 - "$@" <<'PY'
import os, struct, sys
out = bytearray(b"GGUF"); u32 = lambda v: out.extend(struct.pack("<I", v)); u64 = lambda v: out.extend(struct.pack("<Q", v))
def s(t): u64(len(t)); out.extend(t.encode())
meta = [("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256), ("glm5-next.vocab_size", 512), ("glm5-next.expert_count", 4),
        ("glm5-next.expert_used_count", 1), ("glm5-next.expert_feed_forward_length", 256), ("glm5-next.leading_dense_block_count", 1),
        ("glm5-next.block_count", 3), ("glm5-next.attention.head_count_kv", [0, 64, 64]), ("glm5-next.hyper_connection.count", 4)]
if len(sys.argv) > 2: meta.append(("split.count", int(sys.argv[2])))
meta = [m for m in meta if m[0] != os.environ.get("FAM_DROP")]  # a key left out: a GGUF Origami refuses
experts = ["blk.%d.ffn_%s_exps.weight" % (l, p) for l in (1, 2) for p in ("gate", "up", "down")]
small = ["token_embd.weight", "output.weight", "blk.0.ffn_norm.weight"]
size, little = 4 * 256 * 256 // 32 * 34, 1024
u32(3); u64(len(experts) + len(small)); u64(len(meta))
for k, v in meta:
    s(k)
    if isinstance(v, str): u32(8); s(v)
    elif isinstance(v, list): u32(9); u32(5); u64(len(v)); [out.extend(struct.pack("<i", x)) for x in v]
    else: u32(4); u32(v)
for i, name in enumerate(experts): s(name); u32(3); [u64(d) for d in (256, 256, 4)]; u32(8); u64(i * size)
for i, name in enumerate(small): s(name); u32(1); u64(256); u32(0); u64(len(experts) * size + i * little)
out.extend(bytes((32 - len(out) % 32) % 32)); open(sys.argv[1], "wb").write(bytes(out) + bytes(size * len(experts) + little * len(small)))
PY
    }
    mkdir -p "$tmp/fm"; fam_gguf "$tmp/fm/glm.gguf"; fam_gguf "$tmp/fm/two-00001-of-00002.gguf" 2
    head -c 4000 "$tmp/fm/glm.gguf" > "$tmp/fm/cut.gguf"; check 1 "is shorter than its header says" "$bin" inspect "$tmp/fm/cut.gguf"; rm "$tmp/fm/cut.gguf"
    echo "{\"models\": [{\"name\": \"glm\", \"family\": \"glm\", \"path\": \"$tmp/fm/glm.gguf\"}, {\"name\": \"two\", \"family\": \"glm\", \"path\": \"$tmp/fm/two-00001-of-00002.gguf\"}]}" > "$tmp/data/models.json"
    check 0 "ok    model: glm, GLM-5.3-Flash (GGUF" env ORIGAMI_MODEL=glm "$bin" doctor
    check 0 "ok    hot copy: " env ORIGAMI_MODEL=glm "$bin" doctor
    check 0 "ok    memory: up to " env ORIGAMI_MODEL=glm "$bin" doctor
    check 0 "ok    drive speed: " env ORIGAMI_MODEL=glm "$bin" doctor
    env ORIGAMI_MODEL=glm "$bin" doctor 2>&1 | grep -q "is a GGUF" && { echo "FAIL  doctor sends a family GGUF to conversion"; failures=$((failures + 1)); }
    check 0 "Model     glm, GLM-5.3-Flash (GGUF" env ORIGAMI_MODEL=glm "$bin" status
    key=$(ORIGAMI_MODEL=glm "$bin" status --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["model"]["key"])')
    mkdir -p "$tmp/data/family/$key/hot"; head -c 212992 /dev/zero > "$tmp/data/family/$key/hot/experts.bin"
    echo "{\"identity\": \"$key|expert-source\", \"keys\": [4], \"offsets\": [0]}" > "$tmp/data/family/$key/hot/index.json"
    echo '{"4": 10, "5": 5}' > "$tmp/data/family/$key/uses.json"
    check 0 "Hot copy  1 experts (0.0 GB) in $tmp/data/family/$key/hot" env ORIGAMI_MODEL=glm "$bin" status
    check 0 "a copy of 0.0 GB would cover 67%" env ORIGAMI_MODEL=glm "$bin" status
    check 1 "fix   model: $tmp/fm/two-00001-of-00002.gguf: Origami cannot read this GGUF: two has 1 of its 2 parts" env ORIGAMI_MODEL=two "$bin" doctor
    # A server on the port: its slow-reads line only when it serves this model.
    echo '{"status": "ready", "model": {"name": "other", "path": "/elsewhere/other.gguf"}}' > "$tmp/fake-status.json"
    python3 -c '
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = open(sys.argv[2], "rb").read()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()' $port_c "$tmp/fake-status.json" &
    fake=$!
    for _ in $(seq 20); do curl -s -m 1 127.0.0.1:$port_c/health > /dev/null && break; sleep 0.2; done
    check 0 "info  a server on port $port_c is serving other" env ORIGAMI_MODEL=glm ORIGAMI_PORT=$port_c "$bin" doctor
    echo "{\"status\": \"ready\", \"model\": {\"name\": \"glm\", \"path\": \"$tmp/fm/glm.gguf\"}, \"experts_from\": {\"sources\": [{\"role\": \"all\", \"bytes\": 1, \"slow_reads\": {\"share\": 0.01, \"busy_seconds\": 100}}]}}" > "$tmp/fake-status.json"
    check 0 "ok    model drive: 1% of 100 s of reading" env ORIGAMI_MODEL=glm ORIGAMI_PORT=$port_c "$bin" doctor
    kill "$fake"; wait "$fake" 2> /dev/null
    # Saved prompts of GLM and DeepSeek: each epoch folder's marker names its model; another model's (or build's) epoch is
    # listed on its own and pruned; the current one and one without a marker stay.
    build=$(shasum -a 256 "$bin" | cut -c1-16)
    mkdir -p "$tmp/context/aaaaaaaaaaaaaaaa" "$tmp/context/bbbbbbbbbbbbbbbb" "$tmp/context/cccccccccccccccc"
    echo "{\"model\": \"glm\", \"family_key\": \"$key\", \"context\": 8192, \"version\": \"x\", \"build\": \"$build\"}" > "$tmp/context/aaaaaaaaaaaaaaaa/origami-epoch.json"
    echo "{\"model\": \"gone\", \"family_key\": \"0123456789abcdef\", \"context\": 8192, \"version\": \"3.0.1\", \"build\": \"$build\"}" > "$tmp/context/bbbbbbbbbbbbbbbb/origami-epoch.json"
    check 0 "bbbbbbbbbbbbbbbb (gone, 8192 context, version 3.0.1)" "$bin" cache
    check 0 "(saved prompts of gone from version 3.0.1; safe to delete)" "$bin" cache
    "$bin" cache | grep "aaaaaaaaaaaaaaaa\|cccccccccccccccc" | grep -q "safe to delete" && { echo "FAIL  cache lists a current epoch as unused"; failures=$((failures + 1)); }
    check 0 "Removes bbbbbbbbbbbbbbbb" "$bin" cache prune
    [ -e "$tmp/context/bbbbbbbbbbbbbbbb" ] || [ ! -e "$tmp/context/aaaaaaaaaaaaaaaa" ] || [ ! -e "$tmp/context/cccccccccccccccc" ] \
        && { echo "FAIL  cache prune removed a current epoch or kept another model's"; failures=$((failures + 1)); }
    # `setup --model <file>.gguf` makes it the model, as for a Qwen folder: `status` then names it.
    rm -f "$tmp/data/models.json"
    "$bin" setup --model "$tmp/fm/glm.gguf" > /dev/null 2>&1
    check 0 "GLM-5.3-Flash (GGUF" env -u ORIGAMI_MODEL "$bin" status
    # A GGUF whose setup fails is never saved as the default: the one that worked stays.
    FAM_DROP=glm5-next.block_count fam_gguf "$tmp/fm/nogeo.gguf"
    out=$(env -u ORIGAMI_MODEL "$bin" setup --model "$tmp/fm/nogeo.gguf" 2>&1); [ $? -eq 1 ] && case "$out" in *"No model folder"*) false ;; esac || { echo "FAIL  a GGUF without its geometry set up: $out"; failures=$((failures + 1)); }
    grep -q "nogeo" "$ORIGAMI_CONFIG" && { echo "FAIL  a failed GGUF setup was saved as the default"; failures=$((failures + 1)); }
    "$bin" config unset model > /dev/null
    rm -rf "$tmp/context/aaaaaaaaaaaaaaaa" "$tmp/context/cccccccccccccccc" "$tmp/fm" "$tmp/data/family/$key" "$tmp/fake-status.json" "$tmp/data/models.json"
else echo "skip  doctor and status of a family model (no python3)"; fi
# Low-bit experts on an exact GLM GGUF (`model add <experts.gguf> --base <model>`; headers and zeros): a whole GGUF with
# Q2_K banks pairs only with --yes (it carries no tokenizer or routers to compare) and is listed and shown as such; banks
# of another shape are refused; removing the variant keeps both files.
if command -v python3 > /dev/null; then
    lowbit_gguf() { # lowbit_gguf <file> <ggml type> <experts per bank>: a GLM GGUF serving accepts, its banks of that type
        python3 - "$@" <<'PY'
import struct, sys
path, typ, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
block, size = {8: (32, 34), 10: (256, 84)}[typ]
out = bytearray(b"GGUF"); u32 = lambda v: out.extend(struct.pack("<I", v)); u64 = lambda v: out.extend(struct.pack("<Q", v))
def s(t): u64(len(t)); out.extend(t.encode())
meta = [("general.architecture", "glm5-next"), ("glm5-next.embedding_length", 256), ("glm5-next.vocab_size", 512), ("glm5-next.expert_count", 4),
        ("glm5-next.expert_used_count", 1), ("glm5-next.expert_feed_forward_length", 256), ("glm5-next.leading_dense_block_count", 1),
        ("glm5-next.block_count", 3), ("glm5-next.attention.head_count_kv", [0, 64, 64]), ("glm5-next.hyper_connection.count", 4)]
experts = ["blk.%d.ffn_%s_exps.weight" % (l, p) for l in (1, 2) for p in ("gate", "up", "down")]
small = ["token_embd.weight", "output.weight", "blk.0.ffn_norm.weight"]
bank, little = n * 256 * 256 // block * size, 1024
u32(3); u64(len(experts) + len(small)); u64(len(meta))
for k, v in meta:
    s(k)
    if isinstance(v, str): u32(8); s(v)
    elif isinstance(v, list): u32(9); u32(5); u64(len(v)); [out.extend(struct.pack("<i", x)) for x in v]
    else: u32(4); u32(v)
for i, name in enumerate(experts): s(name); u32(3); [u64(d) for d in (256, 256, n)]; u32(typ); u64(i * bank)
for i, name in enumerate(small): s(name); u32(1); u64(256); u32(0); u64(len(experts) * bank + i * little)
out.extend(bytes((32 - len(out) % 32) % 32)); open(path, "wb").write(bytes(out) + bytes(bank * len(experts) + little * len(small)))
PY
    }
    mkdir -p "$tmp/lb" "$tmp/lx"
    lowbit_gguf "$tmp/lb/base.gguf" 8 4; lowbit_gguf "$tmp/lx/lo.gguf" 10 4; lowbit_gguf "$tmp/lx/wide.gguf" 10 8
    check 0 "Added glmbase" "$bin" model add "$tmp/lb/base.gguf" --name glmbase
    # Added again: the settings saved for it stay, and the output says so; another file under its name starts without them.
    "$bin" config set context 16k --model glmbase > /dev/null && "$bin" config set memory_limit 10GB --model glmbase > /dev/null
    check 0 "Kept the settings saved for glmbase: context, memory_limit." "$bin" model add "$tmp/lb/base.gguf" --name glmbase
    check 0 "(saved for this model (models.json))" env ORIGAMI_MODEL=glmbase "$bin" config
    check 0 "were for $tmp/lb/base.gguf and are not kept" "$bin" model add "$tmp/lx/wide.gguf" --name glmbase
    "$bin" model add "$tmp/lb/base.gguf" --name glmbase > /dev/null
    check 1 "pass --yes once you know lo.gguf holds glmbase's experts" "$bin" model add "$tmp/lx/lo.gguf" --base glmbase --name glmlo
    bkey=$(ORIGAMI_MODEL=glmbase "$bin" status --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["model"]["key"])')
    mkdir -p "$tmp/data/family/$bkey"; echo '{"4":9}' > "$tmp/data/family/$bkey/uses.json"; echo '{"4":2}' > "$tmp/data/family/$bkey/misses.json"
    check 0 "glmbase with low-bit experts from lo.gguf (Q2_K ×6; 2.6" "$bin" model add "$tmp/lx/lo.gguf" --base glmbase --name glmlo --yes
    vk=$(ORIGAMI_MODEL=glmlo "$bin" status --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["model"]["key"])')
    cmp -s "$tmp/data/family/$bkey/uses.json" "$tmp/data/family/$vk/uses.json" && cmp -s "$tmp/data/family/$bkey/misses.json" "$tmp/data/family/$vk/misses.json" \
        && [ "$(stat -f %i "$tmp/data/family/$bkey/uses.json")" != "$(stat -f %i "$tmp/data/family/$vk/uses.json")" ] \
        && echo "ok    a new GLM variant's usage counts are copied from its base's" \
        || { echo "FAIL  a new GLM variant's usage counts"; failures=$((failures + 1)); }
    rm -rf "$tmp/data/family/$bkey" "$tmp/data/family/$vk"
    check 0 "glmbase with low-bit experts" "$bin" model list
    check 0 "experts file $tmp/lx/lo.gguf on " "$bin" model list
    check 0 "low-bit experts file)" env ORIGAMI_MODEL=glmlo "$bin" status
    check 0 "experts file: $tmp/lx/lo.gguf on " env ORIGAMI_MODEL=glmlo "$bin" doctor
    check 0 "low-bit experts from lo.gguf" env ORIGAMI_MODEL=glmlo "$bin" status
    check 0 "replies differ from the exact model" env ORIGAMI_MODEL=glmlo "$bin" doctor
    ORIGAMI_MODEL=glmbase "$bin" status | grep -q "low-bit" && { echo "FAIL  the base shows the variant's experts"; failures=$((failures + 1)); }
    check 1 "blk.1.ffn_gate_exps.weight has shape [8, 256, 256], the base's [4, 256, 256]" "$bin" model add "$tmp/lx/wide.gguf" --base glmbase --name glmwide --yes
    # `refresh` of the variant rewrites its experts file too (same bytes); of the base, it re-keys the variant: its
    # usage counts follow its key both times.
    vkey() { ORIGAMI_MODEL=glmlo "$bin" status --json | python3 -c 'import json, sys; print(json.load(sys.stdin)["model"]["key"])'; }
    k1=$(vkey); mkdir -p "$tmp/data/family/$k1"; echo '{"5":3}' > "$tmp/data/family/$k1/uses.json"; sum=$(shasum -a 256 < "$tmp/lx/lo.gguf")
    check 0 "Rewrote 2 files" "$bin" refresh --model glmlo --yes
    k2=$(vkey)
    [ "$k2" != "$k1" ] && [ -f "$tmp/data/family/$k2/uses.json" ] && [ "$(shasum -a 256 < "$tmp/lx/lo.gguf")" = "$sum" ] \
        && echo "ok    refreshing the variant rewrites its experts file and moves its counts to its new key" \
        || { echo "FAIL  refreshing the variant"; failures=$((failures + 1)); }
    # Its experts file unreadable (here: registered where nothing is), refresh refuses rather than refresh the base alone.
    cp "$tmp/data/models.json" "$tmp/models.kept"
    python3 -c 'import json, sys; f = sys.argv[1]; d = json.load(open(f)); [m.update(experts=sys.argv[2]) for m in d["models"] if m.get("name") == "glmlo"]; json.dump(d, open(f, "w"))' "$tmp/data/models.json" "$tmp/unplugged/lo.gguf"
    check 1 "cannot be read: connect its drive" "$bin" refresh --model glmlo --yes
    cp "$tmp/models.kept" "$tmp/data/models.json"
    [ "$(vkey)" = "$k2" ] || { echo "FAIL  a refused refresh changed the variant's key"; failures=$((failures + 1)); }
    check 0 "Rewrote 1 file" "$bin" refresh --model glmbase --yes
    k3=$(vkey)
    [ "$k3" != "$k2" ] && [ -f "$tmp/data/family/$k3/uses.json" ] && [ ! -e "$tmp/data/family/$k2" ] \
        && echo "ok    refreshing the base moves the variant's counts to its new key" \
        || { echo "FAIL  refreshing the base leaves the variant's counts behind"; failures=$((failures + 1)); }
    check 0 "glmbase and $tmp/lx/lo.gguf are untouched" "$bin" model remove glmlo
    [ -f "$tmp/lb/base.gguf" ] && [ -f "$tmp/lx/lo.gguf" ] && ! grep -q glmlo "$tmp/data/models.json" \
        && echo "ok    removing the GLM variant keeps the base and the experts file" \
        || { echo "FAIL  removing the GLM variant"; failures=$((failures + 1)); }
    "$bin" model remove glmbase > /dev/null; rm -rf "$tmp/lb" "$tmp/lx"
else echo "skip  low-bit experts on a GLM GGUF (no python3)"; fi
# The data folder's own store is always used: saving stores drops it, and repeats.
check 0 "stores = " "$bin" config set stores "$tmp/data/store,$tmp/s7,$tmp/s7"
check 0 "$tmp/s7; $tmp/data/store (always used)" "$bin" config get stores
"$bin" config unset stores > /dev/null
# The Experts card's hot copy with no model running: a size or drive is saved, a drive that is not one refused.
hotcopy() { curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/hot-copy -H "Content-Type: application/json" "$@"; }
check 0 '"saved":["hot_copy"]' hotcopy -d '{"size":"off"}'
check 0 "off" "$bin" config get hot_copy
check 0 '"saved":["hot_copy_dir"]' hotcopy -d '{"drive":"auto"}'
check 0 "connected drive" hotcopy -d '{"drive":"/Volumes/origami-check-absent"}'
# A further copy of a Qwen model's experts goes on another drive; the data folder's is always used, a drive not there refused.
check 0 "always uses its data folder's drive" hotcopy -d '{"copy":"/"}'
check 0 "copy must be the full path of a connected drive" hotcopy -d '{"copy":"/Volumes/origami-check-absent"}'
check 0 "own dashboard" hotcopy -H "Origin: http://example.com" -d '{"build":true}'
"$bin" config unset hot_copy > /dev/null
# The model menu's states, checked by the front off the request path: a missing file, a folder without permission, a
# path that does not answer (a FIFO no one writes, as a drive waiting on a permission prompt), an unplugged drive and a
# model that is there (its GGUF size from the check). The first ask answers at once with `checking`.
mkdir -p "$tmp/locked" && touch "$tmp/locked/m.gguf" && chmod 000 "$tmp/locked" && mkfifo "$tmp/slow.gguf" && printf 'GGUF!' > "$tmp/here.gguf"
cat > "$tmp/data/models.json" <<EOF
{"models": [{"name": "gone", "path": "$tmp/gone/m.gguf"}, {"name": "locked", "path": "$tmp/locked/m.gguf"}, {"name": "slow", "path": "$tmp/slow.gguf"},
            {"name": "unplugged", "path": "/Volumes/origami-check-absent/m.gguf"}, {"name": "here", "path": "$tmp/here.gguf"}]}
EOF
state() { # state <model> <expected availability>
    out=$(curl -s -m 5 "127.0.0.1:$port_a/v1/models?all" | python3 -c 'import json, sys; print({e["id"]: e["origami"]["availability"] + "/" + str(e["origami"]["bytes"]) for e in json.load(sys.stdin)["data"]}.get(sys.argv[1]))' "$1")
    case "$out" in "$2"*) echo "ok    $1 is $2" ;; *) echo "FAIL  $1 is $out, expected $2"; failures=$((failures + 1)) ;; esac
}
if command -v python3 > /dev/null; then
    state slow checking; since=$(date +%s)
    sleep 4
    state gone missing; state locked no_permission; state slow slow; state unplugged not_connected; state here available/5
    check 0 '"searching":' curl -s -m 30 127.0.0.1:$port_a/v1/models/found
    # A check waiting 30 s is abandoned for a new one (a drive allowed since answers it), the label `slow` meanwhile.
    rm "$tmp/slow.gguf" && printf 'GGUF!' > "$tmp/slow.gguf"
    wait=$((since + 31 - $(date +%s))); [ "$wait" -gt 0 ] && sleep "$wait"
    state slow slow; sleep 1; state slow available
else echo "skip  the model menu's states (no python3)"; fi
chmod 755 "$tmp/locked"; rm -f "$tmp/data/models.json"
# The storage map lists Origami's items where Origami keeps them: a registered model, its pack, a pack whose model is
# gone and a move left half done. Only those move or go, only from the page, and only into the folder browser's places.
mkdir -p "$tmp/data/models/m1" "$tmp/data/store/m1-abcdef012345" "$tmp/data/store/x-0123456789ab" "$tmp/data/models/.origami-moving-old" "$tmp/data/s9"
echo '{}' > "$tmp/data/models/m1/config.json"; head -c 3000000 /dev/urandom > "$tmp/data/models/m1/model.safetensors"; echo '{}' > "$tmp/data/models/m1/tokenizer.json"
store_fixture "$tmp/data/store/m1-abcdef012345" pack
store_fixture "$tmp/data/store/x-0123456789ab" mirror
echo "{\"models\": [{\"name\": \"m1\", \"path\": \"$tmp/data/models/m1\", \"manifest\": {\"fingerprint\": \"abcdef0123456789\"}}]}" > "$tmp/data/models.json"
echo "[\"$tmp/data/models/.origami-moving-old\"]" > "$tmp/data/moving.json"
# A pack with no model anywhere is removable once the model search (with this registry) is complete.
curl -s -m 30 "127.0.0.1:$port_a/v1/models/found?fresh" > /dev/null
job() { curl -s -m 10 -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" "$@"; }
ended() { for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_a/v1/jobs | grep -q '"running"' || break; sleep 0.5; done; curl -s -m 5 127.0.0.1:$port_a/v1/jobs; }
check 0 '"name":"m1","path"' curl -s -m 20 127.0.0.1:$port_a/v1/storage
check 0 '"its model is not registered"' curl -s -m 20 127.0.0.1:$port_a/v1/storage
check 0 '"an interrupted move' curl -s -m 20 127.0.0.1:$port_a/v1/storage
# A further copy in the hot-copy format (`copy/index.json`, what setup now writes on another drive) is an item of its own.
mkdir -p "$tmp/data/store/m1-abcdef012345/copy" && echo '{"identity":"fixture","keys":[0],"offsets":[0]}' > "$tmp/data/store/m1-abcdef012345/copy/index.json"
check 0 '"note":"further copy"' curl -s -m 20 127.0.0.1:$port_a/v1/storage
rm -rf "$tmp/data/store/m1-abcdef012345/copy"
# Saved prompts of a model file no model has (another fingerprint) are an unused item of their own; m1's stay in the shared one.
if command -v python3 > /dev/null; then
    mkdir -p "$tmp/context/0123456789ab-8192/0123456789abcdef" "$tmp/context/abcdef012345-8192/0123456789abcdef"
    prompts() { curl -s -m 20 127.0.0.1:$port_a/v1/storage | python3 -c 'import json, sys; print(sorted((i["name"], i.get("note"), "remove" in i["actions"]) for d in json.load(sys.stdin)["drives"] for i in d["items"] if i["role"] == "prompts"))'; }
    check 0 "[('0123456789ab-8192', 'saved prompts from an earlier Origami version or another model file; safe to delete', True), ('context', 'saved prompts and long contexts', False)]" prompts
    check 0 "(from an earlier Origami version or another model file; safe to delete)" "$bin" cache
    rm -rf "$tmp/context/0123456789ab-8192" "$tmp/context/abcdef012345-8192"
else echo "skip  saved prompts of another model file (no python3)"; fi
# A copy Origami 2.x made for m1 is named by that version's fingerprint (config, index and shard sizes): it is said to be so.
if command -v python3 > /dev/null; then
    efp=$(python3 -c 'import hashlib, os, sys; d = sys.argv[1]; print(hashlib.sha256(open(d + "/config.json", "rb").read() + ("model.safetensors:%d\n" % os.stat(d + "/model.safetensors").st_size).encode()).hexdigest()[:12])' "$tmp/data/models/m1")
    store_fixture "$tmp/data/store/m1-$efp" mirror
    check 0 '"note":"made by an earlier Origami version for m1; not used by this version; safe to delete (frees ' curl -s -m 20 127.0.0.1:$port_a/v1/storage
    rm -rf "$tmp/data/store/m1-$efp"
else echo "skip  a copy an earlier version made (no python3)"; fi
check 0 "storage map lists" job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/models.json\"}"
check 0 "Only an item nothing uses" job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/models/m1\"}"
check 0 "your home folder or Origami's folders only" job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/models/m1\",\"to\":\"$tmp/outside\"}"
check 0 "own page" job -H "Origin: http://example.com" -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/store/x-0123456789ab/mirror.json\"}"
check 0 '"state":"running"' job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/store/x-0123456789ab/mirror.json\"}"
check 0 '"Removed mirror.json' ended
[ -e "$tmp/data/store/x-0123456789ab" ] && { echo "FAIL  remove left the orphan"; failures=$((failures + 1)); }
check 0 '"state":"running"' job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/models/.origami-moving-old\"}"
check 0 '"done"' ended
# A family hot copy in a store (here the data folder's) sits in `family-<key>/hot` beside that GGUF's packs: removing it
# (as after Rebuild elsewhere) leaves the pack.
fam="$tmp/data/store/family-0123456789abcdef"; mkdir -p "$fam/pack-Q4_K" "$fam/hot"; echo '{}' > "$fam/pack-Q4_K/pack.json"; echo x > "$fam/hot/experts.bin"
"$bin" config set hot_copy_dir "$tmp/data/store" > /dev/null
check 0 '"name":"hot","note":"its model is not registered"' curl -s -m 20 127.0.0.1:$port_a/v1/storage
check 0 '"state":"running"' job -d "{\"kind\":\"remove\",\"path\":\"$fam/hot\"}"
check 0 '"done"' ended
[ -f "$fam/pack-Q4_K/pack.json" ] && [ ! -e "$fam/hot" ] || { echo "FAIL  removing a family hot copy took its pack"; failures=$((failures + 1)); }
"$bin" config unset hot_copy_dir > /dev/null; rm -rf "$fam"
# A pack whose model is on this Mac but not added (a Qwen folder in a Model search folder, matched by its fingerprint)
# is listed for that model, without Remove, and its removal is refused.
if command -v python3 > /dev/null; then
    qwen_fixture "$tmp/found/qm"
    "$bin" config set models_dirs "$tmp/found" > /dev/null
    # The fingerprint covers each file's identity on disk (inode, times), so it is the search's, not one computed here.
    fp=$(curl -s -m 30 "127.0.0.1:$port_a/v1/models/found?fresh" | python3 -c 'import json, sys; print(next((m["fingerprint"][:12] for m in json.load(sys.stdin)["models"] if m["name"] == "qm"), "none"))')
    store_fixture "$tmp/data/store/qm-$fp" pack
    check 0 '"note":"for qm, found on this Mac, not added"' curl -s -m 20 127.0.0.1:$port_a/v1/storage
    acts=$(curl -s -m 20 127.0.0.1:$port_a/v1/storage | python3 -c 'import json, sys; print([i["actions"] for d in json.load(sys.stdin)["drives"] for i in d["items"] if i["path"].endswith("/" + sys.argv[1])])' "qm-$fp/pack.json")
    [ "$acts" = "[[]]" ] && echo "ok    a found model's pack has no actions" || { echo "FAIL  a found model's pack offers $acts"; failures=$((failures + 1)); }
    check 0 " 409" curl -s -m 10 -w ' %{http_code}' -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/store/qm-$fp/pack.json\"}"
    [ -f "$tmp/data/store/qm-$fp/pack.json" ] || { echo "FAIL  a found model's pack was removed"; failures=$((failures + 1)); }
    "$bin" config unset models_dirs > /dev/null; rm -rf "$tmp/found" "$tmp/data/store/qm-$fp"
else echo "skip  a found model's pack (no python3)"; fi
# Locate… on a Qwen model moved by hand: the same model (its path-free identity) is accepted, and its copy follows it to
# the fingerprint it has there.
if command -v python3 > /dev/null; then
    qwen_fixture "$tmp/data/models/lq" banks
    "$bin" model add "$tmp/data/models/lq" > /dev/null 2>&1
    lfp=$(python3 -c 'import json, sys; print(next(m["manifest"]["fingerprint"][:12] for m in json.load(open(sys.argv[1]))["models"] if m["name"] == "lq"))' "$tmp/data/models.json")
    store_fixture "$tmp/data/store/lq-$lfp" pack
    mv "$tmp/data/models/lq" "$tmp/data/models/lq-moved"
    check 0 '"state":"running"' job -d "{\"kind\":\"relink\",\"path\":\"$tmp/data/models/lq\",\"to\":\"$tmp/data/models/lq-moved\"}"
    check 0 '"Location updated' ended
    nfp=$(python3 -c 'import json, sys; print(next(m["manifest"]["fingerprint"][:12] for m in json.load(open(sys.argv[1]))["models"] if m["name"] == "lq"))' "$tmp/data/models.json")
    [ "$nfp" != "$lfp" ] && [ -f "$tmp/data/store/lq-$nfp/pack.json" ] && [ ! -e "$tmp/data/store/lq-$lfp" ] \
        && echo "ok    Locate accepts a moved Qwen model and its copy follows its fingerprint" \
        || { echo "FAIL  Locate of a moved Qwen model ($lfp -> $nfp)"; ls "$tmp/data/store"; failures=$((failures + 1)); }
    # Another copy under the name it would derive again is refused without --name (the first stays registered).
    qwen_fixture "$tmp/other/lq" banks
    check 1 "pass --name <another name>" "$bin" model add "$tmp/other/lq"
    check 0 "$tmp/data/models/lq-moved" "$bin" model list
    "$bin" model remove lq > /dev/null 2>&1; rm -rf "$tmp/data/models/lq-moved" "$tmp/data/store/lq-$nfp" "$tmp/other"
else echo "skip  Locate of a moved Qwen model (no python3)"; fi
# Delete on a model removes only its own files: a folder that also holds the user's files stays with them (and says
# so); a folder left empty goes.
if command -v python3 > /dev/null; then
    qwen_fixture "$tmp/data/models/dq" banks; qwen_fixture "$tmp/data/models/dr" banks
    echo "mine" > "$tmp/data/models/dq/notes.txt"
    "$bin" model add "$tmp/data/models/dq" > /dev/null 2>&1; "$bin" model add "$tmp/data/models/dr" > /dev/null 2>&1
    check 0 '"state":"running"' job -d "{\"kind\":\"delete\",\"path\":\"$tmp/data/models/dq\"}"
    check 0 '"Deleted dq; kept its folder for the other files in it' ended
    [ -f "$tmp/data/models/dq/notes.txt" ] && [ ! -e "$tmp/data/models/dq/config.json" ] && [ ! -e "$tmp/data/models/dq/model.safetensors" ] \
        && echo "ok    Delete keeps the user's files in a model's folder" \
        || { echo "FAIL  Delete of a model: $(ls -A "$tmp/data/models/dq" 2>&1)"; failures=$((failures + 1)); }
    check 0 '"state":"running"' job -d "{\"kind\":\"delete\",\"path\":\"$tmp/data/models/dr\"}"
    check 0 '"Deleted dr"' ended
    [ ! -e "$tmp/data/models/dr" ] && echo "ok    Delete removes a model's folder that only held the model" \
        || { echo "FAIL  Delete left $(ls -A "$tmp/data/models/dr")"; failures=$((failures + 1)); }
    "$bin" model list | grep -q "models/d[qr]" && { echo "FAIL  a deleted model stayed registered"; failures=$((failures + 1)); }
    rm -rf "$tmp/data/models/dq"
    for _ in $(seq 120); do curl -s -m 30 "127.0.0.1:$port_a/v1/models/found" | grep -q '"searching":true' || break; sleep 0.5; done  # the search the delete started
else echo "skip  Delete of a model's folder (no python3)"; fi
# A Model search folder that cannot be read leaves the search incomplete: a family hot copy of a model not registered
# may be that folder's, so it is neither removed nor offered a rebuild.
mkdir -p "$tmp/locked-dir" "$tmp/data/family/0123456789abcdef" && echo x > "$tmp/data/family/0123456789abcdef/e" && chmod 000 "$tmp/locked-dir"
"$bin" config set models_dirs "$tmp/locked-dir" > /dev/null
curl -s -m 30 "127.0.0.1:$port_a/v1/models/found?fresh" > /dev/null
check 0 "locked-dir could not be read" curl -s -m 10 -w ' %{http_code}' -X POST 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/family/0123456789abcdef\"}"
acts=$(curl -s -m 20 127.0.0.1:$port_a/v1/storage | python3 -c 'import json, sys; print([i["actions"] for d in json.load(sys.stdin)["drives"] for i in d["items"] if i["path"].endswith("/family/0123456789abcdef")])')
[ "$acts" = "[[]]" ] && echo "ok    an unknown model's hot copy has no actions" || { echo "FAIL  an unknown model's hot copy offers $acts"; failures=$((failures + 1)); }
[ -f "$tmp/data/family/0123456789abcdef/e" ] || { echo "FAIL  an unknown model's hot copy was removed"; failures=$((failures + 1)); }
chmod 755 "$tmp/locked-dir"; "$bin" config unset models_dirs > /dev/null; rm -rf "$tmp/locked-dir" "$tmp/data/family/0123456789abcdef"
curl -s -m 30 "127.0.0.1:$port_a/v1/models/found?fresh" > /dev/null
# A pack moved on its own drive is renamed, and its new folder joins the stores.
check 0 '"state":"running"' job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/store/m1-abcdef012345/pack.json\",\"to\":\"$tmp/data/s9\"}"
check 0 "now one of the stores" ended
[ -f "$tmp/data/s9/m1-abcdef012345/pack.json" ] && [ ! -e "$tmp/data/store/m1-abcdef012345" ] || { echo "FAIL  the pack did not move"; failures=$((failures + 1)); }
check 0 "$tmp/data/s9" "$bin" config get stores
"$bin" config unset stores > /dev/null
# Across drives (a disk image) a model is copied, read back and compared, then the registry names its new place and the
# original goes; a file that cannot be read stops the move with the original and the registry as they were.
if mkdir -p "$tmp/data/vol" && hdiutil create -quiet -size 20g -type SPARSE -fs APFS -volname OrigamiCheck "$tmp/vol" \
   && hdiutil attach -quiet -nobrowse -mountpoint "$tmp/data/vol" "$tmp/vol.sparseimage"; then
    check 0 '"state":"running"' job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/models/m1\",\"to\":\"$tmp/data/vol\"}"
    check 0 '"Moved m1 to' ended
    cmp -s "$tmp/data/vol/m1/model.safetensors" /dev/null && { echo "FAIL  the moved model is empty"; failures=$((failures + 1)); }
    [ ! -e "$tmp/data/models/m1" ] || { echo "FAIL  the original is still there"; failures=$((failures + 1)); }
    check 0 "data/vol/m1" "$bin" model list
    chmod 000 "$tmp/data/vol/m1/tokenizer.json"
    check 0 '"state":"running"' job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/vol/m1\",\"to\":\"$tmp/data/models\"}"
    check 0 '"failed"' ended
    chmod 644 "$tmp/data/vol/m1/tokenizer.json"
    [ -f "$tmp/data/vol/m1/model.safetensors" ] && [ ! -e "$tmp/data/models/m1" ] && [ -z "$(ls -A "$tmp/data/models")" ] \
        || { echo "FAIL  a failed move changed the original or left its copy"; failures=$((failures + 1)); }
    check 0 "data/vol/m1" "$bin" model list
    # A copy whose drive goes away mid-copy stays listed, to be removed once the drive is back (skipped when the copy
    # was done before the drive went).
    mkdir -p "$tmp/data/models/m3"; echo '{}' > "$tmp/data/models/m3/config.json"; mkfile -n 2000m "$tmp/data/models/m3/model.safetensors"
    echo "{\"models\": [{\"name\": \"m1\", \"path\": \"$tmp/data/vol/m1\"}, {\"name\": \"m3\", \"path\": \"$tmp/data/models/m3\"}]}" > "$tmp/data/models.json"
    curl -s -m 5 -X DELETE 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d '{}' > /dev/null
    job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/models/m3\",\"to\":\"$tmp/data/vol\"}" > /dev/null
    for _ in $(seq 100); do curl -s -m 2 127.0.0.1:$port_a/v1/jobs | grep -q '"Copying and checking"' && break; sleep 0.05; done
    # Meanwhile the move's own copy is a move under way, not an interrupted one to remove.
    curl -s -m 5 127.0.0.1:$port_a/v1/storage | grep -q '"an interrupted move' && curl -s -m 2 127.0.0.1:$port_a/v1/jobs | grep -q '"running"' \
        && { echo "FAIL  a move's own copy is offered for removal"; failures=$((failures + 1)); }
    hdiutil detach -quiet -force "$tmp/data/vol"
    if ended | grep -q '"failed"'; then
        hdiutil attach -quiet -nobrowse -mountpoint "$tmp/data/vol" "$tmp/vol.sparseimage"
        check 0 '"an interrupted move' curl -s -m 20 127.0.0.1:$port_a/v1/storage
        check 0 '"state":"running"' job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/vol/.origami-moving-m3\"}"
        check 0 '"done"' ended
        [ -e "$tmp/data/vol/.origami-moving-m3" ] && { echo "FAIL  the interrupted copy was not removed"; failures=$((failures + 1)); }
        [ -f "$tmp/data/models/m3/model.safetensors" ] || { echo "FAIL  the original went with its drive's copy"; failures=$((failures + 1)); }
    else echo "skip  a drive gone mid-copy (the copy was done first)"; fi
    rm -rf "$tmp/data/models/m3"
    hdiutil detach -quiet "$tmp/data/vol"
else echo "skip  a move across drives (no disk image)"; fi
# A data folder on a drive without file ownership (an exFAT image): saved prompts cannot be kept private or reused
# there, so their default is this Mac's own data folder; an explicit context_dir there stays, and doctor says what it costs.
if mkdir -p "$tmp/fat" && hdiutil create -quiet -size 64m -fs ExFAT -volname OrigamiFat "$tmp/fat.dmg" \
   && hdiutil attach -quiet -nobrowse -mountpoint "$tmp/fat" "$tmp/fat.dmg"; then
    fat() { env ORIGAMI_CONFIG="$tmp/fat-config.json" ORIGAMI_DATA_DIR="$tmp/fat/Origami" "$@"; }
    check 0 "/Library/Application Support/Origami/prompt-cache" fat "$bin" config get context_dir
    check 0 "$tmp/fat/own" fat env ORIGAMI_CONTEXT_DIR="$tmp/fat/own" "$bin" config get context_dir
    check 1 "saved prompts: off in $tmp/fat/own" fat env ORIGAMI_CONTEXT_DIR="$tmp/fat/own" "$bin" doctor
    hdiutil detach -quiet "$tmp/fat"
else echo "skip  saved prompts with an exFAT data folder (no disk image)"; fi
check 0 "$tmp/data/prompt-cache" env ORIGAMI_CONFIG="$tmp/fat-config.json" "$bin" config get context_dir
# What a model reads is never removed: a download folder holding a registered GGUF, a conversion registered under another
# spelling of its path (/private/var for /var), anything while the registry cannot be read. A model holding a relative
# link is copied on its own drive, not renamed, so it does not point nowhere afterwards; the link's target stays.
mkdir -p "$tmp/data/models/dl" "$tmp/data/models/c1" "$tmp/data/models/m2" "$tmp/data/blobs" "$tmp/data/store/y-0123456789ab"
echo '{}' > "$tmp/data/models/dl/.origami-download.json"; head -c 1000 /dev/urandom > "$tmp/data/models/dl/g.gguf"
echo '{}' > "$tmp/data/models/c1/origami-model.json"; echo '{"format": "origami-store-v1"}' > "$tmp/data/store/y-0123456789ab/pack.json"
echo '{}' > "$tmp/data/models/m2/config.json"; head -c 1000 /dev/urandom > "$tmp/data/blobs/w"; ln -s ../../blobs/w "$tmp/data/models/m2/model.safetensors"
echo "{\"models\": [{\"name\": \"g\", \"family\": \"glm\", \"path\": \"$tmp/data/models/dl/g.gguf\"}, {\"name\": \"c1\", \"path\": \"/private$tmp/data/models/c1\"},
  {\"name\": \"m2\", \"path\": \"$tmp/data/models/m2\"}]}" > "$tmp/data/models.json"
check 0 "a model reads files in it" curl -s -m 20 127.0.0.1:$port_a/v1/storage
check 0 "Only an item nothing uses" job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/models/dl\"}"
[ -d "/private$tmp" ] && curl -s -m 20 127.0.0.1:$port_a/v1/storage | grep -q "a conversion no model is registered for" \
    && { echo "FAIL  a conversion registered under another spelling is listed as unused"; failures=$((failures + 1)); }
check 0 '"state":"running"' job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/models/m2\",\"to\":\"$tmp/data/s9\"}"
check 0 '"Moved m2 to' ended
[ -f "$tmp/data/s9/m2/model.safetensors" ] && [ ! -L "$tmp/data/s9/m2/model.safetensors" ] && cmp -s "$tmp/data/s9/m2/model.safetensors" "$tmp/data/blobs/w" \
    && [ ! -e "$tmp/data/models/m2" ] || { echo "FAIL  a model holding a relative link did not move with its file"; failures=$((failures + 1)); }
echo 'not json' > "$tmp/data/models.json"
check 0 "Only an item nothing uses" job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/store/y-0123456789ab\"}"
[ -f "$tmp/data/models/dl/g.gguf" ] && [ -d "$tmp/data/store/y-0123456789ab" ] || { echo "FAIL  a removal deleted what a model reads"; failures=$((failures + 1)); }
rm -rf "$tmp/data/blobs" "$tmp/data/store/y-0123456789ab"
curl -s -m 5 -X DELETE 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d '{}' > /dev/null
rm -rf "$tmp/data/models.json" "$tmp/data/models" "$tmp/data/s9" "$tmp/data/store/m1-abcdef012345"
# Nothing in use is lost through another item or spelling: the saved prompts' former folder in another case is not
# removed, nor an orphan the saved prompts were moved into; two names for one folder both follow its move; a model
# folder holding a store does not move (the store would be left behind); a pack does not go to the hot copy's folder by
# its resolved path; a GGUF a conversion reads its experts from does not move.
mkdir -p "$tmp/data/PC/abcdef012345-8192" "$tmp/data/models/m4" "$tmp/data/models/m5/st" "$tmp/data/store/z-0123456789ab" "$tmp/data/s9" "$tmp/data/hot" \
    "$tmp/data/store/m4-abcdef012345" "$tmp/data/models/c2" "$tmp/data/models/gq"
echo p > "$tmp/data/PC/abcdef012345-8192/x"; echo '{}' > "$tmp/data/models/m4/config.json"; echo '{}' > "$tmp/data/models/m5/config.json"
store_fixture "$tmp/data/store/z-0123456789ab" pack
store_fixture "$tmp/data/store/m4-abcdef012345" pack
printf 'GGUF!' > "$tmp/data/models/gq/q.gguf"; echo '{}' > "$tmp/data/models/c2/origami-model.json"
echo "{\"files\": [{\"path\": \"$tmp/data/models/gq/q.gguf\"}]}" > "$tmp/data/models/c2/gguf-experts.json"
echo "{\"models\": [{\"name\": \"m4\", \"path\": \"$tmp/data/models/m4\", \"manifest\": {\"fingerprint\": \"abcdef0123456789\"}}, {\"name\": \"m4b\", \"path\": \"$tmp/data/models/m4\"},
  {\"name\": \"m5\", \"path\": \"$tmp/data/models/m5\"}, {\"name\": \"c2\", \"path\": \"$tmp/data/models/c2\"}, {\"name\": \"q\", \"path\": \"$tmp/data/models/gq/q.gguf\"}]}" > "$tmp/data/models.json"
"$bin" config set context_dir "$tmp/data/pc" > /dev/null; "$bin" config set context_dir "$tmp/data/PC" > /dev/null
check 0 "Only an item nothing uses" job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/pc\"}"
check 0 '"state":"running"' job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/PC\",\"to\":\"$tmp/data/store/z-0123456789ab\"}"
check 0 "Saved prompts and long contexts are kept there now" ended
check 0 "Only an item nothing uses" job -d "{\"kind\":\"remove\",\"path\":\"$tmp/data/store/z-0123456789ab\"}"
[ -f "$tmp/data/store/z-0123456789ab/abcdef012345-8192/x" ] || { echo "FAIL  a removal deleted the saved prompts"; failures=$((failures + 1)); }
check 0 '"state":"running"' job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/models/m4\",\"to\":\"$tmp/data/s9\"}"
check 0 '"Moved m4 to' ended
[ "$(grep -o 's9\\/m4"' "$tmp/data/models.json" | wc -l)" -eq 2 ] || { echo "FAIL  a second name for a moved model names its old folder"; failures=$((failures + 1)); }
"$bin" config set stores "$tmp/data/models/m5/st" > /dev/null
check 0 "which Origami uses; move that out first" job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/models/m5\",\"to\":\"$tmp/data/s9\"}"
ln -s "$tmp/data/hot" "$tmp/data/hotlink"; "$bin" config set hot_copy_dir "$tmp/data/hotlink" > /dev/null
check 0 "holds the hot copy alone" job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/store/m4-abcdef012345/pack.json\",\"to\":\"$tmp/data/hot\"}"
check 0 "c2 reads its experts from it" job -d "{\"kind\":\"move\",\"path\":\"$tmp/data/models/gq/q.gguf\",\"to\":\"$tmp/data/s9\"}"
"$bin" config unset stores > /dev/null; "$bin" config unset hot_copy_dir > /dev/null; "$bin" config unset context_dir > /dev/null
curl -s -m 5 -X DELETE 127.0.0.1:$port_a/v1/jobs -H "Content-Type: application/json" -d '{}' > /dev/null
rm -rf "$tmp/data/models.json" "$tmp/data/models" "$tmp/data/s9" "$tmp/data/PC" "$tmp/data/store" "$tmp/data/hot" "$tmp/data/hotlink"
# Saved prompts are the running model's: with none running the front says so, and a save starts no model.
check 0 "No model is running" curl -s -m 5 127.0.0.1:$port_a/v1/prompts/0-abc
check 0 "No model is running" curl -s -m 5 -X DELETE 127.0.0.1:$port_a/v1/prompts/0-abc
check 0 "No model is running" curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/prompts -H "Content-Type: application/json" -d '{"text":"Be brief.","name":"b"}'
check 0 '"failed"' curl -s -m 5 127.0.0.1:$port_a/health
# The page's own logic (a job that ended before it loaded, a model switch at Apply) runs in node against a stub page.
# listen local (the default): this Mac's network name and address are refused as a Host and cannot connect; writes
# need no Origin or the page's own (http or https), whatever its host.
lhn=$(scutil --get LocalHostName 2>/dev/null); ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
[ -n "$ip" ] || ip=$(ifconfig | awk '/inet / && $2 !~ /^(127|169\.254)\./ { print $2; exit }')
check 0 "Host must be 127.0.0.1" curl -s -m 5 -H "Host: $lhn.local:$port_a" 127.0.0.1:$port_a/v1/jobs
if [ -n "$ip" ]; then
    check 0 "Host must be 127.0.0.1" curl -s -m 5 -H "Host: $ip:$port_a" 127.0.0.1:$port_a/v1/jobs
    check 7 "" curl -s -m 5 "http://$ip:$port_a/v1/jobs"
fi
saveconf() { curl -s -m 5 -w ' %{http_code}' -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d '{"listen":"local"}' "$@"; }
check 0 '"saved":["listen"]' saveconf -H "Origin: http://127.0.0.1:$port_a"
check 0 '"saved":["listen"]' saveconf -H "Origin: https://127.0.0.1:$port_a"
check 0 " 403" saveconf -H "Origin: http://localhost:$port_a"
check 0 " 403" saveconf -H "Origin: http://evil.example"
check 0 '"saved":["listen"]' saveconf -H "Host: localhost:$port_a" -H "Origin: http://localhost:$port_a"
check 0 '"saved":["lan_key"]' curl -s -m 5 -X POST 127.0.0.1:$port_a/v1/config -H "Content-Type: application/json" -d '{"lan_key":"off"}'
check 0 '"listening":"local"' curl -s -m 5 127.0.0.1:$port_a/v1/lan
check 0 " 409" curl -s -m 5 -w ' %{http_code}' -X POST 127.0.0.1:$port_a/v1/restart -H "Content-Type: application/json" -d '{}'
"$bin" config unset lan_key > /dev/null
curl -s -m 5 127.0.0.1:$port_a/dashboard > "$tmp/page.html"
if command -v node > /dev/null; then check 0 "a declined confirm saves nothing" node "$(dirname "$0")/dashboard-checks.mjs" "$tmp/page.html"
else echo "skip  Tests/dashboard-checks.mjs (no node)"; fi
kill "$front"; wait "$front" 2>/dev/null
# listen lan: the front on every interface; this Mac's name and address are a Host (no other .local name or lookalike),
# another device (here: this Mac's network address, not loopback) needs the key made when listen lan was set, and only
# this Mac sees that key and its QR code.
check 1 "listen is local" "$bin" config set listen everywhere
check 0 "#key=" "$bin" config set listen lan
check 0 "(set)" "$bin" config get lan_key
key=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["lan_key"])' "$ORIGAMI_CONFIG")
"$bin" serve --port $port_d --model "$tmp/no-model" > "$tmp/serve-lan.log" 2>&1 &
lanfront=$!
for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_d/health 2>/dev/null | grep -q '"failed"' && break; sleep 0.5; done
hostcode() { curl -s -m 5 -o /dev/null -w '%{http_code}' -H "Host: $1" -H "Authorization: Bearer $key" 127.0.0.1:$port_d/v1/jobs; }
lower=$(printf %s "$lhn" | tr '[:upper:]' '[:lower:]')
for h in "$lhn.local:$port_d" "$lower.local." "$lhn" ${ip:+"$ip:$port_d"}; do check 0 200 hostcode "$h"; done
for h in evil.example "$lower.local.evil.example" foo.local "$lower.localhost"; do check 0 400 hostcode "$h"; done
check 0 '"listening":"lan"' curl -s -m 5 127.0.0.1:$port_d/v1/lan
check 0 "\"key\":\"$key\"" curl -s -m 5 127.0.0.1:$port_d/v1/lan
check 0 '"qr":"data:image\/png;base64,' curl -s -m 5 127.0.0.1:$port_d/v1/lan
check 0 "also listening on the network" cat "$tmp/serve-lan.log"
check 0 "which travels in clear over plain HTTP" cat "$tmp/serve-lan.log"
check 1 "warn  network: listen lan sends the key in clear" "$bin" doctor  # 1: the model is missing here
check 1 "trusted_proxy is loopback" "$bin" config set trusted_proxy yes
# From loopback under this Mac's network name (a proxy for the network, or that name rebound to 127.0.0.1): another device's.
check 0 " 401" curl -s -m 5 -o /dev/null -w ' %{http_code}' -H "Host: $lhn.local:$port_d" 127.0.0.1:$port_d/v1/models
case "$(curl -s -m 5 -H "Host: $lhn.local:$port_d" -H "x-api-key: $key" 127.0.0.1:$port_d/v1/lan)" in
    *'"key":'*|*'"qr":'*) echo "FAIL  a request naming this Mac's network name was shown the key"; failures=$((failures + 1)) ;;
    *'"key_required":true'*) echo "ok    a loopback request naming the network name sees no key" ;;
    *) echo "FAIL  v1/lan under the network name"; failures=$((failures + 1)) ;;
esac
if [ -n "$ip" ] && curl -s -m 5 -o /dev/null "http://$ip:$port_d/health"; then
    check 0 " 401" curl -s -m 5 -w ' %{http_code}' "http://$ip:$port_d/v1/models"
    check 0 " 401" curl -s -m 5 -w ' %{http_code}' -H "Authorization: Bearer wrong-key-0123456789" "http://$ip:$port_d/v1/jobs"
    check 0 " 200" curl -s -m 5 -o /dev/null -w ' %{http_code}' -H "Authorization: Bearer $key" "http://$ip:$port_d/v1/models"
    check 0 '<title>Origami</title>' curl -s -m 5 "http://$ip:$port_d/dashboard"
    for file in favicon.ico manifest.webmanifest apple-touch-icon.png icon-512.png; do
        check 0 " 200" curl -s -m 5 -o /dev/null -w ' %{http_code}' "http://$ip:$port_d/$file"
    done
    case "$(curl -s -m 5 -H "x-api-key: $key" "http://$ip:$port_d/v1/lan")" in
        *'"key":'*|*'"qr":'*) echo "FAIL  another device was shown the key"; failures=$((failures + 1)) ;;
        *'"key_required":true'*) echo "ok    another device sees no key or QR code" ;;
        *) echo "FAIL  v1/lan from another device"; failures=$((failures + 1)) ;;
    esac
    # Its page saves with its own origin; lan_key off opens the network (read afresh, no restart).
    check 0 '"saved":["lan_key"]' curl -s -m 5 -X POST "http://$ip:$port_d/v1/config" -H "Authorization: Bearer $key" -H "Content-Type: application/json" -H "Origin: http://$ip:$port_d" -d '{"lan_key":"off"}'
    check 0 " 200" curl -s -m 5 -o /dev/null -w ' %{http_code}' "http://$ip:$port_d/v1/models"
    check 0 "lan_key = (set)" "$bin" config set lan_key on
    check 0 " 401" curl -s -m 5 -o /dev/null -w ' %{http_code}' "http://$ip:$port_d/v1/models"
else echo "skip  requests from another device (no network address, or the macOS firewall has not allowed this binary)"; fi
# A server already on 127.0.0.1 at that port: the wildcard socket would bind beside it, so serve refuses instead.
python3 -c 'import socket, sys, time; s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(); time.sleep(20)' "$port_e" & holder=$!
sleep 0.5
"$bin" serve --port $port_e --model "$tmp/no-model" > "$tmp/serve-taken.log" 2>&1 & taken=$!
sleep 3
if kill -0 "$taken" 2>/dev/null; then echo "FAIL  listen lan started beside a server on 127.0.0.1"; failures=$((failures + 1)); kill "$taken"
else check 0 "Port $port_e is already in use" cat "$tmp/serve-taken.log"; fi
kill "$holder" 2>/dev/null; wait "$taken" "$holder" 2>/dev/null
kill "$lanfront"; wait "$lanfront" 2>/dev/null
check 0 "Only this Mac" "$bin" config set listen local
"$bin" config unset lan_key > /dev/null
check 0 "context_on_disk = true" "$bin" config set context_on_disk true
check 1 "true, false or auto" "$bin" config set context_on_disk maybe
check 0 "context_on_disk = auto" "$bin" config set context_on_disk auto
check 0 "gpu_keepalive = false" "$bin" config set gpu_keepalive false
check 1 "true, false or auto" "$bin" config set gpu_keepalive maybe
check 0 "gpu_keepalive = auto" "$bin" config set gpu_keepalive auto
check 0 "false" "$bin" config get gpu_keepalive --gpu-keepalive false
check 0 "list_models = true" "$bin" config set list_models true
check 0 "true" "$bin" config get list_models
check 1 "list_models" "$bin" config set list_models maybe
check 0 "list_models = false" "$bin" config set list_models false
check 0 "Added store" "$bin" store add "$tmp/s1" --size 50GB
check 0 "stores = $tmp/s1" "$bin" config set stores "$tmp/s1"
check 0 "up to" "$bin" config get stores
# `store add` with --store adds to the configured stores rather than to the flag's.
check 0 "Added store" "$bin" store add "$tmp/s2" --store "$tmp/s3"
check 0 "$tmp/s1 up to 50.0 GB, $tmp/s2" "$bin" config get stores
check 0 "Contents not checked" "$bin" store list
check 0 "Removed store $tmp/s2" "$bin" store remove "$tmp/s2/"
check 0 "$tmp/s2" python3 -c 'import json, sys; print(json.load(open(sys.argv[1])))' "$tmp/data/former-folders.json"
# Folders for the hot copy, the model search and a model's download or conversion.
check 0 "hot_copy_dir = $tmp/hot" "$bin" config set hot_copy_dir "$tmp/hot"
check 0 "$tmp/hot" "$bin" config get hot_copy_dir
check 0 "$tmp/flagged-hot" "$bin" config get hot_copy_dir --hot-copy-dir "$tmp/flagged-hot"
check 0 "hot_copy_dir = auto" "$bin" config set hot_copy_dir auto
check 1 "config set hot_copy off" "$bin" config set hot_copy_dir off
check 1 "config set hot_copy off" "$bin" status --hot-copy-dir off
check 0 "(automatic)" "$bin" config get hot_copy_dir
check 0 "models_dirs = " "$bin" config set models_dirs "$tmp/m1, $tmp/m2"
check 0 "$tmp/m1,$tmp/m2" "$bin" config get models_dirs
check 0 "models_dirs reset" "$bin" config unset models_dirs
check 0 "--to <folder>" "$bin" model --help
check 1 "No model folder" "$bin" model add "$tmp/no-model" --to "$tmp/dest"
check 1 "is not connected" "$bin" model add "$tmp/no-model" --to /Volumes/origami-check-no-such-drive/models
# `setup --store` keeps the folder in the stores even when setup stops (here: no model).
"$bin" setup --store "$tmp/s5" > /dev/null 2>&1
check 0 "$tmp/s5" "$bin" config get stores
# Sizes and lengths that would overflow or trap are refused with exit 1.
check 1 "is not a size" "$bin" status --memory-limit -5GB
check 1 "is not a size" env ORIGAMI_MEMORY_LIMIT=nan "$bin" status
check 1 "is a size" "$bin" config set memory_limit nan
check 1 "is a size" "$bin" config set memory_limit inf
check 1 "is a size" "$bin" config set expert_memory 1e11GB
check 1 "is not a size" "$bin" store add "$tmp/s4" --size -1GB
check 1 "token count" "$bin" config set context 9007199254740993k
check 1 "token count" "$bin" status --context -5k
check 0 "context = 100000" "$bin" config set context 100000
# Paths: standardized, and empty refused; speculation and request_log take off.
check 0 "context_dir = $tmp/a/../context" "$bin" config set context_dir "$tmp/a/../context"
check 0 "$tmp/context" "$bin" config get context_dir
check 1 "needs a folder" "$bin" config set context_dir ""
check 0 "speculation = $tmp/mtp.safetensors" "$bin" config set speculation "$tmp/mtp.safetensors"
check 0 "$tmp/mtp.safetensors" "$bin" config get speculation
check 0 "speculation = off" "$bin" config set speculation off
check 0 "off" "$bin" config get speculation
check 0 "request_log = off" "$bin" config set request_log off
check 0 "(off)" "$bin" config get request_log
# Stop and unload, and Start: the front stays up without a model, saves `stopped` (a restart stays stopped), refuses
# requests with a 503 that says so, and starts the model again; only from its own page or the CLI, and with POST.
stopcfg="$tmp/stop-config.json"
# A trial model's throwaway memory history: one of an ended process is removed when a front starts, a running one's kept.
trialtmp="${TMPDIR:-$(getconf DARWIN_USER_TEMP_DIR)}"
# The ended process is one of this run's own, so a run beside it never meets its file.
true & ended=$!; wait "$ended"
: > "$trialtmp/origami-memory-trial-$ended.json"; : > "$trialtmp/origami-memory-trial-$$.json"
ORIGAMI_CONFIG="$stopcfg" "$bin" serve --port $port_f > "$tmp/serve-stop.log" 2>&1 & stopfront=$!
for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_f/health 2>/dev/null | grep -q '"failed"' && break; sleep 0.5; done
[ -e "$trialtmp/origami-memory-trial-$ended.json" ] && { echo "FAIL  an ended trial model's memory history was left"; failures=$((failures + 1)); }
[ -e "$trialtmp/origami-memory-trial-$$.json" ] || { echo "FAIL  a running process's trial history was removed"; failures=$((failures + 1)); }
rm -f "$trialtmp/origami-memory-trial-$ended.json" "$trialtmp/origami-memory-trial-$$.json"
check 0 " 403" curl -s -m 5 -o /dev/null -w ' %{http_code}' -X POST 127.0.0.1:$port_f/v1/stop -H "Origin: http://example.com" -H "Content-Type: application/json" -d '{}'
check 0 " 403" curl -s -m 5 -o /dev/null -w ' %{http_code}' -X POST 127.0.0.1:$port_f/v1/stop -d 'x=1'
check 0 " 405" curl -s -m 5 -o /dev/null -w ' %{http_code}' 127.0.0.1:$port_f/v1/stop
[ -e "$stopcfg" ] && { echo "FAIL  a refused stop was saved"; failures=$((failures + 1)); }
# A memory trial (`origami tune --memory`) loads at its limit until a reload without it; a stop ends it too. 4 GB is a
# trial every Apple Silicon Mac allows (an 8 GB one allows 5 GB), so the check does not depend on this Mac's memory.
check 0 '"reloading":true' curl -s -m 5 -X POST 127.0.0.1:$port_f/v1/reload -H "Content-Type: application/json" -d '{"memory_limit_trial":4000000000}'
for _ in $(seq 40); do grep -q "trial memory limit of 4000 MB" "$tmp/serve-stop.log" && break; sleep 0.25; done
check 0 "trial memory limit of 4000 MB" cat "$tmp/serve-stop.log"
for _ in $(seq 40); do curl -s -m 2 127.0.0.1:$port_f/health 2>/dev/null | grep -q '"failed"' && break; sleep 0.25; done
check 0 "Stopped: qwen3.8-flash-next is unloaded" env ORIGAMI_CONFIG="$stopcfg" "$bin" stop --port $port_f
check 0 '"stopped" : true' cat "$stopcfg"
check 0 "Stopped: " env ORIGAMI_CONFIG="$stopcfg" "$bin" stop --port $port_f  # again: already stopped
stopped='{"error":{"code":"stopped","message":"Origami is stopped; press Start on its dashboard (this server'\''s address in a browser) or run origami start on its Mac","type":"server_unavailable"}} 503'
check 0 "$stopped" curl -s -m 5 -w ' %{http_code}' -X POST 127.0.0.1:$port_f/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}]}'
check 0 "$stopped" curl -s -m 5 -w ' %{http_code}' -X POST 127.0.0.1:$port_f/api/chat -H "Content-Type: application/json" -d '{"model":"origami","messages":[]}'
check 0 "$stopped" curl -s -m 5 -w ' %{http_code}' 127.0.0.1:$port_f/v1/prompts
check 0 "$stopped" curl -s -m 5 -w ' %{http_code}' 127.0.0.1:$port_f/v1/plan
check 0 "X-Should-Retry: false" curl -s -m 5 -D - -o /dev/null -X POST 127.0.0.1:$port_f/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[]}'
check 0 '"code":"stopped"' curl -s -m 5 -X POST 127.0.0.1:$port_f/v1/reload -H "Content-Type: application/json" -d '{}'
check 0 '"status":"stopped"' curl -s -m 5 127.0.0.1:$port_f/v1/status
check 0 " 503" curl -s -m 5 -o /dev/null -w ' %{http_code}' 127.0.0.1:$port_f/health
check 0 '"models":[]' curl -s -m 5 127.0.0.1:$port_f/api/ps
check 0 '"id":"origami"' curl -s -m 5 127.0.0.1:$port_f/v1/models
check 0 '<title>Origami</title>' curl -s -m 5 127.0.0.1:$port_f/dashboard
check 1 "server: stopped on port $port_f" env ORIGAMI_CONFIG="$stopcfg" "$bin" doctor --port $port_f  # 1: the model is missing here
check 1 "did not start: No model at" env ORIGAMI_CONFIG="$stopcfg" "$bin" start --port $port_f
grep "starting qwen" "$tmp/serve-stop.log" | tail -1 | grep -q "trial memory limit" && { echo "FAIL  Start after a stop loaded the memory trial's limit"; failures=$((failures + 1)); }
check 0 "Stopped: " env ORIGAMI_CONFIG="$stopcfg" "$bin" stop --port $port_f
kill "$stopfront"; wait "$stopfront" 2>/dev/null
ORIGAMI_CONFIG="$stopcfg" "$bin" serve --port $port_f > "$tmp/serve-stop2.log" 2>&1 & stopfront=$!
for _ in $(seq 60); do curl -s -m 2 127.0.0.1:$port_f/health 2>/dev/null | grep -q '"status"' && break; sleep 0.5; done
check 0 '"status":"stopped"' curl -s -m 5 127.0.0.1:$port_f/health
grep -q "starting qwen" "$tmp/serve-stop2.log" && { echo "FAIL  a stopped front started its model after a restart"; failures=$((failures + 1)); }
check 1 "did not start: No model at" env ORIGAMI_CONFIG="$stopcfg" "$bin" start --port $port_f  # Start loads it again (here: none to load)
check 0 '"status":"failed"' curl -s -m 5 127.0.0.1:$port_f/health
grep -q stopped "$stopcfg" && { echo "FAIL  Start left stopped saved"; failures=$((failures + 1)); }
# Start after a failed start tries again, as stop then start would (here it fails again: no model to load).
check 1 "The last start failed (qwen3.8-flash-next did not start: No model at" env ORIGAMI_CONFIG="$stopcfg" "$bin" start --port $port_f
check 0 "starting qwen3.8-flash-next again after a failed start" cat "$tmp/serve-stop2.log"
kill "$stopfront"; wait "$stopfront" 2>/dev/null
check 1 "No Origami server answers on port $port_g" "$bin" stop --port $port_g
# The API key: checked, kept in a settings file only this user can read, never shown, and not a flag (`ps` shows flags).
check 1 "api_key is 16 to 256" "$bin" config set api_key short
check 0 "api_key = (set)" "$bin" config set api_key abcdefghijklmnop0123  # the key is never shown
check 0 "(set)" "$bin" config get api_key
case "$("$bin" config --json)" in *abcdefghijklmnop0123*) echo "FAIL  config --json shows the API key"; failures=$((failures + 1)) ;; esac
[ "$(stat -f %Lp "$ORIGAMI_CONFIG")" = 600 ] || { echo "FAIL  the settings file is readable by others"; failures=$((failures + 1)); }
check 1 "api_key must be 16 to 256" env ORIGAMI_API_KEY=short "$bin" status
check 2 "Unknown option --api-key" "$bin" serve --api-key abcdefghijklmnop0123
# The front checks the key before what it does itself: no save, job or restart without it.
saved=$(cat "$ORIGAMI_CONFIG")
"$bin" serve --port $port_h --model "$tmp/no-model" > "$tmp/serve-key.log" 2>&1 &
keyfront=$!
for _ in $(seq 60); do curl -s -m 2 -H "x-api-key: abcdefghijklmnop0123" 127.0.0.1:$port_h/health 2>/dev/null | grep -q '"failed"' && break; sleep 0.5; done
for route in config restart reload jobs hot-copy fs/mkdir stop start; do
    check 0 " 401" curl -s -m 5 -o /dev/null -w ' %{http_code}' -X POST "127.0.0.1:$port_h/v1/$route" -H "Content-Type: application/json" -d '{"context":"16k"}'
done
for route in lan fs storage jobs models/found; do check 0 " 401" curl -s -m 5 -o /dev/null -w ' %{http_code}' "127.0.0.1:$port_h/v1/$route"; done
[ "$(cat "$ORIGAMI_CONFIG")" = "$saved" ] || { echo "FAIL  a save without the API key was written"; failures=$((failures + 1)); }
check 0 " 200" curl -s -m 5 -o /dev/null -w ' %{http_code}' -H "x-api-key: abcdefghijklmnop0123" 127.0.0.1:$port_h/v1/lan
check 0 '<title>Origami</title>' curl -s -m 5 127.0.0.1:$port_h/dashboard
kill "$keyfront"; wait "$keyfront" 2>/dev/null
check 0 "api_key = off" "$bin" config set api_key off
check 0 "(off)" "$bin" config get api_key
# Saved prompts above 8K: 3 to 8, with the disk each choice takes in its meaning.
check 1 "prompt_cache_entries is 3 to 8" "$bin" config set prompt_cache_entries 2
check 0 "prompt_cache_entries = 5" "$bin" config set prompt_cache_entries 5
check 0 "5" "$bin" config get prompt_cache_entries
check 0 "4" "$bin" config get prompt_cache_entries --prompt-cache-entries 4
check 0 "at 256K" "$bin" config
check 0 "prompt_cache_entries reset" "$bin" config unset prompt_cache_entries
# Every setting is also a flag.
check 0 "$tmp/flagged" "$bin" config get context_dir --context-dir "$tmp/flagged"
check 0 "12.0 GB" "$bin" config get expert_memory --expert-memory 12GB
check 0 "20.0 GB" "$bin" config get hot_copy --hot-copy 20GB
check 0 "false" "$bin" config get commands --commands false
check 0 "true" "$bin" config get status_messages --status-messages true
check 0 "$tmp/mtp.safetensors" "$bin" config get speculation --speculation "$tmp/mtp.safetensors"
check 0 "$tmp/requests" "$bin" config get request_log --request-log "$tmp/requests"
check 0 "replaces the configured ones for this run" "$bin" serve --help
check 0 "any up to 262144" "$bin" serve --help
check 0 "xhigh" "$bin" serve --help
# Storage from the command line, in a data folder of its own: a model moved and moved back as the dashboard moves it; a
# move stopped after its rename put back by `move recover`; one whose carry-over waits for a model that cannot be read
# kept until it can; folders on a drive renamed (its volume UUID noted) or found by their contents on another drive
# relinked only once confirmed; unfinished conversions listed; what is said about a model that is not there.
fail() { echo "FAIL  $1"; failures=$((failures + 1)); }
sx() { env ORIGAMI_TEST_VOLUMES="${vols:-}" ORIGAMI_DATA_DIR="$sxd" ORIGAMI_CONFIG="$sxd/config.json" "$bin" "$@"; }
# A stand-in server on port_g answering /health with what fake-health.json holds: a model loaded, or stopped.
echo '{"status": "ready"}' > "$tmp/fake-health.json"
python3 -c '
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = open(sys.argv[2], "rb").read()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()' $port_g "$tmp/fake-health.json" &
fake=$!
for _ in $(seq 20); do curl -s -m 1 127.0.0.1:$port_g/health > /dev/null && break; sleep 0.2; done
sxd=$(mkdir -p "$tmp/sx" && cd "$tmp/sx" && pwd -P)
fp=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
mkdir -p "$sxd/models/m1" "$sxd/away"; echo '{}' > "$sxd/models/m1/config.json"; echo weights > "$sxd/models/m1/w.bin"
printf '{"models":[{"name":"m1","path":"%s","manifest":{"fingerprint":"%s"}}]}' "$sxd/models/m1" $fp > "$sxd/models.json"
check 0 "Moved m1 to $sxd/away." sx model move m1 "$sxd/away"
[ -f "$sxd/away/m1/w.bin" ] && [ ! -e "$sxd/models/m1" ] && tr -d "\\" < "$sxd/models.json" | grep -q "$sxd/away/m1" || fail "model move left its files or its registry entry behind"
check 1 "m1 is already in $sxd/away" sx model move m1 "$sxd/away"
check 1 "No folder at $sxd/nowhere" sx model move m1 "$sxd/nowhere"
check 1 "No model named nope" sx model move nope "$sxd/away"
check 1 "Quit it (Ctrl-C where \`origami serve\` runs" sx model move m1 "$sxd/models" --port $port_g
# As if stopped right after renaming the folder back, before the registry named it: recover renames it back again.
mv "$sxd/away/m1" "$sxd/models/m1"; mkdir -p "$sxd/moves"; set -- $(stat -f '%d %i' "$sxd/models/m1")
printf '{"id":"11111111-2222-3333-4444-555555555555","name":"m1","phase":"copying","renames":true,"kind":"model","model":"m1","folder":"%s","pairs":[{"from":"%s","to":"%s","device":%s,"inode":%s}]}' \
    "$sxd/models" "$sxd/away/m1" "$sxd/models/m1" "$1" "$2" > "$sxd/moves/11111111-2222-3333-4444-555555555555.json"
check 0 "m1  $sxd/away/m1 → $sxd/models  (stopped while copying)" sx move list
check 1 "Quit it (Ctrl-C where \`origami serve\` runs" sx move recover --port $port_g
check 0 "m1: Original locations restored." sx move recover
[ -f "$sxd/away/m1/w.bin" ] && [ ! -e "$sxd/models/m1" ] && [ -z "$(ls "$sxd/moves")" ] || fail "move recover did not put the model back, or kept its journal"
check 0 "No interrupted moves." sx move list
check 0 "Moved m1 to $sxd/models." sx model move m1 "$sxd/models"
tr -d "\\" < "$sxd/models.json" | grep -q "$sxd/models/m1" || fail "the move back left the registry naming the old place"
# A committed move whose model's identity cannot be computed now: nothing is carried over silently, the journal stays.
printf '{"id":"22222222-2222-3333-4444-555555555555","name":"m1","phase":"committed","renames":true,"kind":"model","model":"m1","folder":"%s","pairs":[{"from":"%s","to":"%s"}],"rebind":{"path":"%s","model":"m1","own":"%s","related":{}}}' \
    "$sxd/models" "$sxd/away/m1" "$sxd/models/m1" "$sxd/models/m1" $fp > "$sxd/moves/22222222-2222-3333-4444-555555555555.json"
check 1 "m1: not recovered: m1's files at $sxd/models/m1 cannot be read now" sx move recover
[ -e "$sxd/moves/22222222-2222-3333-4444-555555555555.json" ] || fail "a move whose carry-over failed lost its journal"
rm "$sxd/moves/22222222-2222-3333-4444-555555555555.json"
# A drive renamed: its folders are found by the volume UUID noted while it was there (stand-in folders for volumes).
mkdir -p "$sxd/volA/store/p-0123456789ab" "$sxd/volC/old-store"; mv "$sxd/models/m1" "$sxd/volA/m1"
store_fixture "$sxd/volA/store/p-0123456789ab" pack; store_fixture "$sxd/volC/old-store/p-0123456789ab" pack
printf '{"models":[{"name":"m1","path":"%s"}]}' "$sxd/volA/m1" > "$sxd/models.json"
printf '{"stores":[{"path":"%s"},{"path":"/Volumes/origami-check-absent/old-store"}]}' "$sxd/volA/store" > "$sxd/config.json"
vols="$sxd/volA=UUID-A"; check 0 "Nothing to relink" sx relink
mv "$sxd/volA" "$sxd/volB"; vols="$sxd/volB=UUID-A,$sxd/volC=UUID-C"
check 0 "(missing; at $sxd/volB/m1 on a renamed or remounted drive: \`origami relink\`)" sx model list
check 1 "fix   stores: $sxd/volA/store is missing; it is at $sxd/volB/store (the same drive by its volume UUID" sx doctor
check 1 "Nothing was changed (not confirmed)." sx relink
tr -d "\\" < "$sxd/config.json" | grep -q "$sxd/volA/store" || fail "relink changed a setting without being confirmed"
check 0 "stores: /Volumes/origami-check-absent/old-store → $sxd/volC/old-store" sx relink --yes
tr -d "\\" < "$sxd/config.json" | grep -q "$sxd/volB/store" && tr -d "\\" < "$sxd/models.json" | grep -q "$sxd/volB/m1" || fail "relink did not use the renamed drive's folders"
check 0 "Nothing to relink" sx relink
# Conversions stopped part way are listed with what to do; a model whose drive is not connected says so.
mkdir -p "$sxd/models/q-4bit.partial"; head -c 4096 /dev/zero > "$sxd/models/q-4bit.partial/model-00001.safetensors"; touch "$sxd/models/g-q2_k.gguf.partial"
check 0 "q-4bit.partial  0.0 GB  unfinished conversion (resume with the same \`origami model add\`, or delete)" sx store list
check 0 "g-q2_k.gguf.partial  0.0 GB  unfinished conversion (start again with the same \`origami model add\`, or delete)" sx store list
printf '{"models":[{"name":"ext","path":"/Volumes/origami-check-absent/ext"}]}' > "$sxd/models.json"
check 0 "ext  /Volumes/origami-check-absent/ext  (drive origami-check-absent not connected)" sx model list
check 1 "No model at /Volumes/origami-check-absent/ext: the drive origami-check-absent is not connected" sx setup --model ext
check 1 "No model file at $sxd/typo.gguf: fix the path" sx model add "$sxd/typo.gguf"
check 1 "models/typo may also mean the Hugging Face repo models/typo, not downloaded because ./models is a folder here" sh -c "cd '$sxd' && ORIGAMI_DATA_DIR='$sxd' ORIGAMI_CONFIG='$sxd/config.json' '$(cd "$(dirname "$bin")" && pwd)/origami' model add models/typo"
# cache prune with a server that has no model loaded (`origami stop`): it lists what it removes with sizes, and keeps a
# build with a named conversation unless --yes; with a model loaded it refuses.
mkdir -p "$sxd/prompt-cache/0123456789ab-8192/aaaaaaaaaaaaaaaa" "$sxd/prompt-cache/0123456789ab-8192/bbbbbbbbbbbbbbbb" "$sxd/prompt-cache/family-context"
echo '{"entries": {"0": {"name": "chat-1", "tokens": 9}}}' > "$sxd/prompt-cache/0123456789ab-8192/aaaaaaaaaaaaaaaa/index.json"
echo "{\"entries\": {}}" > "$sxd/prompt-cache/0123456789ab-8192/bbbbbbbbbbbbbbbb/index.json"; touch -t 202001010000 "$sxd/prompt-cache/0123456789ab-8192/aaaaaaaaaaaaaaaa/index.json" "$sxd/prompt-cache/0123456789ab-8192/bbbbbbbbbbbbbbbb/index.json"
mkdir -p "$sxd/prompt-cache/0123456789ab-8192/cccccccccccccccc"; echo "{\"entries\": {}}" > "$sxd/prompt-cache/0123456789ab-8192/cccccccccccccccc/index.json"
echo '{"status": "stopped"}' > "$tmp/fake-health.json"
check 0 "Keeps 0123456789ab-8192/aaaaaaaaaaaaaaaa  0.0 GB: it holds chat-1; \`origami cache prune --yes\` removes it too" sx cache prune --port $port_g
[ -e "$sxd/prompt-cache/0123456789ab-8192/aaaaaaaaaaaaaaaa" ] && [ ! -e "$sxd/prompt-cache/0123456789ab-8192/bbbbbbbbbbbbbbbb" ] && [ ! -e "$sxd/prompt-cache/family-context" ] \
    || fail "cache prune removed a named conversation without --yes, or kept another build or a family context"
echo '{"status": "ready"}' > "$tmp/fake-health.json"
check 1 "Run \`origami stop\`" sx cache prune --yes --port $port_g
echo '{"status": "stopped"}' > "$tmp/fake-health.json"
check 0 "Removes 0123456789ab-8192/aaaaaaaaaaaaaaaa" sx cache prune --yes --port $port_g
kill "$fake"; wait "$fake" 2> /dev/null
rm -rf "$tmp/sx" "$tmp/fake-health.json"
# cache and uninstall touch only Origami's folders in a shared context_dir and store.
mkdir -p "$tmp/context/0123456789ab-8192/0123456789abcdef" "$tmp/context/website" "$tmp/data/store/mine" "$tmp/data/store/m-0123456789ab" "$tmp/s1/m-0123456789ab" "$tmp/s1/n-0123456789ab"
touch "$tmp/s1/n-0123456789ab/experts.bin.partial"
touch "$tmp/context/keep.txt" "$tmp/context/website/index.json" "$tmp/context/0123456789ab-8192/0123456789abcdef/index.json"
echo '{}' > "$tmp/data/store/mine/pack.json"
echo '{"format": "origami-store-v1"}' > "$tmp/data/store/m-0123456789ab/mirror.json"
echo '{"format": "origami-store-v1"}' > "$tmp/s1/m-0123456789ab/pack-3bit.json"
check 0 "0123456789ab-8192/0123456789abcdef" "$bin" cache
check 0 "Keeps 0123456789ab-8192/0123456789abcdef  0.0 GB: its index could not be read, so it may hold named or pinned prompts; \`origami cache prune --all --yes\` removes it too" "$bin" cache prune --all
[ -e "$tmp/context/0123456789ab-8192/0123456789abcdef/index.json" ] || fail "cache prune removed a build whose index could not be read without --yes"
check 0 "Removes 0123456789ab-8192/0123456789abcdef  0.0 GB  (its index could not be read" "$bin" cache prune --all --yes
[ -e "$tmp/context/website/index.json" ] || { echo "FAIL  cache prune --all deleted a file of the user's"; failures=$((failures + 1)); }
# A hot_copy_dir that is also a store lists its folders once (unconfirmed, nothing is deleted): a second delete would fail.
[ "$(ORIGAMI_STORES="$tmp/s1" ORIGAMI_HOT_COPY_DIR="$tmp/s1" "$bin" uninstall < /dev/null 2>&1 | grep -c "/s1/m-0123456789ab$")" -eq 1 ] \
    || { echo "FAIL  uninstall lists a folder twice"; failures=$((failures + 1)); }
# A model's own further copy place and a folder named before (`former-folders.json`) are walked too.
mkdir -p "$tmp/hm/family-0123456789abcdef/hot" "$tmp/fm/x-0123456789ab/hot"; touch "$tmp/hm/family-0123456789abcdef/hot/experts.bin"
echo '{}' > "$tmp/fm/x-0123456789ab/hot/index.json"; echo "[\"$tmp/fm\"]" > "$tmp/data/former-folders.json"
python3 -c 'import json, os, sys; p = sys.argv[1]; m = json.load(open(p)) if os.path.exists(p) else {"models": []}; m["models"].append({"name": "hm", "path": sys.argv[2], "hot_copy_more": json.dumps([{"folder": sys.argv[2], "size": "auto"}])}); json.dump(m, open(p, "w"))' "$tmp/data/models.json" "$tmp/hm"
out=$("$bin" uninstall < /dev/null 2>&1)
case "$out" in *"$tmp/hm/family-0123456789abcdef/hot"*"$tmp/fm/x-0123456789ab"*|*"$tmp/fm/x-0123456789ab"*"$tmp/hm/family-0123456789abcdef/hot"*)
    echo "ok    uninstall lists a model's own copy place and a former folder" ;;
*) echo "FAIL  uninstall missed a copy place or former folder: $out"; failures=$((failures + 1)) ;; esac
# uninstall also removes ~/Library/Logs/Origami, which no variable redirects: skip it where that folder holds real logs.
logs="$(eval echo ~"$(id -un)")/Library/Logs/Origami"
if [ -e "$logs" ]; then echo "skip  uninstall --yes ($logs exists; not deleting real logs)"
else
mkdir -p "$tmp/hc/family-0123456789abcdef/hot" "$tmp/hc/mine"; touch "$tmp/hc/family-0123456789abcdef/hot/experts.bin" "$tmp/hc/mine/x" "$tmp/data/former-folders.json" "$tmp/data/moving.json"
"$bin" config set hot_copy_dir "$tmp/hc" > /dev/null
echo "[\"$tmp/fm\"]" > "$tmp/data/former-folders.json"
check 0 "Left in place: the origami binary" "$bin" uninstall --yes
for kept in "$tmp/context/keep.txt" "$tmp/context/website/index.json" "$tmp/data/store/mine/pack.json" "$tmp/hc/mine/x"; do
    [ -e "$kept" ] || { echo "FAIL  uninstall deleted $kept"; failures=$((failures + 1)); }
done
for gone in "$tmp/context/0123456789ab-8192" "$tmp/data/store/m-0123456789ab" "$tmp/s1/m-0123456789ab" "$tmp/s1/n-0123456789ab" "$tmp/data/config.json" \
    "$tmp/hc/family-0123456789abcdef" "$tmp/data/former-folders.json" "$tmp/data/moving.json" "$tmp/hm/family-0123456789abcdef" "$tmp/fm/x-0123456789ab"; do
    [ -e "$gone" ] && { echo "FAIL  uninstall left $gone"; failures=$((failures + 1)); }
done
fi
# An unwritable data folder: one plain line naming the folder and the fix, not Foundation's own error text.
mkdir -p "$tmp/locked" && chmod 555 "$tmp/locked"
check 1 "Origami cannot write to $tmp/locked: give yourself write access" env ORIGAMI_CONFIG="$tmp/locked/config.json" ORIGAMI_DATA_DIR="$tmp/locked" "$bin" config set context 32k
out=$(ORIGAMI_CONFIG="$tmp/locked/config.json" ORIGAMI_DATA_DIR="$tmp/locked" "$bin" config set context 32k 2>&1)
case "$out" in *NSCocoaErrorDomain*|*"Error Domain"*) echo "FAIL  raw Foundation error: $out"; failures=$((failures + 1)) ;; esac
chmod 755 "$tmp/locked"
# `origami service install|remove` with a test label, port and data folder (never the real service's): it prints the plist
# it wrote, the server answers on its port, a different plist of that label is kept without --yes, remove stops it.
svc=local.inference.origami-recipe-test
if launchctl print "gui/$(id -u)" > /dev/null 2>&1 && ! launchctl print "gui/$(id -u)/$svc" > /dev/null 2>&1 \
    && ! curl -s -m 1 127.0.0.1:11977/health > /dev/null 2>&1; then
    # The port and data folder are options: install carries them into the service's arguments (the folder made whole).
    mkdir -p "$tmp/svc"
    check 0 "<string>$svc</string>" env ORIGAMI_CONFIG="$tmp/svc/config.json" ORIGAMI_LAN_KEY=svc-lan-key-0123456789 "$bin" service install --label $svc --port 11977 --data-dir "$tmp/svc/../svc"
    case "$out" in *"<string>serve</string>"*"<string>--port</string>"*"<string>11977</string>"*"<string>--data-dir</string>"*"<string>$tmp/svc</string>"*)
        echo "ok    service install writes the setting options given into the service's arguments" ;;
        *) fail "service install did not carry --port and --data-dir: $out" ;; esac
    # A key from the shell is stored for the service, readable by this user only, and shown as (set).
    mode=$(stat -f %Lp "$HOME/Library/LaunchAgents/$svc.plist")
    case "$out" in
        *svc-lan-key-0123456789*) fail "service install printed the key from the shell" ;;
        *"<key>ORIGAMI_LAN_KEY</key>"*"<string>(set)</string>"*"only you can read it"*)
            [ "$mode" = 600 ] && grep -q svc-lan-key-0123456789 "$HOME/Library/LaunchAgents/$svc.plist" \
                && echo "ok    service install shows a key from the shell as (set) and writes the plist 0600" \
                || fail "the service plist is mode $mode or lacks the key" ;;
        *) fail "service install did not show the key as (set): $out" ;;
    esac
    up=no; for _ in $(seq 40); do curl -s -m 1 127.0.0.1:11977/health | grep -q status && { up=yes; break; }; sleep 0.5; done
    [ $up = yes ] && echo "ok    the installed service answers on the port its option names" || fail "the installed service does not answer on port 11977"
    # Installed again while it runs: it waits for the old job to go before loading the new one, which answers.
    check 0 "Loaded: it starts now" env ORIGAMI_CONFIG="$tmp/svc/config.json" ORIGAMI_LAN_KEY=svc-lan-key-0123456789 "$bin" service install --label $svc --port 11977 --data-dir "$tmp/svc/../svc"
    up=no; for _ in $(seq 40); do curl -s -m 1 127.0.0.1:11977/health | grep -q status && { up=yes; break; }; sleep 0.5; done
    [ $up = yes ] && echo "ok    a service installed over its running self answers again" || fail "the reinstalled service does not answer on port 11977"
    check 1 "holds a different service; \`origami service install --yes\` replaces it" env ORIGAMI_DATA_DIR="$tmp/svc2" ORIGAMI_PORT=11977 "$bin" service install --label $svc
    check 0 "Removed the service $svc" "$bin" service remove --label $svc
    sleep 1
    curl -s -m 1 127.0.0.1:11977/health > /dev/null && fail "the removed service still answers"
    [ -e "$HOME/Library/LaunchAgents/$svc.plist" ] && fail "service remove left its plist"
    rm -f "$HOME/Library/Logs/Origami/$svc.log"
else echo "skip  service install (no login session for launchd, or the test label or port 11977 is in use)"; fi
[ "$failures" -eq 0 ] && echo "cli-checks: all passed" || { echo "cli-checks: $failures failed"; exit 1; }

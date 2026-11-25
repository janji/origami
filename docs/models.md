# Models

Origami runs three mixture-of-experts models. All measurements on this page come from one Mac: an M2 Pro with
32 GB of memory on macOS 26. No other Mac has run Origami yet.

| Model | What Origami reads | Status |
|---|---|---|
| Qwen3.8-Flash-Next | MLX 4-bit builds directly; bf16, FP8, NVFP4 and GGUF checkpoints after a one-time conversion | Mature: the main, most-tested path. Text and images. |
| GLM-5.3-Flash | GGUF files as they are; FP8 or BF16 safetensors after a one-time conversion to GGUF | Newer. Text only. Slow on 32 GB. |
| DeepSeek-V4.1-Flash | GGUF files as they are (Engram included) | Newer. Text only. Slow on 32 GB. |

Every file linked on this page is published under its publisher's license; the README lists them
([Model licenses](../README.md#model-licenses)).

`origami inspect <folder or file>` shows what a model folder or GGUF holds and what Origami would do with it.

A model's identity is its files on this Mac: their path, size and dates, and its tokenizer and chat template files.
Moving, renaming or rewriting the files, or editing `chat_template.jinja` or `tokenizer*.json`, makes it a new model to
Origami, whose usage counts, hot copy, tuning and saved prompts start over.
`origami inspect --family <file, folder or owner/repo>` does the same for GLM and DeepSeek from the file headers
alone: whether the checkpoint runs, converts or is refused, and why.

## Qwen3.8-Flash-Next

### Builds it runs directly

MLX builds quantized to **4-bit with group size 64 or 32** (with the PLE table at group 32):

- group 64, such as [carloslfu/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/carloslfu/Qwen3.8-Flash-Next-MLX-4bit)
  (tested), and fine-tunes converted the same way;
- group 32, such as [mlx-community/Qwen3.8-Flash-Next-4bit](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit)
  (tested). Its 8-bit router is turned into floats and its norms are shifted to Origami's `1 + w` form when it loads.

Other MLX builds, 2-, 4- or 8-bit affine, are converted once by `origami model add <folder>`.

### Checkpoints it converts once

`origami model add <folder or file>` converts these into the 4-bit layout above. Conversion needs about 105 GB free
in the data folder (`origami config set data_dir …` moves it) and only reads the source.

| Source | Result |
|---|---|
| Official bf16 release, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), and fine-tunes saved the same way | Byte-identical to the MLX conversion of the same weights. |
| Official FP8 release, [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (185 GB, half the download) | Experts differ from the MLX conversion by about the rounding error 4-bit quantization adds anyway (checked on one layer). |
| NVIDIA's [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (133 GB) | Experts are kept as they are: 4-bit floats (E2M1) with an FP8 scale per 16 weights and one scale per matrix, read natively. Same size as the usual 4-bit experts and closer to the original weights (on two layers: 8.1% RMS error against bf16, versus 9.5% for group 64 and 8.3% for group 32). The rest is quantized to 4-bit group 64. |
| llama.cpp GGUF files of this model | Origami reads F32, F16, BF16, Q4_0 to Q8_0, Q2_K to Q6_K, the IQ types, MXFP4 and Q2_0 tensors, undoes llama.cpp's renames and reshapes, and fetches the official config and tokenizer (a few MB). The GGUF's own chat template is kept. |

Notes on GGUF sources:

- A 4-bit GGUF gets quantized twice, so a Q8_0, Q6_K or Q5_K file gives the closest result. On one layer of a
  Q4_K_XL file, the dense Q8_0 matrices were 98% equal to the official conversion and the Q4_K experts 73%.
- The 1- to 3-bit IQ types are read too, but a 2-bit source quantized again to 4-bit is lossy.
- Files with Hadamard rotations folded in are refused.
- Images need the separate mmproj file, which Origami does not read yet, so a converted GGUF serves text only.

### GGUFs with smaller experts

When a GGUF's routed experts average under 4.5 bits per weight (Origami's own 4-bit experts are 4.5),
`origami model add <file.gguf>` converts only the rest of the model (about 37 GB, most of it the PLE table) and
reads the experts from the GGUF while serving, each layer and projection in its own type (IQ1_M to Q6_K, Q2_0
included), computed with llama.cpp's kernels. Keep the GGUF where it is: the converted folder points into it. GGUFs
whose experts are 4.5 bits or more (Q4_K and up) convert as in the table above.

This is what ISTA-DASLab's GSQ-RCO releases are for:

- [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) picks a type
  per tensor under a size budget. Its IQ3_XXS build is 3.0 bits per weight, 99.4% of BF16 on their benchmarks, with
  43 GB of experts instead of 68.
- [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  keeps 256 of the 512 experts per layer (code, tool use and vision); Origami limits the router to them.

Add the transformer shard (the `-00001-of-00002` file, with its PLE shard beside it).

What to expect:

- Replies differ from the MLX 4-bit build (other weights, other arithmetic).
- More experts fit in memory: the IQ3_XXS build's take 2.33 MB each in the cache against 2.76 MB; the Coder's take
  2.66 MB, but it has half as many (33 GB in all).
- Packs and MTP do not apply (`origami setup` says so). The hot copy does, on another drive than the GGUF's: it is
  built while serving, as for the 4-bit model ([settings](settings.md)).
- Measured on eight short benchmark replies at 8K, with the GGUF on an external drive: the IQ3_XXS build decoded at
  7.8 tok/s against 5.8 for 4-bit at the same 18 GB memory limit, with every scored answer right and 89% of 4-bit's
  next tokens. The Coder also got every scored answer right (8.6 tok/s at the default memory; 77% of 4-bit's next
  tokens overall, 85–100% on the code and tool cases, 36% on the multilingual one it was not kept for).

### Low-bit experts on the 4-bit model (opt-in)

A faster variant of the MLX 4-bit model you already have: its routed experts come from a calibrated low-bit GGUF, and
everything else (dense weights, router, PLE table, tokenizer) stays the 4-bit model's. Nothing is converted or copied
and the PLE shard is not needed.

```sh
origami model add Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --base <your 4-bit model's name> --name flash-q2
origami serve --model flash-q2
```

- The file: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` (38 GB) from
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  its `-00002` part (the PLE table) is not used. The IQ3_XXS part works the same way.
- `model add` first checks that the file is the same checkpoint as the base (headers and about 13 MB of reads): no
  folded rotations, the same geometry, every token of the base's tokenizer at the same id, the router and
  shared-expert gate of five layers equal value for value, and expert types and shapes Origami's kernels run. It
  refuses a mismatch and says what differs.
- It changes replies: `origami status`, `doctor`, the dashboard, the server log and the request log say the model uses
  low-bit experts and from which file. Packs and MTP do not apply; the variant gets its own hot copy (the `hot_copy`
  settings, on another drive than the experts file's), built while serving. The base model and
  the default stay exact.
- `origami model remove flash-q2` removes only the variant's small generated folder (in the data folder); the base
  model and the GGUF stay.
- Measured on one 32 GB Mac with the Q2_0 file and the experts on an external drive: 12 of 14 checkable tasks solved,
  the same as exact; 91% of exact's next tokens; about 12 tok/s on a first reply and 14 warm (exact: see the
  README's [What to expect](../README.md#what-to-expect)). With the IQ3_XXS file: 94.9% of exact's next tokens,
  10.9 tok/s.

GLM-5.3-Flash and DeepSeek-V4.1-Flash take the same command: see
[Low-bit experts on an exact GGUF](#low-bit-experts-on-an-exact-gguf-opt-in).

### Smaller experts made by Origami (removed)

The `expert_bits` setting of earlier versions is gone: published low-bit experts (above) do better on every count
measured. A settings file that still sets it starts with the model's own experts and says so;
`origami config unset expert_bits` removes it. Packs it made are no longer read; `origami status`, `origami doctor`
and the dashboard's Models tab list them for deletion. `--expert-bits 2|4|8` remains only for converting GLM-5.3
safetensors.

### MTP speculation (opt-in)

Every release of this model carries a multi-token-prediction (MTP) head that guesses the token after next. With
`origami config set speculation <file>`, Origami drafts one token with the head and checks it together with the
current token in one step; an accepted draft gives two tokens per step (79% were accepted on the benchmark replies).

- The file is a safetensors file holding `language_model.mtp.*`, as in ARC4NUM's MLX-Serve build.
  `python3 scripts/fetch-mtp-head.py mtp.safetensors` fetches only those 1.5 GB.
- It is meant for Macs that hold every expert in memory (96 GB or more). Each reply measures the two-token step
  against a plain one and decodes plainly when speculating would be slower.
- On a 32 GB Mac the step reads both tokens' experts from the drive and costs about 2.5 plain steps, so it stays off.
  Even with every expert in memory, the step costs about 1.7 plain steps in GPU time, so it only pays when most
  drafts are accepted (79% is near break-even).
- At temperature 0, replies can differ slightly from plain decode (the two-token step's arithmetic differs). With
  sampling, a rejected draft is replaced by a token drawn so the distribution is unchanged.

## GLM-5.3-Flash and DeepSeek-V4.1-Flash

### Get the files

Origami has run unsloth's UD-Q4_K_XL builds. Download one with the Hugging Face CLI (`pip install -U huggingface_hub`
gives `hf`) to a drive with room, then register it under a name:

| Model | Repository | Files | Size |
|---|---|---|---|
| GLM-5.3-Flash | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | `UD-Q4_K_XL/*` (6 files) | 199.7 GB |
| DeepSeek-V4.1-Flash | [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | `UD-Q4_K_XL/*` (9 files) | 500.7 GB |

```sh
hf download unsloth/GLM-5.3-Flash-GGUF --include "UD-Q4_K_XL/*" --local-dir /Volumes/MySSD/glm-5.3-flash
origami model add /Volumes/MySSD/glm-5.3-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --name glm

hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-Q4_K_XL/*" --local-dir /Volumes/MySSD/deepseek-v4.1-flash
origami model add /Volumes/MySSD/deepseek-v4.1-flash/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf --name deepseek
```

The drive needs the files' size free, and the hot copy 20–40 GB more to start on its drive (best another one; see
Hot copy below). Then `origami serve --model glm` (or `origami model default glm`). Leave the files where they are: Origami reads
them in place.

### Serving

Serve a GGUF as it is:

```sh
origami serve --model /path/to/model.gguf     # for a split GGUF, any one of its files, or its folder
origami model add /path/to/model.gguf         # or keep it under a name (see below)
```

Origami recognises the model from the file's own metadata and uses every tensor in the type the file stores (the
GGUF types listed above), with the file's own tokenizer and chat template.

- **Text only.** A request with images is refused.
- **GLM-5.3 safetensors** (FP8 or BF16) convert once into a GGUF with 2-bit experts:
  `origami model add <folder>`; add `--expert-bits 4` or `8` for Q4_K or Q8_0 experts.
- **Context in memory or on disk.** The context stays in memory while it takes up to a quarter of what Origami may
  use, else it goes to files on the context drive (`context_dir`), which macOS pages like any file.
  `context_on_disk` `true` or `false` decides it instead. Replies are the same either way.
- **Hot copy.** A copy of the most-used experts (counted while serving) on a second drive, read alongside the
  model's drive. It is automatic (20–40 GB to start, grown while it pays) when the GGUF is on another drive than the data folder, built in the
  background after the first reply, attached between replies (no restart) and kept up to date as the counts change.
  `hot_copy_more` adds places on more drives.
- **Mixed-format experts.** When a model's experts come in several record sizes, the expert cache moves memory
  between the size pools based on observed reuse and read cost, at safe points. Replies are unchanged in exact mode.
- **Shared features.** Saved prompts (the end of each system prompt and of each whole prompt, so a new chat or a
  restart starts from there), `origami save`, `forget`, `prompts` and `X-Origami-Keep`, the dashboard's memory plan
  and dials (context up to the model's own maximum, quality, memory), `/v1/status`, `/v1/plan` and the quality dial
  all work as for Qwen. `memory_limit` sizes their expert slots.

### Low-bit experts on an exact GGUF (opt-in)

The same idea as for Qwen: a GGUF you already serve exactly, with its routed experts read from a published low-bit
GGUF and everything else (attention, dense weights, routers, shared experts, tokenizer, DeepSeek's Engram table) from
your GGUF. Decode on these models mostly waits for expert reads, and 2- to 3-bit experts move fewer bytes per token.

```sh
origami model add <low-bit.gguf> --base <your GGUF's name> --name glm-q2
origami serve --model glm-q2
```

- The file can be a whole low-bit model (for a split one, its first part; parts without experts, such as a separate
  Engram file, are not needed) or a file of experts only. A file that covers only some layers is fine: the other
  layers keep your GGUF's experts.
- `model add` checks the pairing from headers and a few MB: the same model and geometry, every expert bank in your
  GGUF's shape and in a type Origami's kernels run, and, where the file has them, the same tokens and the same router
  values in three layers. It refuses a mismatch and says what differs. An experts-only file names no model and has no
  tokenizer or routers to compare, so `model add` says so and adds it only with `--yes`: pass it once the publisher
  says the file holds experts of the same checkpoint.
- Nothing is copied or converted: the models.json entry points at both files, and `origami model remove` removes
  only that entry. Your GGUF and its default stay exact.
- It changes replies: `status`, `doctor`, the dashboard, the server log and the request log say the model uses low-bit
  experts and from which file. The variant has its own usage counts, hot copy and saved
  prompts; your GGUF's are left as they are. Its usage counts start as a copy of your GGUF's (the router is the same),
  so its hot copy can be built from its first reply; from then on each keeps its own.
- On synthetic models the variant gives bit for bit what one GGUF with those experts in place gives.
- **GLM-5.3-Flash, measured** on one 32 GB Mac with lte-ai's experts-only `experts-q23-fp8.gguf` from
  [lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts](https://huggingface.co/lte-ai/GLM-5.3-Flash-Strata-Q2Q3-Experts) (Q2_K
  gate and up, Q3_K down, layers 3–44; 110 GB) beside unsloth's UD-Q4_K_XL: 12 of 14 checkable tasks solved (exact:
  10), 92% of exact's next tokens, 1.73–1.88 tok/s with a 20 GB hot copy and 39 s to the first token.
- **DeepSeek-V4.1-Flash: no real file has been run yet**, and Origami makes no quality claim for one. Files whose
  headers pass the check include ISTA-DASLab's GSQ-RCO IQ1_M and IQ2_XXS, antirez's `DeepSeek-V4.1-Flash-Q2.gguf` and
  ggml-org's Q2_K; for GLM-5.3-Flash also antirez's `GLM-5.3-Flash-Q2.gguf` and unsloth's UD-IQ2_XXS and UD-Q2_K_XL
  (not measured). A file split into `.partNofM` pieces must be joined first.
- `origami refresh --model <variant>` rewrites the variant's experts file as well as carrying its copies and counts
  ([troubleshooting.md](troubleshooting.md#the-model-drive-got-slow)).

### Speed

These models are much larger than Qwen, and on a 32 GB Mac most of every token's experts come from the drive, so
the time goes to reading experts, not to the GPU. The measured figures are in the README's
[What to expect](../README.md#what-to-expect). Other Macs and drives will differ; a drive that has slowed down on the
model's files makes them much slower ([The model drive got slow](troubleshooting.md#the-model-drive-got-slow)). In earlier short benchmarks with Origami 2.x, `fast` (then called `faster`) kept the answers (90% and 99% of exact's
next tokens for GLM and DeepSeek) and `fastest` did not (62% and 83%, 1 of 4 answers right); neither model offers
`fastest` now (see [Quality dial](#quality-dial)).

### What is checked

- The GLM and DeepSeek kernels are checked bit-for-bit against a CPU reference computed from the dequantized
  weights, in the model-free `engine-checks`.
- An opt-in check compares the engine's logits with a reference run on a real model:
  `engine-checks --only family_reference_logits`, with `ORIGAMI_FAMILY_GGUF` (the model) and
  `ORIGAMI_FAMILY_REFERENCE` (a folder of token ids and reference logits) set. See [switches.md](switches.md).
- No stored end-to-end comparison with the models' reference implementations ships with this repository.

For DeepSeek, `exact` means Origami's own F32/GGUF execution, using every selected expert. DeepSeek's official
reference rounds activations to FP8/FP4 and uses BF16 in places, so its outputs can differ from Origami's.

## Quality dial

One setting trades replies for decode speed. `exact` (the default) gives the model's own replies. The other levels
take shortcuts: they leave out parts of the model (experts) that a reply would otherwise wait for (a lightly weighted
one that is not in memory, or one still being read), so they decode faster and reply differently. Prompts are always
read exactly, so saved prompts serve every level.

Every model offers the same ladder, from the model's own replies to the most shortcuts:

| Level | What it does |
|---|---|
| `exact` (default) | the model's own replies; the slowest |
| `dynamic` | takes shortcuts only where the model is sure of the next word; between exact and fast |
| `fast` | takes the shortcut everywhere; replies differ slightly |
| `fastest` | the most shortcuts; replies differ most. Only where it stayed usable: Qwen |

| Model | Levels offered | Measured effect |
|---|---|---|
| Qwen3.8-Flash-Next | `exact`, `dynamic`, `fast`, `fastest` | `fast` and `fastest` took 62–71% less time than exact; on the most recent 14-task check exact solved 12, `fast` 10 and `fastest` 9 |
| GLM-5.3-Flash | `exact`, `dynamic`, `fast` | `fast` took 34–42% less time, every task right (`fastest` rambled and wrote reasoning after `</think>`, so GLM does not offer it) |
| DeepSeek-V4.1-Flash | `exact`, `dynamic`, `fast` | `dynamic` took about 30% less time, every task right, one reply of 14 with reasoning after `</think>`; `fast` is faster still, but got one task of 14 wrong and wrote reasoning after `</think>` twice |

On Qwen and GLM, `dynamic` stayed right on every task and was slower than `fast`; its speed against exact is not yet
published here. The evidence is short checkable tasks (answers scored right or wrong: ten short tasks and four coding
and long-form ones per model) on one Mac, an M2 Pro with 32 GB reading the models from an external SSD. Treat it as a
guide, not a guarantee.

What the shortcuts are, on Qwen and GLM:

- `fast`: an expert under 10% of the routing weight is left out when it is not in memory;
- `fastest`: the `fast` skip, and an expert still being read is left out;
- `dynamic`: the `fast` skip, but only after a token the model gave at least 0.9 probability; other tokens decode
  exactly.

DeepSeek takes bigger shortcuts at the same names, because the ones above saved it no time: its `dynamic` is the
`fastest` skips after a token of at least 0.9 probability (exact after any other), and its `fast` is the `fastest`
skips everywhere.

An earlier Qwen run of eight benchmark replies at 8K, without a pack, measured `fast` at 1.7× exact's decode speed
(94.8% of exact's next tokens) and `fastest` at 2.1× (90.0%, 3 of 4 answers right).

A level a model does not offer is never refused: a request, setting or chat command naming `fastest` on GLM or
DeepSeek runs their `fast`. Usage, `X-Origami-Mode`, `/v1/status` and the request record always name the level on
the serving model's ladder.

- Levels other than exact can reply differently from run to run, since what is in memory decides which experts count.
- At every level a tool call decodes exactly from its opening marker on, so the call's name and arguments are the
  model's own.
- GLM and DeepSeek read every expert a layer routes before the layer runs, so for them every expert not in memory
  counts as late: the `late_miss skip` knob, and with it `fastest` (and DeepSeek's `dynamic` and `fast`), leaves out
  all of them but the heaviest.
- `faster`, the name of `fast` before 3.1.8, still means `fast`; `dynamic-fastest`, which DeepSeek offered before
  3.1.8, still works and runs as `dynamic`.

Choose a level:

- per request: `"origami_mode": "<level>"`, or the model name `<model>-<level>` (such as `qwen3.8-flash-next-fast`
  or `deepseek-v4.1-flash-dynamic`);
- per chat: the chat command `origami quality <level>`;
- for every request that names none: `origami config set quality <level>`.

`origami config set allow_fast false` forbids every level but exact. The dashboard, `/v1/models`, `/v1/status` and
`/v1/plan` list only the served model's levels; `/v1/status` and `/v1/plan` also report the level and the knobs that
model runs.

### Knobs

Each knob is also a setting of its own, applied on top of the level:

| Knob | What it does |
|---|---|
| `experts_per_token` 1–10 or `all` | each token uses only its heaviest experts of the ten the router picks, weights renormalised (5: 1.5× decode, 89% as exact; 7: 1.4×, 94%) |
| `tail_skip` share or `off` | drops routed experts below this share of the routing weight (the heaviest is kept); 0.1 always: 1.7×, 85% as exact |
| `tail_skip_misses_only` true/false | the tail skip drops only experts not in memory, saving the reads and keeping the free ones (with 0.1: `fast`) |
| `late_miss` `wait`, `skip` or `substitute` | an expert still being read when its layer needs it: wait (exact), leave it out (1.x's fast mode: 1.3×, 91.6% as exact), or use the in-memory expert the router scored highest instead (1.3×, 93%, but two of four answers broke) |
| `confident_above` probability or `off` | the other knobs apply only after a token the model gave at least this probability; other steps decode exactly (`dynamic`: 0.9) |

### Measuring a level yourself

Two request fields compare a level with exact:

- `"origami_return_tokens": true` puts the reply's token IDs in `usage.origami_tokens`.
- `"origami_teacher_tokens": [ids…]` forces the reply along those tokens (for example exact's reply). At each
  position the decode scores its own greedy pick against the given token, then continues from the given one.
  `usage.origami_teacher` reports `positions`, `agreed` and `first_difference`; top-1 agreement is
  agreed / positions.

The IDs are the model's raw vocabulary IDs, special tokens included, so they bypass the chat template; an ID outside
the vocabulary is refused. A teacher-forced reply is never saved to the prompt cache.

## Keeping several models

```sh
origami model add ~/Models/my-fine-tune --name tuned          # a folder, a GGUF, or a Hugging Face repo id
origami model list
origami serve --model tuned                                    # or: origami model default tuned
```

`--to <folder>` writes a download
or conversion there instead of `<data_dir>/models`. Removing a name never touches its files. `origami model move <name>
<folder>` moves a model's files with the server quit, as the dashboard's **Move…** does (its hot copy, counts and saved
prompts follow it); `origami relink` follows a drive that was renamed or mounted under another name. `model list` says
when a model's drive is not connected. Only `owner/repo` is taken for a Hugging Face repo, and not when `owner` is a
folder where you run the command: then both readings are printed and nothing is downloaded. A Hugging Face repo is
checked before anything is downloaded: a repo of GGUF files is refused (download the GGUF you want, then add the file),
and the drive must hold the download and, for a checkpoint Origami converts, the conversion beside it. A name already
registered for another folder is refused unless you pass `--name`. `origami setup --model <file>.gguf` makes that
GGUF the default model.

By default the server lists one model, `origami`: the setup chosen in the dashboard (its model, context, quality and
memory). Every request gets it, whatever model name the client sends, so apps that name the model their own way keep
working; a request can still ask for a quality level.

With `origami config set list_models true` the server lists every model instead, as Ollama does: the configured one
and each name from `origami model list`, with their `-<level>` ids, at `/v1/models` and `/api/tags`. A request that
names another listed model loads it in place of the running one: running requests finish, the old model's process
stops (all its memory comes back), and the new one starts, which takes as long as starting that model. One model is
loaded at a time either way. `origami serve --single-model` serves just the configured model in one process.

[INTERNALS.md](../INTERNALS.md) explains how the engines work and how to add a quantization layout.

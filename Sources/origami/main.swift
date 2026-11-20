import Foundation
import OrigamiEngine

let version = "3.1.2"

/// Options that set a setting; every command that reads settings accepts them. Help lists them in this order: the
/// ones people change most first, the quality knobs last.
let settingOptions = [
    "model", "context", "port", "memory-limit", "listen", "hot-copy", "store", "quality", "reasoning-effort", "max-thinking-tokens",
    "expert-memory", "hot-copy-dir", "hot-copy-more", "data-dir", "context-dir", "config", "models-dirs", "list-models", "commands",
    "status-messages", "request-log", "context-on-disk", "prompt-cache-entries", "gpu-keepalive", "speculation", "allow-fast",
    "experts-per-token", "tail-skip", "tail-skip-misses-only", "late-miss", "confident-above",
]
/// The settings every command's help names; serve's and setup's name them all.
let commonSettingOptions = ["model", "context", "port", "memory-limit", "data-dir", "config"]
/// Options without a value.
let switchOptions: Set<String> = [
    "yes", "deep", "no-pack", "full-pack", "json", "help", "version", "all", "family", "single-model", "recheck", "undo", "memory",
]
let optionLines: [String: String] = [
    "model":
        "--model <name|folder|file>  model to use: a name from `origami model list`, a folder, a .gguf or a Hugging Face repo\n                              (default: \(Models.defaultRepo) if present)",
    "name": "--name <name>               the name to list the model under (default: its folder's name)",
    "store":
        "--store <folder>            a folder for copies of the experts (repeatable); replaces the configured ones for this run\n                              (setup: adds it to them and builds there first)",
    "context": "--context <tokens>          context length, any up to 262144: 8k, 32k, 100000, 256k (default 8k)",
    "port": "--port <n>                  server port (default 11439)",
    "listen": "--listen local|lan          local: this Mac only (default); lan: also other devices on this network, with lan_key",
    "memory-limit": "--memory-limit <size>       memory ceiling, e.g. 20GB (default: automatic)",
    "expert-memory": "--expert-memory <size>      memory for cached experts, e.g. 12GB (default: what memory_limit leaves)",
    "hot-copy": "--hot-copy <size>|off|auto  disk space for a copy of the most-used experts, which reads faster (default auto)",
    "commands": "--commands true|false       answer `origami <verb>` chat messages (default true)",
    "context-on-disk":
        "--context-on-disk true|false|auto\n                              GLM and DeepSeek: their context in files on disk (default auto: when it is large)",
    "list-models": "--list-models true|false    list every model and switch by name (default false: one model, `origami`)",
    "status-messages":
        "--status-messages true|false\n                              show progress of long prompts in streamed replies (default false)",
    "gpu-keepalive":
        "--gpu-keepalive true|false|auto\n                              keep the GPU at full speed while replying (default auto: on, except GLM and DeepSeek in Low Power Mode)",
    "speculation": "--speculation <file>|off    MTP speculative decoding from the model's MTP head file (default off)",
    "request-log": "--request-log <folder>|off  a JSON line per request in that folder (default off)",
    "hot-copy-dir":
        "--hot-copy-dir <folder>|auto\n                              folder for the copy of the most-used experts (default auto)",
    "hot-copy-more":
        "--hot-copy-more <folder=size,…>|off\n                              more places for that copy, one per drive (default none)",
    "models-dirs": "--models-dirs <folder,…>    more folders the dashboard's model search looks in",
    "to": "--to <folder>               where a download or conversion is written (default <data_dir>/models)",
    "base": "--base <model>              with a GGUF: read only its routed experts, the rest from this exact model",
    "root": "--root <folder>             root containing every file listed by a content manifest",
    "receipt": "--receipt <file.json>       verified content receipt to write (parent directory must exist)",
    "context-dir": "--context-dir <folder>      folder for a context kept on disk and saved prompts (default <data_dir>/prompt-cache)",
    "prompt-cache-entries":
        "--prompt-cache-entries <n>\n                              saved prompts kept above an 8K context, 3 to 8 (default 3; up to 6.8 GB each at 256K)",
    "allow-fast": "--allow-fast true|false     allow quality levels other than exact (default true)",
    "quality": "--quality <level>           quality dial for requests that choose none (default exact): "
        + QualityDial.offeredByModel,
    "experts-per-token":
        "--experts-per-token <n>|all\n                              quality knob: routed experts each token uses, the heaviest first (1 to 10)",
    "tail-skip": "--tail-skip <share>|off     quality knob: drop routed experts under this share of the gate weight, e.g. 0.05",
    "tail-skip-misses-only":
        "--tail-skip-misses-only true|false\n                              quality knob: the tail skip drops only experts not in memory",
    "late-miss":
        "--late-miss wait|skip|substitute\n                              quality knob: an expert still being read when it is needed",
    "confident-above": "--confident-above <p>|off   quality knob: the others apply only after a token the model gave at least p",
    "reasoning-effort":
        "--reasoning-effort <level>  thinking when a request asks for none: \(ServingDefaults.efforts.joined(separator: "|")) (default none)",
    "max-thinking-tokens": "--max-thinking-tokens <n>   close the think block after n tokens when a request sets no budget",
    "expert-bits":
        "--expert-bits 2|4|8         converting a GLM-5.3-Flash checkpoint: Q2_K, Q4_K or Q8_0 experts (default 2)",
    "data-dir": "--data-dir <folder>         where caches, the hot copy and downloads go",
    "config": "--config <file>             use another settings file (e.g. one per port)",
    "size": "--size <size>               most space Origami may use in that folder, e.g. 200GB",
    "label": "--label <label>             the service's launchd label (default com.origami.serve)",
    "yes": "-y, --yes                   answer yes to questions (downloads, deletions)",
    "deep": "--deep                      also compare every record with the model's weights",
    "json": "--json                      machine-readable output",
    "single-model":
        "--single-model              serve this one model in this process, without listing or starting the others\n                              (what `serve` starts for each model it runs)",
    "family":
        "--family                    any model (a folder, a .gguf, or a Hugging Face repo id), judged from its headers only:\n                              family, geometry, tensor types, and whether Origami runs it",
    "all": "--all                       remove every saved prompt, this build's too",
    "no-pack": "--no-pack                   do not copy the experts to another drive",
    "full-pack": "--full-pack                 copy the experts even onto the model's own drive",
    "review": "--review true               save a proposal without applying it",
    "tradeoffs": "--tradeoffs true            compare quality modes without changing settings",
    "quality-floor": "--quality-floor <level>     lowest quality to compare (default the model's mildest fast level)",
    "minutes": "--minutes <n>               time limit: 3 to 30 minutes (default 8); with --memory 3 to 60 (default 30)",
    "recheck": "--recheck                   check that the tuned settings are still faster than the defaults",
    "undo": "--undo                      remove the tuned settings and reload with the defaults",
    "memory":
        "--memory                    reload at a few memory limits, time replies at each and suggest the fastest this Mac\n                              keeps without cuts or swap (about 4 minutes for Qwen, 6 for GLM, 10 for DeepSeek)",
]

let help = """
    Origami \(version) runs large mixture-of-experts models (Qwen, GLM, DeepSeek) on Apple Silicon Macs,
    reading their experts from the SSD.

    Start here:
      1. origami setup          find the model on this Mac, or download it (104 GB)
      2. origami serve          start the server at http://127.0.0.1:11439/v1 and leave it running
      3. origami dashboard      open the dashboard (in another terminal)
      A GLM or DeepSeek GGUF instead: origami serve --model /path/to/model.gguf

    Usage: origami <command> [options]

    Server:
      serve                 Start the server (OpenAI, Anthropic and Ollama APIs) and the dashboard
      stop                  Unload the model and free its memory; the dashboard stays up
      start                 Load the model again after `origami stop`
      status                Show the server, the model, memory, drives and expert copies
      dashboard             Open the dashboard in your browser
      service install|remove
                            Start the server at each login (a LaunchAgent), or stop and remove it

    Models and drives:
      setup                 Find or download the model, measure the drives, build the expert copies
      model list|add|remove|move|default
                            Manage named models; `model add <experts.gguf> --base <model>` adds low-bit experts
      store list|add|remove Use folders on other drives for copies of the experts
      move list|recover     Finish or undo a move a stop interrupted
      relink                Follow a drive that was renamed or mounted under another name
      refresh               Rewrite the model's files in place when its drive reads them slowly
      verify [--deep]       Check the expert copies against their checksums
      inspect [folder]      Show what a model folder or GGUF holds and whether Origami runs it
      cache [prune]         List saved prompts, or remove those no model uses

    Settings and speed:
      config                List every setting; `config set <key> <value>`, `config unset <key>`
      tune                  Find faster timing settings on the running server (about 8 minutes)
      tune --memory         Find this Mac's fastest memory limit on the running server
      bench                 Time a short question, a story and a long document on the running server

    Problems:
      doctor                Check everything without loading the model, and say what to fix
      report                Print this Mac's details and a benchmark, to paste into a GitHub issue
      content               Check model files against known checksums (`origami content --help`)
      uninstall             Delete Origami's copies, caches and settings (never your models)

    Common options: --model <name|folder|file.gguf>, --context 32k, --port <n>, --memory-limit 24GB, -y
    Every setting is also an option (--context 32k) and an ORIGAMI_<NAME> variable; `origami config` lists them.
    `origami help <command>` (or `origami <command> --help`) shows a command's options; `origami --version`.
    Exit status: 0 success, 1 failure, 2 usage error.
    """

/// One command's usage line, summary, accepted options and most positional arguments; `main` refuses anything else
/// with exit status 2 before the command runs.
struct CommandSpec {
    let usage, summary: String
    var options: [String] = []
    /// Most arguments after the command name.
    var arguments = 0
    var settings = true

    func help(_ name: String) -> String {
        let all = ["serve", "setup"].contains(name)
        let own = options.map { "  " + optionLines[$0]! } + ["  -h, --help                  show this help"]
        let shared =
            settings
            ? ["", "Settings for this run (`origami config set <key> <value>` keeps one):"]
                + (all ? settingOptions : commonSettingOptions).map { "  " + optionLines[$0]! }
                + (all ? [] : ["  and every other setting as --<key> <value> (`origami serve --help` lists them)"])
            : []
        return "Usage: origami \(usage)\n\n\(summary).\n\nOptions:\n" + (own + shared).joined(separator: "\n")
    }
}
let commands: [String: CommandSpec] = [
    "setup": .init(
        usage: "setup [options]",
        summary:
            "Find the model on this Mac or download it, measure the drives and copy the experts to a faster drive if there is one.\nRun it again after adding a drive",
        options: ["yes", "no-pack", "full-pack"]),
    "serve": .init(
        usage: "serve [options]",
        summary:
            "Start the server at http://127.0.0.1:11439/v1 (OpenAI, Anthropic and Ollama APIs) and the dashboard at /dashboard.\nThe port answers at once; the model loads in the background. Leave it running; Ctrl-C stops it.\n`listen lan` also lets other devices on your network in",
        options: ["single-model"]),
    "stop": .init(
        usage: "stop [options]",
        summary:
            "Unload the running server's model and give its memory back. A running reply gets 30 s to finish.\nThe dashboard stays up, apps get \"Origami is stopped\", and it stays stopped, also after a restart, until `origami start`",
        options: ["port", "config"], settings: false),
    "start": .init(
        usage: "start [options]", summary: "Load the model again after `origami stop`, and wait until it is ready",
        options: ["port", "config"], settings: false),
    "status": .init(
        usage: "status [--json] [options]",
        summary: "Show the server, the model, the memory limit in force, the drives, the expert copies and the saved prompts",
        options: ["json"]),
    "store": .init(
        usage: "store list\n       origami store add <folder> [--size 200GB]\n       origami store remove <folder> [--yes]",
        summary: "List, add or remove folders (on any drive) where Origami keeps copies of the model's experts",
        options: ["size", "yes", "no-pack", "full-pack"], arguments: 2),
    "model": .init(
        usage:
            "model list [--json]\n       origami model add <folder|org/repo|file.gguf> [--name <name>] [--to <folder>] [--expert-bits 2|4|8] [--yes]\n       origami model add <experts.gguf> --base <model> [--name <name>] [--yes]\n       origami model remove <name>\n       origami model move <name> <folder>\n       origami model default <name>",
        summary:
            "List, add, remove, move or choose the default of the models Origami knows by name; `origami serve --model <name>` serves one.\n`model add <experts.gguf> --base <model>` adds a faster variant of an exact model whose experts come from a low-bit file\n(replies differ from the exact model's). `model move` moves a model's files with the server quit, as the dashboard's Move:\nrenamed on the same drive, else copied, read back and compared before the original goes; its hot copy, counts and\nsaved prompts follow it",
        options: ["name", "to", "base", "expert-bits", "json", "yes"], arguments: 3),
    "move": .init(
        usage: "move [list]\n       origami move recover [<id>]",
        summary:
            "List the moves a stop interrupted, or finish or undo them as the dashboard's Recover does (with the server quit):\na move that reached its new place is finished, one that did not is put back. Nothing is deleted that is not checked",
        arguments: 2),
    "service": .init(
        usage: "service install [--label <label>] [--yes] [options]\n       origami service remove [--label <label>]",
        summary:
            "Run `origami serve` in the background at each login (a LaunchAgent): install writes ~/Library/LaunchAgents/<label>.plist\n(this binary, the setting options given, the ORIGAMI_ variables of this shell, a log in ~/Library/Logs/Origami),\nprints it and loads it; a different plist of that label is replaced only with --yes. remove stops it and deletes the plist",
        options: ["label", "yes"], arguments: 1),
    "relink": .init(
        usage: "relink [--yes] [options]",
        summary:
            "When a drive was renamed or mounted under another name, find the folders Origami's settings name on it (models,\nstores, hot copy places, context_dir) by the drive's volume UUID or their own contents, show each, and use the new\nplaces once confirmed; what is keyed on a model moves with it. Quit `origami serve` first",
        options: ["yes"]),
    "config": .init(
        usage:
            "config [--json]\n       origami config get <key>\n       origami config set <key> <value> [--yes]\n       origami config unset <key> [--yes]",
        summary:
            "List every setting with its value and where it comes from, or read, change or reset one.\n--yes with hot_copy_more also deletes the copies in the places it drops (kept otherwise)",
        options: ["json", "yes"], arguments: .max),
    "verify": .init(
        usage: "verify [--deep] [options]", summary: "Read the expert copies again and check every expert's checksum",
        options: ["deep"]),
    "content": .init(
        usage:
            "content verify <manifest.json> --root <folder> --receipt <output.json>\n       origami content import <receipt.json> --root <folder> --receipt <output.json>\n       origami content check <receipt.json>",
        summary:
            "Check model files against a list of SHA-256 checksums you supply, and keep a receipt of the result.\nverify hashes every listed file, import hashes them again at a new folder, check quickly re-checks a receipt's files.\nThe checksums are yours: this does not prove who published the files",
        options: ["root", "receipt", "json"], arguments: 2, settings: false),
    "cache": .init(
        usage: "cache [prune [--all] [--yes]] [options]",
        summary:
            "List the saved prompts per model, context and Origami build (Qwen, GLM and DeepSeek). prune lists, with sizes, then\nremoves those of older builds, of model files no model uses and a GLM or DeepSeek context left on disk; --all removes\nevery one. A build holding pinned or named conversations is kept unless --yes. It runs with the server quit or stopped\n(`origami stop`)",
        options: ["all", "yes"], arguments: 1),
    "refresh": .init(
        usage: "refresh [--model <name>] [--yes] [options]",
        summary:
            "Rewrite the model's files in place, byte for byte and checked, when its drive reads them far below its own speed\n(`origami doctor` says when). Quit `origami serve` first. It takes about as long as reading the model three times,\nis safe to stop and run again, and moves the hot copy and usage counts over to the rewritten files",
        options: ["yes"]),
    "uninstall": .init(
        usage: "uninstall [--yes] [options]",
        summary: "Delete Origami's expert copies, caches and settings. Your models are never deleted", options: ["yes"]),
    "dashboard": .init(usage: "dashboard [options]", summary: "Open the dashboard of the running server in your browser"),
    "doctor": .init(
        usage: "doctor [options]",
        summary:
            "Check the settings, the Mac, the model, its drive, the expert copies, memory and disk without loading the model,\nand say what to fix. Exit status 1 when something needs fixing"
    ),
    "inspect": .init(
        usage:
            "inspect [folder | file.gguf] [--json] [options]\n       origami inspect --family <folder | file.gguf | owner/repo> [--json]",
        summary: "Show what a model folder or GGUF holds (formats and sizes) and whether Origami runs it",
        options: ["json", "family"], arguments: 1),
    "bench": .init(
        usage: "bench [options]", summary: "Time a short question, a story and a 6K-token document on the running server"),
    "tune": .init(
        usage: "tune [--minutes N] [options]\n       origami tune --memory [--minutes N]\n       origami tune --recheck | --undo",
        summary:
            "On the running server, try timing settings that leave replies unchanged and keep the one that is faster every time\n(about 8 minutes, longer if the model needs it; a request from an app stops it, and the next run continues). Origami then uses it for this model and context on this Mac.\n--memory instead finds this Mac's fastest memory limit; it suggests it and changes no setting",
        options: ["memory", "minutes", "recheck", "undo", "review", "tradeoffs", "quality-floor", "port", "config", "data-dir"],
        settings: false),
    "report": .init(
        usage: "report [options]",
        summary: "Print this Mac, the memory plan, the drives and a short benchmark of the running server, to paste into a GitHub issue"),
    "help": .init(usage: "help [command]", summary: "Show help for Origami or for one command", arguments: 1, settings: false),
    "version": .init(usage: "version", summary: "Print the version", settings: false),
]

/// `--key value`, `--key=value`, bare switches and `-h`/`-V`/`-y`; everything else is positional.
func parse(_ arguments: [String]) throws -> (positional: [String], flags: [String: [String]], switches: Set<String>) {
    var positional: [String] = []
    var flags: [String: [String]] = [:]
    var seen = Set<String>()
    var index = 0
    while index < arguments.count {
        let argument = arguments[index]
        if argument.hasPrefix("--") {
            let body = String(argument.dropFirst(2))
            if let equals = body.firstIndex(of: "=") {
                let name = String(body[..<equals])
                guard !switchOptions.contains(name) else { throw CLIError.usage("--\(name) takes no value") }
                flags[name, default: []].append(String(body[body.index(after: equals)...]))
            } else if switchOptions.contains(body) {
                seen.insert(body)
            } else if index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") {
                flags[body, default: []].append(arguments[index + 1])
                index += 1
            } else if optionLines[body] != nil {
                throw CLIError.usage("--\(body) needs a value")
            } else {
                throw CLIError.usage("Unknown option --\(body). See `origami help`.")
            }
        } else if let short = ["-h": "help", "-V": "version", "-y": "yes"][argument] {
            seen.insert(short)
        } else if argument.hasPrefix("-") && argument.count > 1 {
            throw CLIError.usage("Unknown option \(argument). See `origami help`.")
        } else {
            positional.append(argument)
        }
        index += 1
    }
    return (positional, flags, seen)
}

// Whole lines as they are printed, also into a pipe (the dashboard's jobs read a command's progress from one).
setvbuf(stdout, nil, _IOLBF, 0)
do {
    let (positional, flags, switches) = try parse(Array(CommandLine.arguments.dropFirst()))
    let name = positional.first ?? (switches.contains("version") && !switches.contains("help") ? "version" : "help")
    guard let spec = commands[name] else { throw CLIError.usage("Unknown command \"\(name)\". See `origami help`.") }
    // `--help` and `--version` never run the command.
    if switches.contains("help") {
        print(positional.isEmpty ? help : spec.help(name))
        exit(0)
    }
    if switches.contains("version") {
        print("origami \(version)")
        exit(0)
    }
    // A dashboard job's tune gets the server's own options, which describe the model it measures.
    let job = name == "tune" && ProcessInfo.processInfo.environment["ORIGAMI_JOB"] != nil
    let allowed = Set((spec.settings || job ? settingOptions : []) + spec.options)
    if let option = (Array(flags.keys) + Array(switches)).sorted().first(where: { !allowed.contains($0) }) {
        // tune measures the server as it runs: a setting given here would only label the result for a model it does not run.
        if name == "tune", settingOptions.contains(option) {
            throw CLIError.usage(
                "tune measures the running server with its own settings; --\(option) would only mislabel the result. "
                    + "Change it with `origami config set \(option.replacingOccurrences(of: "-", with: "_")) <value>`, reload, then tune.")
        }
        throw CLIError.usage(
            optionLines[option] == nil && !switchOptions.contains(option)
                ? "Unknown option --\(option). See `origami \(name) --help`."
                : "\(name) does not take --\(option). See `origami \(name) --help`.")
    }
    let rest = Array(positional.dropFirst())
    guard rest.count <= spec.arguments else {
        throw CLIError.usage("Unexpected argument \"\(rest[spec.arguments])\". See `origami \(name) --help`.")
    }
    switch name {
    case "setup": try await Commands.setup(flags: flags, switches: switches)
    case "serve": try await Commands.serve(flags: flags, single: switches.contains("single-model"))
    case "status": try Commands.status(flags: flags, json: switches.contains("json"))
    case "stop", "start": try await Commands.power(start: name == "start", flags: flags)
    case "store": try await Commands.store(rest, flags: flags, switches: switches)
    case "move": try Commands.moves(rest, flags: flags)
    case "relink": try Commands.relink(flags: flags, yes: switches.contains("yes"))
    case "service": try Commands.service(rest, flags: flags, yes: switches.contains("yes"))
    case "config": try Commands.config(rest, flags: flags, json: switches.contains("json"), yes: switches.contains("yes"))
    case "model": try await Commands.model(rest, flags: flags, switches: switches)
    case "verify": try Commands.verify(flags: flags, deep: switches.contains("deep"))
    case "content": try ContentCommand.run(rest, flags: flags, json: switches.contains("json"))
    case "cache": try Commands.cache(rest, flags: flags, all: switches.contains("all"), yes: switches.contains("yes"))
    case "dashboard": try Diagnostics.dashboard(flags: flags)
    case "doctor": try await Diagnostics.doctor(flags: flags)
    case "inspect":
        try switches.contains("family")
            ? Diagnostics.inspectFamily(rest, json: switches.contains("json"))
            : Diagnostics.inspect(rest, flags: flags, json: switches.contains("json"))
    case "bench": try await Diagnostics.bench(flags: flags)
    case "report": try await Diagnostics.report(flags: flags)
    case "tune":
        try await Tuning.run(
            flags: flags, recheck: switches.contains("recheck"), undo: switches.contains("undo"), memory: switches.contains("memory"))
    case "refresh": try Refresh.run(flags: flags, yes: switches.contains("yes"))
    case "uninstall": try Commands.uninstall(flags: flags, yes: switches.contains("yes"))
    case "version": print("origami \(version)")
    default:
        guard let topic = rest.first else {
            print(help)
            break
        }
        guard let command = commands[topic] else { throw CLIError.usage("Unknown command \"\(topic)\". See `origami help`.") }
        print(command.help(topic))
    }
} catch {
    Terminal.progress(nil)
    // One line: a server shows a model's last `origami:` line as the reason it did not start. A bare out-of-memory error
    // gets what to do, with this Mac's maximum; one that already says what to do (after a "; ") is left as it is.
    let maximum = OrigamiPlan.gigabytes(WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory))
    let text =
        (error as? EngineError)?.code == "insufficient_memory" && !"\(error)".contains("; ")
        ? "the model does not fit in the memory limit (\(error)): raise memory_limit (at most \(maximum) on this Mac: "
            + "`origami config set memory_limit \(maximum.replacingOccurrences(of: " ", with: ""))`), or close other apps"
        : Terminal.fileProblem(error) ?? "\(error)"
    FileHandle.standardError.write(Data("origami: \(text)\n".utf8))
    exit((error as? CLIError)?.exitCode ?? 1)
}

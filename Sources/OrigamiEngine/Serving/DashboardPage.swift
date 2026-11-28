import CryptoKit
import Foundation

/// The dashboard served at `/` (for browsers) and `/dashboard`, in four views: Now (the state and the one next action, the
/// speed, where the last request's time, the expert reads and the memory went, a health line per thing that can need you,
/// the requests and the saved prompts), Setup (model, context, quality, thinking, memory and expert width as dials, with this
/// Mac's memory answering each move; tuning; sampling), Models (the models and their low-bit variants, where experts are read
/// from, the drives and files) and Connect (an app's settings, who can reach Origami, the log). One self-contained file, no
/// external requests, relative URLs so it also works behind a proxy path. It reads `v1/status`, `v1/plan`, `v1/models?all`
/// (and `v1/models` for Connect an app), `v1/models/found`, `v1/models/check`, `v1/fs` (its one folder browser), `v1/storage`
/// (where Origami's files are), `v1/prompts/<id>` (a saved prompt's text), `v1/lan` (who can reach it) and `v1/jobs`, and writes
/// `v1/config` then `v1/reload` (or `v1/restart` for access), `v1/fs/mkdir`, `v1/prompts` (rename, keep, delete, a system prompt edited into a new one),
/// `v1/presets` (save, update, rename, delete or use a preset of the Setup settings), `v1/memory/learned` (forgets the learned memory budget), `v1/memory/wired` (saves the budget and raises its macOS ceiling)
/// and `v1/jobs` (preparing smaller experts, adding a model or a low-bit variant, moving or removing a stored item, tuning).
public enum DashboardPage {
    /// The CSP `script-src` for the page's one inline script, by its SHA-256: a script injected into the page
    /// would not match and does not run.
    package static let scriptHash: String = {
        let start = html.range(of: "<script>")!
        let end = html.range(of: "</script>", range: start.upperBound..<html.endIndex)!
        return "'sha256-" + Data(SHA256.hash(data: Data(html[start.upperBound..<end.lowerBound].utf8))).base64EncodedString() + "'"
    }()
    /// The page's icon, a white paper crane on black, also served at `/favicon.svg` and `/favicon.ico` (browsers ask
    /// for the latter whatever the page says). A few flat facets in three tones, no lines, so it still reads at 16 px.
    public static let icon =
        ##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7"/><path fill="#999" d="M13 19 19 3l3 16z"/><path fill="#fff" d="M15 20 27 5l-4 15z"/><path fill="#fff" d="M8 22 6 8l8 12z"/><path fill="#ccc" d="M6 8 2 14l6-1z"/><path fill="#ccc" d="M21 21 30 15l-8 1z"/><path fill="#fff" d="M7 22l9-4 9 4z"/><path fill="#aaa" d="M7 22h18l-9 5z"/></svg>"##
    public static let html = #"""
        <!doctype html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
        <meta name="color-scheme" content="dark light">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="Origami">
        <meta name="theme-color" content="#0b0b0a" media="(prefers-color-scheme: dark)">
        <meta name="theme-color" content="#efeee9" media="(prefers-color-scheme: light)">
        <title>Origami</title>
        <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,\#(icon.addingPercentEncoding(withAllowedCharacters: .alphanumerics)!)">
        <link rel="manifest" href="manifest.webmanifest">
        <link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
        <style>
        \#(DashboardStyles.css)
        </style>
        </head>
        <body>
        <header class="topbar" id="top">
          <div class="wrap bar">
            <span class="wordmark">ORIGAMI</span><span class="label muted" id="version"></span>
            <div class="live"><span class="live-text" id="live-text" aria-live="polite"></span>
              <a class="chip x-chip" id="x-chip" href="#experts" hidden><i></i><span class="x-chip-name"><span class="x-hot">Hot </span>copy</span><span id="x-chip-text"></span></a>
              <span class="chip" id="chip" data-state="down"><i></i><span id="chip-text">Connecting</span></span>
              <button type="button" class="power" id="power" hidden aria-label="Stop and unload">Stop<span class="power-more"> and unload</span></button></div>
          </div>
          <nav class="views wrap" aria-label="Dashboard"><button type="button" data-view="run" aria-current="page">Now</button><button type="button" data-view="setup">Setup<i id="setup-dot" aria-hidden="true"></i></button><button type="button" data-view="library">Models</button><button type="button" data-view="settings">Connect</button></nav>
          <div class="readline" id="progress" hidden role="progressbar" aria-label="Prompt read" aria-valuemin="0" aria-valuemax="100"><span id="reused"></span><span id="read"></span></div>
        </header>
        <main>
          <section class="notice keyask wrap" id="keyask" hidden aria-labelledby="keyask-title">
            <div class="notice-box"><h2 id="keyask-title">Origami needs its key here</h2>
              <p>Paste the key the Mac shows in Origami's dashboard, under Connect, or scan the QR code there. On the Mac itself, <code>origami dashboard</code> opens this page with its key.</p>
              <div class="add-row"><input type="text" id="keyask-input" aria-label="Key" placeholder="Paste the key" spellcheck="false" autocomplete="off" autocapitalize="off">
                <button type="button" id="keyask-use">Use this key</button></div>
              <p class="note" id="keyask-note" role="status"></p></div>
          </section>
        <fieldset id="page">
          <section class="hero wrap" aria-label="Model">
            <canvas class="art" id="art" aria-hidden="true"></canvas>
            <div class="hero-main">
              <div class="hero-name"><h1 id="title">Origami</h1><p class="lede" id="lede"></p></div>
              <p class="badges" id="badges"></p>
              <div class="picker" id="picker"><label class="label" for="model">Model</label>
                <span class="select" id="model-select"><select id="model"></select><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 4.5 6 8.5 10 4.5"/></svg></span></div>
              <div class="first" id="first" hidden></div>
              <button type="button" class="primary start" id="use-model" hidden>Load model</button><p class="note" id="use-model-why" role="status"></p>
              <p class="note" id="model-note" role="status"></p>
            </div>
          </section>
          <section class="notice wrap" id="notice" hidden role="alert">
            <div class="notice-box"><h2 id="notice-title"></h2><p id="notice-text"></p>
              <ol class="steps" id="notice-steps" hidden>
                <li>On this Mac, open System Settings › Privacy &amp; Security › Files &amp; Folders.</li>
                <li>Under origami (or under the program that starts it, when the message above names one), turn on Removable Volumes. If macOS shows a prompt instead, click Allow. A new origami binary started on its own needs this again; one started by a fixed launcher does not.</li>
                <li>Come back here and choose Check again.</li>
              </ol>
              <div class="notice-actions"><a class="button" id="notice-settings" href="x-apple.systempreferences:com.apple.preference.security?Privacy_FilesAndFolders" hidden>Open Privacy settings</a>
                <button type="button" id="notice-retry">Check again</button><button type="button" id="notice-setup" hidden>Change the setup</button></div>
            </div>
          </section>
          <section class="notice wrap" id="banner" hidden role="status"><div class="notice-box"><p id="banner-text"></p></div></section>
          <section class="notice wrap" id="pending" hidden role="status"><div class="notice-box"><p id="pending-text"></p>
            <div class="actions"><button type="button" class="primary" id="pending-reload">Reload to apply</button><button type="button" id="pending-discard">Discard the saved change</button></div></div></section>
          <section class="job wrap" id="job" hidden aria-live="polite"><div class="job-in">
            <div class="job-row"><div><h2 id="job-title"></h2><p class="job-stage" id="job-stage"></p></div><p class="job-figure" id="job-figure"></p>
              <button type="button" id="job-button">Cancel</button></div>
            <div class="job-track" role="progressbar" aria-labelledby="job-title" aria-valuemin="0" aria-valuemax="100" id="job-track"><span id="job-fill"></span></div>
          </div></section>
        <div id="view-run">
          <p class="loadline wrap" role="status">Loading the model. Requests wait until it is ready.</p>
          <section class="notice slowed wrap" id="slowed" hidden role="status" aria-labelledby="slowed-title"><div class="notice-box"><h2 id="slowed-title">Slowed by</h2><ul id="slowed-list"></ul></div></section>
          <section class="now wrap running-only" id="now" aria-label="Speed">
            <div class="gauge" aria-live="polite">
              <p class="big" id="speed-line"><span id="speed">–</span><small>tok/s</small></p>
              <p class="spark" id="spark" aria-hidden="true"></p>
              <p class="basis"><span id="basis"></span> <button type="button" class="measure" id="measure">Measure</button></p>
              <dl class="figures" id="figures"></dl>
            </div>
            <div class="flows"><div class="flow" id="flow-time"></div><div class="flow" id="flow-reads"></div><div class="flow" id="flow-mem"></div></div>
          </section>
          <section class="block wrap" id="health-block" hidden><h2>Health</h2><div id="health"></div><p class="note" id="health-note" role="status"></p><p class="note" id="health-why" role="status"></p><button type="button" class="small" id="health-reload" hidden>Reload to apply</button></section>
          <section class="block wrap running-only"><h2>Requests</h2><div id="recent"></div></section>
          <details class="block wrap running-only" id="prompts-block"><summary><h2>Saved prompts</h2><span class="sum-hint" id="prompts-count"></span></summary><div id="prompts"></div><p class="note" id="prompt-note" role="status"></p></details>
        </div>
        <div id="view-setup" hidden>
          <p class="empty wrap" id="setup-empty" hidden>The dials show once a model runs.</p>
          <section class="tune wrap" id="tune" aria-label="Setup">
            <fieldset class="dials" id="dials">
              <div class="dial" id="d-context">
                <div class="row"><label class="label" for="context">Context</label><output id="o-context" for="context"></output></div>
                <input type="range" id="context" min="0" max="4" step="1" value="0">
                <div class="ticks" id="t-context"></div><p class="hint" id="h-context"></p><p class="dial-effect" id="e-context"></p>
              </div>
              <div class="dial" id="d-quality">
                <div class="row"><label class="label" for="quality">Quality</label><output id="o-quality" for="quality"></output></div>
                <input type="range" id="quality" min="0" max="0" step="1" value="0">
                <div class="ticks" id="t-quality"></div><p class="hint" id="h-quality"></p><p class="dial-effect" id="e-quality"></p>
              </div>
              <div class="dial" id="d-reasoning">
                <div class="row"><label class="label" for="reasoning">Thinking</label><output id="o-reasoning" for="reasoning"></output></div>
                <input type="range" id="reasoning" min="0" max="4" step="1" value="0">
                <div class="ticks" id="t-reasoning"></div><p class="hint" id="h-reasoning"></p><p class="dial-effect" id="e-reasoning"></p>
              </div>
              <div class="dial" id="d-memory">
                <div class="row"><span><label class="label" for="memory">Memory limit</label><button type="button" class="auto" id="auto" aria-pressed="true">Automatic</button></span><output id="o-memory" for="memory"></output></div>
                <input type="range" id="memory" min="0" max="1" step="1" value="0" aria-describedby="h-memory wired-help">
                <div class="ticks" id="t-memory"></div><p class="hint" id="h-memory"></p><p class="dial-effect" id="e-memory"></p>
                <div class="wired-memory" id="wired-memory" hidden>
                  <p class="note" id="wired-state"></p>
                  <button type="button" id="wired-restore" hidden>Restore macOS ceiling</button><p class="note" id="wired-note" role="status"></p><p class="note" id="wired-restore-why" role="status"></p>
                  <details id="memory-more"><summary>Details</summary>
                    <dl><dt>Saved limit</dt><dd id="wired-saved"></dd><dt>Loaded limit</dt><dd id="wired-active"></dd>
                      <dt>Allowed right now</dt><dd id="wired-usable"></dd><dt>Origami's memory</dt><dd id="wired-footprint"></dd>
                      <dt>macOS ceiling</dt><dd id="wired-system"></dd><dt>The GPU's own suggestion</dt><dd id="wired-metal"></dd></dl>
                    <p class="note" id="wired-help"></p></details>
                </div>
              </div>
            </fieldset>
            <div class="readout" aria-live="polite">
              <div class="effects" id="effects" hidden><p class="overall" id="overall"></p><div id="effect-rows"></div></div>
              <div class="mem" id="mem"><span class="label" id="mem-label">This Mac's memory</span><canvas id="memstrip" role="img" aria-label="This Mac's memory"></canvas><div class="keys" id="keys"></div></div>
              <p class="basis" id="ml" hidden><span id="ml-text"></span> <button type="button" class="measure" id="ml-reset">Forget the learned start</button></p>
              <p class="fits" id="fits"></p>
            </div>
          </section>
          <details class="block wrap" id="selftune" hidden open><summary><h2>Tune for this Mac</h2><span class="sum-hint" id="st-line"></span></summary>
            <div class="sf"><div class="tune-options">
              <label>Find<select id="st-mode"><option value="memory">The fastest memory limit</option><option value="tradeoffs">The speed of each quality level</option></select></label>
              <label id="st-budget-label">Time limit<select id="st-budget"><option value="" selected>8 min</option><option value="15">15 min</option><option value="30">30 min</option></select></label>
              <label id="st-floor-label" hidden>Lowest quality<select id="st-floor"></select></label></div>
              <p class="note" id="st-scope">Reloads the model at a few memory limits (10 to 20 minutes) and suggests the highest this Mac keeps without cutting or swapping. Your setting is not changed.</p>
              <details class="tuning-help"><summary>What is tested</summary><p class="note">The fastest memory limit: a long prompt, then two writing prompts at each limit; a limit macOS cut or swapped for is out, and your saved limit is reloaded at the end. The speed of each quality level: writing prompts at each level with your thinking setting, after a warm-up, at temperature 0; saved settings stay unchanged until you apply one.</p></details>
              <div class="actions"><button type="button" class="primary" id="st-go">Run tuning</button></div>
              <p class="note" id="st-why" role="status"></p><button type="button" class="small" id="st-reload" hidden>Reload to apply</button>
              <p class="note" id="st-note" role="status"></p><div id="st-result"></div>
            </div></details>
          <details class="block wrap running-only" id="sampling"><summary><h2>Sampling</h2><span class="sum-hint" id="sm-hint"></span></summary><div class="sf">
            <fieldset id="sm-group">
              <div class="drives acc" id="sm-choice" role="radiogroup" aria-label="Sampling">
                <label><input type="radio" name="sampling" id="sm-recommended" value="recommended"><span>Model recommended</span><span id="sm-source"></span></label>
                <label><input type="radio" name="sampling" id="sm-custom" value="custom"><span>Custom</span><span></span></label></div>
              <div class="sm-modes"><table><thead><tr><th>The model's</th><th class="r">Thinking</th><th class="r">Not thinking</th><th class="r">With tools</th></tr></thead>
                <tbody id="sm-table"></tbody></table></div>
              <fieldset class="sm-dials" id="sm-dials">
                <div class="dial"><div class="row"><span><label class="label" for="sm-temperature">Creativity</label><span class="tech">temperature</span></span><output id="o-sm-temperature" for="sm-temperature"></output></div>
                  <input type="range" id="sm-temperature" min="0" max="2" step="0.05" value="0.7"><span class="sm-mark" id="m-sm-temperature" aria-hidden="true" hidden></span><p class="hint">0 = always the most likely word, the same reply every time; higher = more varied.</p></div>
                <div class="dial"><div class="row"><span><label class="label" for="sm-top_p">Word pool</label><span class="tech">top_p</span></span><output id="o-sm-top_p" for="sm-top_p"></output></div>
                  <input type="range" id="sm-top_p" min="0.05" max="1" step="0.05" value="0.8"><span class="sm-mark" id="m-sm-top_p" aria-hidden="true" hidden></span><p class="hint">Draws from the likeliest words that together hold this share of the chance; 1 = every word.</p></div>
                <div class="dial"><div class="row"><span><label class="label" for="sm-top_k">Shortlist</label><span class="tech">top_k</span></span><output id="o-sm-top_k" for="sm-top_k"></output></div>
                  <input type="range" id="sm-top_k" min="0" max="100" step="1" value="20"><span class="sm-mark" id="m-sm-top_k" aria-hidden="true" hidden></span><p class="hint">Draws from this many of the likeliest words; 0 = no limit.</p></div>
                <div class="dial"><div class="row"><span><label class="label" for="sm-min_p">Minimum chance</label><span class="tech">min_p</span></span><output id="o-sm-min_p" for="sm-min_p"></output></div>
                  <input type="range" id="sm-min_p" min="0" max="0.5" step="0.01" value="0"><span class="sm-mark" id="m-sm-min_p" aria-hidden="true" hidden></span><p class="hint">Leaves out words below this share of the likeliest word's chance; 0 = none.</p></div>
                <div class="dial"><div class="row"><span><label class="label" for="sm-presence_penalty">Repetition brake</label><span class="tech">presence_penalty</span></span><output id="o-sm-presence_penalty" for="sm-presence_penalty"></output></div>
                  <input type="range" id="sm-presence_penalty" min="0" max="2" step="0.1" value="0"><span class="sm-mark" id="m-sm-presence_penalty" aria-hidden="true" hidden></span><p class="hint">Above 0, words already used come back less often.</p></div>
              </fieldset>
              <label class="acc-check"><input type="checkbox" id="sm-override"><span>Use these even when an app sends its own</span></label>
            </fieldset>
            <div class="actions"><button type="button" id="sm-apply" hidden disabled aria-disabled="true">Apply and reload</button><button type="button" id="sm-reset" disabled aria-disabled="true">Reset to the recommendation</button></div><p class="note" id="sm-apply-why" role="status"></p>
            <p class="note" id="sm-note" role="status"></p>
          </div></details>
          <details class="block wrap" id="presets-block"><summary><h2>Presets</h2><span class="sum-hint" id="presets-count"></span></summary>
            <div id="presets"></div>
            <div class="add-row preset-new"><input type="text" id="preset-name" aria-label="Name for a new preset" placeholder="A name, such as Long chats" maxlength="64" spellcheck="false" autocomplete="off">
              <button type="button" id="preset-save">Save as preset</button></div>
            <p class="note" id="preset-note" role="status"></p><p class="note" id="preset-why" role="status"></p>
          </details>
          <div class="wrap restore-row"><button type="button" id="restore" hidden>Restore last working settings</button><p class="note" id="restore-note"></p><p class="note" id="restore-why" role="status"></p></div>
        </div>
        <div id="view-library" hidden>
          <section class="block wrap" id="models-block"><div class="library-head"><h2>Models</h2><button type="button" id="add-open" aria-expanded="false" aria-controls="add">Add a model</button></div>
            <div id="model-list"></div><p class="note" id="model-list-why" role="status"></p>
            <div class="add" id="add" hidden>
              <div class="add-row" id="add-row"><input type="text" id="add-source" aria-label="Model to add" placeholder="owner/name, or a folder or .gguf path" spellcheck="false" autocomplete="off" autocapitalize="off">
                <button type="button" id="add-check">Check</button><button type="button" id="add-browse">Browse…</button></div>
              <div class="add-row" id="add-base-row" hidden><label class="label" for="add-base">Add as</label><select id="add-base"></select></div>
              <p class="add-info" id="add-info" role="status"></p>
              <button type="button" class="primary go" id="add-go" hidden>Add model</button> <button type="button" class="go" id="add-to" hidden>Write it elsewhere…</button>
            </div>
          </section>
          <div id="expert-details"><section class="block wrap" id="experts" hidden aria-labelledby="x-title">
            <h2 id="x-title">Where experts are read from<span class="x-model" id="x-model"></span></h2>
            <p class="x-lede" id="x-lede"></p>
            <div id="x-sources"></div>
            <div class="rsplit" aria-hidden="true"><span id="rs-mem"></span><span id="rs-hot"></span><span id="rs-drive"></span></div>
            <p class="x-said" id="x-said" role="status"></p>
            <p class="x-said" id="x-speed"></p>
            <p class="x-said x-unused" id="x-unused"></p>
            <div class="x-acts" id="x-acts"></div><p class="note" id="x-acts-why" role="status"></p>
            <fieldset class="prepare" id="x-change" hidden><h3 id="x-change-head">Where the hot copy goes</h3>
              <p id="x-change-text">Choose a drive; Origami makes the folder on it.</p>
              <div class="drives" id="x-drives" role="radiogroup" aria-labelledby="x-change-head"></div>
              <div class="x-size" id="x-size-row"><label class="label" for="x-size">Size</label>
                <span class="select"><select id="x-size"></select><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 4.5 6 8.5 10 4.5"/></svg></span>
                <span class="x-unit" id="x-unit" role="group" aria-label="Size in"><button type="button" class="small" id="x-unit-gb" data-unit="gb" aria-pressed="true">GB</button><button type="button" class="small" id="x-unit-pct" data-unit="pct" aria-pressed="false">% of experts</button></span>
                <div class="x-copy"><div class="x-graph" id="x-graph" hidden><span class="x-axis" id="x-yaxis">uses per expert</span><div class="x-plot" id="x-plot"></div>
                  <div class="x-axis x-ranks" aria-hidden="true"><span>most used</span><span>experts by rank</span><span>least used</span></div></div>
                  <input type="range" id="x-cap" min="0" max="1" step="1" value="0" aria-label="Hot copy size" aria-describedby="x-size-note x-size-edge">
                  <span class="sub" id="x-size-note"></span><span class="sub" id="x-size-edge"></span></div></div>
              <label class="x-keep" id="x-keep-row" hidden><input type="checkbox" id="x-keep"> Keep this size when another model is loaded</label>
              <div class="x-shared" id="x-shared"></div>
              <button type="button" class="primary" id="x-use">Use this</button>
              <div class="x-places" id="x-places-row" hidden><h3 id="x-places-head">More drives</h3>
                <div class="drives" id="x-places" role="group" aria-labelledby="x-places-head"></div>
                <button type="button" id="x-places-save" aria-describedby="x-places-why">Save drives</button><span class="x-why" id="x-places-why"></span></div></fieldset>
            <p class="note" id="experts-note" role="status"></p>
          </section></div>
          <section class="block wrap" id="store-folders">
            <div class="library-head"><h2>Drives and files</h2><span id="storage-total" class="muted"></span><button type="button" class="small" id="storage-refresh">Check again</button></div>
            <div id="storage-recovery"></div>
            <div id="storage"></div><div class="x-shared" id="disk-budget"></div><p class="note" id="storage-why" role="status"></p><p class="note" id="storage-read" role="status"></p><p class="note" id="storage-note" role="status"></p>
            <details class="folder-settings"><summary>Folders Origami uses</summary><div id="folders"></div><p class="note" id="folders-note" role="status"></p>
            <div class="actions"><button type="button" id="folders-reload" hidden>Reload now</button></div></details></section>
        </div>
        <div id="view-settings" hidden>
          <section class="block wrap conn" id="connect"><h2>Connect an app</h2>
            <p class="fhint">Any OpenAI-compatible app: this address, <span id="c-key">any API key</span>, this model.</p>
            <div class="conn-row"><span class="label" id="c-url-label">Address</span><code id="c-url" aria-labelledby="c-url-label"></code><button type="button" class="small" data-copy="c-url" aria-label="Copy the base URL">Copy</button></div>
            <div class="conn-row"><span class="label" id="c-model-label">Model</span><code id="c-model" aria-labelledby="c-model-label"></code><button type="button" class="small" data-copy="c-model" aria-label="Copy the model id">Copy</button></div>
            <details class="tuning-help"><summary>Thinking levels, and Pi</summary>
              <p class="fhint">Send <code>reasoning_effort</code> none, minimal, low, medium, high, xhigh or max to choose how long it thinks; apps that read <code>/v1/models</code> find the levels there. Pi reads a model's thinking levels only from its own settings: put this in <code>~/.pi/agent/models.json</code>, choose the model with <code>/model</code>, and <code>/thinking</code> offers every level.</p>
              <pre class="pv-text" id="c-pi" tabindex="0" aria-label="Pi's models.json"></pre><button type="button" class="small" data-copy="c-pi" aria-label="Copy Pi's models.json">Copy</button></details>
            <p class="note" id="connect-note" role="status"></p></section>
          <section class="block wrap" id="access"><h2>Who can reach Origami</h2><div class="sf">
            <div class="drives acc" id="acc-listen" role="radiogroup" aria-label="Who can reach Origami">
              <label><input type="radio" name="listen" id="acc-local" value="local"><span>This Mac only</span><span id="acc-local-hint">Apps and browsers on this Mac. The default.</span></label>
              <label><input type="radio" name="listen" id="acc-lan" value="lan"><span>This Mac and devices on this network</span><span>Also an iPad, a phone or another computer on the same Wi-Fi, with a key.</span></label></div>
            <div id="acc-key-row" hidden><label class="acc-check"><input type="checkbox" id="acc-key" aria-describedby="acc-key-hint"><span>Other devices need the key</span></label>
              <p class="fhint" id="acc-key-hint">Without it, anything on this network can use Origami and change its settings.</p></div>
            <div class="actions"><button type="button" id="acc-save">Save</button><button type="button" id="acc-restart" hidden>Restart Origami now</button></div><p class="note" id="acc-save-why" role="status"></p>
            <p class="note" id="acc-note" role="status"></p>
            <div id="acc-open-box" hidden><h3 id="acc-open-head">Open it on another device</h3>
              <p class="fhint" id="acc-open-hint"></p>
              <div class="acc-open"><img class="acc-qr" id="acc-qr" alt="" hidden>
                <div class="conn-row"><span class="label" id="acc-url-label">Address</span><code id="acc-url" aria-labelledby="acc-url-label"></code><button type="button" class="small" data-copy="acc-url" aria-label="Copy the address">Copy</button></div>
                <div class="conn-row" id="acc-more-row"><span class="label" id="acc-more-label">Or</span><code id="acc-more" aria-labelledby="acc-more-label"></code></div>
                <div class="conn-row" id="acc-key-show"><span class="label" id="acc-keytext-label">Key</span><code id="acc-keytext" aria-labelledby="acc-keytext-label"></code><button type="button" class="small" data-copy="acc-keytext" aria-label="Copy the key">Copy</button></div></div>
              <p class="note" id="acc-copy-note" role="status"></p></div>
          </div></section>
          <details class="block wrap running-only"><summary><h2>Log</h2></summary><div id="events"></div></details>
        </div>
        <div class="savebar" id="savebar" hidden><div class="wrap save-row"><div><span id="save-state" role="status">Connecting…</span><p class="note" id="apply-note" role="status"></p><p class="note" id="save-why" role="status"></p>
         <div id="review" hidden><p class="note" id="review-text" role="status"></p><div class="actions"><button type="button" class="primary" id="review-keep">Keep mine</button><button type="button" id="review-take">Take theirs</button></div></div></div>
         <div class="save-buttons"><button type="button" id="discard">Discard</button><button type="button" id="save">Save</button><button type="button" class="primary" id="apply" disabled>Save &amp; reload</button><button type="button" id="reload-saved" hidden>Reload to apply</button><button type="button" id="reset" hidden>Reset</button></div></div></div></fieldset></main>
        <footer><div class="wrap" id="footer"></div></footer>
        <dialog class="fs" id="fs" aria-labelledby="fs-title">
          <div class="fs-head"><h2 id="fs-title">Choose a folder</h2><button type="button" id="fs-close">Close</button></div>
          <nav aria-label="Folder path"><ol class="fs-crumbs" id="fs-crumbs"></ol></nav>
          <p class="note" id="fs-host"></p><div class="fs-path"><input type="text" id="fs-path" aria-label="Folder path on the server" placeholder="Paste a host folder path"><button type="button" id="fs-go">Go</button></div><input type="search" id="fs-search" aria-label="Filter folders" placeholder="Filter this folder">
          <div class="fs-body">
            <div class="fs-places"><span class="label" id="fs-places-label">Drives and places</span><ul id="fs-roots" aria-labelledby="fs-places-label"></ul></div>
            <div><span class="label" id="fs-here">Folders</span><ul id="fs-list" aria-labelledby="fs-here"></ul></div>
          </div>
          <button type="button" id="fs-more" hidden>Next page</button>
          <p class="note" id="fs-note" role="status"></p>
          <div class="fs-new" id="fs-new" hidden><input type="text" id="fs-name" aria-label="New folder name" placeholder="New folder name" spellcheck="false" autocomplete="off" autocapitalize="off">
            <button type="button" id="fs-make">Create</button></div>
          <div class="actions"><a class="button" id="fs-settings" href="x-apple.systempreferences:com.apple.preference.security?Privacy_FilesAndFolders" hidden>Open Privacy settings</a>
            <button type="button" id="fs-mkdir">New folder</button><button type="button" class="primary" id="fs-choose">Choose this folder</button></div>
        </dialog>
        <dialog class="fs" id="storage-confirm" aria-labelledby="storage-confirm-title"><div class="fs-head"><h2 id="storage-confirm-title"></h2><button type="button" id="storage-cancel">Cancel</button></div><div id="storage-confirm-body"></div><div class="actions"><button type="button" class="primary" id="storage-accept">Continue</button></div></dialog>
        <dialog class="fs pv" id="pv" aria-labelledby="pv-title">
          <div class="fs-head"><h2 id="pv-title">Saved prompt</h2><button type="button" id="pv-close">Close</button></div>
          <p class="note" id="pv-meta"></p>
          <pre class="pv-text" id="pv-text" tabindex="0" aria-label="Saved text"></pre>
          <textarea class="pv-text" id="pv-edit" aria-label="System prompt text" spellcheck="false" hidden></textarea>
          <div class="add-row"><input type="text" id="pv-name" aria-label="Name" placeholder="Name" spellcheck="false" autocomplete="off" autocapitalize="off">
            <button type="button" id="pv-rename">Rename</button></div>
          <p class="note" id="pv-note" role="status"></p>
          <div class="actions" id="pv-actions"><button type="button" id="pv-keep" aria-pressed="false">Keep</button>
            <button type="button" id="pv-edit-start">Edit text</button><button type="button" id="pv-delete">Delete</button></div>
          <div class="actions" id="pv-confirm" hidden><span>Delete this saved prompt and its files?</span>
            <button type="button" id="pv-delete-yes">Delete</button><button type="button" id="pv-delete-no">Cancel</button></div>
        </dialog>
        <script>
        \#(DashboardScript.js)
        </script>
        </body>
        </html>
        """#
}

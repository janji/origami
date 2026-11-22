enum DashboardStyles {
    static let css = #"""
        /* Paper and ink: one monospace face, hairlines, one copper accent for what is live or waits for you, green for
           ready and red for failed. Light is the same sheet turned over. */
        :root {
          --ground: #0b0b0a; --raise: #141413; --ink: #ecebe6; --ink-2: #a09f98; --line: #2b2b29; --dim: #62615c;
          --accent: #b4560f; --accent-ink: #e58a45; --ok: #9ccc7c; --bad: #ff6b5c; --track: #1e1e1c;
          --art-hi: 236, 235, 230; --art-lo: 72, 72, 69; --art-dot: 98, 97, 92; --art-bg: 11, 11, 10; --art-accent: 229, 138, 69;
          --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;
          /* Added to the home screen, the page runs full screen under a see-through status bar (and a notch, sideways). */
          --safe-top: env(safe-area-inset-top, 0px); --safe-right: env(safe-area-inset-right, 0px);
          --safe-bottom: env(safe-area-inset-bottom, 0px); --safe-left: env(safe-area-inset-left, 0px);
        }
        @media (prefers-color-scheme: light) {
          :root {
            --ground: #efeee9; --raise: #e5e4de; --ink: #141412; --ink-2: #55544f; --line: #cdccc5; --dim: #a09f98;
            --accent: #a4500c; --accent-ink: #944707; --ok: #35702a; --bad: #b3261a; --track: #d9d8d1;
            --art-hi: 20, 20, 18; --art-lo: 160, 159, 152; --art-dot: 180, 179, 172; --art-bg: 239, 238, 233; --art-accent: 164, 80, 12;
          }
        }
        * { box-sizing: border-box; }
        [hidden] { display: none !important; }
        html { -webkit-text-size-adjust: 100%; scrollbar-color: var(--line) var(--ground); }
        body { margin: 0; background: var(--ground); color: var(--ink); font: 14px/1.55 var(--mono); caret-color: var(--accent-ink); font-variant-numeric: tabular-nums; }
        ::selection { background: var(--ink); color: var(--ground); }
        :focus-visible { outline: 2px solid var(--accent-ink); outline-offset: 2px; }
        .label, th, summary, button, .button, .chip { font-size: 11px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; }
        .muted { color: var(--ink-2); }
        .wrap { max-width: 1180px; margin: 0 auto; padding: 0 max(32px, var(--safe-right)) 0 max(32px, var(--safe-left)); }
        fieldset { margin: 0; padding: 0; border: 0; min-width: 0; }
        code { font: inherit; font-size: 12px; color: var(--ink); background: var(--raise); padding: 1px 6px; overflow-wrap: anywhere; }
        a { color: var(--ink); text-underline-offset: 3px; }

        /* Top bar: the name, the state, Stop or Start, and the four views */
        .topbar { position: sticky; top: 0; z-index: 3; background: var(--ground); border-bottom: 1px solid var(--line); }
        /* The status bar's white text sits on black in both themes. */
        .topbar::before { content: ""; display: block; height: var(--safe-top); background: #000; }
        .topbar .bar { display: flex; align-items: center; gap: 16px; height: 52px; }
        .wordmark { font-size: 13px; font-weight: 600; letter-spacing: .42em; }
        .live { margin-left: auto; display: flex; align-items: center; gap: 14px; min-width: 0; }
        .live-text { font-size: 12px; color: var(--ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .chip { display: inline-flex; align-items: center; gap: 9px; border: 1px solid var(--line); padding: 6px 11px; white-space: nowrap; }
        .chip i { width: 8px; height: 8px; background: var(--dim); }
        .chip[data-state="ready"] i { background: var(--ok); }
        .chip[data-state="busy"] i, .chip[data-state="reloading"] i { background: var(--accent-ink); }
        /* Loading is a hollow square, working a filled one, failed a red one with a ring: they differ without colour too. */
        .chip[data-state="loading"] i { background: transparent; border: 1px solid var(--accent-ink); }
        .chip[data-state="failed"] i { background: var(--bad); outline: 1px solid var(--bad); outline-offset: 2px; }
        .chip[data-state="failed"] { border-color: var(--bad); }
        .chip[data-state="stopping"] i { background: transparent; border: 1px solid var(--ink-2); }
        /* The hot copy's chip shows only while the copy is being built; Health says the rest. */
        a.chip { color: var(--ink); text-decoration: none; }
        a.chip:not([data-state="busy"]) { display: none; }
        .x-chip-name { color: var(--ink-2); }
        .power { white-space: nowrap; padding: 6px 11px; }
        .power[data-starts="true"]:not(:disabled) { background: var(--ink); color: var(--ground); border-color: var(--ink); }
        @media (hover: hover) { .power[data-starts="true"]:hover:not(:disabled) { background: var(--ground); color: var(--ink); } }
        .readline { position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; }
        .readline span { position: absolute; top: 0; height: 2px; transition: width .25s ease-out, left .25s ease-out; }
        #reused { left: 0; background: var(--dim); }
        #read { background: var(--accent-ink); }
        /* The live text's key to the bar's reused part. */
        .key { display: inline-block; width: 7px; height: 7px; background: var(--dim); margin-right: 5px; }
        .views { display: flex; gap: 4px; margin-bottom: -1px; overflow-x: auto; scrollbar-width: none; }
        .views button { padding: 9px 14px 10px; border: 0; border-bottom: 2px solid transparent; color: var(--ink-2); white-space: nowrap; }
        .views button:first-child { margin-left: -14px; }
        .views button[aria-current="page"] { color: var(--ink); border-bottom-color: var(--accent-ink); }
        .views button i { font-style: normal; color: var(--accent-ink); margin-left: 6px; }
        .views button i:empty { display: none; }

        /* The crane, the model and the one next action */
        .hero { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 32px; align-items: center; padding-top: 22px; padding-bottom: 22px; }
        .hero > * { min-width: 0; }
        .art { display: block; width: 100%; height: 178px; }
        h1 { font-size: 26px; line-height: 1.15; font-weight: 600; letter-spacing: -0.02em; margin: 0; overflow-wrap: anywhere; }
        .lede { margin: 6px 0 0; color: var(--ink-2); font-size: 12px; }
        .lede:empty, .note:empty, .badges:empty { display: none; }
        .badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 0; }
        .badge { font-size: 11px; letter-spacing: .06em; padding: 3px 8px; border: 1px solid var(--line); color: var(--ink-2); white-space: nowrap; }
        .badge b { color: var(--ink); font-weight: 500; }
        .badge[data-b="differs"] { border-color: var(--accent-ink); color: var(--accent-ink); }
        .badge[data-b="differs"] b { color: inherit; }
        .picker { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; margin-top: 16px; }
        .picker .label { color: var(--ink-2); }
        .picker .select { flex: 1; max-width: 620px; }
        .select { position: relative; display: inline-flex; min-width: 0; max-width: 100%; }
        select { appearance: none; -webkit-appearance: none; font: inherit; font-size: 13px; color: var(--ink); background: var(--ground); border: 1px solid var(--line);
          border-radius: 0; padding: 8px 34px 8px 11px; min-width: 0; max-width: 100%; text-overflow: ellipsis; cursor: pointer; transition: border-color .15s ease-out;
          background-image: linear-gradient(45deg, transparent 50%, var(--ink-2) 50%), linear-gradient(135deg, var(--ink-2) 50%, transparent 50%);
          background-position: calc(100% - 17px) 50%, calc(100% - 12px) 50%; background-size: 5px 5px; background-repeat: no-repeat; }
        .select select { width: 100%; background-image: none; padding: 9px 40px 9px 12px; font-size: 14px; }
        select:hover:not(:disabled) { border-color: var(--ink-2); }
        .select svg { position: absolute; right: 12px; top: 50%; width: 12px; height: 12px; margin-top: -6px; pointer-events: none; stroke: var(--ink); fill: none; stroke-width: 1.5; }
        select:disabled { cursor: not-allowed; color: var(--ink-2); }
        input[type="text"], input[type="search"] { font: inherit; font-size: 14px; color: var(--ink); background: var(--ground); border: 1px solid var(--line); border-radius: 0; padding: 9px 12px;
          min-width: 0; flex: 1; transition: border-color .15s ease-out; }
        input[type="text"]:hover, input[type="search"]:hover { border-color: var(--ink-2); }
        input::placeholder { color: var(--ink-2); }
        .start { margin-top: 16px; }
        /* First run: what Origami found, each one press away */
        .first { margin-top: 16px; max-width: 620px; }
        .first p { margin: 0 0 10px; color: var(--ink-2); font-size: 13px; }
        .first .actions { margin-top: 12px; flex-wrap: wrap; }

        /* Notices: what keeps Origami from working, and how to fix it */
        .notice { padding-bottom: 22px; }
        .notice-box { border: 1px solid var(--accent-ink); padding: 16px 18px; }
        .notice[data-kind="failed"] .notice-box { border-color: var(--bad); }
        #banner .notice-box { border-color: var(--line); }
        #pending .actions { margin-top: 12px; }
        #st-reload, #health-reload { margin-top: 8px; }
        .notice h2, .job h2, .prepare h3 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }
        .notice p { margin: 6px 0 0; color: var(--ink-2); max-width: 78ch; font-size: 13px; overflow-wrap: anywhere; }
        #banner p { margin: 0; }
        .steps { margin: 12px 0 0; padding-left: 3ch; max-width: 78ch; font-size: 13px; }
        .steps li { padding: 2px 0; }
        .steps li::marker { color: var(--ink-2); }
        .notice-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
        /* Slowed by: one compact line per condition (what, then the fix), so the speed stays on the first screen */
        .slowed { padding-bottom: 16px; }
        .slowed .notice-box { padding: 10px 14px; }
        .slowed h2 { font-size: 13px; }
        .slowed[data-calm="1"] .notice-box { border-color: var(--line); }
        .slowed[data-calm="1"] h2, .slowed[data-calm="1"] li { color: var(--dim); }
        .slowed ul { margin: 4px 0 0; padding: 0; list-style: none; font-size: 13px; max-width: 120ch; }
        .slowed li { padding: 2px 0; color: var(--ink-2); overflow-wrap: anywhere; }
        .slowed li b { color: var(--ink); font-weight: 600; }

        /* A job: adding, converting, tuning, moving, with its progress */
        .job { padding-bottom: 22px; }
        .job-in { border: 1px solid var(--line); padding: 14px 18px 16px; }
        .job[data-state="running"] .job-in { border-color: var(--accent-ink); }
        .job-row { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; align-items: center; gap: 8px 22px; }
        .job-stage { margin: 4px 0 0; font-size: 12px; color: var(--ink-2); overflow-wrap: anywhere; }
        .job-figure { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }
        .job-track { position: relative; height: 3px; margin-top: 14px; background: var(--track); }
        .job-track span { position: absolute; left: 0; top: 0; height: 100%; background: var(--accent-ink); transition: width .25s ease-out; }
        .job[data-state="done"] .job-track span { background: var(--ok); }
        .job[data-state="failed"] .job-track, .job[data-state="cancelled"] .job-track { display: none; }

        /* Sections: a small heading that runs into a rule, as a terminal's divider does */
        .block { padding-top: 6px; padding-bottom: 26px; }
        .block h2, .rule { display: flex; align-items: center; gap: 12px; font: inherit; font-size: 11px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; margin: 0 0 14px; white-space: nowrap; }
        .block h2::after, .rule::after { content: ""; flex: 1; border-top: 1px solid var(--line); }
        .block h2 > *, .rule > * { order: 2; }
        details.block > summary { display: flex; align-items: center; gap: 10px; cursor: pointer; list-style: none; }
        details.block > summary h2 { flex: 1; margin: 0; }
        details.block > summary h2::before { content: "+"; width: 1ch; color: var(--accent-ink); font-weight: 400; }
        details.block[open] > summary h2::before { content: "−"; }
        details.block[open] > summary { margin-bottom: 14px; }
        summary::-webkit-details-marker { display: none; }
        /* The small disclosures open with the sections' mark: + closed, − open. */
        .tuning-help > summary, .folder-settings > summary, #memory-more > summary, .file-path > summary { list-style: none; }
        .tuning-help > summary::before, .folder-settings > summary::before, #memory-more > summary::before, .file-path > summary::before { content: "+"; display: inline-block; width: 2ch; color: var(--accent-ink); }
        .tuning-help[open] > summary::before, .folder-settings[open] > summary::before, #memory-more[open] > summary::before, .file-path[open] > summary::before { content: "−"; }
        .sum-hint { font-size: 12px; font-weight: 400; letter-spacing: 0; text-transform: none; color: var(--ink-2); white-space: normal; }
        .sum-hint:empty { display: none; }
        .note { margin: 10px 0 0; color: var(--ink-2); font-size: 12px; }
        .fhint { margin: 4px 0 8px; color: var(--ink-2); font-size: 12px; max-width: 72ch; }
        .empty { color: var(--ink-2); font-size: 13px; margin: 0; }
        .sub { display: block; color: var(--ink-2); font-size: 12px; }
        .tag { color: var(--ink-2); }
        .tag.kept { color: var(--accent-ink); }
        .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }

        /* Buttons, and Origami's state on them without words. Busy: the control at work turns a ring (with reduced motion a
           still, half-folded square) and takes no second press. Failed: red and a cross on the control that failed. Waiting
           for you: the accent border on the control to use. Disabled: greyed, no hover, the not-allowed cursor. */
        button, .button { font-family: inherit; color: var(--ink); background: transparent; border: 1px solid var(--line); border-radius: 0; padding: 7px 12px; cursor: pointer;
          transition: background-color .15s ease-out, color .15s ease-out, border-color .15s ease-out; }
        .button { display: inline-block; text-decoration: none; }
        /* Hover only where a pointer hovers: a touch screen keeps the last tap's hover, so a tapped Start would look filled as Stop. */
        @media (hover: hover) { button:hover:not(:disabled), .button:hover { background: var(--ink); color: var(--ground); border-color: var(--ink); }
          .views button:hover:not(:disabled) { background: transparent; color: var(--ink); border-color: transparent; border-bottom-color: var(--line); }
          .views button[aria-current="page"]:hover:not(:disabled) { border-bottom-color: var(--accent-ink); } }
        button:disabled { color: var(--dim); cursor: not-allowed; }
        button.primary { padding: 11px 18px; background: var(--ink); color: var(--ground); border-color: var(--ink); font-weight: 600; }
        @media (hover: hover) { button.primary:hover:not(:disabled) { background: var(--ground); color: var(--ink); } }
        button.primary:disabled { background: transparent; color: var(--dim); border-color: var(--line); }
        button.small, .measure { padding: 4px 9px; }
        @keyframes turn { to { transform: rotate(1turn); } }
        button[aria-busy="true"]::before, #job[aria-busy="true"] #job-title::before, #add-info[aria-busy="true"]::before,
        #fs[aria-busy="true"] #fs-here::before, #pv-text[aria-busy="true"]::before, #storage-read[aria-busy="true"]::before {
          content: ""; display: inline-block; width: 12px; height: 12px; margin-right: 10px; vertical-align: -2px;
          border: 1px solid var(--accent-ink); background: linear-gradient(135deg, var(--accent-ink) 50%, transparent 50%); }
        @media (prefers-reduced-motion: no-preference) {
          button[aria-busy="true"]::before, #job[aria-busy="true"] #job-title::before, #add-info[aria-busy="true"]::before,
          #fs[aria-busy="true"] #fs-here::before, #pv-text[aria-busy="true"]::before, #storage-read[aria-busy="true"]::before,
          .chip[data-state="loading"] i, .chip[data-state="reloading"] i, .chip[data-state="busy"] i, .chip[data-state="stopping"] i {
            background: none; border: 2px solid var(--accent-ink); border-right-color: transparent; border-radius: 50%; animation: turn .8s linear infinite; }
          .chip[data-state="loading"] i, .chip[data-state="reloading"] i, .chip[data-state="busy"] i, .chip[data-state="stopping"] i { width: 11px; height: 11px; margin: -2px; border-width: 1.5px; }
        }
        button[aria-busy="true"], button.primary[aria-busy="true"] { color: var(--ink-2); border-color: var(--ink-2); background: transparent; }
        button[data-failed] { color: var(--bad); border-color: var(--bad); background: transparent; }
        button[data-failed]:not([aria-busy="true"])::before, .job[data-state="failed"] #job-title::before { content: "×"; content: "×" / ""; margin-right: 8px; color: var(--bad); }
        input[aria-invalid="true"] { border-color: var(--bad); }
        select[data-ask], .button[data-ask], button[data-ask] { border-color: var(--accent-ink); box-shadow: 0 0 0 1px var(--accent-ink); }
        .button[data-ask], button[data-ask] { color: var(--accent-ink); }

        /* Now: the speed, then where the last request's time, the expert reads and the memory went */
        .now { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 40px; padding-top: 10px; padding-bottom: 30px; }
        .big { margin: 0; font-size: 92px; line-height: .95; font-weight: 600; letter-spacing: -0.04em; white-space: nowrap; }
        .big small { font-size: 15px; font-weight: 400; letter-spacing: 0; margin-left: 12px; color: var(--ink-2); }
        .big.none { font-size: 30px; line-height: 1.2; letter-spacing: -0.01em; color: var(--ink-2); white-space: normal; }
        .big.none small { display: none; }
        /* One mark per recent reply, as tall as it was fast. */
        .spark { margin: 14px 0 0; font: 22px/1 var(--mono); letter-spacing: .12em; color: var(--accent-ink); overflow: hidden; white-space: nowrap; }
        .spark:empty { display: none; }
        .basis { margin: 10px 0 0; color: var(--ink-2); font-size: 12px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
        .figures { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 22px 0 0; }
        .figures:empty { display: none; }
        .figures div { border-top: 1px solid var(--line); padding-top: 8px; min-width: 0; }
        .figures dt { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-2); }
        .figures dd { margin: 2px 0 0; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; }
        .figures dd small { font-size: 12px; font-weight: 400; color: var(--ink-2); margin-left: 4px; }
        .flows { display: grid; grid-template-columns: minmax(0, 1fr); gap: 20px; align-content: start; }
        .flow:empty { display: none; }
        .flow-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; font-size: 12px; color: var(--ink-2); }
        .flow-head .label { color: var(--ink); }
        /* A bar drawn in characters, one per part, as the memory strip is: the row clips a long run of each part's character. */
        .abar { display: flex; height: 1.3em; margin: 6px 0; font-size: 13px; line-height: 1.3; overflow: hidden; white-space: nowrap; background: var(--track); user-select: none; }
        .abar span { flex: none; overflow: hidden; transition: width .3s ease-out; }
        .abar.tall { height: 2.6em; white-space: normal; word-break: break-all; }
        [data-k="a"] { color: var(--ink); } [data-k="b"] { color: var(--accent-ink); } [data-k="c"] { color: var(--ink-2); } [data-k="d"] { color: var(--dim); } [data-k="e"] { color: var(--bad); }
        .akeys { display: flex; flex-wrap: wrap; gap: 2px 18px; font-size: 12px; color: var(--ink-2); }
        .akeys i { font-style: normal; margin-right: 6px; }
        .akeys b { color: var(--ink); font-weight: 500; }
        /* Every model's copies on a shared drive, and the disk budget beside storage. */
        .x-shared:empty { display: none; }
        .x-shared { margin: 14px 0; font-size: 12px; }
        .x-shared table { width: 100%; }
        .x-shared .x-keep { white-space: nowrap; }
        /* A phone: name and size on one line, what it keeps and gave below, the box last. */
        @media (max-width: 600px) {
          .x-shared tr { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 12px; padding: 8px 0; }
          .x-shared td { display: block; padding: 0; border: 0; }
          .x-shared td:nth-child(3), .x-shared td:nth-child(4) { grid-column: 1 / -1; }
        }
        #disk-budget p { margin: 4px 0; }
        /* The last request's breakdown and its verdict, under its bar. */
        .why { margin-top: 8px; font-size: 12px; color: var(--ink-2); }
        .why p { margin: 2px 0; overflow-wrap: anywhere; }
        .why b { color: var(--ink); font-weight: 500; }
        .why .verdict { margin-top: 6px; color: var(--ink); }
        .why .verdict[data-slow] { color: var(--accent-ink); }

        /* While a model loads or reloads, Now shows that alone: the numbers were the last model's. */
        .loadline { display: none; margin-top: 0; margin-bottom: 30px; color: var(--ink-2); }
        #page[data-state="loading"] .loadline, #page[data-state="reloading"] .loadline { display: block; }
        #page[data-state="loading"] #view-run > :not(.loadline), #page[data-state="reloading"] #view-run > :not(.loadline),
        #page[data-state="loading"] .badges, #page[data-state="reloading"] .badges { display: none; }

        /* Health: one line per thing that can need you; its mark says ready, working, look here or failed */
        .hrow { display: grid; grid-template-columns: 14px 11ch minmax(0, 1fr) auto; align-items: baseline; gap: 4px 14px; padding: 10px 0; border-top: 1px solid var(--line); font-size: 13px; }
        .hrow:first-child { border-top: 0; padding-top: 0; }
        .hrow .label { color: var(--ink-2); }
        .hrow b { font-weight: 600; }
        .hrow .sub { margin-top: 2px; max-width: 86ch; }
        .hrow .hact { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
        .mark { display: inline-block; width: 8px; height: 8px; margin-right: 9px; vertical-align: 1px; background: var(--dim); }
        .hrow .mark { margin: 6px 0 0; align-self: start; }
        .mark[data-m="ink"], .mark[data-m="ok"] { background: var(--ok); }
        .mark[data-m="live"], .mark[data-m="warn"] { background: var(--accent-ink); }
        .mark[data-m="warn"] { transform: rotate(45deg); }
        .mark[data-m="fail"] { background: var(--bad); outline: 1px solid var(--bad); outline-offset: 2px; }
        .mark[data-m="none"] { background: transparent; border: 1px solid var(--dim); }
        .hbar { display: block; position: relative; height: 3px; max-width: 46ch; margin: 7px 0 3px; background: var(--track); }
        .hbar span { position: absolute; left: 0; top: 0; height: 100%; background: var(--ink); }
        .hbar[data-m="warn"] span { background: var(--accent-ink); }
        .hbar i { position: absolute; top: -3px; width: 1px; height: 9px; background: var(--ink-2); }

        /* Tables: requests, saved prompts, sources, comparisons */
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; color: var(--ink-2); padding: 0 14px 8px 0; border-bottom: 1px solid var(--line); white-space: nowrap; }
        td { padding: 9px 14px 9px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
        tr:last-child td { border-bottom: 0; }
        td.r, th.r { text-align: right; white-space: nowrap; }
        td:last-child, th:last-child { padding-right: 0; }
        #recent .ending { display: none; }
        #recent td.t-split { width: 26%; max-width: 0; }
        #recent .abar { margin: 2px 0 0; font-size: 11px; background: none; }
        .events { max-height: 280px; overflow: auto; font-size: 12px; }
        .events div { display: grid; grid-template-columns: 64px 1fr; gap: 14px; padding: 6px 0; border-bottom: 1px solid var(--line); }
        .events time { color: var(--ink-2); }

        /* Setup: the dials and what each move does here */
        .tune { display: grid; grid-template-columns: 7fr 5fr; gap: 40px; padding-bottom: 12px; }
        .dial { padding: 16px 0 14px; border-bottom: 1px solid var(--line); }
        .dial:first-child { padding-top: 6px; }
        .dial:last-child, .dial:has(+ .dial[hidden]:last-child) { border-bottom: 0; }
        .dial .row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 6px 16px; }
        .dial output { font-size: 17px; font-weight: 600; letter-spacing: -0.01em; text-align: right; }
        /* A dial's words show when it is in use, switched off, or set to something that changes replies. */
        .dial .hint { display: none; margin: 8px 0 0; color: var(--ink-2); font-size: 12px; }
        .dial[data-off] .hint, .dial[data-differs] .hint, .dial:has(input:focus-visible) .hint, .dial:hover .hint, #d-memory:has(details[open]) .hint { display: block; }
        .dial[data-differs] output { color: var(--accent-ink); }
        .dial[data-off] output, .dial[data-off] .ticks, .dials:disabled output, .dials:disabled .ticks, .dials:disabled .ticks .on, .sm-dials:disabled output { color: var(--dim); }
        .dial-effect { margin: 6px 0 0; font-size: 12px; color: var(--accent-ink); }
        .dial-effect:empty { display: none; }
        .auto { margin-left: 10px; vertical-align: 2px; padding: 4px 9px; }
        .auto[aria-pressed="true"] { background: var(--ink); color: var(--ground); border-color: var(--ink); }
        .auto[aria-pressed="true"]:disabled { background: var(--dim); color: var(--ground); border-color: var(--dim); }
        input[type="range"] { -webkit-appearance: none; appearance: none; display: block; width: 100%; height: 22px; margin: 10px 0 0; background: transparent; cursor: pointer; }
        input[type="range"]:disabled { cursor: not-allowed; }
        input[type="range"]:disabled::-webkit-slider-runnable-track { background: linear-gradient(to right, var(--dim) 0 var(--fill, 0%), var(--track) var(--fill, 0%) 100%); }
        input[type="range"]:disabled::-moz-range-progress { background: var(--dim); }
        input[type="range"]::-webkit-slider-runnable-track { height: 2px; background: linear-gradient(to right, var(--ink) 0 var(--fill, 0%), var(--track) var(--fill, 0%) 100%); }
        input[type="range"]::-moz-range-track { height: 2px; background: var(--track); }
        input[type="range"]::-moz-range-progress { height: 2px; background: var(--ink); }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; margin-top: -6px; background: var(--ink); border: 0; border-radius: 0; transition: transform .15s ease-out; }
        input[type="range"]::-moz-range-thumb { width: 14px; height: 14px; background: var(--ink); border: 0; border-radius: 0; }
        input[type="range"]:hover::-webkit-slider-thumb { transform: scale(1.15); }
        input[type="range"]:focus-visible { outline: none; }
        input[type="range"]:focus-visible::-webkit-slider-thumb { outline: 2px solid var(--accent-ink); outline-offset: 3px; }
        input[type="range"]:focus-visible::-moz-range-thumb { outline: 2px solid var(--accent-ink); outline-offset: 3px; }
        input[type="range"]:disabled::-webkit-slider-thumb { background: var(--dim); transform: none; }
        input[type="range"]:disabled::-moz-range-thumb { background: var(--dim); }
        .ticks { position: relative; height: 18px; margin-top: 6px; font-size: 11px; color: var(--ink-2); }
        .ticks span { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; transition: color .15s ease-out; }
        .ticks span:first-child { transform: none; }
        .ticks span:last-child { transform: translateX(-100%); }
        .ticks span:only-child { left: auto !important; right: 0; transform: none; }
        .ticks .on { color: var(--ink); }
        .ticks .far { text-decoration: underline dotted; text-underline-offset: 3px; }
        .wired-memory { margin-top: 12px; font-size: 12px; }
        .wired-memory > details > summary { cursor: pointer; color: var(--ink-2); width: fit-content; }
        .wired-memory dl { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 16px; margin: 10px 0 0; }
        .wired-memory dt { color: var(--ink-2); }
        .wired-memory dd { margin: 0; text-align: right; }
        .wired-memory .note { max-width: 66ch; }
        #wired-state { color: var(--accent-ink); }
        .readout { position: sticky; top: 104px; align-self: start; padding-top: 6px; }
        .effects { border: 1px solid var(--accent-ink); padding: 12px 14px; margin-bottom: 20px; }
        .overall { margin: 0 0 8px; font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }
        .effect { display: grid; grid-template-columns: minmax(0, 1fr) auto 15ch; align-items: center; gap: 12px; padding: 4px 0; font-size: 12px; }
        .effect .what { color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .effect .word { text-align: right; white-space: nowrap; }
        /* A model switch is never cut short: both names show in full. */
        .effect .what.whole { color: var(--ink); white-space: normal; overflow-wrap: anywhere; }
        .scale { letter-spacing: .32em; color: var(--ink-2); white-space: nowrap; }
        .scale b { color: var(--accent-ink); font-weight: 600; }
        .mem .label { display: block; color: var(--ink-2); margin-bottom: 8px; }
        .mem canvas { display: block; width: 100%; height: 42px; }
        .keys { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; margin-top: 10px; font-size: 12px; color: var(--ink-2); }
        .keys span { white-space: nowrap; }
        .keys i { font-style: normal; margin-right: 8px; }
        .keys b { color: var(--ink); font-weight: 500; }
        .k-fixed { color: var(--ink); } .k-experts, .k-context { color: var(--accent-ink); } .k-others { color: var(--dim); } .k-freed { color: var(--ink-2); }
        .fits { margin: 16px 0 0; font-size: 12px; color: var(--ink-2); }
        .fits[data-state="warn"] { color: var(--accent-ink); }
        .fits:empty { display: none; }
        .prepare { margin: 18px 0 0; padding: 14px; border: 1px solid var(--accent-ink); }
        .prepare h3 { font-size: 15px; }
        .prepare p { margin: 8px 0 0; font-size: 13px; color: var(--ink-2); }
        .prepare button { margin-top: 12px; width: 100%; }
        .prepare .other { width: auto; }
        .drives { margin-top: 10px; font-size: 12px; }
        .drives label { display: grid; grid-template-columns: auto minmax(0, 1fr) auto 15ch; align-items: baseline; gap: 12px; padding: 6px 0; border-top: 1px solid var(--line); cursor: pointer; }
        .drives input, .acc-check input { margin: 0; accent-color: var(--accent); }
        .drives label > span:nth-of-type(1) { min-width: 0; overflow-wrap: anywhere; }
        .drives .no { color: var(--ink-2); }
        .drives .no, fieldset:disabled label { cursor: not-allowed; }
        .drives label > span:last-child { text-align: right; }
        /* Sampling: the model's recommendation per mode, or custom dials for every mode */
        .sm-modes { overflow-x: auto; margin-top: 12px; }
        .sm-modes th, .sm-modes td { padding-top: 6px; padding-bottom: 6px; font-size: 12px; }
        .sm-dials { margin-top: 12px; max-width: 72ch; }
        .tech { margin-left: 8px; color: var(--ink-2); font-size: 11px; letter-spacing: 0; text-transform: none; }
        #sm-source, .sm-mark { display: none; }
        #sm-choice label span:empty { display: none; }
        /* Tuning */
        .tune-options { display: flex; gap: 18px; flex-wrap: wrap; }
        .tune-options label { display: grid; gap: 6px; font-size: 12px; color: var(--ink-2); }
        .tuning-help { margin-top: 10px; font-size: 12px; }
        .tuning-help summary { cursor: pointer; color: var(--ink-2); width: fit-content; text-transform: none; letter-spacing: 0; font-size: 12px; }
        #st-result { margin-top: 16px; } #st-result:empty { display: none; }
        #st-result > p:first-child { margin: 0; font-size: 20px; font-weight: 600; }
        #st-result > button { margin-top: 10px; }

        /* Models: each one a row, the running one marked, a low-bit variant saying its replies differ */
        .library-head { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
        .library-head h2 { flex: 1; margin: 0; }
        .mrow { display: grid; grid-template-columns: 14px minmax(0, 1fr) auto; align-items: center; gap: 4px 14px; padding: 11px 0; border-top: 1px solid var(--line); }
        .mrow:first-child { border-top: 0; padding-top: 0; }
        .mrow .mark { margin: 0; }
        .mrow strong { font-weight: 600; overflow-wrap: anywhere; }
        .mrow .badge { margin-left: 8px; vertical-align: 1px; }
        .mrow .hact { display: flex; gap: 8px; }
        .mhead { margin: 18px 0 0; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-2); font-weight: 500; }
        .add { margin-top: 18px; padding: 14px; border: 1px solid var(--line); }
        .add-row { display: flex; gap: 10px; }
        .add-row + .add-row { margin-top: 10px; align-items: center; }
        .add-info { margin: 10px 0 0; font-size: 13px; color: var(--ink-2); }
        .add-info:empty { display: none; }
        .add .go { margin-top: 12px; }
        /* Experts: where the running model's experts are read from, each drive's role and state, and how reads split */
        #experts { scroll-margin-top: 110px; }
        .x-model { font-size: 12px; font-weight: 400; letter-spacing: 0; text-transform: none; color: var(--ink-2); white-space: normal; overflow-wrap: anywhere; }
        .x-lede { display: none; }
        #x-sources td { vertical-align: baseline; }
        #x-sources .x-role { white-space: nowrap; }
        #x-sources .x-state .sub { max-width: 60ch; }
        .x-acts [aria-expanded="true"] { background: var(--ink); color: var(--ground); border-color: var(--ink); }
        .job-track.x-track { display: block; max-width: 46ch; margin: 8px 0 6px; }
        .rsplit { display: flex; height: 3px; margin-top: 14px; background: var(--track); }
        .rsplit span { flex: none; height: 100%; width: 0; transition: width .25s ease-out; }
        #rs-mem { background: var(--accent-ink); } #rs-hot { background: var(--ink); } #rs-drive { background: var(--dim); }
        .x-said { margin: 8px 0 0; font-size: 12px; color: var(--ink-2); }
        .x-said:empty, .x-acts:empty { display: none; }
        .x-acts { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
        /* Why a button cannot act yet, beside it: a tooltip never shows on touch. */
        .x-pair { display: inline-flex; align-items: center; gap: 10px; }
        .x-why { font-size: 12px; color: var(--ink-2); }
        .x-unused button { margin-left: 6px; padding: 3px 9px; vertical-align: baseline; }
        .x-size { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-top: 12px; }
        .x-size .label { color: var(--ink-2); }
        /* The hot copy's size over the experts' popularity: bars inside the size solid, outside faint; the chosen edge copper, the
           copy now dashed; no room hatched. The plot is inset by half the slider's thumb so the thumb sits on the edge it sets. */
        .x-unit { display: inline-flex; }
        .x-unit button { white-space: nowrap; }
        .x-unit button + button { margin-left: -1px; }
        .x-unit [aria-pressed="true"] { background: var(--ink); color: var(--ground); border-color: var(--ink); }
        .x-copy { flex: 1 1 100%; max-width: 760px; }
        .x-copy .sub { display: block; margin-top: 4px; }
        .x-graph { margin: 4px 7px 0; }
        .x-axis { display: block; font-size: 11px; color: var(--ink-2); }
        .x-ranks { display: flex; justify-content: space-between; gap: 12px; margin-top: 4px; }
        .x-ranks span:nth-child(2) { text-align: center; }
        .x-plot { position: relative; height: 128px; margin-top: 4px; border-bottom: 1px solid var(--line); }
        .x-plot svg { display: block; width: 100%; height: 100%; }
        .x-plot rect { fill: var(--dim); opacity: .4; transition: opacity .12s ease-out; }
        .x-plot rect.in { fill: var(--ink); opacity: 1; }
        .x-noroom { position: absolute; top: 0; bottom: 0; right: 0; border-left: 1px solid var(--ink-2);
          background: repeating-linear-gradient(135deg, var(--line) 0 1px, transparent 1px 6px); }
        .x-noroom span { position: absolute; top: 20px; right: 4px; max-width: calc(100% - 8px); padding: 0 4px; font-size: 11px; line-height: 1.3; text-align: right;
          color: var(--ink-2); background: var(--ground); }
        .x-now, .x-edge { position: absolute; top: 0; bottom: -1px; width: 0; }
        .x-now { border-left: 1px dashed var(--ink-2); }
        .x-now::after { content: "now"; position: absolute; top: 2px; left: 4px; font-size: 10px; color: var(--ink-2); background: var(--ground); padding: 0 2px; }
        .x-edge { border-left: 2px solid var(--accent-ink); margin-left: -1px; }
        /* More drives: ranks a faster drive holds are hatched and named; each drive its own size. */
        .x-plot rect.on { fill: var(--ink-2); opacity: .55; }
        .x-on { position: absolute; top: 0; bottom: 0; background: repeating-linear-gradient(45deg, var(--line) 0 1px, transparent 1px 5px); border-right: 1px solid var(--ink-2); }
        .x-on span { position: absolute; top: 2px; left: 4px; max-width: calc(100% - 8px); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
          font-size: 10px; color: var(--ink-2); background: var(--ground); padding: 0 2px; }
        .x-places { margin-top: 18px; }
        .x-places h3 { margin: 0; font-size: 13px; }
        .x-place .x-copy { box-sizing: border-box; padding: 0 0 10px 26px; }
        .x-place > .sub { display: block; padding: 0 0 8px 26px; color: var(--ink-2); }
        .x-place input[type=range] { width: 100%; }
        .x-place label { grid-template-columns: auto minmax(0, 1fr) auto; }
        #x-places-save { margin-top: 10px; margin-right: 8px; }
        #x-cap { margin-top: 2px; }
        #x-cap:not(:disabled)::-webkit-slider-thumb { background: var(--accent-ink); }
        #x-cap:not(:disabled)::-moz-range-thumb { background: var(--accent-ink); }
        /* Storage: each drive a bar of what Origami's items take by role, the drive's other files and its free space, then the items */
        .drive-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 24px; margin-bottom: 20px; }
        .sdrive-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 2px 12px; font-size: 12px; color: var(--ink-2); }
        .sdrive h3 { margin: 0; font-size: 13px; font-weight: 600; color: var(--ink); }
        .sbar { display: flex; height: 8px; margin: 8px 0; background: var(--track); overflow: hidden; }
        .sbar span { flex: none; height: 100%; }
        .skeys { display: flex; flex-wrap: wrap; gap: 2px 14px; font-size: 11px; color: var(--ink-2); }
        .skeys i { display: inline-block; width: 8px; height: 8px; margin-right: 6px; }
        [data-role="model"] { background: var(--ink); } [data-role="pack"] { background: var(--ink-2); } [data-role="hot copy"] { background: var(--accent-ink); }
        [data-role="prompts"] { background: var(--accent); } [data-role="origami"] { background: var(--dim); } [data-role="other"] { background: var(--line); }
        .model-files { border-top: 1px solid var(--line); padding: 12px 0 4px; }
        .model-files h3 { margin: 0; font-size: 14px; overflow-wrap: anywhere; }
        .model-files > summary { display: flex; gap: 12px; align-items: baseline; letter-spacing: 0; text-transform: none; cursor: pointer; list-style: none; }
        .model-files > summary::before { content: "+"; color: var(--accent-ink); }
        .model-files[open] > summary::before { content: "−"; }
        .model-files > summary span { margin-left: auto; font-size: 12px; color: var(--ink-2); white-space: nowrap; }
        .sitems { list-style: none; margin: 6px 0 0; padding: 0 0 0 2ch; }
        .frow { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 9px 0; font-size: 13px; }
        .frow > div { min-width: 0; overflow-wrap: anywhere; }
        .sitems .frow { flex-wrap: wrap; border-top: 1px solid var(--line); }
        .file-title { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
        .file-location { font-size: 12px; color: var(--ink-2); }
        .file-path { margin-top: 2px; font-size: 11px; color: var(--ink-2); overflow-wrap: anywhere; }
        .file-path summary { font-size: 10px; display: inline; letter-spacing: .06em; cursor: pointer; }
        .file-path p { margin: 4px 0 0; }
        .sact { display: flex; flex-wrap: wrap; gap: 8px; }
        .sact button { font-size: 10px; padding: 6px 9px; letter-spacing: .06em; }
        .folder-settings { margin-top: 18px; }
        .folder-settings > summary { cursor: pointer; width: fit-content; color: var(--ink-2); }
        .folder-settings .fhint { display: none; }
        .fgroup { padding: 14px 0 0; }
        .fgroup h3 { margin: 0; font-size: 13px; font-weight: 600; }
        .fgroup > button { margin-top: 4px; }
        #storage-recovery p { margin: 0 0 12px; color: var(--accent-ink); }

        /* Connect an app, and Access: who reaches Origami, the address other devices open, with its QR code */
        .conn h3, #acc-open-box h3 { margin: 22px 0 0; font-size: 12px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
        .conn-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; margin: 8px 0; }
        .conn-row .label { min-width: 9ch; color: var(--ink-2); }
        .conn .pv-text { min-height: 0; max-height: 40vh; margin-bottom: 8px; }
        .acc label { grid-template-columns: auto minmax(0, 1fr); }
        .acc label > span:last-child { grid-column: 2; text-align: left; color: var(--ink-2); }
        .acc-check { display: flex; align-items: baseline; gap: 12px; margin-top: 14px; font-size: 12px; cursor: pointer; }
        .acc-check + .fhint { margin-left: 25px; }
        .acc-open { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 22px; align-items: start; margin-top: 8px; }
        .acc-qr { grid-row: span 3; width: 172px; height: 172px; padding: 8px; background: #fff; image-rendering: pixelated; }
        .acc-open .conn-row { margin: 0; }
        .acc-open code { white-space: pre-line; }
        .keyask { padding-top: 22px; }
        .keyask .add-row { margin-top: 14px; max-width: 60ch; }

        /* Dialogs: the folder browser, a confirmation, a saved prompt */
        .fs { width: min(780px, calc(100vw - 32px)); max-height: calc(100vh - 32px); padding: 20px; color: var(--ink); background: var(--ground); border: 1px solid var(--ink); }
        .fs::backdrop { background: rgba(0, 0, 0, .6); }
        /* iPadOS before 15.4 has no <dialog>: closed ones stay hidden, and the script opens one as a fixed overlay. */
        dialog:not([open]) { display: none; }
        dialog[data-fallback] { position: fixed; z-index: 10; top: calc(16px + var(--safe-top)); left: 0; right: 0; margin: 0 auto; overflow: auto; box-shadow: 0 0 0 100vmax rgba(0, 0, 0, .6); }
        .fs[open] { display: flex; flex-direction: column; gap: 12px; }
        .fs-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        .fs-head h2 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: -0.01em; overflow-wrap: anywhere; }
        .fs ul, .fs ol { list-style: none; margin: 0; padding: 0; }
        .fs-crumbs { display: flex; flex-wrap: wrap; font-size: 12px; }
        .fs-crumbs li + li::before { content: "/"; color: var(--dim); padding: 0 5px; }
        .fs-crumbs button, .fs-body li button { border: 0; text-transform: none; letter-spacing: 0; font-weight: 400; text-align: left; overflow-wrap: anywhere; }
        .fs-crumbs button { padding: 2px 0; font-size: 12px; color: var(--ink-2); }
        .fs-crumbs [aria-current] { color: var(--ink); font-weight: 600; }
        .fs-path { display: flex; gap: 8px; }
        #fs-search { flex: none; width: 100%; }
        .fs-body { display: grid; grid-template-columns: 210px minmax(0, 1fr); gap: 18px; min-height: 0; }
        .fs-body > div { min-width: 0; }
        .fs-body .label { display: block; color: var(--ink-2); margin-bottom: 6px; }
        .fs-body ul { max-height: 42vh; overflow: auto; }
        .fs-body li button { display: block; width: 100%; padding: 8px 4px; border-bottom: 1px solid var(--line); font-size: 13px; }
        .fs-body li i { font-style: normal; color: var(--ink-2); margin-right: 6px; }
        .fs-body li button:hover i, .fs-body li button:hover .sub { color: inherit; }
        .fs-new { display: flex; gap: 10px; }
        .fs .actions { margin-top: 0; }
        #storage-confirm-body p { overflow-wrap: anywhere; margin: 0 0 8px; }
        .pv-text { margin: 0; min-height: 120px; max-height: 46vh; overflow: auto; padding: 12px; font: 12px/1.5 var(--mono); color: var(--ink);
          background: var(--raise); border: 1px solid var(--line); white-space: pre-wrap; overflow-wrap: anywhere; }
        textarea.pv-text { width: 100%; height: 46vh; resize: vertical; }
        .pv .actions { align-items: center; }
        .pv [aria-pressed="true"] { color: var(--accent-ink); border-color: var(--accent-ink); }

        /* The save bar shows only while there is something to save, reload or review */
        .savebar { position: sticky; bottom: 0; z-index: 2; border-top: 1px solid var(--accent-ink); background: var(--ground); padding-bottom: var(--safe-bottom); }
        .save-row { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding-top: 12px; padding-bottom: 12px; }
        .save-buttons { display: flex; gap: 8px; flex-shrink: 0; }
        .save-buttons .primary { padding: 9px 14px; }
        .savebar .note { margin: 2px 0 0; }
        #save-state { font-size: 13px; }
        .restore-row { padding-bottom: 22px; }
        footer .wrap { padding-top: 18px; padding-bottom: calc(36px + var(--safe-bottom)); color: var(--ink-2); font-size: 11px; letter-spacing: .06em; border-top: 1px solid var(--line); }

        @media (max-width: 860px) {
          .wrap { padding-left: max(16px, var(--safe-left)); padding-right: max(16px, var(--safe-right)); }
          .live-text { display: none; }
          .hero { grid-template-columns: 104px minmax(0, 1fr); gap: 14px; padding-top: 16px; padding-bottom: 16px; align-items: start; }
          .art { height: 96px; }
          h1 { font-size: 19px; }
          .picker .label { display: none; }
          .picker, .badges, .start, .first, #model-note { grid-column: 1 / -1; }
          .hero-main { display: contents; }
          .hero-name { min-width: 0; align-self: center; }
          .now, .tune { grid-template-columns: minmax(0, 1fr); gap: 24px; }
          .big { font-size: 68px; }
          .readout { position: static; }
          .keys { grid-template-columns: 1fr; }
          .wide { display: none; }
          #recent td { white-space: nowrap; }
          #recent th { white-space: normal; }
          #recent .sub { display: none; }
          /* The ending column is hidden here: an interrupted or failed request says so under its prompt. */
          #recent .sub.ending { display: block; white-space: normal; }
          .hrow { grid-template-columns: 14px minmax(0, 1fr); }
          .hrow .label { grid-column: 2; }
          .hrow > div, .hrow .hact { grid-column: 2; justify-content: flex-start; }
          .mrow { grid-template-columns: 14px minmax(0, 1fr); }
          .mrow .hact { grid-column: 2; }
          .job-row { grid-template-columns: minmax(0, 1fr) auto; }
          .job-row button { grid-column: 1 / -1; justify-self: start; }
          .add-row { flex-wrap: wrap; }
          .save-row { align-items: stretch; flex-direction: column; gap: 8px; }
          .save-buttons button { flex: 1; padding: 9px; }
          .sitems { padding-left: 0; }
          .sitems .frow { flex-direction: column; align-items: flex-start; }
          /* Each place the experts are read from becomes a block: drive and role (reads at the right), then size and state. */
          #x-sources thead { display: none; }
          #x-sources table, #x-sources tbody { display: block; }
          #x-sources tr { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 2px 14px; padding: 11px 0; border-bottom: 1px solid var(--line); }
          #x-sources tr:last-child { border-bottom: 0; }
          #x-sources td { padding: 0; border: 0; }
          #x-sources .x-reads { grid-column: 3; grid-row: 1; }
          #x-sources .x-role { white-space: normal; }
          #x-sources .x-reads[data-l]::before { content: attr(data-l) " "; color: var(--ink-2); }
          #x-sources .x-sz { grid-column: 1; grid-row: 2; text-align: left; color: var(--ink-2); }
          #x-sources .x-state { grid-column: 2 / -1; grid-row: 2; }
          .drives label { grid-template-columns: auto minmax(0, 1fr) auto; }
          .drives label > span:last-child { grid-column: 2 / -1; text-align: left; color: var(--ink-2); }
          .sm-modes th, .sm-modes th.r { white-space: normal; vertical-align: bottom; }
          .fs { padding: 16px; }
          .fs-body { grid-template-columns: 1fr; gap: 12px; }
          .fs-places ul { display: flex; flex-wrap: wrap; gap: 6px; max-height: none; }
          .fs-places li button { border: 1px solid var(--line); padding: 5px 9px; }
          .fs-body #fs-list { max-height: 38vh; }
          .acc-open { grid-template-columns: 1fr; }
          .acc-qr { grid-row: auto; }
        }
        /* A phone's header keeps the state and the first word of Stop and unload. */
        @media (max-width: 520px) {
          #version, .power-more, .x-hot, .x-chip-name { display: none; }
          .topbar .bar { gap: 10px; }
          .live { gap: 8px; }
          .wordmark { letter-spacing: .3em; }
          .chip { padding: 6px 8px; letter-spacing: .06em; gap: 7px; }
          .views button { padding-left: 10px; padding-right: 10px; }
          .views button:first-child { margin-left: -10px; }
          .figures dd { font-size: 16px; }
          .figures dd small { display: block; margin: 0; }
          .big { font-size: 56px; }
        }
        /* A finger gets a taller target than a pointer needs. */
        @media (pointer: coarse) { button.small, .measure, .auto, .sact button { min-height: 36px; } .views button { min-height: 44px; } }
        @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
        """#
}

#!/usr/bin/env python3
"""Isolated, model-free dashboard API checks. Never connects to an existing server."""
import base64, hashlib, json, os, pathlib, re, shutil, socket, struct, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request, uuid
binary = str(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.build/debug/origami').resolve())
checks = 0

def check(condition, message):
    global checks
    assert condition, message
    checks += 1
    print('ok   ', message, flush=True)

def lowbit_gguf(path, typ, n):
    """A GLM GGUF of headers and zeros (as in cli-checks.sh): its expert banks of ggml type `typ`, `n` experts each."""
    block, size = {8: (32, 34), 10: (256, 84)}[typ]
    out = bytearray(b'GGUF'); u32 = lambda v: out.extend(struct.pack('<I', v)); u64 = lambda v: out.extend(struct.pack('<Q', v))
    def s(t): u64(len(t)); out.extend(t.encode())
    meta = [('general.architecture', 'glm5-next'), ('glm5-next.embedding_length', 256), ('glm5-next.vocab_size', 512), ('glm5-next.expert_count', 4),
            ('glm5-next.expert_used_count', 1), ('glm5-next.expert_feed_forward_length', 256), ('glm5-next.leading_dense_block_count', 1),
            ('glm5-next.block_count', 3), ('glm5-next.attention.head_count_kv', [0, 64, 64]), ('glm5-next.hyper_connection.count', 4)]
    experts = ['blk.%d.ffn_%s_exps.weight' % (l, p) for l in (1, 2) for p in ('gate', 'up', 'down')]
    small = ['token_embd.weight', 'output.weight', 'blk.0.ffn_norm.weight']
    bank, little = n * 256 * 256 // block * size, 1024
    u32(3); u64(len(experts) + len(small)); u64(len(meta))
    for k, v in meta:
        s(k)
        if isinstance(v, str): u32(8); s(v)
        elif isinstance(v, list): u32(9); u32(5); u64(len(v)); [out.extend(struct.pack('<i', x)) for x in v]
        else: u32(4); u32(v)
    for i, name in enumerate(experts): s(name); u32(3); [u64(d) for d in (256, 256, n)]; u32(typ); u64(i * bank)
    for i, name in enumerate(small): s(name); u32(1); u64(256); u32(0); u64(len(experts) * bank + i * little)
    out.extend(bytes((32 - len(out) % 32) % 32)); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out) + bytes(bank * len(experts) + little * len(small)))

with tempfile.TemporaryDirectory(prefix='origami-control-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'; data.mkdir()
    models = [data / 'models' / name for name in ('first', 'second')]
    for model in models:
        model.mkdir(parents=True); (model / 'config.json').write_text('{}'); (model / 'weights.bin').write_bytes(b'model fixture')
    registry = data / 'models.json'
    variant = data / 'models' / 'second-q2'; variant.mkdir(); (variant / 'config.json').write_text('{}')
    registry.write_text(json.dumps({'models': [{'name': model.name, 'path': str(model), 'manifest': {'fingerprint': ('a' if model.name == 'first' else 'b') * 64}} for model in models]
        + [{'name': 'second-q2', 'path': str(variant), 'base': 'second', 'experts': str(variant / 'experts.gguf')}]}))
    store = data / 'store'; payload = store / ('first-' + 'a' * 12); payload.mkdir(parents=True)
    for stem in ('pack', 'pack-3bit', 'mirror'):
        (payload / (stem + '.bin')).write_bytes((stem + ' fixture').encode())
        (payload / (stem + '.json')).write_text(json.dumps({'format': 'origami-store-v1', 'expert_file': stem + '.bin'}))
    (payload / 'hot').mkdir(); (payload / 'hot' / 'index.json').write_text('{}')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = root / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'model': 'first', 'context': '32k', 'sampling': 'custom', 'temperature': 1.1, 'stopped': True, 'models_dirs': '[]', 'port': port}))
    config.chmod(0o600)
    env = {k:v for k,v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_TEST_MOVE_PAUSE'] = str(root / 'move-pause')  # pauses a copying move only while that file exists
    log = open(root / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    base = f'http://127.0.0.1:{port}/'
    def call(path, body=None, revision=None):
        headers = {'Content-Type': 'application/json', 'Origin': base.rstrip('/')}
        if revision: headers['If-Match'] = revision
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as error: return error.code, json.load(error)
    def state(): return call('v1/dashboard')[1]['dashboard']
    def job(body):
        code, result = call('v1/jobs', body); assert code == 202, (code,result)
        until = time.time() + 20
        while time.time() < until:
            current = call('v1/jobs')[1]['job']
            if current['state'] != 'running':
                assert current['state'] == 'done', current
                return current
            time.sleep(.05)
        raise AssertionError('operation timed out')
    def items(): return [item for drive in call('v1/storage')[1]['drives'] for item in drive['items']]
    try:
        for _ in range(100):
            try: snapshot = state(); break
            except (OSError, urllib.error.URLError): time.sleep(.05)
        else: raise AssertionError('isolated server did not start')
        check(not snapshot['can_tune'] and snapshot['configured_model'] == 'first', 'stopped control plane never loads a model')
        check(snapshot['pending_changes'] == [] and snapshot['pending_revert'] == {} and not snapshot['reload_required'], 'with no model loaded nothing waits for a reload')
        with urllib.request.urlopen(base + 'dashboard') as response:
            html = response.read().decode(); csp = response.headers['Content-Security-Policy']
        script = re.search(r'<script>\n([\s\S]*?)</script>', html).group(1)
        digest = base64.b64encode(hashlib.sha256(('\n'+script).encode()).digest()).decode()
        check("'sha256-" + digest + "'" in csp, 'assembled dashboard script matches the served CSP')
        page = root / 'dashboard.html'; page.write_text(html)
        if shutil.which('node'):
            result = subprocess.run(['node', str(pathlib.Path(__file__).with_name('dashboard-checks.mjs')), str(page)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert result.returncode == 0, result.stdout
            check(True, 'dashboard logic and state regression checks pass against the served page')
        else: print('skip  Tests/dashboard-checks.mjs (no node)')
        listed = {entry['id']: entry['origami'] for entry in call('v1/models?all')[1]['data'] if entry.get('origami')}
        check(listed['second-q2'].get('variant_of') == 'second' and 'variant_of' not in listed['second'], 'a low-bit expert variant names its exact base in the model listing')
        code, refusal = call('v1/jobs', {'kind': 'add', 'source': str(models[1]), 'base': '--yes'})
        check(code == 400 and 'base' in refusal['error']['message'], 'an add job refuses a base that could be read as an option')
        # A low-bit experts file for a GLM GGUF: the pairing is checked before any job; what it cannot verify is asked,
        # a refusal says why, and only an acknowledged add registers it.
        kept_registry = registry.read_bytes()
        lowbit_gguf(root / 'glm' / 'base.gguf', 8, 4); lowbit_gguf(root / 'lx' / 'lo.gguf', 10, 4); lowbit_gguf(root / 'lx' / 'wide.gguf', 10, 8)
        listed_models = json.loads(kept_registry)['models'] + [{'name': 'glmbase', 'path': str(root / 'glm' / 'base.gguf'), 'family': 'glm'}]
        registry.write_text(json.dumps({'models': listed_models}))
        code, asked = call('v1/jobs', {'kind': 'add', 'source': str(root / 'lx' / 'lo.gguf'), 'base': 'glmbase'})
        check(code == 200 and "only its banks' shapes say it is glmbase" in asked.get('confirm', '') and call('v1/jobs')[1]['job'] is None
              and all(m['name'] != 'lo' for m in json.loads(registry.read_text())['models']), 'an unverifiable experts file is asked about first; nothing starts or is registered')
        code, refusal = call('v1/jobs', {'kind': 'add', 'source': str(root / 'lx' / 'wide.gguf'), 'base': 'glmbase', 'acknowledged': True})
        check(code == 409 and 'the base\'s [4, 256, 256]' in refusal['error']['message'] and call('v1/jobs')[1]['job'] is None,
              'a pairing refused is answered with its reason, before any job')
        job({'kind': 'add', 'source': str(root / 'lx' / 'lo.gguf'), 'base': 'glmbase', 'acknowledged': True})
        check(any(m.get('name') == 'lo' and m.get('base') == 'glmbase' for m in json.loads(registry.read_text())['models']),
              'an acknowledged add registers the variant')
        # Its path is the base's GGUF: whichever comes first in models.json, the storage map's item there is the base's,
        # and unlisting the base keeps the variant (deleting the variant must never delete the base's files).
        added = json.loads(registry.read_text())['models']
        registry.write_text(json.dumps({'models': [m for m in added if m.get('name') == 'lo'] + [m for m in added if m.get('name') != 'lo']}))
        at_base = [i for i in items() if i['role'] == 'model' and i['path'] == str(root / 'glm' / 'base.gguf')]
        check(len(at_base) == 1 and at_base[0].get('model') == 'glmbase', "a GLM variant gets no storage item of its own on its base's file")
        job({'kind': 'forget', 'path': at_base[0]['path'], 'revision': state()['revision']})
        check([m['name'] for m in json.loads(registry.read_text())['models'] if m['name'] in ('lo', 'glmbase')] == ['lo']
              and (root / 'glm' / 'base.gguf').exists(), 'unlisting the base keeps its variant registered')
        # Its experts file unreadable (a drive unplugged): its key is unknown, so no count or copy is listed as an orphan.
        counts = data / 'family' / '0123456789abcdef'; counts.mkdir(parents=True); (counts / 'uses.json').write_text('{}')
        registry.write_text(json.dumps({'models': [dict(m, experts=str(root / 'unplugged' / 'lo.gguf')) if m.get('name') == 'lo' else m for m in added]}))
        check(not any(i['path'] == str(counts) for i in items()), "a variant whose experts file cannot be read leaves unknown keys unknown")
        shutil.rmtree(counts)
        registry.write_bytes(kept_registry)
        revision = snapshot['revision']
        code, _ = call('v1/config', {'context': '64k', 'sampling': 'recommended', 'temperature': ''}, revision)
        check(code == 200 and json.loads(config.read_text())['context'] == '32k', 'per-model save preserves global defaults')
        saved = state()['saved_settings']
        check(saved['context']['value'] == '64k' and saved['sampling']['value'] == 'recommended', 'model overrides and recommended sampling round-trip')
        code, refusal = call('v1/config', {'context': '16k'}, revision)
        check(code == 409 and refusal['changes'] == [{'key': 'context', 'base': '32k', 'current': '64k'}] and state()['saved_settings']['context']['value'] == '64k',
              'a write based on an older revision is refused when its key changed since, naming the value it was based on and the current one')
        config.write_text(json.dumps(dict(json.loads(config.read_text()), reasoning_effort='high')))
        code, refusal = call('v1/config', {'reasoning_effort': 'low'}, revision)
        check(code == 409 and refusal['changes'][0]['key'] == 'reasoning_effort', 'a change made outside the page (an edit of the settings file) counts too')
        code, _ = call('v1/config', {'quality': 'exact'}, revision)
        check(code == 200, 'a write based on an older revision goes through when none of its keys changed since')
        revision = state()['revision']
        config.write_text(json.dumps({k: v for k, v in json.loads(config.read_text()).items() if k != 'reasoning_effort'}))
        wired = state()['wired_memory']
        check(0 < wired['safe_max_budget_bytes'] < wired['physical_memory_bytes']
              and wired['os_headroom_bytes'] == wired['physical_memory_bytes'] - wired['safe_max_budget_bytes']
              and wired['persistent'] is False and wired['active_budget_bytes'] is None,
              'wired memory reports machine-derived headroom, reboot lifetime and no active budget while stopped')
        files_before = (config.read_bytes(), registry.read_bytes())
        system_before = wired['system_limit_mib']
        above_safe = f"{(wired['safe_max_budget_bytes'] + 1048576) / 1e9:.9f}GB"
        for values in ({'memory_limit': above_safe}, {'memory_limit': '0'},
                       {'memory_limit': '1GB; exit 0'}, {'context': '64k'},
                       {'memory_limit': '1GB', 'context': 'invalid'}):
            code, _ = call('v1/memory/wired', values, state()['revision'])
            assert code == 400, (values, code)
            assert (config.read_bytes(), registry.read_bytes()) == files_before
        check(state()['wired_memory']['system_limit_mib'] == system_before,
              'invalid wired-memory changes are rejected before settings or system mutation')
        code, _ = call('v1/memory/wired', {'memory_limit': '1GB', 'reasoning_effort': 'low'}, revision)
        check(code == 409, 'wired-memory saves reject stale revisions where a key changed since')
        if wired['supported']:
            code, result = call('v1/memory/wired', {'memory_limit': '1GB', 'context': '64k'}, state()['revision'])
            check(code == 200 and result['wired_memory']['configured_budget_bytes'] == 1_000_000_000
                  and result['wired_memory']['system_limit_mib'] == system_before
                  and result['reload_required'] is False and result['wired_memory']['active_budget_bytes'] is None,
                  'coordinated save updates model budget without authorization or reload when stopped and ceiling already suffices')
            code, result = call('v1/memory/wired', {'memory_limit': 'auto'}, state()['revision'])
            check(code == 200 and result['wired_memory']['configured_budget_bytes'] is None
                  and result['wired_memory']['system_limit_mib'] == system_before,
                  'automatic budgeting never lowers the system ceiling')
        call('v1/config', {'model': 'second'})
        check(state()['saved_settings']['context']['value'] == '32k', 'a second model retains its own defaults')
        call('v1/config', {'model': 'first'})
        comma = data / 'Models, archived'; comma.mkdir()
        call('v1/config', {'models_dirs': json.dumps([str(comma)])})
        check(call('v1/fs')[1]['settings']['models_dirs'][0]['path'] == str(comma), 'folder lists support commas in paths')
        code, _ = call('v1/config', {'models_dirs': '["relative/path"]'})
        check(code == 400, 'relative host paths are rejected')
        listing = items(); copies = [x for x in listing if x['role'] in ('pack', 'hot copy')]
        retired = [x for x in copies if x['path'].endswith('pack-3bit.json')]
        copies = [x for x in copies if x not in retired]
        check(len(retired) == 1 and retired[0]['unused'] and 'move' not in retired[0]['actions'] and 'earlier Origami' in retired[0]['note'],
              'a 3-bit pack an earlier version made is listed as unused, never moved')
        check(len(copies) == 3 and all('move' in x['actions'] for x in copies) and any(x['path'].endswith('/hot') for x in copies)
              and any(x['path'].endswith('mirror.json') and x['note'].startswith('offline mirror') for x in copies), 'full and hot copies (and a 3.0 mirror) are separate movable items')
        destination = data / 'destination'; destination.mkdir()
        hot = next(x for x in copies if x['role'] == 'hot copy')
        code, preview = call('v1/storage/plan', {'kind': 'move', 'path': hot['path'], 'to': str(destination)})
        check(code == 200 and preview['copy_bytes'] == 0, 'same-volume move preview needs no duplicate disk space')
        job({'kind': 'move', 'path': hot['path'], 'to': str(destination), 'revision': preview['revision']})
        moved = destination / payload.name / 'mirror.json'
        check(moved.exists() and not pathlib.Path(hot['path']).exists() and (payload / 'pack.bin').exists(), 'moving a hot copy preserves sibling full copies')
        check(state()['saved_settings']['hot_copy_dir']['value'] == str(destination), 'cache placement follows the move')
        job({'kind': 'delete', 'path': str(moved)})
        check(not moved.exists() and state()['saved_settings']['hot_copy']['value'] == 'off', 'explicit hot-copy deletion disables automatic regeneration')
        # Further hot copy places (`hot_copy_more`), saved from the card as the full list of places after the first.
        place = data / 'second-place'
        code, saved = call('v1/hot-copy', {'places': [{'folder': str(place), 'size': '40GB'}]}, state()['revision'])
        check(code == 200 and saved['saved'] == ['hot_copy_more'] and state()['saved_settings']['hot_copy_more']['value'] == str(place) + '=40.0 GB',
              'a further hot copy place is saved with its own cap')
        # A copy in that further place: Move moves that place and Delete turns that place off; the first place is untouched.
        further = place / ('first-' + 'a' * 12) / 'hot'; further.mkdir(parents=True); (further / 'index.json').write_text('{}')
        elsewhere = data / 'place-moved'; elsewhere.mkdir()
        first_place = {k: state()['saved_settings'][k]['value'] for k in ('hot_copy', 'hot_copy_dir')}
        job({'kind': 'move', 'path': str(further), 'to': str(elsewhere)})
        saved = state()['saved_settings']
        check((elsewhere / further.parent.name / 'hot' / 'index.json').exists() and saved['hot_copy_more']['value'] == str(elsewhere) + '=40.0 GB'
              and {k: saved[k]['value'] for k in first_place} == first_place, 'moving a copy in a further place moves that place, not the first')
        job({'kind': 'delete', 'path': str(elsewhere / further.parent.name / 'hot')})
        saved = state()['saved_settings']
        check(saved['hot_copy_more']['value'] == str(elsewhere) + '=off' and {k: saved[k]['value'] for k in first_place} == first_place,
              'deleting a copy in a further place turns that place off, not the first')
        # Another registered model's copy in its own further place: Move and Delete change that model's own list.
        def second_places():
            entry = next(m for m in json.loads(registry.read_text())['models'] if m['name'] == 'second')
            return json.loads(entry['hot_copy_more'])
        own, own_moved = data / 'second-own', data / 'second-moved'; own_moved.mkdir()
        listed = json.loads(registry.read_text())
        next(m for m in listed['models'] if m['name'] == 'second')['hot_copy_more'] = json.dumps([{'folder': str(own), 'size': 30000000000}])
        registry.write_text(json.dumps(listed))
        copy = own / ('second-' + 'b' * 12) / 'hot'; copy.mkdir(parents=True); (copy / 'index.json').write_text('{}')
        job({'kind': 'move', 'path': str(copy), 'to': str(own_moved)})
        check(second_places() == [{'folder': str(own_moved), 'size': 30000000000}] and state()['saved_settings']['hot_copy_more']['value'] == str(elsewhere) + '=off',
              "moving another model's copy moves that model's own place: " + repr(second_places()))
        job({'kind': 'delete', 'path': str(own_moved / copy.parent.name / 'hot')})
        check(second_places() == [{'folder': str(own_moved), 'size': 0}] and state()['saved_settings']['hot_copy_more']['value'] == str(elsewhere) + '=off',
              "deleting another model's copy turns that model's own place off: " + repr(second_places()))
        listed = json.loads(registry.read_text()); next(m for m in listed['models'] if m['name'] == 'second').pop('hot_copy_more')
        registry.write_text(json.dumps(listed))
        code, saved = call('v1/hot-copy', {'places': [{'folder': str(place), 'size': '40GB'}]}, state()['revision'])
        code, saved = call('v1/hot-copy', {'places': [{'folder': str(place)}, {'folder': str(data / 'third')}]}, state()['revision'])
        check(code == 409 and 'same drive' in saved['error']['message'], 'a second place on one drive is refused')
        code, saved = call('v1/hot-copy', {'places': []}, state()['revision'])
        check(code == 200 and state()['saved_settings']['hot_copy_more']['value'] == '(none)', 'removing every further place')
        # The size slider sends a size in GB to a tenth (a share of the experts is converted on the page): saved as bytes.
        sizes = {}
        for size in ('18.5GB', 'auto', '40GB', 'off'):
            code, _ = call('v1/hot-copy', {'size': size}); assert code == 200, code
            sizes[size] = state()['saved_settings']['hot_copy']['value']
        check(sizes == {'18.5GB': '18.5 GB', 'auto': '(automatic)', '40GB': '40.0 GB', 'off': 'off'},
              'hot_copy keeps its meaning: a size in decimal GB, automatic or off: ' + repr(sizes))
        script = re.search(r'<script>\n([\s\S]*?)</script>', urllib.request.urlopen(base + 'dashboard').read().decode()).group(1)
        check(all(field in script for field in ('popularity', 'room_bytes', 'counted_uses', 'held_bytes', 'tokens')), 'the page reads the hot copy row\'s popularity and each drive\'s room')
        target = data / 'relocated-models'; target.mkdir()
        job({'kind': 'move', 'path': str(models[0]), 'to': str(target)})
        check((target / 'first' / 'weights.bin').read_bytes() == b'model fixture', 'model move preserves contents and updates its registered path')
        check(json.loads(registry.read_text())['models'][0]['path'] == str(target / 'first'), 'registry points to the verified destination')
        # A moved GLM model keeps what is keyed on its identity (path, file and dates): its counts, its hot copy and its
        # tuning go to the new key; a stop between "committed" and the journal's end is finished by Recover.
        kept_models = registry.read_bytes()
        glm_home = root / 'glm-move'; lowbit_gguf(glm_home / 'm.gguf', 8, 4)
        call('v1/config', {'models_dirs': json.dumps([str(glm_home)])})
        for _ in range(20):  # a search under way answers with the last one; asked again until this file is found
            key = next((m['key'] for m in call('v1/models/found?fresh')[1]['models'] if m['path'] == str(glm_home / 'm.gguf')), None)
            if key: break
            time.sleep(.5)
        registry.write_text(json.dumps({'models': json.loads(kept_models)['models'] + [{'name': 'glmmv', 'path': str(glm_home / 'm.gguf'), 'family': 'glm'}]}))
        hot_dir = pathlib.Path(state()['saved_settings']['hot_copy_dir']['value'])
        def seed(k):
            (data / 'family' / k).mkdir(parents=True); (data / 'family' / k / 'uses.json').write_text('{"3": 7}')
            (hot_dir / ('family-' + k) / 'hot').mkdir(parents=True); (hot_dir / ('family-' + k) / 'hot' / 'index.json').write_text(json.dumps({'identity': k + '|expert-source'}))
            (data / 'tuning.json').write_text(json.dumps({k[:12] + '@32768': {'fixture': True}}))
        def carried(old, what):
            keys = [p.name for p in (data / 'family').iterdir() if re.fullmatch('[0-9a-f]{16}', p.name)]
            new = keys[0] if len(keys) == 1 else None
            check(new is not None and new != old and (data / 'family' / new / 'uses.json').read_text() == '{"3": 7}'
                  and new in (hot_dir / ('family-' + new) / 'hot' / 'index.json').read_text() and not (hot_dir / ('family-' + old)).exists()
                  and new[:12] in (data / 'tuning.json').read_text() and old[:12] not in (data / 'tuning.json').read_text(), what + ': ' + repr(keys))
            return new
        seed(key)
        moved_to = data / 'glm-moved'; moved_to.mkdir()
        job({'kind': 'move', 'path': str(glm_home / 'm.gguf'), 'to': str(moved_to)})
        key = carried(key, 'a moved model keeps its counts, hot copy and tuning under its new key')
        # As if stopped right after "committed": the file renamed and the registry rewritten, the journal left.
        again = data / 'glm-again'; again.mkdir(); os.rename(moved_to / 'm.gguf', again / 'm.gguf')
        registry.write_text(registry.read_text().replace(str(moved_to / 'm.gguf'), str(again / 'm.gguf')))
        ident = str(uuid.uuid4()); (data / 'moves').mkdir(exist_ok=True)
        (data / 'moves' / (ident + '.json')).write_text(json.dumps({'id': ident, 'name': 'glmmv', 'phase': 'committed', 'renames': True, 'kind': 'model', 'model': 'glmmv',
            'pairs': [{'from': str(moved_to / 'm.gguf'), 'to': str(again / 'm.gguf')}],
            'rebind': {'path': str(again / 'm.gguf'), 'model': 'glmmv', 'own': key, 'related': {}}}))
        job({'kind': 'recover', 'id': ident})
        key = carried(key, 'a move stopped between committed and finish is re-bound by Recover')
        # A move that fails (the copy's folder at the destination is read-only) puts everything back and leaves no journal.
        blocked = data / 'ro'; (blocked / ('family-' + key)).mkdir(parents=True); (blocked / ('family-' + key)).chmod(0o555)
        code, _ = call('v1/jobs', {'kind': 'move', 'path': str(hot_dir / ('family-' + key) / 'hot'), 'to': str(blocked)})
        until = time.time() + 20
        while call('v1/jobs')[1]['job']['state'] == 'running' and time.time() < until: time.sleep(.05)
        failed = call('v1/jobs')[1]['job']
        (blocked / ('family-' + key)).chmod(0o755)
        check(code == 202 and failed['state'] == 'failed' and (hot_dir / ('family-' + key) / 'hot' / 'index.json').exists()
              and not list((data / 'moves').glob('*.json')), 'a failed move puts the copy back and leaves no journal: ' + repr(failed.get('error')))
        shutil.rmtree(data / 'family'); (data / 'tuning.json').unlink()
        registry.write_bytes(kept_models); call('v1/config', {'models_dirs': '[]'})
        # A Qwen variant's generated folder is links to its base's files: no Move. Any folder linking into another
        # item's files is refused a move, which would copy them whole.
        kept_models = registry.read_bytes()
        vq = data / 'variants' / 'vq'; vq.mkdir(parents=True); (vq / 'config.json').write_text('{}'); (vq / 'weights.bin').symlink_to(models[1] / 'weights.bin')
        linky = data / 'models' / 'linky'; linky.mkdir(); (linky / 'config.json').write_text('{}'); (linky / 'weights.bin').symlink_to(models[1] / 'weights.bin')
        listed = json.loads(kept_models)['models']
        registry.write_text(json.dumps({'models': listed + [{'name': 'vq', 'path': str(vq), 'base': 'second'}, {'name': 'linky', 'path': str(linky)}]}))
        at = {x.get('model'): x for x in items() if x['role'] == 'model'}
        check('move' not in at['vq']['actions'] and 'variant' in (at['vq'].get('note') or ''), "a variant's folder of links offers no Move, saying why")
        code, refusal = call('v1/storage/plan', {'kind': 'move', 'path': str(linky), 'to': str(target)})
        check(code == 409 and 'links to' in refusal['error']['message'] and not (target / 'linky').exists(),
              "a folder linking into another model's files is not moved")
        job({'kind': 'forget', 'path': str(vq)})
        check(not vq.exists() and (models[1] / 'weights.bin').read_bytes() and linky.exists(),
              "unlisting a variant removes its generated folder of links, not the files they point to")
        registry.write_bytes(kept_models)
        # A move that copies (a folder holding a link is copied, not renamed) and whose source file is written during the
        # copy keeps the original, removes the copy and names the file.
        kept_models = registry.read_bytes()
        linked = data / 'models' / 'hflike'; linked.mkdir(); (linked / 'config.json').write_text('{}')
        (linked / 'blob.bin').write_bytes(b'original bytes'); (linked / 'weights.bin').symlink_to('blob.bin')
        registry.write_text(json.dumps({'models': json.loads(kept_models)['models'] + [{'name': 'hflike', 'path': str(linked)}]}))
        pause = root / 'move-pause'; pause.write_text('')
        code, _ = call('v1/jobs', {'kind': 'move', 'path': str(linked), 'to': str(target)})
        until = time.time() + 20
        while not (root / 'move-pause.copied').exists() and time.time() < until: time.sleep(.05)
        with open(linked / 'blob.bin', 'ab') as writer: writer.write(b' and more')
        pause.unlink()
        while call('v1/jobs')[1]['job']['state'] == 'running' and time.time() < until: time.sleep(.05)
        failed = call('v1/jobs')[1]['job']
        check(code == 202 and failed['state'] == 'failed' and 'blob.bin changed while it was copied' in (failed.get('error') or '')
              and (linked / 'blob.bin').read_bytes() == b'original bytes and more' and not (target / 'hflike').exists()
              and not list(target.glob('.origami-moving-*')) and json.loads(registry.read_text())['models'][-1]['path'] == str(linked),
              'a source file written during a copying move keeps the original and removes the copy: ' + repr(failed.get('error')))
        job({'kind': 'move', 'path': str(linked), 'to': str(target)})
        check((target / 'hflike' / 'weights.bin').read_bytes() == b'original bytes and more' and not linked.exists(),
              'the same move with no writer copies and deletes the original')
        shutil.rmtree(target / 'hflike'); registry.write_bytes(kept_models)
        # Recover a completed cross-volume copy interrupted before original cleanup.
        original = data / 'old-copy'; copied = data / 'verified-copy'; original.write_bytes(b'verified bytes'); copied.write_bytes(original.read_bytes())
        recovery = data / 'moves'; recovery.mkdir(exist_ok=True); ident = str(uuid.uuid4())
        (recovery / (ident+'.json')).write_text(json.dumps({'id':ident,'name':'fixture','phase':'committed','renames':False,'pairs':[{'from':str(original),'to':str(copied)}]}))
        job({'kind':'recover','id':ident})
        check(not original.exists() and copied.read_bytes() == b'verified bytes', 'interrupted move cleanup rechecks both copies before deleting the source')
        original.write_bytes(b'original'); copied.write_bytes(b'different'); ident = str(uuid.uuid4())
        (recovery / (ident+'.json')).write_text(json.dumps({'id':ident,'name':'fixture','phase':'committed','renames':False,'pairs':[{'from':str(original),'to':str(copied)}]}))
        try: job({'kind':'recover','id':ident})
        except AssertionError: pass
        check(original.exists() and copied.exists() and (recovery / (ident+'.json')).exists(), 'recovery preserves both locations when verification fails')
        job({'kind':'forget','path':str(target/'first')})
        check((target/'first').exists() and not any(x['name']=='first' for x in json.loads(registry.read_text())['models']), 'unlisting keeps model files on disk')
        code, _ = call('v1/config', {'memory_auto_measure': 'true'}, state()['revision'])
        check(code == 200 and state()['saved_settings']['memory_auto_measure']['value'] == 'true', "the Memory line's switch saves memory_auto_measure")
        check('starting ' not in (root/'server.log').read_text(), 'all checks ran without starting an inference process')
    except Exception:
        log.flush(); print((root/'server.log').read_text(), file=sys.stderr); raise
    finally:
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()

# Saved versus running with a model "loaded": a stub worker stands in for the model process (ORIGAMI_TEST_WORKER) and
# answers as a ready model, so the server's own comparison of saved and running settings is exercised without a model.
WORKER = r"""#!/usr/bin/env python3
import http.server, json, sys
port = int(sys.argv[sys.argv.index('--port') + 1])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({'status': 'ready', 'queue': {'busy': False, 'queued': 0}} if self.path.startswith('/v1/status') else {'status': 'ready'}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
"""
with tempfile.TemporaryDirectory(prefix='origami-pending-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'; model = data / 'models' / 'first'; model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (data / 'models.json').write_text(json.dumps({'models': [{'name': 'first', 'path': str(model), 'manifest': {'fingerprint': 'a' * 64}}]}))
    worker = root / 'worker.py'; worker.write_text(WORKER); worker.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = root / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'model': 'first', 'context': '32k', 'models_dirs': '[]', 'port': port})); config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_TEST_WORKER'] = str(worker)
    log = open(root / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    base = f'http://127.0.0.1:{port}/'
    def call(path, body=None, revision=None):
        headers = {'Content-Type': 'application/json', 'Origin': base.rstrip('/')}
        if revision: headers['If-Match'] = revision
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as error: return error.code, json.load(error)
    try:
        for _ in range(200):
            try:
                snapshot = call('v1/dashboard')[1]['dashboard']
                if snapshot['active_model'] and snapshot['tune_block'] == '': break
            except (OSError, urllib.error.URLError, KeyError): pass
            time.sleep(.05)
        else: raise AssertionError('the stub worker did not become ready')
        check(not snapshot['reload_required'] and snapshot['pending_changes'] == [] and snapshot['can_tune'], 'a model running its saved settings waits for nothing')
        code, _ = call('v1/config', {'context': '64k', 'memory_limit': '1GB'}, snapshot['revision'])
        assert code == 200, code
        snapshot = call('v1/dashboard')[1]['dashboard']
        changes = {c['key']: (c['saved'], c['running']) for c in snapshot['pending_changes']}
        check(snapshot['reload_required'] and changes == {'context': ('64k', '32k'), 'memory_limit': ('1.0 GB', 'automatic')},
              'saved settings that differ in memory and context are listed with their saved and running values: ' + repr(changes))
        check(snapshot['pending_revert'] == {'context': '32k', 'memory_limit': 'auto'}, 'Discard the saved change offers the running values: ' + repr(snapshot['pending_revert']))
        check(snapshot['tune_block'] == 'Reload to apply saved settings first' and not snapshot['can_tune'], 'tuning waits for the reload, in the words the page shows')
        code, refusal = call('v1/jobs', {'kind': 'tune', 'mode': 'memory'})
        check(code == 409 and refusal['error']['message'] == 'Reload to apply saved settings first', 'the server refuses a tuning job with the same words')
        for arguments in (['tune'], ['tune', '--memory']):
            result = subprocess.run([binary, *arguments, '--config', str(config)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            check(result.returncode == 1 and 'Reload to apply saved settings first' in result.stdout, 'origami ' + ' '.join(arguments) + ' from the command line is refused with the same words')
        code, _ = call('v1/config', snapshot['pending_revert'], snapshot['revision'])
        snapshot = call('v1/dashboard')[1]['dashboard']
        check(code == 200 and not snapshot['reload_required'] and snapshot['pending_changes'] == [] and snapshot['pending_revert'] == {},
              'posting pending_revert makes the saved settings the running ones again')
        # Every setting a model reads once asks for a reload (`Settings.needsReload`), not only the per-model ones.
        config.write_text(json.dumps(dict(json.loads(config.read_text()), max_thinking_tokens=512, gpu_keepalive=False))); code = 200  # set with origami config
        snapshot = call('v1/dashboard')[1]['dashboard']
        check(code == 200 and snapshot['reload_required'] and {c['key'] for c in snapshot['pending_changes']} == {'max_thinking_tokens', 'gpu_keepalive'},
              'a saved max_thinking_tokens or gpu_keepalive asks for a reload: ' + repr((snapshot['pending_changes'], snapshot['reload_required'], snapshot['saved_settings'].get('max_thinking_tokens'), json.loads(config.read_text()))))
        # One listed model (`list_models` off): any name is the setup's, so it reaches the model (this stub has no POST: 501).
        try: urllib.request.urlopen(urllib.request.Request(base + 'v1/chat/completions', data=b'{"model":"gpt-4o","messages":[]}', headers={'Content-Type': 'application/json'}), timeout=15); code = 200
        except urllib.error.HTTPError as error: code = error.code
        check(code == 501, 'with list_models off, any model name reaches the model')
    except Exception:
        log.flush(); print((root / 'server.log').read_text(), file=sys.stderr); raise
    finally:
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()

# `origami start` after a failed start (a drive not readable yet): it starts the model again, as stop then start does.
FLAKY = WORKER.replace("port = int(", """import os, pathlib
marker = pathlib.Path(os.environ['FLAKY_MARKER'])
if not marker.exists():
    marker.write_text('failed once')
    print('origami: the model drive is not readable yet', file=sys.stderr, flush=True); sys.exit(1)
port = int(""", 1)
with tempfile.TemporaryDirectory(prefix='origami-restart-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'; model = data / 'models' / 'first'; model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (data / 'models.json').write_text(json.dumps({'models': [{'name': 'first', 'path': str(model), 'manifest': {'fingerprint': 'a' * 64}}]}))
    worker = root / 'worker.py'; worker.write_text(FLAKY); worker.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = root / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'model': 'first', 'context': '32k', 'models_dirs': '[]', 'port': port})); config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_TEST_WORKER'] = str(worker); env['FLAKY_MARKER'] = str(root / 'failed-once')
    log = open(root / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    try:
        for _ in range(200):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response: health = json.load(response)
            except urllib.error.HTTPError as error: health = json.load(error)
            except (OSError, urllib.error.URLError): health = {}
            if health.get('status') == 'failed': break
            time.sleep(.05)
        else: raise AssertionError('the first start did not fail')
        result = subprocess.run([binary, 'start', '--config', str(config)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        check(result.returncode == 0 and 'The last start failed' in result.stdout and 'not readable yet' in result.stdout and 'Ready: first runs again.' in result.stdout,
              'origami start after a failed start starts the model again and says so: ' + result.stdout.strip())
    except Exception:
        log.flush(); print((root / 'server.log').read_text(), file=sys.stderr); raise
    finally:
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()

# The front switching models by name (`list_models` on), with a stub worker per model: `broken` exits as it starts,
# `unplugged` is on a drive that is not connected, and a request names what it gets.
FRONT = WORKER.replace("port = int(", """import os, time
model = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'first'
if model.startswith('slow'): print('reading the dense weights', file=sys.stderr, flush=True); time.sleep(12)
if model.endswith('broken'):
    print('origami: the model drive is not readable yet', file=sys.stderr, flush=True); sys.exit(1)
port = int(""", 1).replace("    def do_GET(self):", """    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        if model == 'crasher': os.kill(os.getpid(), 30)  # SIGUSR1: ends the process like a crash, without macOS's crash dialog
        if body.get('stream'):
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Connection', 'close'); self.end_headers()
            self.wfile.write(('data: ' + json.dumps({'model': model, 'pid': os.getpid()}) + '\\n\\ndata: [DONE]\\n\\n').encode()); return
        reply = json.dumps({'model': model, 'pid': os.getpid(), 'limit': body.get('origami_time_limit')}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(reply))); self.end_headers(); self.wfile.write(reply)
    def do_GET(self):""", 1)
with tempfile.TemporaryDirectory(prefix='origami-front-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'
    entries = []
    for name in ('first', 'second', 'broken', 'slow', 'slowbroken', 'crasher'):
        folder = data / 'models' / name; folder.mkdir(parents=True); (folder / 'config.json').write_text('{}')
        entries.append({'name': name, 'path': str(folder), 'manifest': {'fingerprint': 'a' * 64}})
    entries.append({'name': 'unplugged', 'path': '/Volumes/origami-check-absent/unplugged', 'manifest': {'fingerprint': 'b' * 64}})
    (data / 'models.json').write_text(json.dumps({'models': entries}))
    worker = root / 'worker.py'; worker.write_text(FRONT); worker.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = root / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'model': 'first', 'context': '32k', 'models_dirs': '[]', 'port': port, 'list_models': True}))
    config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_TEST_WORKER'] = str(worker)
    log = open(root / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    base = f'http://127.0.0.1:{port}/'
    def ask(model, timeout=60):
        request = urllib.request.Request(base + 'v1/chat/completions', data=json.dumps({'model': model, 'messages': []}).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as error: return error.code, json.load(error)
    try:
        for _ in range(200):
            try:
                code, served = ask('first', timeout=5)
                if code == 200: break
            except (OSError, urllib.error.URLError): pass
            time.sleep(.05)
        else: raise AssertionError('the stub worker did not become ready')
        code, refusal = ask('unplugged')
        again = ask('first')[1]
        check(code == 503 and 'is not connected' in refusal['error']['message'] and again['pid'] == served['pid'],
              'a model that cannot start (its drive gone) never stops the one that serves: ' + refusal['error']['message'])
        code, refusal = ask('broken')
        back = ask('first')[1]
        check(code == 503 and 'did not start' in refusal['error']['message'] and back['model'] == 'first'
              and 'starting first' in (root / 'server.log').read_text().split('broken did not start')[-1],
              'a model that fails once the old one stopped brings the old one back, and its request gets the start error')
        code, refusal = ask('gpt-4o')
        check(code == 404 and refusal['error'].get('code') == 'model_not_found', 'with list_models on, an unlisted name is 404 model_not_found')
        check(ask('origami')[0] == 200 and ask('second')[1]['model'] == 'second', "`origami` and a listed name are served")
        # A model process that dies under a request (a crash): the client gets a 502, not an empty reply, the log names
        # the signal, and the next request starts a model again.
        code, refusal = ask('crasher')
        for _ in range(100):
            if 'crashed with signal 30' in (root / 'server.log').read_text(): break
            time.sleep(.05)
        check(code == 502 and 'stopped while answering' in refusal['error']['message']
              and 'crashed with signal 30' in (root / 'server.log').read_text() and ask('second')[1]['model'] == 'second',
              'a model process that crashes under a request gives a 502 and a log line with its signal: ' + refusal['error']['message'])
        # A stream waiting on a switch (`slow` takes 12 s to listen) gets its headers and keepalives, then the model's
        # own stream without a second status line; a start that fails after the headers is a stream error event.
        def stream(model, path='v1/chat/completions', half_close=False):
            body = json.dumps({'model': model, 'messages': [], 'stream': True}).encode()
            with socket.create_connection(('127.0.0.1', port), timeout=90) as sock:
                sock.sendall(f'POST /{path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)
                if half_close: sock.shutdown(socket.SHUT_WR)
                started, first, data = time.time(), None, b''
                while chunk := sock.recv(65536):
                    first = first or time.time() - started
                    data += chunk
            return first, data.decode()
        # While it loads: the status, `origami status` and a waiting client's 503 say for how long, with its last line.
        loading = {}
        def watch():
            time.sleep(4)
            try: urllib.request.urlopen(base + 'v1/status', timeout=5)
            except urllib.error.HTTPError as error: loading['status'] = json.load(error)
            else:
                with urllib.request.urlopen(base + 'v1/status', timeout=5) as response: loading['status'] = json.load(response)
            loading['cli'] = subprocess.run([binary, 'status', '--config', str(config)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30).stdout
            try: urllib.request.urlopen(base + 'v1/plan', timeout=5)
            except urllib.error.HTTPError as error: loading['refusal'] = json.load(error)['error']['message']
        watcher = threading.Thread(target=watch); watcher.start()
        first, text = stream('slow')
        watcher.join()
        progress = (loading.get('status') or {}).get('loading') or {}
        check(loading['status']['status'] == 'loading' and progress.get('seconds', 0) >= 3 and progress.get('line') == 'reading the dense weights'
              and 'Live      loading slow for ' in loading['cli'] and ': reading the dense weights' in loading['cli']
              and 'last: reading the dense weights' in loading.get('refusal', ''),
              'loading says its seconds and the model\'s last line: ' + repr((loading.get('status'), loading.get('refusal'))))
        check(first is not None and first < 11.5 and text.startswith('HTTP/1.1 200') and ': keepalive' in text and text.count('HTTP/1.1') == 1
              and '"model": "slow"' in text and text.endswith('data: [DONE]\n\n'),
              'a stream waiting on a switch gets its headers and a keepalive, then the model\'s stream: ' + repr(text[:400]))
        first, text = stream('slowbroken', 'v1/messages')
        check(text.startswith('HTTP/1.1 200') and ': keepalive' in text and 'event: error' in text and 'did not start' in text
              and ask('first')[1]['model'] == 'first', 'a start that fails after the headers arrives as a stream error in the request\'s format: ' + repr(text[-300:]))
        # A stream whose client shuts its sending side after the request still gets all of it; a request that is not a
        # stream, waiting in the queue, whose client closed is dropped (no switch is made for it).
        first, text = stream('second', half_close=True)
        check('"model": "second"' in text and text.endswith('data: [DONE]\n\n'), 'a half-closed stream client gets its whole reply: ' + repr(text[-200:]))
        slow = threading.Thread(target=lambda: ask('slow')); slow.start()
        time.sleep(1)
        before = (root / 'server.log').read_text().count('starting first')
        body = json.dumps({'model': 'first', 'messages': []}).encode()
        with socket.create_connection(('127.0.0.1', port), timeout=10) as sock:
            sock.sendall(f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)
        slow.join(); time.sleep(3)
        check((root / 'server.log').read_text().count('starting first') == before, 'a queued non-stream request whose client closed is dropped')
        # `origami_time_limit` counts from the request's arrival: a cold switch's wait comes off it, and one that outlives
        # it is the 408 it promises, before the model sees it.
        def limited(model, limit):
            request = urllib.request.Request(base + 'v1/chat/completions', data=json.dumps({'model': model, 'messages': [], 'origami_time_limit': limit}).encode(),
                                             headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response: return response.status, json.load(response)
            except urllib.error.HTTPError as error: return error.code, json.load(error)
        ask('first')  # the check before this one may have left `slow` loaded: start from another model
        code, seen = limited('slow', 60)
        check(code == 200 and 45 < seen['limit'] <= 48.5, 'a time limit counts from arrival: the model got what the 12 s switch left of 60 s: ' + repr(seen))
        code, seen = limited('first', 60)
        check(code == 200 and 59 < seen['limit'] <= 60, 'a short switch takes only its own seconds off the limit: ' + repr(seen))
        code, refusal = limited('slow', 5)
        check(code == 408 and refusal['error'].get('type') == 'request_timeout', 'a limit that passed during the switch is a 408 request_timeout')
        # J59: `start_on_request` and `idle_unload_minutes` (both off by default).
        def health():
            try:
                with urllib.request.urlopen(base + 'health', timeout=5) as response: return json.load(response)
            except urllib.error.HTTPError as error: return json.load(error)
        def until(state, seconds=30):
            end = time.time() + seconds
            while time.time() < end:
                if health().get('status') == state: return True
                time.sleep(.1)
            return False
        stop = urllib.request.Request(base + 'v1/stop', data=b'{}', headers={'Content-Type': 'application/json', 'Origin': base.rstrip('/')})
        urllib.request.urlopen(stop, timeout=10).close()
        check(until('stopped') and ask('first')[0] == 503, 'by default a request to a stopped server gets 503 stopped')
        config.write_text(json.dumps(dict(json.loads(config.read_text()), start_on_request=True)))
        code, served = ask('second')
        check(code == 200 and served['model'] == 'second' and 'stopped' not in json.loads(config.read_text()),
              'with start_on_request a request after a stop starts its model, and the stop is no longer saved')
        config.write_text(json.dumps(dict(json.loads(config.read_text()), idle_unload_minutes=0.02)))
        check(until('stopped', 15) and json.loads(config.read_text()).get('stopped') is True and 'idle_unload_minutes' in (root / 'server.log').read_text(),
              'idle_unload_minutes unloads the model after that long without a request, saved as a stop is')
        code, served = ask('first')
        check(code == 200 and served['model'] == 'first', 'and the next request starts it again with start_on_request')
        # The idle time counts from the last of a request, a start and a load's or reply's end: an explicit start is not
        # unloaded at the next look, nor a model whose load and reply took longer than the setting once its reply ends.
        config.write_text(json.dumps(dict(json.loads(config.read_text()), idle_unload_minutes=0.1)))
        check(until('stopped', 20), 'idle_unload_minutes of 6 s unloads the idle model')
        time.sleep(2)
        urllib.request.urlopen(urllib.request.Request(base + 'v1/start', data=b'{}', headers={'Content-Type': 'application/json', 'Origin': base.rstrip('/')}), timeout=10).close()
        time.sleep(3)
        check(health().get('status') != 'stopped' and 'stopped' not in json.loads(config.read_text()), 'an explicit start counts as activity: not unloaded at once')
        code, served = ask('slow')
        time.sleep(3)
        check(code == 200 and health().get('status') != 'stopped', 'a load and reply longer than idle_unload_minutes: the idle time counts from the reply\'s end')
        config.write_text(json.dumps({k: v for k, v in json.loads(config.read_text()).items() if k not in ('start_on_request', 'idle_unload_minutes')}))
    except Exception:
        log.flush(); print((root / 'server.log').read_text(), file=sys.stderr); raise
    finally:
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
# The memory search (`origami tune --memory`, the real one, as the dashboard's job) against a stub that answers as a model
# would: a reply takes 3 s, and a restart without a trial limit takes 1.5 s to listen, so the closing reload can be seen.
MEMORY = r"""#!/usr/bin/env python3
import http.server, json, os, sys, time
port = int(sys.argv[sys.argv.index('--port') + 1])
trial = any(a.startswith('--memory-limit') for a in sys.argv)
if not trial: time.sleep(1.5)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, body, kind='application/json'):
        self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        time.sleep(3 if 'origami_tune' in body else 0)
        said = 'trial' if trial else 'saved'
        events = [{'choices': [{'delta': {'content': said}}]}, {'choices': [], 'usage': {'prompt_tokens': 9, 'completion_tokens': 1}}]
        self.send(''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode() + b'data: [DONE]\n\n', 'text/event-stream')
    def do_GET(self):
        self.send(json.dumps({'status': 'ready', 'queue': {'busy': False, 'queued': 0}, 'trial': trial,
                              'tuning': {'key': 'stub@32768', 'identity': {}, 'knobs': {}}, 'memory_learned': {'minimum_bytes': 1000000000},
                              'model': {'name': 'Stub-35B'}, 'memory_pressure_cuts': 2, 'memory': {'budget_bytes': 18e9, 'recovery_target_bytes': 22e9, 'temporarily_reduced': True,
                                         'last_cut': {'reason': 'the system swap counter increased', 'at': '2026-11-17T10:00:00Z'}},
                              'experts_from': {'split': {'reads': 100, 'memory': .6, 'hot_copy': .3, 'model_drive': .1},
                                               'sources': [{'role': 'hot', 'drive': 'Fast', 'state': 'ready', 'bytes': 8e9, 'read_share': .3}]},
                              'requests': {'recent': [{'seconds': 40, 'memory_budget_deficit_gb_seconds': 100,
                                                      'prompt_restore': {'wanted': 7403, 'restored': 6912, 'reason': None}}]}}).encode())
http.server.ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
"""
with tempfile.TemporaryDirectory(prefix='origami-memory-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'; model = data / 'models' / 'first'; model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (data / 'models.json').write_text(json.dumps({'models': [{'name': 'first', 'path': str(model), 'manifest': {'fingerprint': 'a' * 64}}]}))
    worker = root / 'worker.py'; worker.write_text(MEMORY); worker.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = root / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'model': 'first', 'context': '32k', 'models_dirs': '[]', 'port': port, 'memory_auto_measure': True}))
    config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_TEST_WORKER'] = str(worker); env['ORIGAMI_AUTO_MEASURE_IDLE'] = '1'
    log = open(root / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    base = f'http://127.0.0.1:{port}/'
    def call(path, body=None, method=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), method=method,
                                         headers={'Content-Type': 'application/json', 'Origin': base.rstrip('/')})
        try:
            with urllib.request.urlopen(request, timeout=60) as response: return response.status, response.read()
        except urllib.error.HTTPError as error: return error.code, error.read()
    def line():
        try: return json.loads((data / 'tuning.json').read_text())['entries']['memory-auto@first@32768'].get('line')
        except (OSError, ValueError, KeyError): return None
    def trial_loaded(seconds=60):
        until = time.time() + seconds
        while time.time() < until:
            if 'at a trial memory limit' in (root / 'server.log').read_text(): return True
            time.sleep(.05)
        return False
    def job_state(): return (json.loads(call('v1/jobs')[1])['job'] or {}).get('state')
    def running_trial(): return json.loads(call('v1/status')[1]).get('trial')
    try:
        # J71: an automatic search, cancelled during a trial: the line says it restores during the closing reload, and the
        # final state after it; never "measuring" once it has ended.
        check(trial_loaded(), 'an idle server starts the memory search by itself and loads a trial limit')
        time.sleep(.5)
        call('v1/jobs', method='DELETE')
        seen, until = [], time.time() + 40
        while time.time() < until:
            now = line()
            if not seen or seen[-1] != now: seen.append(now)
            if job_state() != 'running' and now and 'restoring' not in now and now != 'memory: measuring automatically while idle; a request stops it': break
            time.sleep(.02)
        after = seen[seen.index(next(l for l in seen if l and 'restoring' in l)):] if any(l and 'restoring' in l for l in seen) else []
        check(after and after[-1] == 'memory: automatic measurement stopped: cancelled; will try again tomorrow'
              and not any('measuring automatically' in (l or '') for l in after) and running_trial() is False,
              'Cancel: "cancelled; restoring your saved limit" during the reload, then the final line, at the saved limit: ' + repr(seen))
        # J55: an app's request during a trial is answered at the saved limit, after the trial ended; the search stops.
        config.write_text(json.dumps(dict(json.loads(config.read_text()), memory_auto_measure=False)))
        code, _ = call('v1/jobs', {'kind': 'tune', 'mode': 'memory'})
        log.seek(0); log.truncate()
        check(code == 202 and trial_loaded(), 'a memory search from the dashboard loads a trial limit')
        time.sleep(2.5)  # its first reply is under way
        code, reply = call('v1/chat/completions', {'model': 'first', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True})
        until = time.time() + 40
        while job_state() == 'running' and time.time() < until: time.sleep(.1)
        text = (root / 'server.log').read_text()
        check(code == 200 and b'saved' in reply and 'a request arrived during the memory search' in text and running_trial() is False
              and text.count('at a trial memory limit') == 1 and job_state() in ('interrupted', 'failed', 'cancelled'),
              "an app's request during a trial is answered at the saved limit, the trial ended before it, and no trial loads again")
        # A memory search from a terminal (no job: only its trial reloads) that an app's request interrupted: its next
        # trial is refused, and once it has ended (its closing restore) the next search's first trial loads again.
        def ready(seconds=60):
            until = time.time() + seconds
            while time.time() < until:
                if json.loads(call('v1/status')[1]).get('status') == 'ready': return True
                time.sleep(.1)
            return False
        log.seek(0); log.truncate()
        first, _ = call('v1/reload', {'memory_limit_trial': 4000000000})
        loaded = trial_loaded() and ready()
        call('v1/chat/completions', {'model': 'first', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True})
        refused, _ = call('v1/reload', {'memory_limit_trial': 4000000000})
        call('v1/reload', {'restore_memory': True})
        again, _ = call('v1/reload', {'memory_limit_trial': 4000000000})
        call('v1/reload', {'restore_memory': True})
        check(first == 202 and loaded and refused == 409 and again == 202 and ready(),
              'a terminal search an app interrupted is refused its next trial; once it ended, a new search loads one: ' + repr((first, loaded, refused, again)))
        # J67: `origami status` with a server answering: what runs now, from its status.
        said = subprocess.run([binary, 'status', '--config', str(config)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30).stdout
        wanted = ['Live      Stub-35B loaded, ready',
                  'Live      memory: may use 18.0 GB of 22.0 GB now (lower for now: macOS took memory back); macOS took memory back 2 times since the model loaded, the last at ',
                  ' because the Mac had started to use swap (memory written to disk)', 'Live      expert reads: memory 60% · Fast copy 30% · model drive 10%',
                  'Live      copies: Fast ready 8.0 GB', 'Live      last request: 6912 of 7403 prompt tokens reused from a saved copy',
                  "Live      last request: slower than usual: macOS took back about 2.5 GB of Origami's memory during this reply, so more of the model had to be read from the drive"]
        check(all(w in said for w in wanted), 'origami status shows what runs now: ' + said)
    except Exception:
        log.flush(); print((root / 'server.log').read_text(), file=sys.stderr); raise
    finally:
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
# J39: the model's own drive goes away and comes back (a disk image under /Volumes, hidden from Finder): the status
# says degraded with what to do, then that it is back and to Reload; a reload clears it.
with tempfile.TemporaryDirectory(prefix='origami-drive-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'; data.mkdir()
    name = 'OrigamiDriveCheck' + uuid.uuid4().hex[:6]
    image = root / 'drive.dmg'
    attach = lambda: subprocess.run(['hdiutil', 'attach', '-quiet', '-nobrowse', str(image)], check=True)
    detach = lambda: subprocess.run(['hdiutil', 'detach', '-quiet', '/Volumes/' + name], check=False)
    if subprocess.run(['hdiutil', 'create', '-quiet', '-size', '4m', '-fs', 'HFS+', '-volname', name, str(image)]).returncode != 0:
        print('skip  the model drive going away (hdiutil cannot make an image here)')
    else:
        attach()
        model = pathlib.Path('/Volumes/' + name + '/first'); model.mkdir(); (model / 'config.json').write_text('{}')
        (data / 'models.json').write_text(json.dumps({'models': [{'name': 'first', 'path': str(model), 'manifest': {'fingerprint': 'a' * 64}}]}))
        worker = root / 'worker.py'; worker.write_text(WORKER); worker.chmod(0o755)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        config = root / 'config.json'
        config.write_text(json.dumps({'data_dir': str(data), 'model': 'first', 'context': '32k', 'models_dirs': '[]', 'port': port})); config.chmod(0o600)
        env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
        env['ORIGAMI_TEST_WORKER'] = str(worker)
        log = open(root / 'server.log', 'w+')
        process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
        base = f'http://127.0.0.1:{port}/'
        def health():
            try:
                with urllib.request.urlopen(base + 'v1/status', timeout=5) as response: return json.load(response)
            except urllib.error.HTTPError as error: return json.load(error)
            except (OSError, urllib.error.URLError): return {}
        def until(test, seconds=40):
            end = time.time() + seconds
            while time.time() < end:
                state = health()
                if test(state): return state
                time.sleep(.2)
            return health()
        try:
            check(until(lambda s: s.get('status') == 'ready').get('status') == 'ready', 'a model on a drive under /Volumes runs')
            detach()
            gone = until(lambda s: s.get('status') == 'degraded')
            cli = subprocess.run([binary, 'status', '--config', str(config)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30).stdout
            check(gone.get('degraded') == "The model's drive is not answering: reconnect it; reading resumes by itself, with no reload" and 'not answering: reconnect it' in cli and 'loaded, drive gone' in cli,
                  'the model drive gone: degraded, in the status and "drive gone" in origami status, with what to do: ' + repr(gone.get('degraded')))
            attach()
            check(until(lambda s: s.get('status') == 'ready' and 'degraded' not in s).get('status') == 'ready', 'back again: ready by itself, no reload')
        except Exception:
            log.flush(); print((root / 'server.log').read_text(), file=sys.stderr); raise
        finally:
            process.terminate()
            try: process.wait(timeout=8)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            log.close()
            detach()

# Slowed by: a stub model whose status says (in a server's own fields) memory started at a learned budget, its drive reads slowly and the hot copy is being built;
# the front turns that into `slowed_by` (with this Mac's largest apps), in /v1/status, /health and the dashboard, and
# `origami status` prints each with its fix, the costliest first.
SLOW = WORKER.replace("{'status': 'ready', 'queue': {'busy': False, 'queued': 0}}", """{'status': 'ready', 'queue': {'busy': False, 'queued': 0},
    'memory': {'available_bytes': 10121428992, 'budget_bytes': 14679362304, 'budget_mode': 'automatic', 'cap_bytes': 24051816858, 'floor_bytes': 4294967296,
        'footprint_bytes': 11620984456, 'last_cut': None, 'recovery_target_bytes': 24051816858, 'learned_start_bytes': 14679362304, 'reserved_bytes': 12565218464, 'start_bytes': 17589764720,
        'target_bytes': 24051816858, 'temporarily_reduced': True},
    'memory_learned': {'applied': True, 'bytes': 14679362304, 'cap_bytes': 24051816858, 'days': 12.28, 'minimum_bytes': 15679362304, 'regrowths': 292, 'trims': 129,
        'in_force': 'memory: automatic; this Mac has changed since the memory limit was measured (cut on 3 days since): run `origami tune --memory`',
        'limit_advice': {'applied': False, 'limit_bytes': None, 'measured': '2026-11-16T23:29:01Z', 'stale': 'cut on 3 days since', 'suggested_bytes': 25500000000, 'text': 'Chosen: 25.5GB'}},
    'experts_from': {'sources': [
        {'role': 'all', 'drive': 'Models', 'state': 'reading', 'slow_reads': {'high': True, 'share': 0.31, 'hour_share': 0.2, 'bytes': 1e12},
         'detail': 'The model drive read its files below a third of its own speed for 31% of its reading time. (at least 25 min for 54.8 GB).'},
        {'role': 'hot', 'drive': 'internal drive', 'state': 'building', 'progress': {'done_bytes': 12000000000, 'total_bytes': 40000000000, 'seconds_left': 600}}]}}""", 1)
assert SLOW != WORKER
with tempfile.TemporaryDirectory(prefix='origami-slowed-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'; model = data / 'models' / 'first'; model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (data / 'models.json').write_text(json.dumps({'models': [{'name': 'first', 'path': str(model), 'manifest': {'fingerprint': 'a' * 64}}]}))
    worker = root / 'worker.py'; worker.write_text(SLOW); worker.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = root / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'model': 'first', 'context': '32k', 'models_dirs': '[]', 'port': port})); config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_TEST_WORKER'] = str(worker)
    log = open(root / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    base = f'http://127.0.0.1:{port}/'
    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=15) as response: return json.load(response)
        except (OSError, urllib.error.URLError, ValueError): return {}
    try:
        for _ in range(200):
            if get('v1/status').get('status') == 'ready': break
            time.sleep(.05)
        status = get('v1/status')
        ids = [item['id'] for item in status.get('slowed_by', []) if item['id'] not in ('mac_hot', 'low_power')]
        check(ids == ['memory_low', 'model_drive_slow', 'hot_copy_building'] and all(item.get('since') for item in status['slowed_by']),
              'the status lists memory, the slow drive and the hot copy, costliest first, each with since: ' + repr(status.get('slowed_by')))
        memory = status['slowed_by'][0]
        check(memory['text'] == "Memory: Origami holds itself to 14.7 GB of the 24.1 GB it may use: it started at what this Mac usually had free over the last 12 days and grows as memory frees"
              and memory['fix'].startswith('Other apps hold ') and 'grows by up to 2 GB a step' in memory['fix'] and len(status.get('memory_holders') or []) >= 1
              and isinstance(status.get('mac_now', {}).get('low_power'), bool),
              'memory started at a learned budget says it grows as memory frees and names the apps that hold the most; the status keeps memory_holders and mac_now: ' + repr(memory))
        check([item['id'] for item in get('health').get('slowed_by', [])] == [item['id'] for item in status['slowed_by']]
              and [item['id'] for item in get('v1/dashboard').get('slowed_by', [])] == [item['id'] for item in status['slowed_by']],
              '/health and the dashboard carry the same list')
        cli = subprocess.run([binary, 'status', '--config', str(config)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60).stdout
        lines = cli.splitlines()
        slowed = [i for i, line in enumerate(lines) if line.startswith('Slowed by ')]
        check(len(slowed) >= 3 and lines[slowed[0]].startswith('Slowed by Memory: Origami holds itself to 14.7 GB of the 24.1 GB') and lines[slowed[0] + 1].startswith('          Other apps hold ')
              and any(lines[i].startswith("Slowed by The model's drive reads slowly: 31%") and 'run `origami refresh`' in lines[i + 1] and '(at least 25 min)' in lines[i + 1] for i in slowed)
              and any(lines[i].startswith('Slowed by The hot copy is still being built (12.0 GB of 40.0 GB)') and 'in about 10 min' in lines[i + 1] for i in slowed)
              and slowed[0] < min(i for i, line in enumerate(lines) if line.startswith('Live')),
              'origami status prints each condition and its fix, before the live lines: ' + repr([lines[i] for i in slowed]))
    except Exception:
        log.flush(); print((root / 'server.log').read_text(), file=sys.stderr); raise
    finally:
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
# Remove unusable saved prompts from the page while a "model" is loaded (the stub worker): another build's folders go, as
# do a model file's no model uses; this build's, the running one and one a second server holds (a held `.lock`) stay; a
# folder with a pinned conversation stays until its own tick.
import fcntl
with tempfile.TemporaryDirectory(prefix='origami-prune-') as temporary:
    root = pathlib.Path(temporary).resolve(); data = root / 'data'; model = data / 'models' / 'first'; model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (data / 'models.json').write_text(json.dumps({'models': [{'name': 'first', 'path': str(model), 'manifest': {'fingerprint': 'a' * 64}}]}))
    worker = root / 'worker.py'; worker.write_text(WORKER); worker.chmod(0o755)
    context = root / 'context'; group = context / ('a' * 12 + '-32768')
    build = hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest()[:16]
    arithmetic = re.search(r'value = "([0-9a-f]{64})"', (pathlib.Path(__file__).resolve().parent.parent / 'Sources/OrigamiEngine/Core/ArithmeticDigest.swift').read_text()).group(1)
    identity = hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest() + ' Version %s (Build %s)' % tuple(
        subprocess.check_output(['sw_vers', flag], text=True).strip() for flag in ('-productVersion', '-buildVersion'))
    def epoch(folder, marker_build=None, index=None):
        folder.mkdir(parents=True); (folder / 'entry-0').mkdir(); (folder / 'entry-0' / 'checkpoint.origami').write_bytes(b'x' * 4096)
        # This build's own marker also names it as the canary's fields do (`PromptCanary`): the arithmetic digest and the
        # executable on this macOS.
        mine = {'arithmetic': arithmetic, 'builds': [identity]} if marker_build == build else {}
        if marker_build: (folder / 'origami-epoch.json').write_text(json.dumps({'model': 'first', 'context': 32768, 'version': '3.1.3', 'build': marker_build, **mine}))
        if index is not None: (folder / 'index.json').write_text(json.dumps(index))
        return folder
    running = epoch(group / ('1' * 16), build, {'epoch': '1', 'entries': {'0': {'tokens': 600}}})
    older = epoch(group / ('2' * 16), '0' * 16, {'epoch': '2', 'entries': {}})
    unmarked = epoch(group / ('3' * 16))
    pinned = epoch(group / ('4' * 16), '0' * 16, {'epoch': '4', 'entries': {'0': {'tokens': 600, 'pinned': True}}})
    held = epoch(group / ('5' * 16), '0' * 16)
    mine = epoch(group / ('6' * 16), build)
    orphan = epoch(context / ('b' * 12 + '-32768') / ('7' * 16), build)
    # The running server's lock and a second server's.
    locks = [open(folder / '.lock', 'w') for folder in (running, held)]
    for lock in locks: fcntl.flock(lock, fcntl.LOCK_EX)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    config = root / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'context_dir': str(context), 'model': 'first', 'context': '32k', 'models_dirs': '[]', 'port': port})); config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    env['ORIGAMI_TEST_WORKER'] = str(worker)
    log = open(root / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    base = f'http://127.0.0.1:{port}/'
    def call(path, body=None, origin=None):
        headers = {'Content-Type': 'application/json', 'Origin': origin or base.rstrip('/')}
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as error: return error.code, json.load(error)
    def job(body):
        code, result = call('v1/jobs', body); assert code == 202, (code, result)
        until = time.time() + 20
        while time.time() < until:
            current = call('v1/jobs')[1]['job']
            if current['state'] != 'running':
                assert current['state'] == 'done', current
                return current
            time.sleep(.05)
        raise AssertionError('operation timed out')
    try:
        for _ in range(200):
            try:
                if call('v1/dashboard')[1]['dashboard']['active_model']: break
            except (OSError, urllib.error.URLError, KeyError): pass
            time.sleep(.05)
        else: raise AssertionError('the stub worker did not become ready')
        states = {pathlib.Path(b['path']).name: (b['state'], b.get('reason')) for b in call('v1/storage')[1]['saved_prompts']}
        check(states == {'1' * 16: ('in_use', None), '2' * 16: ('unusable', 'made by another Origami build (version 3.1.3)'),
                         '3' * 16: ('unusable', 'made by an earlier Origami build'), '4' * 16: ('unusable', 'made by another Origami build (version 3.1.3)'),
                         '5' * 16: ('in_use', None), '6' * 16: ('usable', None), '7' * 16: ('unusable', 'a model file no model uses')},
              'with a model loaded, the storage map says which saved prompts are in use, usable by this build or unusable and why: ' + repr(states))
        code, refusal = call('v1/jobs', {'kind': 'prune', 'yes': []}, origin='http://elsewhere.example')
        check(code == 403 and refusal['error']['message'] == "Start jobs from this server's own page" and older.exists(), 'Remove unusable starts only from the page itself')
        code, plan = call('v1/storage/plan', {'kind': 'prune'})
        going = {pathlib.Path(b['path']).name: b['kept'] for b in plan['folders']}
        check(code == 200 and going == {'2' * 16: [], '3' * 16: [], '4' * 16: ['a pinned prompt'], '7' * 16: []}, 'the preview lists the unusable folders, the pinned one with what it keeps: ' + repr(going))
        done = job({'kind': 'prune', 'yes': [], 'revision': plan['revision']})
        check(not older.exists() and not unmarked.exists() and not orphan.exists() and all(f.exists() for f in (running, held, mine, pinned))
              and 'Kept ' + str(group.name) + '/' + '4' * 16 in done.get('message', ''),
              'with the model loaded, older builds and a model file no model uses go; the running, a held, this build\'s and the pinned folder stay: ' + repr(done.get('message')))
        done = job({'kind': 'prune', 'yes': [str(pinned)], 'revision': plan['revision']})
        check(not pinned.exists() and running.exists() and held.exists() and (running / 'entry-0' / 'checkpoint.origami').exists(),
              'the pinned folder goes once ticked; the held ones are untouched: ' + repr(done.get('message')))
        code, refusal = call('v1/jobs', {'kind': 'prune', 'yes': []})
        check(code == 409 and refusal['error']['message'] == 'Only an item nothing uses is removed', 'with nothing unusable left, it refuses in the storage map\'s words: ' + repr(refusal))
    except Exception:
        log.flush(); print((root / 'server.log').read_text(), file=sys.stderr); raise
    finally:
        for lock in locks: lock.close()
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
print(f'{checks} control-plane checks passed')

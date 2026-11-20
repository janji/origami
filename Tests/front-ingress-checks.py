#!/usr/bin/env python3
"""Model-free checks of the front's ingress limits against real, stopped `origami serve` fronts (the front answers
itself; no model loads), each on its own port and data folder. Slow clients (silent, trickled headers, trickled bodies)
run while a normal client is served; the budget the dashboard reports stays within its maximum; wrong keys are rate
limited per client; X-Forwarded-For is honoured only from loopback with trusted_proxy set.
Usage: Tests/front-ingress-checks.py [.build/debug/origami]
Another device's limits need this Mac's network address (connections to it count as another device); without one,
or where the macOS firewall keeps this binary from being reached there, those parts are skipped."""
import json, os, pathlib, socket, subprocess, sys, tempfile, threading, time

binary = str(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.build/debug/origami').resolve())
KEY = 'k' * 20
checks = 0


def check(condition, message):
    global checks
    assert condition, message
    checks += 1
    print('ok   ', message, flush=True)


def lan_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('192.0.2.1', 9))  # no packet is sent: this only picks the outgoing interface
            address = probe.getsockname()[0]
        return None if address.startswith(('127.', '0.')) else address
    except OSError:
        return None


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def connect(host, port):
    sock = socket.create_connection((host, port), timeout=20)
    return sock


def request(host, port, path='/v1/models', method='GET', headers=None, body=b'', timeout=20):
    """One request; returns (status, json body or None, seconds)."""
    began = time.time()
    sock = connect(host, port)
    sock.settimeout(timeout)
    lines = [f'{method} {path} HTTP/1.1', f'Host: {host}:{port}']
    for name, value in (headers or {}).items():
        lines.append(f'{name}: {value}')
    if body or method == 'POST':
        lines += ['Content-Type: application/json', f'Content-Length: {len(body)}']
    try:
        sock.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode() + body)
    except (BrokenPipeError, ConnectionResetError):
        pass  # refused from its headers: the reply is already there
    return read_reply(sock, began)


def read_reply(sock, began):
    data = b''
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    except (ConnectionResetError, socket.timeout):
        pass
    finally:
        sock.close()
    if not data:
        return None, None, time.time() - began
    head, _, rest = data.partition(b'\r\n\r\n')
    status = int(head.split(b' ')[1])
    try:
        payload = json.loads(rest)
    except ValueError:
        payload = None
    return status, payload, time.time() - began


def code(payload):
    return ((payload or {}).get('error') or {}).get('code')


def start(root, name, settings):
    data = root / name
    data.mkdir()
    port = free_port()
    config = data / 'config.json'
    config.write_text(json.dumps({'data_dir': str(data), 'model': str(root / 'no-model'), 'stopped': True,
                                  'models_dirs': '[]', 'port': port, **settings}))
    config.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ORIGAMI_')}
    log = open(data / 'server.log', 'w+')
    process = subprocess.Popen([binary, 'serve', '--config', str(config)], env=env, stdout=log, stderr=log)
    for _ in range(200):
        try:
            if request('127.0.0.1', port, headers={'Authorization': 'Bearer ' + KEY}, timeout=2)[0]:
                return process, port
        except OSError:
            pass
        time.sleep(0.05)
    process.kill()
    raise AssertionError(f'{name} front did not start: ' + (data / 'server.log').read_text())


def budget(port):
    status, payload, _ = request('127.0.0.1', port, '/v1/dashboard', headers={'Authorization': 'Bearer ' + KEY})
    assert status == 200, (status, payload)
    return payload['front_connections']


def within(snapshot):
    return (snapshot['header_reservation_bytes'] == snapshot['connections'] * 49152
            and snapshot['header_reservation_bytes'] + snapshot['buffered_body_reservation_bytes'] <= snapshot['maximum_reserved_bytes'])


class Slow(threading.Thread):
    """A client that sends `opening`, then `piece` every `every` seconds, then reads the reply's status."""
    def __init__(self, host, port, opening, piece=b'', every=1.0):
        super().__init__(daemon=True)
        self.sock = connect(host, port)
        self.opening, self.piece, self.every = opening, piece, every
        self.status = self.payload = None
        self.seconds = 0

    def run(self):
        began = time.time()
        try:
            self.sock.sendall(self.opening)
            while self.piece:
                self.sock.settimeout(self.every)
                try:
                    if self.sock.recv(65536, socket.MSG_PEEK):
                        break  # the reply came
                except socket.timeout:
                    pass
                self.sock.sendall(self.piece)
        except OSError:
            pass
        self.sock.settimeout(20)
        self.status, self.payload, self.seconds = read_reply(self.sock, began)


def post_head(host, port, length):
    return (f'POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\nAuthorization: Bearer {KEY}\r\n'
            f'Content-Type: application/json\r\nContent-Length: {length}\r\n\r\n').encode()


with tempfile.TemporaryDirectory(prefix='origami-ingress-') as temporary:
    root = pathlib.Path(temporary).resolve()
    lan = lan_address()
    servers = []
    try:
        front, port = start(root, 'lan', {'listen': 'lan', 'api_key': KEY, 'trusted_proxy': 'loopback'})
        servers.append(front)
        if lan is not None:
            # With the macOS firewall on, a binary it has not allowed is not reached at this Mac's own address.
            try:
                with socket.create_connection((lan, port), timeout=3) as probe:
                    probe.settimeout(3)
                    probe.sendall(f'GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {KEY}\r\n\r\n'.encode())
                    if not probe.recv(16): lan = None
            except OSError: lan = None
        key = {'Authorization': 'Bearer ' + KEY}
        normal = lambda: request('127.0.0.1', port, headers=key)

        # Slow request lines and silent connections from this Mac: cut at the 10 s header deadline with a 408.
        slow = [Slow('127.0.0.1', port, b'GET /v1/mod', b'e', 1.0) for _ in range(6)]
        slow += [Slow('127.0.0.1', port, b'') for _ in range(6)]
        # Uploads that announce 24 MiB and stall: two take the 64 MiB body pool, the third finds it full.
        uploads = [Slow('127.0.0.1', port, post_head('127.0.0.1', port, 24 << 20) + b'{"model":"x"', b'', 1) for _ in range(2)]
        for client in slow + uploads:
            client.start()
        time.sleep(0.3)
        status, payload, _ = request('127.0.0.1', port, '/v1/chat/completions', 'POST', key, b'{"pad":"' + b'a' * (17 << 20) + b'"}')
        check(status == 503 and code(payload) == 'server_busy', 'a large body that finds the pool held by fresh uploads gets 503 server_busy')
        snapshot = budget(port)
        check(snapshot['ingress'] >= 14 and within(snapshot), f'header bytes of unclassified connections are counted and within the budget: {snapshot}')
        # A second later the stalled uploads are behind pace: a request that needs their room takes it.
        time.sleep(1.2)
        status, payload, seconds = request('127.0.0.1', port, '/v1/chat/completions', 'POST', key, b'{"pad":"' + b'a' * (20 << 20) + b'"}')
        check(status == 503 and code(payload) == 'stopped' and seconds < 5,
              f'a large upload is served (the stopped reply) by taking a stalled upload\'s room ({seconds:.1f} s)')
        for _ in range(5):
            status, _, seconds = normal()
            check(status == 200 and seconds < 2, f'a normal request is served while slow clients hold connections ({seconds:.2f} s)')
        for client in uploads:
            client.join(15)
        statuses = sorted((client.status, code(client.payload)) for client in uploads)
        check(statuses[0] == (408, 'request_timeout') and statuses[1] == (503, 'server_busy'),
              f'one stalled upload loses its room (503), the other is cut by the rate floor (408): {statuses}')
        for client in slow:
            client.join(15)
        check(all(client.status == 408 and 9 <= client.seconds < 13 for client in slow),
              f'slow headers and silent connections get 408 at the header deadline: {[(c.status, round(c.seconds, 1)) for c in slow]}')
        snapshot = budget(port)
        check(snapshot['connections'] == 1 and snapshot['buffered_body_reservation_bytes'] == 0, f'every reservation is returned: {snapshot}')

        # Wrong keys: 10 a minute per client; this Mac is never locked out; a trusted proxy's clients are told apart.
        wrong = {'Authorization': 'Bearer ' + 'w' * 20}
        replies = [request('127.0.0.1', port, headers=wrong)[0] for _ in range(12)]
        check(set(replies) == {401}, 'wrong keys from this Mac are refused 401, never locked out')
        replies = [request('127.0.0.1', port, headers={**wrong, 'X-Forwarded-For': '203.0.113.9'})[0] for _ in range(11)]
        check(replies == [401] * 10 + [429], f'a proxied client is locked out after 10 wrong keys: {replies}')
        status = request('127.0.0.1', port, headers={**key, 'X-Forwarded-For': '203.0.113.9'})[0]
        check(status == 429, 'the locked-out client is refused even with the right key until the minute ends')
        status = request('127.0.0.1', port, headers={**key, 'X-Forwarded-For': '203.0.113.10'})[0]
        check(status == 200, 'another proxied client is not locked out')
        status = request('127.0.0.1', port, headers={**key, 'X-Forwarded-For': 'not-an-address'})[0]
        check(status == 400, 'a malformed X-Forwarded-For from a trusted proxy is refused')

        if lan is None:
            print('skip  another device\'s limits: no network address, or the macOS firewall has not allowed this binary')
        else:
            # Another device: at most 8 unclassified connections from one address, refused 503 past that.
            held = [connect(lan, port) for _ in range(8)]
            time.sleep(0.2)
            status, payload, _ = request(lan, port, headers=key)
            check(status == 503, 'a 9th unclassified connection from one address is refused')
            status, _, seconds = normal()
            check(status == 200 and seconds < 2, 'this Mac is still served while one address holds its share')
            for sock in held:
                sock.close()
            time.sleep(0.3)
            # Its X-Forwarded-For is ignored: varying it does not escape the lockout of its address.
            replies = [request(lan, port, headers={**wrong, 'X-Forwarded-For': f'198.51.100.{i}'})[0] for i in range(11)]
            check(replies == [401] * 10 + [429], f'another device cannot name its own client address: {replies}')

        # Without trusted_proxy the header means nothing; with it, a proxied request counts as another device.
        plain, plain_port = start(root, 'plain', {})
        servers.append(plain)
        status = request('127.0.0.1', plain_port, headers={'X-Forwarded-For': '203.0.113.9'})[0]
        check(status == 200, 'X-Forwarded-For is ignored without trusted_proxy')
        proxied, proxied_port = start(root, 'proxied', {'trusted_proxy': 'loopback'})
        servers.append(proxied)
        check(request('127.0.0.1', proxied_port)[0] == 200, 'this Mac needs no key without api_key')
        status, payload, _ = request('127.0.0.1', proxied_port, headers={'X-Forwarded-For': '203.0.113.9'})
        check(status == 401 and code(payload) == 'invalid_api_key', 'a proxied client needs the key (api_key) with trusted_proxy loopback')
    finally:
        for process in servers:
            process.terminate()
        for process in servers:
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
print(f'{checks} front ingress checks passed')

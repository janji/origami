# Security policy

## What the server exposes

`origami serve` listens on 127.0.0.1 only and answers only requests whose `Host` is a loopback address. Cross-site
browser requests are refused. An optional API key (`origami config set api_key …`) is then required on every
endpoint. `origami config set listen lan` opens the port to your local network; other devices must then send a key
on every request (Origami creates one if none is set). Traffic is plain HTTP. Do not expose the port through a proxy
or tunnel without a key. The details are in [docs/integration.md](docs/integration.md#security).

## Reporting a vulnerability

Please report it privately through a GitHub security advisory on this repository (Security tab, "Report a
vulnerability"), not in a public issue. Include the version (`origami --version`), what you did and what happened.

## Supported versions

Only the latest release receives security fixes.

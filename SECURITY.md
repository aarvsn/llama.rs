# Security Policy

## Supported versions

Security fixes target the latest released version on the `main` branch.

## Reporting a vulnerability

Please report vulnerabilities privately via
[GitHub security advisories](https://github.com/aarvsn/llama.rs/security/advisories/new).
Do not open public issues for security problems.

Include reproduction steps and, where possible, a minimal file or request
that triggers the issue. You will receive an acknowledgement within a few
days; fixes ship as patch releases once validated.

## Scope

llama.rs parses untrusted model files and (optionally) serves an HTTP API.
Both are treated as attack surfaces:

- **GGUF parsing** runs with strict bounds checking, sanity limits on
  metadata sizes and counts, and no `unsafe` interpretation of mapped
  bytes. Malformed files must produce errors, never panics or memory
  unsafety. If you find a panic on a crafted GGUF, that is a bug worth
  reporting (fuzzing contributions welcome).
- **The HTTP server** binds to 127.0.0.1 by default. Exposing it on a
  network is the operator's decision; it implements request validation
  and graceful shutdown, but no authentication - treat it like llama.cpp's
  server.
- **Generated content** is the model's output; llama.rs makes no safety
  guarantees about model behavior.

## Non-goals

Model-weight provenance and licensing are the user's responsibility; the
project ships no model files.

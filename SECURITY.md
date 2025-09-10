# Security Policy

## Supported versions

The latest released minor version receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | ✅        |
| < 0.3   | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Instead, use GitHub's private vulnerability reporting ("Report a vulnerability"
under the Security tab) or email the maintainer. Include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- affected version(s).

You can expect an initial acknowledgement within a few days. Once a fix is
available it will be released and the reporter credited (unless anonymity is
requested).

## Scope notes

chunkasr is offline-first and performs no network I/O. The most relevant
concerns are therefore around untrusted inputs: malformed audio files, crafted
manifests, and checkpoints. Checkpoints are stored as `.npz` with
`allow_pickle=False`, so loading them does not execute arbitrary code.

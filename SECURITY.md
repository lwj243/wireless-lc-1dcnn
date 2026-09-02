# Security Policy

## Supported version

Only the latest commit on the default branch is supported.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Report them privately to the repository owner through GitHub's private vulnerability reporting feature.

## Safe use

- Never commit `.env` files, access tokens, raw participant data, unpublished documents, or private COMSOL projects.
- Do not load third-party PyTorch checkpoints with unrestricted pickle deserialization.
- This project loads checkpoints with `torch.load(..., weights_only=True)` and validates expected keys and tensor shapes.
- Install dependencies only in an isolated environment and review dependency updates before merging.
- Respect the minimum security versions in `pyproject.toml`; the historical PyTorch 2.5.1 training runtime is not an approved deployment environment.
- Generated training files belong in ignored `outputs/` or `runs/` directories.

## Dependency audit

The five direct dependencies were checked on 2026-09-02 with `pip-audit`. The reviewed set used PyTorch 2.13.0, Pillow 12.3.0, NumPy 1.26.4, pandas 2.0.3, and Matplotlib 3.7.2; no known direct-dependency vulnerabilities were reported at that time. This is a dated snapshot, not a guarantee about future advisories or transitive dependencies.

## Threat model

The code is an offline research pipeline. It does not expose a web server, accept network requests, execute shell commands, or evaluate user-supplied code. The main remaining risks are malicious model files, compromised dependencies, accidental credential commits, and accidental publication of private research data.

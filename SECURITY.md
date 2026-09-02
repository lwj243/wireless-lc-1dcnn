# Security Policy

## Supported version

Only the latest commit on the default branch is supported.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Report them privately to the repository owner through GitHub's private vulnerability reporting feature.

## Safe use

- Keep the GitHub repository private unless its contents have been approved for public release.
- Never commit `.env` files, access tokens, raw participant data, unpublished documents, or private COMSOL projects.
- Do not load third-party PyTorch checkpoints with unrestricted pickle deserialization.
- This project loads checkpoints with `torch.load(..., weights_only=True)` and validates expected keys and tensor shapes.
- Install dependencies only in an isolated environment and review dependency updates before merging.
- Generated training files belong in ignored `outputs/` or `runs/` directories.

## Threat model

The code is an offline research pipeline. It does not expose a web server, accept network requests, execute shell commands, or evaluate user-supplied code. The main remaining risks are malicious model files, compromised dependencies, accidental credential commits, and accidental publication of private research data.


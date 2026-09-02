# Contributing

Contributions that improve reproducibility, testing, documentation, or model robustness are welcome.

## Development workflow

1. Create a branch from `main`.
2. Install the project in an isolated Python 3.10 environment with `python -m pip install -e ".[dev]"`.
3. Run `python -m ruff check .`, `python -m bandit -q -r src`, and `python -m pytest`.
4. Open a pull request describing the scientific and software impact of the change.

Do not submit credentials, private research data, unpublished participant data, proprietary COMSOL files, or untrusted serialized model files.

Scientific contributions must distinguish simulated results from experimental, ex-vivo, in-vivo, or clinical validation.


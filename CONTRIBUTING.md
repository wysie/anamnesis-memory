# Contributing to Anamnesis

Anamnesis is built test-first.

Before opening a PR:

```bash
python -m py_compile src/anamnesis/*.py
pytest -q
```

Memory-system rules:

- Core recall must work without an LLM.
- Scope filters must run before ranking or generation.
- Invalidated/superseded memories must not be recalled by default.
- Local LLM workers may propose, but deterministic policy decides.
- New lifecycle/governance behavior needs tests.

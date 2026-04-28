---
name: Bug report
about: Report a problem with fomodynamics
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of what the bug is.

## Reproducer

```python
# Minimal Python snippet that triggers the bug. 10–20 lines is ideal.
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. If there is a traceback, paste it here in a fenced
code block.

## Environment

Run these and paste the output:

```
python --version
python -c "import jax; print(jax.devices(), jax.__version__)"
python -c "import fmd; print(fmd.__file__)"
uv --version       # if using uv
```

OS / kernel:

Install method (uv sync / pip install / from source):

## Additional context

Anything else that might help — related issues, recent changes, scenarios
where the bug does *not* reproduce, etc.

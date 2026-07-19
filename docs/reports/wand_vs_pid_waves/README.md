# wand_vs_pid_waves — canonical fmd example report

Compares a mechanical wand-to-flap linkage against three wand-only
feedback configurations (natural-trim PID, deeper-trim PID, and a tuned
proportional-only controller) under SF Bay moderate waves, with a P
speed governor holding boatspeed and each controller calibrated at its
own pinned trim. This is the canonical
"recipe + regenerator + interpretation skill" example for the fmd
package. Use it as a starting point for your own controller tuning
and wave-condition evaluations.

## Regenerate

```bash
JAX_PLATFORMS=cpu uv run --no-sync python docs/reports/wand_vs_pid_waves/run.py
# Quick smoke run (5 seeds instead of 50):
JAX_PLATFORMS=cpu uv run --no-sync python docs/reports/wand_vs_pid_waves/run.py --quick
```

The script writes `metrics.json`, `report_guidelines.txt`, and PNGs
into `plots/` (all in this folder). Re-running overwrites them.

## Regenerate the written report

Point any AI agent (Claude Code, Cursor, ChatGPT desktop, etc.) at
[`interpretation_skill.md`](interpretation_skill.md). The skill is
harness-agnostic — it tells the agent to read the artifacts in this
folder and produce `report.md` here. No special tooling required;
any agent that can read files and write markdown can follow it.

For what the simulation captures (and abstracts away), read
[`model_setup.md`](model_setup.md). For the design choices, gains, and what
to tune, read [`recipe.md`](recipe.md).

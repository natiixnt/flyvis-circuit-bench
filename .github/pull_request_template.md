## What and why

<!-- What changes, and why. Link the issue if there is one. -->

## Checklist

- [ ] `uv run ruff check .` and `uv run pytest` pass
- [ ] New code has tests that run without the measurement package
- [ ] No published number changed, or the change is explained here and in `CHANGELOG.md`
- [ ] With the measurement package: `uv run fcb reproduce --data <dir> --check` passes

## New leaderboard row (only if you add a method)

- [ ] Link to the code that produces the scores file:
- [ ] The row is the unedited output of `fcb score`, added to `README.md` and `README.pl.md` in AUROC order
- [ ] The method does not use the ablation effects, and was not tuned on them (or the row is marked as tuned)
- [ ] What the method needs (connectome only, weights, forward or backward passes):

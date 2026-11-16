# Local ms-swift source

This directory is a shallow checkout of `modelscope/ms-swift` at tag `v4.5.2`,
the version used by `swift` for the AURORA
OPSD training script.

The local checkout contains two project-local fixes required by this experiment:

- `swift/pipelines/train/rlhf.py` loads `args.teacher_adapters` for the teacher
  model instead of `reward_adapters`.
- `swift/arguments/rlhf_args.py` keeps a separate teacher model when explicit
  teacher adapters are supplied, even when teacher and student use the same
  base-model path.

The checkout is installed in editable mode in the `swift46` environment:

```bash
python -m pip install --no-deps -e \
  third_party/ms-swift
```

Changes under `third_party/ms-swift/swift` therefore take effect immediately;
reinstallation is only needed after changing package metadata or entry points.
For VS Code, select `python` and keep
`justMyCode: false` so breakpoints inside this source checkout are active.

# Task: Run-From-V2 Path Fix

Tags: #progress #bugfix #paths #cli

Related notes: [[README]] [[2026-05-06_fresh_ag_news_dataset]] [[2026-05-06_constrained_paraphrase_pipeline]]

## 1. Goal

Fix path handling after the user ran scripts from inside `SetConCA_V2` and the code created nested `SetConCA_V2` or `setconca_v2/data` folders.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-06 |
| Problem | Relative paths behaved differently depending on current working directory. |
| Main fix | `src/setconca_v2/paths.py` |
| Affected scripts | `scripts/download_news_dataset.py`, `scripts/generate_constrained_sets.py` |

## 3. Hypothesis Or Rationale

CLI paths should resolve against the V2 project root, not accidentally duplicate the project folder when the user runs commands from inside `SetConCA_V2`.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Added `project_root()`. | Needed a canonical V2 root. | It resolves to the `SetConCA_V2` directory from package location. | Succeeded |
| 2 | Added `resolve_project_path()`. | Needed consistent CLI path handling. | Absolute paths pass through; relative paths resolve under project root. | Succeeded |
| 3 | Stripped redundant leading project folder names. | Needed to support commands that still pass `SetConCA_V2/data/...`. | Prevents nested `SetConCA_V2/SetConCA_V2/...` outputs. | Succeeded |
| 4 | Updated scripts to use the resolver. | Needed both download and generation CLIs to behave consistently. | Both scripts resolve input/config/output paths through `resolve_project_path`. | Succeeded |
| 5 | Moved misplaced raw files. | Needed to repair existing nested output. | Files were moved into `data/raw`. | Succeeded |
| 6 | Ran tests and a dry-run generation check. | Needed to verify the fix from inside V2. | Prior session reported `11 passed` and dry-run output saved under the intended V2 folder. | Succeeded |

## 5. Code And Pseudocode

```text
def resolve_project_path(raw):
    path = Path(raw)
    if path.is_absolute():
        return path

    root = project_root()
    if path.parts[0].lower() == root.name.lower():
        path = path without first part

    return root / path
```

## 6. Results

### Verified Commands

Tests:

```powershell
python -m pytest tests -q
```

Reported result from the prior session:

```text
11 passed
```

Dry-run generation from inside `SetConCA_V2`:

```powershell
python scripts\generate_constrained_sets.py `
  --models-config configs\rewrite_models.example.json `
  --input data\raw\original_sentences.example.jsonl `
  --out-dir data\generated\cwd_test `
  --dry-run `
  --include-disabled `
  --max-originals 1
```

Reported result: output saved under `SetConCA_V2\data\generated\cwd_test`.

## 7. Interpretation

This was a practical reproducibility fix. It reduces user friction and prevents silently writing experimental artifacts into the wrong folder.

## 8. Successes

The fix succeeded because it centralizes path behavior and keeps both supported invocation styles:

```powershell
python scripts\download_news_dataset.py --out data\raw\ag_news_train_10.jsonl
python SetConCA_V2\scripts\download_news_dataset.py --out SetConCA_V2\data\raw\ag_news_train_10.jsonl
```

## 9. Failures Or Limits

The dry-run command in the prior session referenced `data\raw\original_sentences.example.jsonl`; that file is not present in the current observed raw-data folder. The current confirmed raw dataset is `data/raw/ag_news_train.jsonl`.

## 10. External Works And Papers

No external paper or method was used. This was local CLI path engineering.

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `src/setconca_v2/paths.py` | Added project-root and path-resolution helpers. | Avoid nested outputs and cwd-sensitive behavior. |
| `scripts/download_news_dataset.py` | Uses `resolve_project_path`. | Resolve output path consistently. |
| `scripts/generate_constrained_sets.py` | Uses `resolve_project_path`. | Resolve config, input, and output paths consistently. |
| `README.md` | Updated commands for running inside V2. | Give the user correct invocation examples. |
| `data/raw/ag_news_train_10.jsonl` | Prior session reported moving misplaced pilot file. | Repair nested-path output. |
| `data/raw/ag_news_train_10.manifest.json` | Prior session reported moving misplaced pilot manifest. | Repair nested-path output. |

## 12. Follow-Up

- [ ] Add a regression test for `resolve_project_path`.
- [ ] Avoid committing generated `__pycache__` files.

# Task 2: Repository and data versioning

## Design

| Concern | Solution |
|---|---|
| Code & config | Git (`main`), small commits with conventional messages (`feat`, `fix`, `test`, `data`, `ci`, `docs`, …) |
| Data | **DVC.** Git stores only the pointer `data/raw/credit_card_customers.csv.dvc` (md5 + size). The CSV itself is ignored by Git (`data/raw/.gitignore`). |
| Data remote | DVC content-addressed storage (`files/md5/xx/…`) kept on an **orphan branch `dvc-storage`**, checked out as a worktree at `.dvc-storage/` and configured as the DVC remote `storage`. |
| Human-readable versions | [`data/versions.yaml`](../data/versions.yaml) maps md5 → `v1`/`v2`; every run logs `data_md5`, `data_version`, `data_rows`. |
| Restore points | Annotated tags: `data-v1` (v1 enters), `baseline-v1` (code + config + data of the first baseline), `data-v2` (v2 merged). |
| Original files | `datasets/` (as supplied) is git-ignored. Data enters the project only via `dvc add`. |

### Why an orphan branch as the DVC remote

The brief requires free tools, no secrets and reproducibility from a fresh clone, including in CI. The DVC remote is:

- **Free and secret-free.** CI reads it with the built-in `GITHUB_TOKEN` (`actions/checkout` of `dvc-storage`). Google Drive, S3 or DagsHub would need credentials.
- **Not ordinary Git history.** No commit on `main` contains a CSV; checking out any `main` commit yields only the `.dvc` pointer. The storage branch shares no history with `main` and holds opaque content-addressed blobs written only by `dvc push`.
- **Replaceable.** Swapping in S3/GDrive/DagsHub is one `dvc remote modify` command, with no code change.

Trade-off accepted: the storage branch grows with each data version (42 KB + 58 KB here). For large data you would move to object storage.

## Repository layout

See the [README](../README.md#repository-layout).

## Version 2 enters as a tracked change

```bash
git switch -c data/v2
cp <new file> data/raw/credit_card_customers.csv
dvc add data/raw/credit_card_customers.csv       # updates the .dvc pointer (new md5)
# register the label in data/versions.yaml
scripts/dvc_storage.sh push                     # dvc push + commit to the dvc-storage branch
git commit -am "data: dataset v2 (v1 + 300 new customers) enters as a tracked DVC change"
# open a PR -> the same quality gate runs on the new data
```

The diff of that commit is two lines in the pointer file plus the registry entry. That diff *is* the data version change, and it is reviewed and gated like code.

## Restoring an earlier commit with its exact data

```bash
git checkout baseline-v1          # any commit, tag or branch
dvc checkout                      # (or `dvc pull` in a fresh clone) -> data pinned by that commit
python -m credit_risk.train --config configs/ci.yaml --no-tracking --out /tmp/m.json
python scripts/update_baseline.py --check --candidate /tmp/m.json
```

[`scripts/restore_demo.sh`](../scripts/restore_demo.sh) performs exactly this from a **fresh clone** in a temporary directory and requires an exact match (tolerance 1e-6) with the result recorded at that commit. Logs: [`reports/restore/`](../reports/restore).

```text
== restore demo: ref=baseline-v1
data md5   : 107f23e3bb76e1e34faaf8ddcc25a8e0
Baseline reproduces exactly from this commit's code, config and data.
recorded   : AP=0.876207 recall=0.970588 data=v1
reproduced : AP=0.876207 recall=0.970588 data=v1
```

Run after v2 has merged, the same demo restores v1 code + v1 data even though `main` now points to v2 (see `reports/restore/restore_baseline-v1.log`). `restore_main.log` shows the v2 state reproducing its own baseline.

Reproducibility relies on: pinned full lock file (`requirements.lock`), deterministic hash split, seeded CV and models, `n_jobs`-independent algorithms, and data pinned by md5.

## Commit history

`git log --oneline` shows small, single-purpose commits (scaffold → DVC → data v1 → config → data module → features → models → training → gate → registry → tests → experiments → baseline → CI → docs). The PR branches each carry one focused change.

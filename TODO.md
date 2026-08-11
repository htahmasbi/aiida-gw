# TODO

## Code polish — from pre-HT review (Aug 2026)

### Correctness / robustness
- [ ] **1. `relaxation.py` / `single_point.py`** access `outputs.output_structure` unguarded in `finalize`/`inspect_scf` — missing structure crashes the workchain instead of a clean exit code. Apply the same guard we used in `gw.py`.
- [ ] **2. `cli.py run`** parses `--supercell`, `--kpoints`, `--kpoints-w` with bare `int(x)` — garbage input raises a cryptic traceback; wrap with a helpful message.
- [ ] **3. `run --group`** calls `s.get_pymatgen()` on every node and re-fetches OPTIMADE every invocation — guard against non-structure nodes and allow reusing already-stored structures.
- [ ] **4. Re-running `aiida-gw run` duplicates** `prepared_*` StructureData nodes — add dedup keyed on `optimade_id`/extras.

### Refactoring / DRY
- [ ] **5. `cli.py` `--json-dir` and `--json-files` branches** are ~100 near-identical lines → extract one `_load_structures_from_json(files, ...)` helper.
- [ ] **6. `get_kinds_section_qs`** element-loop (override / resolve / yaml branches) is convoluted → simplify.
- [ ] **7. `_resolve_*_for_element` helpers** in builders do `logger.error` **and** `print(..., stderr)` — duplicate reporting; keep one.

### Config validation
- [ ] **8. Pydantic validators:** warn on `periodic` vs `cell_periodic` inconsistency; sanity-check `memory_per_proc` vs node count; type `memory_per_machine` as `int | None` instead of string `"0"`.

### Tests / CI
- [ ] **9. `ci.yml` ignores `tests/test_builders.py`** — install `aiida-core` + `aiida-cp2k` in CI so all tests run.
- [ ] **10. No tests for `cli.py`, `workflows/*`, or `cp2k_parsers.py`** — add coverage for run/results paths and parser branches.
- [ ] **11. Add `conftest.py` fixtures** to de-duplicate the temp-file helpers in `test_data_reader.py`/`test_builders.py`.

### Tooling / hygiene
- [ ] **12. `.ruff_cache` / `.mypy_cache` exist but no config** (`pyproject.toml`/`ruff.toml`/`mypy.ini`) — add config, pre-commit, run `ruff check` in CI.
- [ ] **13. Remove deprecated `reentry_register`** from `setup.json` (aiida-core 2.x uses native entry points — confirmed not needed).
- [ ] **14. Clean strays:** `workflows/archive/` (legacy step1-3.py pipeline), root `run_dir/`, `build/`, `TODO.md` (done items), `.egg-info`.
- [ ] **15. `utils/` scripts** (`export_data.py`, `setup_cluster.sh`, ...) are ad-hoc — document or consolidate.

### Docs
- [ ] **16. README** verify it matches current behavior (resolution check, `--max-structures` on `--json-dir`, results columns).

### Pending features / HT (in session todo list)
- [ ] **17. Opt-in `V_HARTREE_CUBE`/`E_DENSITY_CUBE` PRINT + retrieve list** for band alignment — only after test runs validate.
- [ ] **18. Confirm `periodic` vs `cell_periodic` in `config.toml`** (XZ vs XYZ) before HT.
- [ ] **19. Confirm `memory_per_proc = 96`** (GB/rank → ~1.5 TB/node) for `cpu-genoa` nodes.
- [ ] **20. Test runs on P group** with fixed parser; verify results + band-structure path.
- [ ] **21. Launch HT GW calculations.**

## Done

- [x] **1. Rename `auto_resolve_ri` flag** → `resolve_from_files` with updated description
- [x] **3. Smart selection for orbital basis** — handled by `orb_basis` filter + `_first_token`
- [x] **4. Cache parsed data files** — `functools.lru_cache` on file-path parsing
- [x] **5. More robust error reporting** — `logger.error` + print to stderr on resolution failures
- [x] **6. Tests** — 48 tests across 4 files, CI badge in README
- [x] **7. Default `resolve_from_files` to `True`** — changed from `False`
- [x] **8. Smart RI accuracy selection** — `ri_basis_accuracy_target` picks closest error value
- [x] **9. First-token resolution** — strip aliases from potential/orbital names
- [x] **10. `fetch-json` CLI command** — saves MC2D structures grouped by element count into JSON files
- [x] **Bugfix: numpy missing in `classify_from_spacegroup`**
- [x] **Bugfix: `kpoints_w` can be `None` → crash**
- [x] **Bugfix: `Cp2kEFSParser` doesn't handle `RUN_TYPE ENERGY`**
- [x] **Bugfix: `BasisEntry.__post_init__` treats `0.0` as falsy**
- [x] **12. `save_mc2d_by_nelements` handles `None` nelements** — `None` values are skipped with a warning; no more `TypeError` on `sorted()`.
- [x] **13. `get_kinds_section_qs` / `get_kinds_section_sirius` graceful fallback** — `.get()` with warning + `"DEFAULT"` fallback instead of `KeyError`.
- [x] **14. Better error logging in `Cp2kEFSParser._parse_efs` ENERGY branch** — Exception is logged before returning `ERROR_OUTPUT_MISSING`.
- [x] **15. `save_mc2d_by_nelements` skips `None` nelements** — Addressed together with #12; structures with `nelements=None` are skipped with a warning.
- [x] **16. `get_file_section_qs` warns when no files found** — Logs a warning if no standard CP2K data files exist.
- [x] **17. Removed unused `basis_set_mapping`/`potential_mapping`** — Fields removed from `Cp2kConfig`.
- [x] **18. Removed unused `resource_preset` and `RESOURCE_PRESETS`** — Field removed from `ProjectConfig`, dict removed from `enums.py`, CLI display line removed.
- [x] **19. Dead code in `parsers.py` cleaned up** — Removed 3 commented-out sections (PBC parsing, bohr conversion).
- [x] **20. OPTIMADE fetch retry logic** — `fetch_mc2d_structures` now retries up to 3× with exponential backoff on transient failures.

## Remaining

*(None — all items completed)*

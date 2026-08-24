# TODO

## Code polish — pre-HT review (Aug 2026)

### Correctness / robustness
- [ ] **4. Dedup `prepared_*` StructureData nodes** — re-running `aiida-gw run` duplicates prepared structures. Add dedup keyed on `optimade_id`/extras in `cli.py` run loop.

### Refactoring / DRY
- [ ] **5. `cli.py` `--json-dir` and `--json-files` branches** are ~100 near-identical lines → extract one `_load_structures_from_json(files, ...)` helper.
- [ ] **6. `get_kinds_section_qs`** element-loop (override / resolve / yaml branches) is convoluted → simplify.
- [x] **7. `_resolve_*_for_element` helpers** do `logger.error` + `print(stderr)` — duplicate reporting. **DECLINED** — keep `logger.error` only; stderr is useful for shell users. *(no change needed)*

### Config validation
- [ ] **8. Pydantic validators** — `memory_per_proc` sanity-check vs node count not done. `periodic` vs `cell_periodic` inconsistency not checked.

### Tests / CI
- [ ] **9. `ci.yml` ignores `tests/test_builders.py`** — install `aiida-core` + `aiida-cp2k` in CI so all tests run.
- [ ] **10. No tests for `cli.py` or `workflows/*`** — only `Cp2kBuilder` is covered. Add CLI run/results path tests.
- [ ] **11. Add `conftest.py` fixtures** to de-duplicate the temp-file helpers in `test_data_reader.py`.

### Tooling / hygiene
- [ ] **12. Add ruff/mypy config** — no `pyproject.toml`/`ruff.toml`/`mypy.ini` exists. Add config + pre-commit + run ruff in CI.
- [x] **13. Remove deprecated `reentry_register`** from `setup.json` — **DECLINED** — harmless with aiida-core 2.x, no action needed. *(no change needed)*
- [ ] **14. Clean strays:** `workflows/archive/` (legacy step1-3.py pipeline), root `build/`, `.egg-info`. (`run_dir/` and `TODO.md` done items already cleaned.)
- [ ] **15. Document `utils/` scripts** (`export_data.py`, `setup_cluster.sh`, ...) — ad-hoc cluster utilities need a README or consolidation.

### Docs
- [ ] **16. Verify README** matches current behavior (resolution check, `--max-structures` on `--json-dir`, results columns, `--json-files` support).

---

## Pending features / HT preparation

- [x] **17. Test GW parser with real calculation** — validated against run 307 output: DOS parsers OK (1298 pts); `read_bandstructure()` rewritten for real xTP format (61×52 DFT/G0W0, 61×104 SOC) in afdf0a5.
- [x] **18. Verify `periodic` vs `cell_periodic` in `config.toml`** (XZ vs XYZ) — confirmed via test run 358 reproducing reference gaps with current config.toml.
- [x] **19. Confirm `memory_per_proc = 96`** (GB/rank → ~1.5 TB/node) for `cpu-genoa` nodes — checked previously.
- [x] **20. Test runs on P group** with fixed parser; verify results + band-structure path — workchain 358: exit 0, all gaps match reference exactly (SCF 1.541 / G0W0 2.259 / G0W0+SOC 1.962 eV), bandstructures 61×52 + 61×104 stored.
- [ ] **21. Launch HT GW calculations.**

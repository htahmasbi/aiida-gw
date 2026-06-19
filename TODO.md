# TODO

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

### Feature: element_settings (from earlier TODO)

- [ ] **2. Add `element_settings` override** — explicitly pin potential/RI/orbital per element in config

### Testing

- [ ] **11. Test Fetch Json files** — do some test runs and do calculations using these json files.

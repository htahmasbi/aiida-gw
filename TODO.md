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

## Remaining

- [ ] **2. Add `element_settings` override** — explicitly pin potential/RI/orbital per element in config

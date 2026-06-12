# TODO

## 1. Rename `auto_resolve_ri` flag
Misleading — it controls orbital/potential file resolution, but RI is always resolved.
Rename to `auto_resolve` or `resolve_from_files` and update description in `config.py`.

## 2. Add `element_settings` override
Like `aiida-relax-project`: allows explicitly pinning potential/RI/orbital per element.
Should take priority over auto-resolve when set.

## 3. Smart selection for orbital basis
`resolve_orbital_basis_name` just picks the first match. If an element has multiple
ORB entries (e.g. SZV and DZVP), it might pick the wrong one. Filter by pattern
or pick the most diffuse.

## 4. Cache parsed data files
Each element re-parses the full file from scratch. Use `functools.lru_cache` on
`parse_cp2k_data_file` for multi-element structures.

## 5. More robust error reporting
When `basis_set_file` / `potential_file` are missing or can't be parsed, the
warning goes to the logger but CLI users may never see it. Log at ERROR level
or print to stderr.

## 6. Tests
No test suite. At minimum: `parse_cp2k_data_file`, `resolve_ri_basis_name`,
`resolve_potential_name` with sample file content.

## 7. Default `auto_resolve_ri` to `True`
Currently `False` — users with custom files but no explicit flag silently fall
back to YAML values, which may be wrong for their element set.

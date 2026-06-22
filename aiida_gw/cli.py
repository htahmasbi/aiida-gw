from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from aiida_gw.core.config import ProjectConfig, get_config, load_config
from aiida_gw.core.enums import CalculationMode
from aiida_gw.core.logging import get_logger, setup_logging

app = typer.Typer(
    name="aiida-gw",
    help="AiiDA GW workflows for CP2K on 2D materials",
    add_completion=False,
)

console = Console()
logger = get_logger("cli")


def _parse_exclude_elements(exclude_str: str | None) -> set[str] | None:
    """Parse ``--exclude-elements`` into a set, or return ``None``."""
    if exclude_str:
        return set(e.strip() for e in exclude_str.split(",") if e.strip())
    return None


def _detect_supported_elements(config: ProjectConfig) -> set[str] | None:
    """Auto-detect elements with full basis/potential/RI support from data files."""
    if not config.gw.resolve_from_files:
        return None
    try:
        from aiida_gw.codes.cp2k.data_reader import get_supported_elements

        supported = get_supported_elements(
            config.gw.basis_set_file,
            config.gw.ri_basis_set_file,
            config.gw.potential_file,
        )
        if supported:
            logger.info(f"Auto-detected {len(supported)} supported elements from data files")
            return supported
    except Exception as exc:
        logger.warning(f"Could not auto-detect supported elements: {exc}")
    return None


def _has_unsupported_elements(
    struct_elements: set[str],
    exclude_elements: set[str] | None,
    supported_elements: set[str] | None,
) -> bool:
    """Check if *struct_elements* contains excluded or unsupported elements."""
    if exclude_elements and (struct_elements & exclude_elements):
        return True
    if supported_elements and not (struct_elements <= supported_elements):
        return True
    return False


@app.command()
def run(
    mode: Annotated[
        CalculationMode,
        typer.Option("--mode", "-m", help="Calculation mode"),
    ] = CalculationMode.SINGLE_POINT,
    code_label: Annotated[
        str | None,
        typer.Option("--code", help="Code label (e.g. cp2k@localhost)"),
    ] = None,
    structure_file: Annotated[
        str | None,
        typer.Option("--structure", "-s", help="Structure file (cif/POSCAR)"),
    ] = None,
    group: Annotated[
        str | None,
        typer.Option("--group", help="AiiDA group with structures (fetched from OPTIMADE or stored locally)"),
    ] = None,
    max_structures: Annotated[
        int,
        typer.Option("--max-structures", help="Max structures (OPTIMADE / volume scan)"),
    ] = 10,
    vacuum: Annotated[
        float | None,
        typer.Option("--vacuum", help="Vacuum gap (Å) for 2D materials"),
    ] = None,
    supercell: Annotated[
        str | None,
        typer.Option(
            "--supercell",
            help="Supercell as nx,ny,nz (e.g. 3,3,1)",
        ),
    ] = None,
    kpoints: Annotated[
        str | None,
        typer.Option("--kpoints", help="K-point mesh as kx,ky,kz"),
    ] = None,
    kpoints_w: Annotated[
        str | None,
        typer.Option("--kpoints-w", help="GW k-point mesh as kx,ky,kz"),
    ] = None,
    elements: Annotated[
        str | None,
        typer.Option("--elements", help="Comma-separated elements for OPTIMADE filter (e.g. B,N)"),
    ] = None,
    optimade_filter: Annotated[
        str | None,
        typer.Option("--filter", help="Raw OPTIMADE filter string (overrides --elements)"),
    ] = None,
    exclude_elements: Annotated[
        str | None,
        typer.Option("--exclude-elements", help="Comma-separated elements to exclude (e.g. La,Ce,Pr)"),
    ] = None,
    json_dir: Annotated[
        str | None,
        typer.Option("--json-dir", help="Directory with mc2d_*.json files from fetch-json"),
    ] = None,
    formula: Annotated[
        str | None,
        typer.Option("--formula", help="Run only the structure with this chemical formula (e.g. MoS2)"),
    ] = None,
) -> None:
    """Run a calculation workflow."""
    from aiida import load_profile
    from aiida.engine import submit
    from aiida.orm import Float, List, load_code, Str, StructureData
    from aiida.plugins import DataFactory

    from aiida_gw.transformations.structures import prepare_2d_structure

    config = get_config()
    code_label = code_label or config.code_label

    try:
        load_profile()
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to load AiiDA profile: {e}")
        raise typer.Exit(1)

    try:
        code = load_code(code_label)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Code '{code_label}' not found: {exc}")
        console.print("Run 'verdi code list' to see available codes.")
        raise typer.Exit(1)

    kpoints_mesh = None
    if kpoints:
        kpoints_mesh = [int(x) for x in kpoints.split(",")]

    kpoints_w_mesh = None
    if kpoints_w:
        kpoints_w_mesh = [int(x) for x in kpoints_w.split(",")]

    metadata = config.metadata_options.to_dict()

    if mode == CalculationMode.GW:
        if not structure_file and not group and not json_dir:
            console.print("[red]Error:[/red] Provide --structure, --group, or --json-dir for GW mode")
            raise typer.Exit(1)

        gw_config = config.gw
        vacuum_val = vacuum or gw_config.vacuum
        supercell_val = [int(x) for x in (supercell or ",".join(map(str, gw_config.supercell))).split(",")]

        if group:
            from aiida_gw.datasets.mc2d_optimade import fetch_and_store_mc2d

            element_list = elements.split(",") if elements else None
            user_excl = _parse_exclude_elements(exclude_elements)
            supported = _detect_supported_elements(config) if user_excl is None else None
            fetch_and_store_mc2d(
                group_label=group,
                max_structures=max_structures,
                elements=element_list,
                optimade_filter=optimade_filter,
                exclude_elements=user_excl,
                supported_elements=supported,
            )

            from aiida.orm import Group

            group_node = Group.collection.get(label=group)
            structures = list(group_node.nodes)[:max_structures]
            if user_excl or supported:
                filtered = []
                for s in structures:
                    comp = s.get_pymatgen().composition
                    elems = {str(e) for e in comp.elements}
                    if _has_unsupported_elements(elems, user_excl, supported):
                        logger.info(f"Skipping structure {s.pk} ({s.label}) — unsupported elements: {elems}")
                    else:
                        filtered.append(s)
                structures = filtered
            console.print(f"[green]Fetched {len(structures)} structures from OPTIMADE[/green]")
        elif json_dir:
            from pathlib import Path as PPath
            import json as _json

            from pymatgen.core import Structure

            json_path = PPath(json_dir)
            if not json_path.is_dir():
                console.print(f"[red]Error:[/red] '{json_dir}' is not a directory")
                raise typer.Exit(1)

            json_files = sorted(json_path.glob("mc2d_*.json"))
            if not json_files:
                console.print(f"[red]Error:[/red] No mc2d_*.json files found in '{json_dir}'")
                raise typer.Exit(1)

            user_excl = _parse_exclude_elements(exclude_elements)
            supported = _detect_supported_elements(config) if user_excl is None else None
            target_elements = set(elements.split(",")) if elements else None

            structures = []
            for json_file in json_files:
                with open(json_file) as f:
                    entries = _json.load(f)
                for entry in entries:
                    entry_formula = entry.get("formula", "")
                    if formula and entry_formula != formula:
                        continue
                    elems = set(entry.get("elements", []))
                    if target_elements and elems != target_elements:
                        logger.info(f"Skipping {entry_id} — elements {elems} != {target_elements}")
                        continue
                    if user_excl and (elems & user_excl):
                        logger.info(f"Skipping {entry_id} — excluded elements: {elems & user_excl}")
                        continue
                    if supported and not (elems <= supported):
                        logger.info(f"Skipping {entry_id} — unsupported elements: {elems - supported}")
                        continue
                    try:
                        pmg_struct = Structure.from_dict(entry["structure"])
                    except Exception as e:
                        logger.warning(f"Failed to parse structure {entry.get('id', 'unknown')}: {e}")
                        continue
                    try:
                        node = StructureData(pymatgen=pmg_struct)
                        node.store()
                        node.base.extras.set("source", "mc2d_optimade")
                        node.base.extras.set("optimade_id", entry.get("id", ""))
                        node.base.extras.set("formula", entry.get("formula", ""))
                        structures.append(node)
                    except Exception as e:
                        logger.warning(f"Failed to store structure {entry.get('id', 'unknown')}: {e}")
                        continue

            if not structures:
                msg = f"No structure with formula '{formula}' found" if formula else "No valid structures found in JSON files"
                console.print(f"[red]Error:[/red] {msg}")
                raise typer.Exit(1)

            console.print(f"[green]Loaded {len(structures)} structures from JSON files[/green]")
        else:
            from aiida.orm import StructureData as OrmStructureData

            try:
                sm = OrmStructureData.get_or_create(structure_file)
                structures = [sm[0]]
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to load structure: {e}")
                raise typer.Exit(1)

        if structures:
            from aiida_gw.core.builders import get_bandstructure_path

            pk_list = []
            for i, struct in enumerate(structures):
                original_pmg = struct.get_pymatgen()
                prepared = prepare_2d_structure(
                    original_pmg,
                    vacuum=vacuum_val,
                    supercell=supercell_val,
                )
                prepared_node = StructureData(pymatgen=prepared).store()
                prepared_node.label = f"prepared_{i}"

                from aiida_gw.workflows.gw import GwWorkChain

                builder = GwWorkChain.get_builder()
                builder.structure = prepared_node
                builder.code = code
                builder.protocol_name = Str("protocol_GW.yml")
                if kpoints_mesh:
                    builder.kpoints_mesh = List(list=kpoints_mesh)
                if kpoints_w_mesh:
                    builder.kpoints_w_mesh = List(list=kpoints_w_mesh)
                if gw_config.ri_basis_accuracy_target is not None:
                    builder.ri_basis_accuracy_target = Float(gw_config.ri_basis_accuracy_target)

                # Compute bandstructure path from the ORIGINAL structure
                # (before vacuum/supercell) using pymatgen SpacegroupAnalyzer
                bs_path = get_bandstructure_path(
                    prepared_node,
                    original_pmg_structure=original_pmg,
                )
                builder.bandstructure_path = List(list=bs_path)

                node = submit(builder)
                pk_list.append(node.pk)
                console.print(f"[green]Submitted[/green] GwWorkChain<{node.pk}> for structure {i}")

            console.print(f"Submitted {len(pk_list)} GW calculations")

    elif mode == CalculationMode.SINGLE_POINT:
        if not structure_file:
            console.print("[red]Error:[/red] Provide --structure for single-point mode")
            raise typer.Exit(1)

        from aiida.orm import StructureData as OrmStructureData

        sm = OrmStructureData.get_or_create(structure_file)
        structure = sm[0]

        from aiida_gw.workflows.single_point import SinglePointWorkChain

        builder = SinglePointWorkChain.get_builder()
        builder.structure = structure
        builder.code = code
        if kpoints_mesh:
            builder.kpoints_mesh = List(list=kpoints_mesh)

        node = submit(builder)
        console.print(f"[green]Submitted[/green] SinglePointWorkChain<{node.pk}>")

    elif mode == CalculationMode.RELAX:
        if not structure_file:
            console.print("[red]Error:[/red] Provide --structure for relaxation mode")
            raise typer.Exit(1)

        from aiida.orm import StructureData as OrmStructureData

        sm = OrmStructureData.get_or_create(structure_file)
        structure = sm[0]

        from aiida_gw.workflows.relaxation import RelaxWorkChain

        builder = RelaxWorkChain.get_builder()
        builder.structure = structure
        builder.code = code
        if kpoints_mesh:
            builder.kpoints_mesh = List(list=kpoints_mesh)

        node = submit(builder)
        console.print(f"[green]Submitted[/green] RelaxWorkChain<{node.pk}>")


@app.command()
def config_show(
    as_json: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Show current configuration."""
    config = get_config()

    if as_json:
        console.print_json(data=config.model_dump())
    else:
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("code_label", config.code_label)
        table.add_row("", "")

        table.add_row("cp2k.cutoff", str(config.cp2k.cutoff))
        table.add_row("cp2k.kpoints_mesh", str(config.cp2k.kpoints_mesh))
        table.add_row("", "")

        table.add_row("gw.kpoints_mesh", str(config.gw.kpoints_mesh))
        table.add_row("gw.vacuum", str(config.gw.vacuum))
        table.add_row("gw.supercell", str(config.gw.supercell))
        if config.gw.element_settings:
            for el, ovr in config.gw.element_settings.items():
                parts = []
                if ovr.orb_basis:
                    parts.append(f"orb={ovr.orb_basis}")
                if ovr.potential:
                    parts.append(f"pot={ovr.potential}")
                if ovr.ri_basis:
                    parts.append(f"ri={ovr.ri_basis}")
                table.add_row(f"gw.element_settings.{el}", ", ".join(parts))
        else:
            table.add_row("gw.element_settings", "(none)")

        console.print(table)


@app.command()
def fetch(
    group_label: Annotated[str, typer.Option("--group", help="Group label for storing structures")] = "mc2d_structures",
    max_structures: Annotated[int, typer.Option("--max", help="Maximum number of structures")] = 10,
    elements: Annotated[str | None, typer.Option("--elements", help="Comma-separated element list (e.g. B,N)")] = None,
    optimade_filter: Annotated[
        str | None,
        typer.Option("--filter", help="Raw OPTIMADE filter string (overrides --elements)"),
    ] = None,
    exclude_elements: Annotated[
        str | None,
        typer.Option("--exclude-elements", help="Comma-separated elements to exclude (e.g. La,Ce,Pr)"),
    ] = None,
) -> None:
    """Fetch 2D structures from MC2D database via OPTIMADE."""
    from aiida import load_profile

    load_profile()

    element_list = elements.split(",") if elements else None
    from aiida_gw.datasets.mc2d_optimade import fetch_and_store_mc2d

    config = get_config()
    user_excl = _parse_exclude_elements(exclude_elements)
    supported = _detect_supported_elements(config) if user_excl is None else None

    data = fetch_and_store_mc2d(
        group_label=group_label,
        max_structures=max_structures,
        elements=element_list,
        optimade_filter=optimade_filter,
        exclude_elements=user_excl,
        supported_elements=supported,
    )
    console.print(f"[green]Fetched {len(data)} MC2D structures into group '{group_label}'[/green]")


@app.command()
def fetch_json(
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = ".",
    max_structures: Annotated[int | None, typer.Option("--max", help="Maximum number of structures")] = None,
    exclude_elements: Annotated[
        str | None,
        typer.Option("--exclude-elements", help="Comma-separated elements to exclude (e.g. La,Ce,Pr)"),
    ] = None,
    structures_per_file: Annotated[
        int | None,
        typer.Option("--structures-per-file", help="Max structures per JSON file (splits large groups into numbered chunks)"),
    ] = None,
) -> None:
    """Fetch all MC2D structures and save as JSON files grouped by element count."""
    from aiida_gw.datasets.mc2d_optimade import save_mc2d_by_nelements

    config = get_config()
    user_excl = _parse_exclude_elements(exclude_elements)
    supported = _detect_supported_elements(config) if user_excl is None else None

    paths = save_mc2d_by_nelements(
        output_dir=output_dir,
        max_structures=max_structures,
        exclude_elements=user_excl,
        supported_elements=supported,
        structures_per_file=structures_per_file,
    )
    for n, entry in sorted(paths.items()):
        if isinstance(entry, list):
            for p in entry:
                console.print(f"[green]{n} element(s):[/green] {p}")
        else:
            console.print(f"[green]{n} element(s):[/green] {entry}")


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress output")] = False,
) -> None:
    """AiiDA GW workflows for CP2K on 2D materials."""
    global console
    if quiet:
        console = Console(quiet=True)
        logging.getLogger("aiida_gw").setLevel(logging.WARNING)
    elif verbose:
        logging.getLogger("aiida_gw").setLevel(logging.DEBUG)

    setup_logging()


if __name__ == "__main__":
    app()

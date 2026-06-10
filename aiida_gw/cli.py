from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from aiida_gw.core.config import get_config, load_config
from aiida_gw.core.enums import CalculationMode
from aiida_gw.core.logging import get_logger, setup_logging

app = typer.Typer(
    name="aiida-gw",
    help="AiiDA GW workflows for CP2K on 2D materials",
    add_completion=False,
)

console = Console()
logger = get_logger("cli")


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
    optimade_group: Annotated[
        str | None,
        typer.Option("--optimade-group", help="Fetch structures from OPTIMADE into group"),
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
) -> None:
    """Run a calculation workflow."""
    from aiida import load_profile
    from aiida.engine import submit
    from aiida.orm import load_code, Str, StructureData
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

    metadata = config.metadata_options.to_dict()

    if mode == CalculationMode.GW:
        if not structure_file and not optimade_group:
            console.print("[red]Error:[/red] Provide --structure or --optimade-group for GW mode")
            raise typer.Exit(1)

        gw_config = config.gw
        vacuum_val = vacuum or gw_config.vacuum
        supercell_val = [int(x) for x in (supercell or ",".join(map(str, gw_config.supercell))).split(",")]

        if optimade_group:
            from aiida_gw.datasets.mc2d_optimade import fetch_and_store_mc2d

            element_list = elements.split(",") if elements else None
            fetch_and_store_mc2d(
                group_label=optimade_group,
                max_structures=max_structures,
                elements=element_list,
            )

            from aiida.orm import Group

            group = Group.collection.get(label=optimade_group)
            structures = group.nodes
            console.print(f"[green]Fetched {len(structures)} structures from OPTIMADE[/green]")
        else:
            from aiida.orm import StructureData as OrmStructureData

            try:
                sm = OrmStructureData.get_or_create(structure_file)
                structures = [sm[0]]
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to load structure: {e}")
                raise typer.Exit(1)

        if structures:
            pk_list = []
            for i, struct in enumerate(structures):
                pymatgen = struct.get_pymatgen()
                prepared = prepare_2d_structure(
                    pymatgen,
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
        table.add_row("resource_preset", config.resource_preset)
        table.add_row("", "")

        table.add_row("cp2k.cutoff", str(config.cp2k.cutoff))
        table.add_row("cp2k.kpoints_mesh", str(config.cp2k.kpoints_mesh))
        table.add_row("", "")

        table.add_row("gw.kpoints_mesh", str(config.gw.kpoints_mesh))
        table.add_row("gw.vacuum", str(config.gw.vacuum))
        table.add_row("gw.supercell", str(config.gw.supercell))

        console.print(table)


@app.command()
def fetch(
    group_label: Annotated[str, typer.Option("--group", help="Group label for storing structures")] = "mc2d_structures",
    max_structures: Annotated[int, typer.Option("--max", help="Maximum number of structures")] = 10,
    elements: Annotated[str | None, typer.Option("--elements", help="Comma-separated element list (e.g. B,N)")] = None,
) -> None:
    """Fetch 2D structures from MC2D database via OPTIMADE."""
    from aiida import load_profile

    load_profile()

    element_list = elements.split(",") if elements else None
    from aiida_gw.datasets.mc2d_optimade import fetch_and_store_mc2d

    data = fetch_and_store_mc2d(
        group_label=group_label,
        max_structures=max_structures,
        elements=element_list,
    )
    console.print(f"[green]Fetched {len(data)} MC2D structures into group '{group_label}'[/green]")


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress output")] = False,
) -> None:
    """AiiDA GW workflows for CP2K on 2D materials."""
    if quiet:
        logging.getLogger("aiida_gw").setLevel(logging.WARNING)
    elif verbose:
        logging.getLogger("aiida_gw").setLevel(logging.DEBUG)

    setup_logging()


if __name__ == "__main__":
    app()

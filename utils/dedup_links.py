"""Report and (optionally) remove duplicate output links left by double execution.

Before the RabbitMQ ``consumer_timeout`` fix, a long job could be executed twice,
leaving two ``CREATE`` links (on the ``CalcJobNode``) and/or two ``RETURN`` links
(on the workchains) that share a label such as ``retrieved`` or
``output_parameters``. AiiDA rebuilds output namespaces from those links and
raises ``KeyError: duplicate label ...`` as soon as a label repeats within a
single link type, which is what used to break workchain finalization:

  * ``CalculationNode.outputs`` walks ``CREATE`` links, so duplicate ``CREATE``
    links on a ``CalcJobNode`` are the problem there;
  * ``WorkflowNode.outputs`` walks ``RETURN`` links, so duplicate ``RETURN``
    links on a workchain are the problem there.

The runtime now tolerates duplicates (see ``aiida_gw.core.nodes.get_output_node``),
so this script is only needed to make the affected historical nodes inspectable
again via ``verdi process show`` / ``node.outputs``.

AiiDA exposes no API to delete a link, so ``--apply`` removes the extra rows from
``db_dblink`` directly (psql_dos storage). Stop the daemon and take a database
backup first. When the duplicate links point at different output nodes, the one
that looks more complete is kept (a non-empty ``output_parameters`` dict, or the
``retrieved`` folder with more files); ties keep the earliest link.

  python3 utils/dedup_links.py                       # report only
  python3 utils/dedup_links.py --pk 638 3868         # report specific sources
  python3 utils/dedup_links.py --pk 638 --apply      # remove (daemon stopped!)
"""
import argparse
import sys

from aiida import load_profile, orm
from aiida.common.links import LinkType
from aiida.manage import get_manager
from aiida.storage.psql_dos.models.node import DbLink

# Only these link types are surfaced through ``node.outputs`` and can collide.
OUTPUT_LINK_TYPES = (LinkType.CREATE.value, LinkType.RETURN.value)


def find_duplicate_groups(session):
    """Return ``(input_id, label, type, count)`` rows whose count is greater than one."""
    from sqlalchemy import func

    return (
        session.query(DbLink.input_id, DbLink.label, DbLink.type, func.count(DbLink.id))
        .filter(DbLink.type.in_(OUTPUT_LINK_TYPES))
        .group_by(DbLink.input_id, DbLink.label, DbLink.type)
        .having(func.count(DbLink.id) > 1)
        .all()
    )


def links_for(session, input_id, label, type_):
    """All link rows for one ``(source, label, type)`` group, oldest first."""
    return (
        session.query(DbLink)
        .filter(DbLink.input_id == input_id, DbLink.label == label, DbLink.type == type_)
        .order_by(DbLink.id)
        .all()
    )


def _content_score(output_pk: int, label: str) -> int:
    """A crude "how complete does this output look" score; 0 when empty or unknown."""
    try:
        node = orm.load_node(output_pk)
    except Exception:
        return 0
    try:
        if label == "output_parameters":
            return 1 if node.get_dict() else 0
        if label == "retrieved":
            return len(node.base.repository.list_object_names())
    except Exception:
        return 0
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="delete the duplicate links (default: report only)")
    parser.add_argument("--pk", nargs="*", type=int, default=None, help="restrict to these source node PKs")
    args = parser.parse_args()

    load_profile()
    session = get_manager().get_profile_storage().get_session()

    groups = find_duplicate_groups(session)
    if args.pk:
        wanted = set(args.pk)
        groups = [group for group in groups if group[0] in wanted]

    if not groups:
        print("No duplicate same-type output links found.")
        return

    header = f"{'source':>8}  {'label':<20} {'type':<7} {'n':>2}  output pks (score; kept marked *)"
    print(header)
    print("-" * len(header))

    removed = 0
    for input_id, label, type_, count in sorted(groups):
        rows = links_for(session, input_id, label, type_)
        scored = [(row, _content_score(row.output_id, label)) for row in rows]
        best_row = max(scored, key=lambda item: (item[1], -item[0].id))[0]
        rendered = ", ".join(
            f"{row.output_id}{'*' if row is best_row else ''}({score})" for row, score in scored
        )
        print(f"{input_id:>8}  {label:<20} {type_:<7} {count:>2}  {rendered}")
        if args.apply:
            for row in rows:
                if row is not best_row:
                    session.delete(row)
                    removed += 1

    if args.apply:
        session.commit()
        print(f"\nRemoved {removed} duplicate link(s); kept 1 per group.")
    else:
        print("\nDry run. Re-run with --apply to remove duplicates (stop the daemon first).")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

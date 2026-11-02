#!/usr/bin/env python3
import argparse, csv, json, logging, re
from collections import defaultdict

from aiida.manage.configuration import load_profile
load_profile()

from aiida import orm
from aiida.orm import CalcJobNode, Dict, StructureData, WorkflowNode

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

def _last_calcjob(node):
    calcs = [n for n in node.called_descendants if isinstance(n, CalcJobNode)]
    return max(calcs, key=lambda c: c.ctime) if calcs else None

def _get_retrieved(calc):
    try:
        return [out.node for out in calc.base.links.get_outgoing() if out.link_label == "retrieved"]
    except Exception:
        return []

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pks", nargs="+", type=int)
    parser.add_argument("--csv", help="CSV with gw_pk")
    args = parser.parse_args(argv)
    
    pks = []
    if args.pks:
        pks = args.pks
    if args.csv:
        with open(args.csv) as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get('gw_pk'):
                    pks.append(int(row['gw_pk']))
    
    for pk in pks:
        try:
            node = orm.load_node(pk)
            calc = _last_calcjob(node)
            print(f"\n{'='*60}")
            print(f"PK {pk} | calc {calc.pk if calc else None} | exit {calc.exit_status if calc else None}")
            if not calc:
                continue
            for ret in _get_retrieved(calc):
                try:
                    content = ret.base.repository.get_object_content('aiida.out')
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    # Look for SCF info
                    lines = content.splitlines()
                    # Find mixing info
                    for i,l in enumerate(lines):
                        if 'MIXING' in l or 'BROYDEN' in l or 'DIRECT_P_MIXING' in l or 'ADDED_MOS' in l:
                            # print surrounding
                            for j in range(max(0,i-2), min(len(lines), i+8)):
                                print(lines[j])
                            print("---")
                    # Look for oscillation signs
                    # count SCF iterations
                    scf_lines = [l for l in lines if 'SCF run' in l or 'OT DIIS' in l or re.search(r'SCF CYCLE', l)]
                    if scf_lines:
                        print(f"  SCF-related lines: {len(scf_lines)}")
                        # print last few
                        for l in scf_lines[-5:]:
                            print(f"    {l[:100]}")
                    # Check for charges/mixing issues
                    if 'charge' in content.lower() or 'slosh' in content.lower():
                        print("  MENTIONS charge/slosh")
                    # Look for eigenvalues jumping
                except Exception as e:
                    print(f"  err {e}")
        except Exception as e:
            print(f"ERR {pk}: {e}")

if __name__ == '__main__':
    main()

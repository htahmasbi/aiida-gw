#!/usr/bin/env python3
"""Plot SCF and G0W0 indirect gaps (with/without SOC) for finished runs."""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="CSV from make_gw_mc2d_table.py")
    parser.add_argument("--out", default="gw_gaps_comparison.png", help="Output plot")
    parser.add_argument("--exclude-negative", action="store_true", help="Exclude runs with negative GW gaps")
    args = parser.parse_args(argv)

    data = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                scf = row.get('scf_gap_indirect_eV')
                scf_soc = row.get('scf_soc_gap_indirect_eV')
                gw = row.get('g0w0_gap_indirect_eV')
                gw_soc = row.get('g0w0_soc_gap_indirect_eV')
                if not scf and not scf_soc and not gw and not gw_soc:
                    continue
                scf = float(scf) if scf and scf.strip() else np.nan
                scf_soc = float(scf_soc) if scf_soc and scf_soc.strip() else np.nan
                gw = float(gw) if gw and gw.strip() else np.nan
                gw_soc = float(gw_soc) if gw_soc and gw_soc.strip() else np.nan
                if args.exclude_negative:
                    if (gw < 0 and not np.isnan(gw)) or (gw_soc < 0 and not np.isnan(gw_soc)):
                        continue
                data.append({
                    'formula': row.get('formula',''),
                    'scf': scf,
                    'scf_soc': scf_soc,
                    'gw': gw,
                    'gw_soc': gw_soc,
                    'gw_pk': row.get('gw_pk','')
                })
            except Exception:
                continue

    if not data:
        print("No data to plot")
        return 1

    # Sort by formula or gw
    data.sort(key=lambda x: (x['gw'] if not np.isnan(x['gw']) else x['scf']))

    x = np.arange(len(data))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(data)*0.3), 8))
    ax.bar(x - 1.5*width, [d['scf'] for d in data], width, label='scf_gap_indirect', alpha=0.8)
    ax.bar(x - 0.5*width, [d['scf_soc'] for d in data], width, label='scf_soc_gap_indirect', alpha=0.8)
    ax.bar(x + 0.5*width, [d['gw'] for d in data], width, label='g0w0_gap_indirect', alpha=0.8)
    ax.bar(x + 1.5*width, [d['gw_soc'] for d in data], width, label='g0w0_soc_gap_indirect', alpha=0.8)

    ax.set_ylabel('Gap (eV)')
    ax.set_xlabel('Structure')
    ax.set_title('Indirect Gaps: SCF vs G0W0 (with/without SOC)')
    ax.set_xticks(x)
    ax.set_xticklabels([d['formula'] for d in data], rotation=90, fontsize=6)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")
    return 0

if __name__ == '__main__':
    sys.exit(main())

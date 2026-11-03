#!/usr/bin/env python3
"""Scatter plot of SCF and G0W0 indirect gaps (with/without SOC) for finished runs."""

import argparse
import csv
import logging
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="CSV from make_gw_mc2d_table.py")
    parser.add_argument("--out", default="gw_indirect_gaps_scatter.png", help="Output plot")
    parser.add_argument("--csv-out", help="Save plotted data to CSV")
    parser.add_argument("--exclude-negative", action="store_true", help="Exclude runs with negative GW gaps")
    parser.add_argument("--include-negative", action="store_true", help="Include runs with negative GW gaps")
    args = parser.parse_args(argv)

    if args.include_negative:
        args.exclude_negative = False

    data = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                failure = (row.get('failure_reason') or row.get('failure_detailed') or '').upper()
                bad = ['SCF_NOT_CONVERGED', 'TIME_LIMIT', 'EXCEPTED', 'CRASHED', 'KILLED']
                if any(b in failure for b in bad) and not row.get('g0w0_gap_indirect_eV'):
                    continue

                scf = row.get('scf_gap_indirect_eV')
                scf_soc = row.get('scf_soc_gap_indirect_eV')
                gw = row.get('g0w0_gap_indirect_eV')
                gw_soc = row.get('g0w0_soc_gap_indirect_eV')
                if not any([scf, scf_soc, gw, gw_soc]):
                    continue

                scf_v = float(scf) if scf and str(scf).strip() not in ['', 'None', 'nan'] else np.nan
                scf_soc_v = float(scf_soc) if scf_soc and str(scf_soc).strip() not in ['', 'None', 'nan'] else np.nan
                gw_v = float(gw) if gw and str(gw).strip() not in ['', 'None', 'nan'] else np.nan
                gw_soc_v = float(gw_soc) if gw_soc and str(gw_soc).strip() not in ['', 'None', 'nan'] else np.nan

                if args.exclude_negative:
                    if (not np.isnan(gw_v) and gw_v < 0) or (not np.isnan(gw_soc_v) and gw_soc_v < 0):
                        continue

                data.append({
                    'gw_pk': row.get('gw_pk', ''),
                    'formula': row.get('formula', ''),
                    'mc2d_id': row.get('mc2d_id', ''),
                    'group': row.get('group', ''),
                    'scf_gap_indirect_eV': scf_v,
                    'scf_soc_gap_indirect_eV': scf_soc_v,
                    'g0w0_gap_indirect_eV': gw_v,
                    'g0w0_soc_gap_indirect_eV': gw_soc_v,
                })
            except Exception:
                continue

    if not data:
        print("No data to plot")
        return 1

    if args.csv_out:
        try:
            fieldnames = list(data[0].keys())
            with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for d in data:
                    outd = dict(d)
                    for k in fieldnames:
                        if isinstance(outd.get(k), float) and np.isnan(outd[k]):
                            outd[k] = ''
                    w.writerow(outd)
            LOGGER.info("Saved %s", args.csv_out)
        except Exception as exc:
            LOGGER.warning("Failed to write csv-out: %s", exc)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each point
    for d in data:
        x = np.arange(4)
        y = [d['scf_gap_indirect_eV'], d['scf_soc_gap_indirect_eV'], d['g0w0_gap_indirect_eV'], d['g0w0_soc_gap_indirect_eV']]
        ax.plot(x, y, 'o-', alpha=0.5, markersize=4)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['SCF indirect', 'SCF+SOC indirect', 'G0W0 indirect', 'G0W0+SOC indirect'], rotation=45)
    ax.set_ylabel('Gap (eV)')
    ax.set_title('Indirect Gaps: SCF vs G0W0 (with/without SOC)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    LOGGER.info("Saved %s", args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

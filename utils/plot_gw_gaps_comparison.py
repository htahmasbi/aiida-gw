#!/usr/bin/env python3
"""Plot SCF and G0W0 gaps (direct/indirect, with/without SOC) for finished runs."""

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
    parser.add_argument("--out", default="gw_gaps_comparison.png", help="Output plot")
    parser.add_argument("--csv-out", help="Save plotted data to CSV")
    parser.add_argument("--type", choices=["indirect", "direct", "both"], default="indirect", help="Which gap type to plot")
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
                if any(b in failure for b in bad):
                    # still allow if we have gw values
                    pass

                scf_d = row.get('scf_gap_direct_eV')
                scf_soc_d = row.get('scf_soc_gap_direct_eV')
                gw_d = row.get('g0w0_gap_direct_eV')
                gw_soc_d = row.get('g0w0_soc_gap_direct_eV')
                scf_i = row.get('scf_gap_indirect_eV')
                scf_soc_i = row.get('scf_soc_gap_indirect_eV')
                gw_i = row.get('g0w0_gap_indirect_eV')
                gw_soc_i = row.get('g0w0_soc_gap_indirect_eV')

                if args.type == "direct":
                    if not any([scf_d, scf_soc_d, gw_d, gw_soc_d]):
                        continue
                elif args.type == "indirect":
                    if not any([scf_i, scf_soc_i, gw_i, gw_soc_i]):
                        continue
                else:
                    if not any([scf_d, scf_soc_d, gw_d, gw_soc_d, scf_i, scf_soc_i, gw_i, gw_soc_i]):
                        continue

                scf_dv = float(scf_d) if scf_d and str(scf_d).strip() not in ['', 'None', 'nan'] else np.nan
                scf_soc_dv = float(scf_soc_d) if scf_soc_d and str(scf_soc_d).strip() not in ['', 'None', 'nan'] else np.nan
                gw_dv = float(gw_d) if gw_d and str(gw_d).strip() not in ['', 'None', 'nan'] else np.nan
                gw_soc_dv = float(gw_soc_d) if gw_soc_d and str(gw_soc_d).strip() not in ['', 'None', 'nan'] else np.nan
                scf_iv = float(scf_i) if scf_i and str(scf_i).strip() not in ['', 'None', 'nan'] else np.nan
                scf_soc_iv = float(scf_soc_i) if scf_soc_i and str(scf_soc_i).strip() not in ['', 'None', 'nan'] else np.nan
                gw_iv = float(gw_i) if gw_i and str(gw_i).strip() not in ['', 'None', 'nan'] else np.nan
                gw_soc_iv = float(gw_soc_i) if gw_soc_i and str(gw_soc_i).strip() not in ['', 'None', 'nan'] else np.nan

                if args.exclude_negative:
                    if args.type == "direct":
                        if (not np.isnan(gw_dv) and gw_dv < 0) or (not np.isnan(gw_soc_dv) and gw_soc_dv < 0):
                            continue
                    elif args.type == "indirect":
                        if (not np.isnan(gw_iv) and gw_iv < 0) or (not np.isnan(gw_soc_iv) and gw_soc_iv < 0):
                            continue
                    else:
                        neg = False
                        if (not np.isnan(gw_dv) and gw_dv < 0) or (not np.isnan(gw_soc_dv) and gw_soc_dv < 0):
                            neg = True
                        if (not np.isnan(gw_iv) and gw_iv < 0) or (not np.isnan(gw_soc_iv) and gw_soc_iv < 0):
                            neg = True
                        if neg:
                            continue

                data.append({
                    'gw_pk': row.get('gw_pk', ''),
                    'formula': row.get('formula', ''),
                    'mc2d_id': row.get('mc2d_id', ''),
                    'group': row.get('group', ''),
                    'scf_gap_direct_eV': scf_dv,
                    'scf_soc_gap_direct_eV': scf_soc_dv,
                    'g0w0_gap_direct_eV': gw_dv,
                    'g0w0_soc_gap_direct_eV': gw_soc_dv,
                    'scf_gap_indirect_eV': scf_iv,
                    'scf_soc_gap_indirect_eV': scf_soc_iv,
                    'g0w0_gap_indirect_eV': gw_iv,
                    'g0w0_soc_gap_indirect_eV': gw_soc_iv,
                })
            except Exception:
                continue

    if not data:
        print("No data to plot")
        return 1

    if args.type == "direct":
        data.sort(key=lambda x: (x['g0w0_gap_direct_eV'] if not np.isnan(x['g0w0_gap_direct_eV']) else x['scf_gap_direct_eV']))
    elif args.type == "indirect":
        data.sort(key=lambda x: (x['g0w0_gap_indirect_eV'] if not np.isnan(x['g0w0_gap_indirect_eV']) else x['scf_gap_indirect_eV']))
    else:
        data.sort(key=lambda x: ((x['g0w0_gap_direct_eV'] if not np.isnan(x['g0w0_gap_direct_eV']) else 1e9), (x['g0w0_gap_indirect_eV'] if not np.isnan(x['g0w0_gap_indirect_eV']) else 1e9)))

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

    x = np.arange(len(data))
    width = 0.2

    if args.type == "both":
        fig, axes = plt.subplots(1, 2, figsize=(max(20, len(data) * 0.3), 9))
        # direct
        ax = axes[0]
        ax.bar(x - 1.5 * width, [d['scf_gap_direct_eV'] for d in data], width, label='scf_gap_direct', alpha=0.8)
        ax.bar(x - 0.5 * width, [d['scf_soc_gap_direct_eV'] for d in data], width, label='scf_soc_gap_direct', alpha=0.8)
        ax.bar(x + 0.5 * width, [d['g0w0_gap_direct_eV'] for d in data], width, label='g0w0_gap_direct', alpha=0.8)
        ax.bar(x + 1.5 * width, [d['g0w0_soc_gap_direct_eV'] for d in data], width, label='g0w0_soc_gap_direct', alpha=0.8)
        ax.set_title('Direct Gaps')
        ax.set_xlabel('Structure')
        ax.set_ylabel('Gap (eV)')
        ax.set_xticks(x)
        ax.set_xticklabels([d['formula'] for d in data], rotation=90, fontsize=5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        # indirect
        ax = axes[1]
        ax.bar(x - 1.5 * width, [d['scf_gap_indirect_eV'] for d in data], width, label='scf_gap_indirect', alpha=0.8)
        ax.bar(x - 0.5 * width, [d['scf_soc_gap_indirect_eV'] for d in data], width, label='scf_soc_gap_indirect', alpha=0.8)
        ax.bar(x + 0.5 * width, [d['g0w0_gap_indirect_eV'] for d in data], width, label='g0w0_gap_indirect', alpha=0.8)
        ax.bar(x + 1.5 * width, [d['g0w0_soc_gap_indirect_eV'] for d in data], width, label='g0w0_soc_gap_indirect', alpha=0.8)
        ax.set_title('Indirect Gaps')
        ax.set_xlabel('Structure')
        ax.set_ylabel('Gap (eV)')
        ax.set_xticks(x)
        ax.set_xticklabels([d['formula'] for d in data], rotation=90, fontsize=5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    elif args.type == "direct":
        fig, ax = plt.subplots(figsize=(max(14, len(data) * 0.25), 9))
        ax.bar(x - 1.5 * width, [d['scf_gap_direct_eV'] for d in data], width, label='scf_gap_direct', alpha=0.8)
        ax.bar(x - 0.5 * width, [d['scf_soc_gap_direct_eV'] for d in data], width, label='scf_soc_gap_direct', alpha=0.8)
        ax.bar(x + 0.5 * width, [d['g0w0_gap_direct_eV'] for d in data], width, label='g0w0_gap_direct', alpha=0.8)
        ax.bar(x + 1.5 * width, [d['g0w0_soc_gap_direct_eV'] for d in data], width, label='g0w0_soc_gap_direct', alpha=0.8)
        ax.set_title('Direct Gaps: SCF vs G0W0 (with/without SOC)')
        ax.set_xlabel('Structure')
        ax.set_ylabel('Gap (eV)')
        ax.set_xticks(x)
        ax.set_xticklabels([d['formula'] for d in data], rotation=90, fontsize=5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(max(14, len(data) * 0.25), 9))
        ax.bar(x - 1.5 * width, [d['scf_gap_indirect_eV'] for d in data], width, label='scf_gap_indirect', alpha=0.8)
        ax.bar(x - 0.5 * width, [d['scf_soc_gap_indirect_eV'] for d in data], width, label='scf_soc_gap_indirect', alpha=0.8)
        ax.bar(x + 0.5 * width, [d['g0w0_gap_indirect_eV'] for d in data], width, label='g0w0_gap_indirect', alpha=0.8)
        ax.bar(x + 1.5 * width, [d['g0w0_soc_gap_indirect_eV'] for d in data], width, label='g0w0_soc_gap_indirect', alpha=0.8)
        ax.set_title('Indirect Gaps: SCF vs G0W0 (with/without SOC)')
        ax.set_xlabel('Structure')
        ax.set_ylabel('Gap (eV)')
        ax.set_xticks(x)
        ax.set_xticklabels([d['formula'] for d in data], rotation=90, fontsize=5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    LOGGER.info("Saved %s", args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())

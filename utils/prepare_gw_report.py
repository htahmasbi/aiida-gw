#!/usr/bin/env python3
"""Clean, deduplicate, classify and plot GW results for report preparation."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

try:
    from pymatgen.core import Composition
except Exception:
    Composition = None


GAP_FIELDS = [
    "scf_gap_indirect_eV",
    "scf_soc_gap_indirect_eV",
    "g0w0_gap_indirect_eV",
    "g0w0_soc_gap_indirect_eV",
    "hf_gap_direct_eV",
]
GW_FIELDS = ["g0w0_gap_indirect_eV", "g0w0_soc_gap_indirect_eV"]
SCF_FIELDS = ["scf_gap_indirect_eV", "scf_soc_gap_indirect_eV"]
NULLS = {"", "-", "none", "nan", "null", "n/a", "na"}

TABLE_SLICES = {
    "gw_pk": (0, 7),
    "calc_pk": (9, 16),
    "status": (18, 27),
    "physics": (28, 76),
    "wall_h": (77, 84),
    "scf_gap_indirect_eV": (86, 98),
    "scf_soc_gap_indirect_eV": (99, 111),
    "g0w0_gap_indirect_eV": (112, 124),
    "g0w0_soc_gap_indirect_eV": (125, 137),
    "hf_gap_direct_eV": (138, 150),
    "mc2d_bg": (151, 163),
    "energy": (164, 176),
    "formula": (178, 192),
    "failure_reason": (194, None),
}

OUT_FIELDS = [
    "decision",
    "reason",
    "report_key",
    "gw_pk",
    "calc_pk",
    "group",
    "mc2d_id",
    "structure_uuid",
    "prepared_structure_uuid",
    "formula",
    "magnetic_state",
    "unstable",
    "band_gap_match",
    "mc2d_bg_eV",
    "status",
    "wall_h",
    "scf_gap_indirect_eV",
    "scf_soc_gap_indirect_eV",
    "g0w0_gap_indirect_eV",
    "g0w0_soc_gap_indirect_eV",
    "hf_gap_direct_eV",
    "energy",
    "failure_reason",
    "physics_issues",
    "recommended_eps_filter",
    "recommended_num_time_freq",
    "recommended_kpoints_distance",
    "run_group",
]


def clean(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in NULLS:
        return ""
    return text


def number(value):
    text = clean(value)
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def integer(value):
    value = number(value)
    return int(value) if value is not None and value.is_integer() else None


def parse_table(text):
    rows = []
    for line in text.splitlines():
        padded = line.ljust(260)
        first = padded[:7].strip()
        if not first.isdigit():
            continue
        if len(line) < 40:
            continue
        row = {}
        for key, (start, stop) in TABLE_SLICES.items():
            row[key] = padded[start:stop].strip() if stop is not None else padded[start:].strip()
        row["bandgap_mc2d_overview_eV"] = row.pop("mc2d_bg")
        physics = row.pop("physics", "")
        row["physics_issues"] = physics if physics.lower().startswith("bad") else ""
        rows.append(row)
    return rows


def parse_csv(text):
    return list(csv.DictReader(text.splitlines()))


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    first = next((line for line in text.splitlines() if line.strip()), "")
    if "," in first and ("gw_pk" in first or "PK" in first or "status" in first):
        return parse_csv(text)
    return parse_table(text)


def load_crosswalk(path):
    if not path:
        return {}
    crosswalk = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pk = integer(row.get("pk") or row.get("gw_pk"))
            if pk is None:
                continue
            crosswalk[pk] = {key: clean(row.get(key)) for key in (
                "mc2d_id",
                "structure_uuid",
                "band_gap",
                "magnetic_state",
                "unstable",
                "band_gap_match",
                "mc2d_candidates",
            )}
    return crosswalk


def normalize_formula(formula):
    formula = clean(formula)
    if not formula:
        return ""
    if Composition is not None:
        try:
            return Composition(formula).reduced_formula
        except Exception:
            pass
    return formula


def augment_rows(rows, crosswalk):
    for row in rows:
        row["prepared_structure_uuid"] = clean(row.get("structure_uuid"))
        pk = integer(row.get("gw_pk"))
        hit = crosswalk.get(pk, {}) if pk is not None else {}
        for key in ("mc2d_id", "band_gap", "magnetic_state", "unstable", "band_gap_match", "mc2d_candidates"):
            if not clean(row.get(key)):
                row[key] = hit.get(key, "")
        if not clean(row.get("mc2d_bg")):
            row["mc2d_bg"] = clean(row.get("band_gap")) or clean(row.get("bandgap_mc2d_overview_eV"))
        if hit.get("structure_uuid"):
            row["structure_uuid"] = hit["structure_uuid"]
        elif not clean(row.get("structure_uuid")):
            row["structure_uuid"] = ""
        row["mc2d_bg_eV"] = clean(row.get("mc2d_bg"))
        row["gw_pk_i"] = pk
        row["formula_norm"] = normalize_formula(row.get("formula"))
        row["failure_reason"] = clean(row.get("failure_reason"))
        row["physics_issues"] = clean(row.get("physics_issues"))
        row["status"] = clean(row.get("status")).lower()
        row["group"] = clean(row.get("group"))
    return rows


def report_key(row):
    for key in ("structure_uuid", "mc2d_id"):
        value = clean(row.get(key))
        if value:
            return f"{key}:{value}"
    formula = clean(row.get("formula_norm"))
    if formula:
        return f"formula:{formula}"
    pk = integer(row.get("gw_pk"))
    return f"pk:{pk}" if pk is not None else "unknown"


def is_open_shell(row):
    value = clean(row.get("magnetic_state")).lower()
    if not value:
        return False
    if value in {"nm", "nonmagnetic", "non-magnetic", "none", "false", "0", "0.0"}:
        return False
    if value in {"fm", "afm", "ferromagnetic", "antiferromagnetic", "magnetic", "spin", "spin-polarized", "true"}:
        return True
    try:
        return float(value) > 0
    except ValueError:
        return True


def gap_problem(row, args):
    problems = []
    for field in GW_FIELDS:
        value = number(row.get(field))
        if value is None:
            continue
        if value < 0:
            problems.append(f"{field}={value:.3f}")
        elif value <= 1e-3:
            problems.append(f"{field.replace('_eV', '')}_zero={value:.3f}")
    hf = number(row.get("hf_gap_direct_eV"))
    if hf is not None and hf < args.max_hf_gap:
        problems.append(f"hf_gap_direct={hf:.3f}")
    return problems


def failure_bucket(row):
    text = f"{row.get('status', '')} {row.get('failure_reason', '')} {row.get('physics_issues', '')}".upper()
    if "OPEN_SHELL" in text:
        return "open_shell"
    if "LOW_MC2D_BG" in text:
        return "low_mc2d_bg"
    if "SCF_NOT_CONVERGED" in text or "450" in text:
        return "scf_not_converged"
    if "TIME_LIMIT" in text or "TIMEOUT" in text or "SIGALRM" in text or "3584" in text:
        return "time_limit"
    if "OOM" in text or "SIGKILL" in text or "35072" in text:
        return "oom"
    if "NO_LINK" in text or "EXIT_300" in text or "ERROR_OUTPUT" in text:
        return "missing_output"
    if "402" in text:
        return "base_retries_exhausted"
    if "NEG_GAP" in text or "NEGATIVE" in text:
        return "negative_gap"
    if "RUNNING" in text:
        return "running"
    if "INCOMPLETE" in text:
        return "incomplete_output"
    return "other"


def classify(row, args):
    key = report_key(row)
    row["report_key"] = key
    bg = number(row.get("mc2d_bg"))
    if is_open_shell(row):
        return "excluded_open_shell", "open_shell_magnetic_state"
    if bg is None:
        if not args.allow_unknown_mc2d_bg:
            return "excluded_low_mc2d_bg", "unknown_mc2d_bg"
    elif bg <= args.min_mc2d_bg:
        return "excluded_low_mc2d_bg", f"mc2d_bg={bg:.3f}"
    problems = gap_problem(row, args)
    if problems:
        return "negative_gw", ";".join(problems)
    if row["status"] == "running":
        return "running", "running"
    if row["status"] != "ok":
        return "failed", failure_bucket(row)
    issues = clean(row.get("physics_issues"))
    if issues:
        return "failed", failure_bucket(row)
    failure = clean(row.get("failure_reason"))
    if failure:
        return "failed", failure_bucket(row)
    missing = [field for field in GAP_FIELDS if number(row.get(field)) is None]
    if missing:
        return "incomplete", "missing:" + ",".join(missing)
    bad_scf = [field for field in SCF_FIELDS if number(row.get(field)) <= 0]
    if bad_scf:
        return "negative_gw", ";".join(f"{field}={number(row.get(field)):.3f}" for field in bad_scf)
    return "report", "ok"


def quality(row):
    score = 0
    if row.get("status") == "ok":
        score += 10
    if not clean(row.get("physics_issues")):
        score += 4
    if not clean(row.get("failure_reason")):
        score += 4
    for field in GAP_FIELDS:
        value = number(row.get(field))
        if value is not None and value > 0:
            score += 1
    return score


def choose(rows):
    best = {}
    for row in rows:
        key = row["report_key"]
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        candidate = (quality(row), integer(row.get("gw_pk")) or 0)
        incumbent = (quality(current), integer(current.get("gw_pk")) or 0)
        if candidate > incumbent:
            best[key] = row
    return list(best.values())


def output_row(row, decision, reason, args):
    out = {field: clean(row.get(field)) for field in OUT_FIELDS}
    out.update(
        {
            "decision": decision,
            "reason": reason,
            "report_key": row["report_key"],
            "gw_pk": clean(row.get("gw_pk")),
            "calc_pk": clean(row.get("calc_pk")),
            "mc2d_bg_eV": clean(row.get("mc2d_bg")),
            "failure_reason": clean(row.get("failure_reason")),
            "physics_issues": clean(row.get("physics_issues")),
        }
    )
    if decision == "negative_gw":
        out.update(
            {
                "recommended_eps_filter": "1e-8",
                "recommended_num_time_freq": "12",
                "recommended_kpoints_distance": "0.05",
                "run_group": args.resubmit_group,
            }
        )
    return out


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(outdir, report_rows, negative_rows, classified):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: plotting skipped: {exc}", file=sys.stderr)
        return

    decisions = Counter(row["decision"] for row in classified)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = sorted(decisions)
    ax.bar(names, [decisions[name] for name in names], color="#4472c4")
    ax.set_ylabel("Runs")
    ax.set_title("GW run classification")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(outdir / "classification.png", dpi=200)
    plt.close(fig)

    points = [(row, "report") for row in report_rows] + [(row, "negative") for row in negative_rows]
    usable = []
    for row, group in points:
        x = number(row.get("mc2d_bg"))
        y = number(row.get("g0w0_gap_indirect_eV"))
        if x is not None and y is not None:
            usable.append((x, y, group, clean(row.get("formula"))))

    if usable:
        fig, ax = plt.subplots(figsize=(7, 7))
        for group, color in (("report", "#2ca02c"), ("negative", "#d62728")):
            subset = [item for item in usable if item[2] == group]
            if subset:
                ax.scatter([item[0] for item in subset], [item[1] for item in subset], s=28, label=group, color=color, alpha=0.8)
        low = min(min(item[0] for item in usable), min(item[1] for item in usable))
        high = max(max(item[0] for item in usable), max(item[1] for item in usable))
        ax.plot([low, high], [low, high], "--", color="gray", linewidth=1)
        ax.set_xlabel("MC2D PBE band gap (eV)")
        ax.set_ylabel("G0W0 indirect gap (eV)")
        ax.set_title("GW vs MC2D PBE gap")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / "gw_vs_mc2d.png", dpi=200)
        plt.close(fig)

        soc = []
        for row in report_rows + negative_rows:
            x = number(row.get("g0w0_gap_indirect_eV"))
            y = number(row.get("g0w0_soc_gap_indirect_eV"))
            if x is not None and y is not None:
                soc.append((x, y))
        if soc:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter([item[0] for item in soc], [item[1] for item in soc], s=25, alpha=0.8)
            low = min(min(item[0] for item in soc), min(item[1] for item in soc))
            high = max(max(item[0] for item in soc), max(item[1] for item in soc))
            ax.plot([low, high], [low, high], "--", color="gray", linewidth=1)
            ax.set_xlabel("G0W0 indirect gap (eV)")
            ax.set_ylabel("G0W0+SOC indirect gap (eV)")
            ax.set_title("SOC vs non-SOC G0W0 gap")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(outdir / "gw_soc_vs_gw.png", dpi=200)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        values = [number(row.get("g0w0_gap_indirect_eV")) for row in report_rows]
        values = [value for value in values if value is not None]
        if values:
            ax.hist(values, bins=30, color="#4472c4", alpha=0.8, label="G0W0 indirect")
        soc_values = [number(row.get("g0w0_soc_gap_indirect_eV")) for row in report_rows]
        soc_values = [value for value in soc_values if value is not None]
        if soc_values:
            ax.hist(soc_values, bins=30, color="#ed7d31", alpha=0.6, label="G0W0+SOC indirect")
        ax.set_xlabel("Band gap (eV)")
        ax.set_ylabel("Structures")
        ax.set_title("Reported GW gap distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "gap_distribution.png", dpi=200)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="all_results", help="CSV from make_gw_mc2d_table.py or fixed-width all_results text")
    parser.add_argument("--crosswalk", help="CSV report from annotate_mc2d_bandgaps.py --nodes")
    parser.add_argument("--outdir", default="gw_report_2el", help="Output directory")
    parser.add_argument("--min-mc2d-bg", type=float, default=0.010, help="Require MC2D PBE gap > VALUE")
    parser.add_argument("--allow-unknown-mc2d-bg", action="store_true", help="Keep rows with unknown MC2D gap")
    parser.add_argument("--exclude-unknown-magnetic", action="store_true", help="Exclude rows with unknown magnetic state")
    parser.add_argument("--max-hf-gap", type=float, default=0.0, help="Flag HF gap below VALUE as numerical failure")
    parser.add_argument("--resubmit-group", default="gw_runs_2el_neg_gw_r3", help="Group name in generated resubmit commands")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting")
    args = parser.parse_args(argv)

    raw_rows = load_rows(args.input)
    rows = augment_rows(raw_rows, load_crosswalk(args.crosswalk))

    classified = []
    for row in rows:
        decision, reason = classify(row, args)
        if args.exclude_unknown_magnetic and decision == "report" and not clean(row.get("magnetic_state")):
            decision, reason = "excluded_unknown_magnetic", "unknown_magnetic_state"
        row["decision"] = decision
        row["reason"] = reason
        classified.append(output_row(row, decision, reason, args))

    report_rows = choose([row for row in rows if row["decision"] == "report"])
    report_keys = {row["report_key"] for row in report_rows}
    report_rows = [output_row(row, "report", "ok", args) for row in report_rows]

    negative_candidates = [row for row in rows if row["decision"] == "negative_gw" and row["report_key"] not in report_keys]
    negative_rows = choose(negative_candidates)
    negative_keys = {row["report_key"] for row in negative_rows}
    negative_rows = [output_row(row, "negative_gw", row["reason"], args) for row in negative_rows]

    other_rows = []
    for row in rows:
        if row["report_key"] in report_keys or row["report_key"] in negative_keys:
            continue
        if row["decision"] in {"report", "negative_gw"}:
            continue
        other_rows.append(output_row(row, row["decision"], row["reason"], args))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "report.csv", report_rows)
    write_csv(outdir / "negative_gw_resubmit.csv", negative_rows)
    write_csv(outdir / "failed_or_excluded.csv", other_rows)
    write_csv(outdir / "all_classified.csv", sorted(classified, key=lambda row: integer(row.get("gw_pk")) or 0))

    decisions = Counter(row["decision"] for row in rows)
    buckets = Counter(failure_bucket(row) for row in rows if row["decision"] in {"failed", "incomplete", "running"})
    summary = {
        "input_rows": len(rows),
        "unique_report_structures": len(report_rows),
        "unique_negative_gw_structures": len(negative_rows),
        "other_rows": len(other_rows),
        "decisions": dict(decisions),
        "failure_buckets": dict(buckets),
        "min_mc2d_bg": args.min_mc2d_bg,
        "unknown_mc2d_bg_allowed": args.allow_unknown_mc2d_bg,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if negative_rows:
        pks = " ".join(str(integer(row.get("gw_pk"))) for row in negative_rows if integer(row.get("gw_pk")))
        base = f"python utils/resubmit_neg_gap_kdenser.py --pks {pks} --config config.toml --run-group {args.resubmit_group}"
        commands = [base + " --dry-run", base]
        (outdir / "resubmit_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    if not args.no_plots:
        make_plots(outdir, report_rows, negative_rows, rows)

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Classify GwWorkChains that finished with exit 300.

For each GwWorkChain in a group with exit_status 300, find its final
Cp2kCalculation and report why it did not finish-ok:

  RAN-OK / RECOVERABLE : scheduler exit code 0 -> only the parser crashed
                         (typically the nan/inf serialization bug). After the
                         parser fix, re-run with: verdi calcjob res <pk>.
  CP2K FAILED          : scheduler exit code != 0 -> genuine failure
                         (SCF not converged, metallic/gap 0, GW/SOC crash).
                         Inspect aiida.out for the real error.

Run inside the aiida environment on casusvm:

  python3 utils/classify_300s.py [--group gw_runs_2el2] [--pks 2092 2304]
"""
import argparse
import re
import sys

from aiida import load_profile, orm
from aiida.orm import QueryBuilder


def get_retrieved(calc):
    for out in calc.get_outgoing().all():
        if out.link_label == "retrieved":
            return out.node
    return None


def get_scheduler_exit_code(calc):
    retrieved = get_retrieved(calc)
    if retrieved is None:
        return None
    try:
        content = retrieved.base.repository.get_object_content("_scheduler-stdout.txt")
    except Exception:
        return None
    match = re.search(r"Exit code:\s*(\d+)", content)
    return int(match.group(1)) if match else None


def has_nan_error(calc):
    try:
        logs = list(calc.log_messages)
    except Exception:
        logs = []
    return any("nan and inf/-inf" in (log.message or "") for log in logs)


def has_output(calc, label):
    return any(out.link_label == label for out in calc.get_outgoing().all())


def find_calcjobs(gw):
    calcjobs = []
    for called in gw.called:
        process_type = called.process_type or ""
        if process_type.endswith("Cp2kCalculation"):
            calcjobs.append(called)
    return sorted(calcjobs, key=lambda c: c.ctime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="gw_runs_2el2")
    parser.add_argument("--pks", nargs="*", type=int, default=None)
    args = parser.parse_args()

    load_profile()

    chains = []
    if args.pks:
        for pk in args.pks:
            node = orm.load_node(pk)
            if node.process_type and node.process_type.endswith("GwWorkChain"):
                chains.append(node)
    else:
        group = orm.load_group(args.group)
        for node in group.nodes:
            if (node.process_type or "").endswith("GwWorkChain"):
                chains.append(node)

    header = (f"{'gw':>7} {'calc':>7} {'calc_state':<22} {'job_exit':>8} "
              f"{'nan?':>4} {'parms':>5}  class")
    print(header)
    print("-" * len(header))

    for gw in sorted(chains, key=lambda n: n.pk):
        if gw.exit_status != 300:
            continue
        calcjobs = find_calcjobs(gw)
        if not calcjobs:
            print(f"{gw.pk:>7} {'-':>7} {'-':<22} {'-':>8} {'-':>4} {'-':>5}  NO CALCJOB")
            continue
        calc = calcjobs[-1]
        state = "excepted" if calc.is_excepted else ("finished" if calc.is_finished_ok else calc.process_state)
        job_exit = get_scheduler_exit_code(calc)
        nan_err = has_nan_error(calc)
        has_params = has_output(calc, "output_parameters")

        if job_exit == 0:
            cls = "RECOVERABLE" if (nan_err or not has_params) else "RAN-OK (no params?)"
        elif job_exit is None:
            cls = "UNKNOWN (no scheduler stdout)"
        else:
            cls = f"CP2K FAILED (exit {job_exit})"

        print(f"{gw.pk:>7} {calc.pk:>7} {state:<22} {str(job_exit):>8} "
              f"{'Y' if nan_err else 'n':>4} {'Y' if has_params else 'n':>5}  {cls}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
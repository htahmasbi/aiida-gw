"""Classify GwWorkChains that finished with exit 300.

For each GwWorkChain in a group with exit_status 300, find its final
Cp2kCalculation and report why it did not finish-ok:

  DONE-OK / RECOVERABLE : scheduler exit code 0 -> only the parser crashed
                         (typically the nan/inf serialization bug). After the
                         parser fix, re-run with: verdi calcjob res <pk>.
  CP2K FAILED           : scheduler exit code != 0 -> genuine failure
                         (SCF not converged, metallic/gap 0, GW/SOC crash).
                         Inspect aiida.out for the real error.
  TIMEOUT / TRUNCATED   : no output_parameters although the wrapper reported
                         exit 0. SLURM keeps that 0 when it kills a job at the
                         walltime limit, so the kill is detected from the
                         scheduler job state or from a truncated aiida.out.
                         Raise max_wallclock_seconds and re-run.

The scheduler wrapper exit code alone cannot distinguish a parser crash from a
walltime kill, so both are cross-checked against the retrieved files.

Run inside the aiida environment on casusvm:

  python3 utils/classify_300s.py [--group gw_runs_2el2] [--pks 2092 2304]
"""
import argparse
import os
import re
import sys

from aiida import load_profile, orm
from aiida.orm import QueryBuilder

# Markers written by CP2K at the very end of a cleanly finished run.
CP2K_END_MARKERS = ("Program ended normally", "ABORT")

# Keywords the SLURM scheduler uses for a job that was cut short.
TIMEOUT_KEYWORDS = ("TIMEOUT", "TIMED_OUT", "CANCELLED", "CANCELED", "NODE_FAIL", "PREEMPTED")

# Substrings identifying a parser/serialization failure in the log messages.
PARSER_ERROR_NEEDLES = (
    "nan and inf",
    "can not be serialized",
    "validationerror",
    "no parsers exit successfully",
    "parsererror",
)

TAIL_BYTES = 65536


def _as_text(content):
    if content is None:
        return None
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return content


def get_retrieved(calc):
    for out in calc.base.links.get_outgoing():
        if out.link_label == "retrieved":
            return out.node
    return None


def _repo_text(retrieved, name):
    if retrieved is None:
        return None
    try:
        return _as_text(retrieved.base.repository.get_object_content(name))
    except Exception:
        return None


def _repo_tail(retrieved, name, nbytes=TAIL_BYTES):
    """Read the last ``nbytes`` of a retrieved file, or None if unreadable."""
    if retrieved is None:
        return None
    try:
        with retrieved.base.repository.open(name, mode="rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - nbytes), os.SEEK_SET)
            data = handle.read()
    except Exception:
        return None
    return _as_text(data)


def get_scheduler_exit_code(calc):
    retrieved = get_retrieved(calc)
    match = re.search(r"Exit code:\s*(\d+)", _repo_text(retrieved, "_scheduler-stdout.txt") or "")
    return int(match.group(1)) if match else None


def get_scheduler_text(calc):
    """Combined stdout+stderr of the scheduler wrapper, as text."""
    retrieved = get_retrieved(calc)
    parts = [
        _repo_text(retrieved, "_scheduler-stdout.txt"),
        _repo_text(retrieved, "_scheduler-stderr.txt"),
    ]
    return "\n".join(part for part in parts if part)


def scheduler_timed_out(calc):
    text = get_scheduler_text(calc).upper()
    return any(keyword in text for keyword in TIMEOUT_KEYWORDS)


def aiida_out_ended_normally(calc):
    """True/False if aiida.out was readable, None if it was not retrievable.

    False means the CP2K output stops mid-run, which is what a scheduler kill
    looks like on disk: the job died before printing its termination marker.
    """
    tail = _repo_tail(get_retrieved(calc), "aiida.out")
    if tail is None:
        return None
    return any(marker in tail for marker in CP2K_END_MARKERS)


def has_parser_error(calc):
    try:
        logs = list(calc.log_messages)
    except Exception:
        return False
    for log in logs:
        message = (log.message or "").lower()
        if any(needle in message for needle in PARSER_ERROR_NEEDLES):
            return True
    return False


def has_output(calc, label):
    return any(out.link_label == label for out in calc.base.links.get_outgoing())


def params_state(calc):
    """Report the output_parameters link as 'Y', 'empty', 'n' or 'err'.

    Distinguishing an empty Dict from a readable one matters: a run whose
    results are present but unreadable must not be conflated with a run that
    never stored anything.
    """
    if not has_output(calc, "output_parameters"):
        return "n"
    try:
        node = next(
            out.node for out in calc.base.links.get_outgoing() if out.link_label == "output_parameters"
        )
        return "Y" if node.get_dict() else "empty"
    except Exception:
        return "err"


def find_calcjobs(gw):
    return sorted(
        (n for n in gw.called_descendants if n.process_label == "Cp2kCalculation"),
        key=lambda c: c.ctime,
    )


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
            if node.process_label == "GwWorkChain":
                chains.append(node)
    else:
        group = orm.load_group(args.group)
        for node in group.nodes:
            if node.process_label == "GwWorkChain":
                chains.append(node)

    header = (f"{'gw':>7} {'calc':>7} {'calc_state':<18} {'job_exit':>8} "
              f"{'err?':>4} {'parms':>5} {'cp2k_end':>8}  class")
    print(header)
    print("-" * len(header))

    for gw in sorted(chains, key=lambda n: n.pk):
        if gw.exit_status != 300:
            continue
        calcjobs = find_calcjobs(gw)
        if not calcjobs:
            print(f"{gw.pk:>7} {'-':>7} {'-':<18} {'-':>8} {'-':>4} {'-':>5} {'-':>8}  NO CALCJOB")
            continue
        calc = calcjobs[-1]
        if calc.is_excepted:
            state = "excepted"
        elif calc.is_finished_ok:
            state = "finished"
        else:
            state = str(getattr(calc.process_state, "value", calc.process_state))
        job_exit = get_scheduler_exit_code(calc)
        parser_err = has_parser_error(calc)
        parms = params_state(calc)
        ended = aiida_out_ended_normally(calc)
        timed_out = scheduler_timed_out(calc)

        if parms == "Y":
            cls = "DONE-OK"
        elif timed_out:
            cls = "TIMEOUT (raise max_wallclock_seconds)"
        elif ended is False:
            cls = "TRUNCATED (walltime kill? check sacct Elapsed)"
        elif job_exit == 0 and parser_err:
            cls = "RECOVERABLE (parser crash)"
        elif job_exit == 0:
            cls = "RECOVERABLE (no params?)"
        elif job_exit is None:
            cls = "UNKNOWN (no scheduler stdout)"
        else:
            cls = f"CP2K FAILED (exit {job_exit})"

        print(f"{gw.pk:>7} {calc.pk:>7} {state:<18} {str(job_exit):>8} "
              f"{'Y' if parser_err else 'n':>4} {parms:>5} {str(ended):>8}  {cls}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
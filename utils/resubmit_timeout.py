#!/usr/bin/env python3
"""Resubmit GwWorkChains killed by walltime/time limit."""

import argparse
import logging
import sys

from aiida.manage.configuration import load_profile
from aiida import orm
from aiida.engine import submit

load_profile()

LOGGER = logging.getLogger("resubmit_timeout")
WORKCHAIN_LABELS = {"GwWorkChain"}
TIMEOUT_KEYWORDS = ("TIMEOUT", "TIMED_OUT", "DUE TO TIME", "WALLTIME", "WALL TIME", "TIME LIMIT", "CANCELLED BY WALLTIME", "EXCEEDED")


def _last_calcjob(node):
    calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
    if not calcs:
        return None
    return max(calcs, key=lambda c: c.ctime)


def _get_retrieved(calc):
    try:
        return [out.node for out in calc.base.links.get_outgoing() if out.link_label == "retrieved"]
    except Exception:
        return []


def _is_timeout(calc):
    if calc is None:
        return False
    # Check all retrieved folders
    for retrieved in _get_retrieved(calc):
        # Check scheduler files
        for name in ("_scheduler-stdout.txt", "_scheduler-stderr.txt", "_scheduler-stderr", "_scheduler-stdout"):
            try:
                content = retrieved.base.repository.get_object_content(name)
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                if content:
                    c = content.upper()
                    if any(k in c for k in TIMEOUT_KEYWORDS):
                        # If it says SCF not converged and also timeout, prefer timeout if timeout is explicit
                        if "SCF" in c and ("NOT CONVERGED" in c) and "TIME LIMIT" not in c and "WALLTIME" not in c and "TIMED_OUT" not in c:
                            continue
                        return True
            except Exception:
                continue
        # Check aiida.out
        try:
            content = retrieved.base.repository.get_object_content("aiida.out")
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if content:
                c = content.upper()
                # Look for timeout anywhere
                if any(k in c for k in ["TIME LIMIT", "WALLTIME", "TIMED OUT", "DUE TO TIME", "EXCEEDED"]):
                    # Make sure it's not just in some other context
                    if "SCF RUN NOT CONVERGED" in c and "TIME" not in c:
                        pass  # might be separate
                    # More specific check - look in last part
                    lines = content.splitlines()
                    # Check last 50 lines
                    for l in lines[-50:]:
                        lu = l.upper()
                        if any(k in lu for k in ["TIME LIMIT", "WALLTIME", "TIMED OUT", "DUE TO TIME", "EXCEEDED"]):
                            return True
                    # Also check whole content for explicit timeout
                    if "SLURM" in c and ("TIMEOUT" in c or "CANCELLED" in c and "TIME" in c):
                        return True
        except Exception:
            continue
        # Check slurm files
        for name in [n for n in []]:
            pass
    return False

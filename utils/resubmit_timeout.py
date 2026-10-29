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
TIMEOUT_KEYWORDS = ("TIMEOUT", "TIMED_OUT", "DUE TO TIME", "WALLTIME", "WALL TIME", "TIME LIMIT", "CANCELLED BY WALLTIME", "EXCEEDED", "SLURM.*TIME")


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
    for retrieved in _get_retrieved(calc):
        # Check all files in retrieved
        try:
            for filename in retrieved.list_object_names():
                if filename.startswith('_scheduler') or filename.endswith('.out') or filename == 'aiida.out':
                    try:
                        content = retrieved.base.repository.get_object_content(filename)
                        if isinstance(content, bytes):
                            content = content.decode("utf-8", errors="replace")
                        if content:
                            c = content.upper()
                            # Look for explicit timeout indicators
                            if any(x in c for x in ["TIME LIMIT", "WALLTIME", "TIMED OUT", "DUE TO TIME", "EXCEEDED TIME", "WALL TIME LIMIT"]):
                                return True
                            # SLURM timeout patterns
                            if "SLURM" in c and ("TIMEOUT" in c or ("CANCELLED" in c and "TIME" in c)):
                                return True
                            # Check last 30 lines more specifically
                            lines = content.splitlines()
                            for l in lines[-30:]:
                                lu = l.upper()
                                if any(x in lu for x in ["TIME LIMIT", "WALLTIME", "TIMED OUT", "DUE TO TIME"]):
                                    return True
                    except Exception:
                        continue
        except Exception:
            # Fallback to specific names
            for name in ("_scheduler-stderr.txt", "_scheduler-stdout.txt", "_scheduler-stderr", "_scheduler-stdout", "aiida.out"):
                try:
                    content = retrieved.base.repository.get_object_content(name)
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="replace")
                    if content:
                        c = content.upper()
                        if any(x in c for x in ["TIME LIMIT", "WALLTIME", "TIMED OUT", "DUE TO TIME"]):
                            return True
                        lines = content.splitlines()
                        for l in lines[-20:]:
                            lu = l.upper()
                            if any(x in lu for x in ["TIME LIMIT", "WALLTIME", "TIMED OUT", "DUE TO TIME"]):
                                return True
                except Exception:
                    continue
    return False

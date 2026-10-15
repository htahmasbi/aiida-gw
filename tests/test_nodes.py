"""Tests for output link resolution that must survive duplicate labels.

A job that was double executed leaves two ``CREATE`` links with the same label
behind, which makes ``node.outputs`` and ``get_outgoing().nested()`` raise
``KeyError``. ``get_output_node`` reads the link triples directly and must keep
working in that situation.
"""

import pytest

pytest.importorskip("aiida")

from unittest.mock import MagicMock  # noqa: E402

from aiida.common.links import LinkType  # noqa: E402

from aiida_gw.core.nodes import get_output_node  # noqa: E402


class _Triple:
    def __init__(self, node, link_label, link_type=LinkType.CREATE):
        self.node = node
        self.link_label = link_label
        self.link_type = link_type


def _make_node(triples):
    node = MagicMock()
    node.base.links.get_outgoing.return_value = triples
    return node


class TestGetOutputNode:
    def test_returns_matching_node(self):
        target = object()
        node = _make_node([_Triple(target, "output_structure")])
        assert get_output_node(node, "output_structure") is target

    def test_missing_label_returns_none(self):
        node = _make_node([_Triple(object(), "output_structure")])
        assert get_output_node(node, "output_parameters") is None

    def test_tolerates_duplicate_label(self):
        first, second = object(), object()
        node = _make_node(
            [
                _Triple(first, "retrieved"),
                _Triple(second, "retrieved"),
                _Triple(object(), "output_parameters"),
            ]
        )
        assert get_output_node(node, "retrieved") is first

    def test_requests_only_output_link_types(self):
        node = _make_node([])
        get_output_node(node, "output_parameters")
        _, kwargs = node.base.links.get_outgoing.call_args
        assert set(kwargs["link_type"]) == {LinkType.CREATE, LinkType.RETURN}

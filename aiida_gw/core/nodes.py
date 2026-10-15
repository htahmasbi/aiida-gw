"""Helpers for reading process outputs in a way that survives duplicate links."""
from aiida.common.links import LinkType


def get_output_node(node, label, link_types=(LinkType.CREATE, LinkType.RETURN)):
    """Return the node that ``node`` exposes as output ``label``, or ``None``.

    This deliberately avoids both ``node.outputs.<label>`` and
    ``ProcessNode.base.links.get_outgoing(...).nested()``: the latter rebuilds the
    whole output namespace and raises ``KeyError: duplicate label ...`` as soon as
    *any* output label is duplicated. A duplicated label is exactly what a job that
    was double executed leaves behind (two ``CREATE`` links for ``retrieved`` or
    ``output_parameters``), and that exception used to escape ``hasattr`` and take
    down the parent workchain. Iterating the link triples only ever inspects the
    requested label, so a duplicate elsewhere is harmless and a duplicate of this
    one is tolerated by returning the first match.

    :param node: a ``ProcessNode``
    :param label: output link label to look up
    :param link_types: link types that should be considered outputs
    :returns: the linked node, or ``None`` when there is no such output
    """
    for link in node.base.links.get_outgoing(link_type=link_types):
        if link.link_label == label:
            return link.node
    return None

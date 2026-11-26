"""Compressed byte trie for already validated token tapes."""
from __future__ import annotations


class _Node:
    def __init__(self, edge=b"", value=None):
        self.edge = edge
        self.value = value
        self.children = {}


class PrefixRadix:
    def __init__(self):
        self.root = _Node()

    def put(self, tape, value):
        node = self.root
        tail = tape
        while tail:
            first = tail[0]
            child = node.children.get(first)
            if child is None:
                node.children[first] = _Node(tail, value)
                return
            common = min(len(child.edge), len(tail))
            if not tail.startswith(child.edge):
                # Compare prefixes in C and bisect the first differing byte.
                low, high = 1, common
                while low < high:
                    middle = (low + high + 1) // 2
                    if tail.startswith(child.edge[:middle]):
                        low = middle
                    else:
                        high = middle - 1
                split = _Node(child.edge[:low])
                child.edge = child.edge[low:]
                split.children[child.edge[0]] = child
                node.children[first] = child = split
                common = low
            node = child
            tail = tail[common:]
        node.value = value

    def remove(self, tape):
        path = []
        node = self.root
        pos = 0
        while pos < len(tape):
            child = node.children.get(tape[pos])
            if child is None or not tape.startswith(child.edge, pos):
                return
            path.append((node, child))
            pos += len(child.edge)
            node = child
        node.value = None
        for parent, child in reversed(path):
            if child.value is not None or len(child.children) > 1:
                break
            first = child.edge[0]
            if not child.children:
                del parent.children[first]
            else:
                leaf = next(iter(child.children.values()))
                leaf.edge = child.edge + leaf.edge
                parent.children[first] = leaf

    def matches(self, tape):
        """Matching stored values in ascending byte length."""
        node, pos = self.root, 0
        while pos < len(tape):
            child = node.children.get(tape[pos])
            if child is None or not tape.startswith(child.edge, pos):
                return
            pos += len(child.edge)
            node = child
            if node.value is not None:
                yield pos, node.value

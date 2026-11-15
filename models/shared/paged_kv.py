"""Paged KV block table with shared pages and sink/recent pinning (blueprint LCT-01/LCT-07).

A conversation's KV grows in fixed-size pages instead of one contiguous tensor.
A logical page maps to a physical page through the block table, so a fork (a
retried tool call, a second agent session on the same repository prefix) shares
the physical pages read-only and copies only the page it appends to. Sink and
recent pages can be pinned in RAM, which is the layout a later SSD cold-tier
(LCT-03) and the QSA sparse fetch (LCT-04) need.

This module owns storage only; it does not compute attention. ``gather``
returns the visible K/V so a test or a fallback path can run dense SDPA. The
arrays are opaque to it (MLX or NumPy), and the token axis is ``-2`` for KV
shaped ``[batch, heads, tokens, dim]``.
"""

from __future__ import annotations


def _slice_axis(axis: int, start: int, stop: int, ndim: int):
    return tuple(slice(start, stop) if index == axis % ndim else slice(None) for index in range(ndim))


class _Store:
    """Physical pages and refcounts shared by every fork of one cache."""

    def __init__(self):
        self.pages: dict[int, dict] = {}
        self.refs: dict[int, int] = {}
        self.free: list[int] = []
        self.pins: set[int] = set()
        self.epoch = 0
        self.next_id = 0

    def allocate(self) -> int:
        if self.free:
            return self.free.pop()
        page = self.next_id
        self.next_id += 1
        return page

    def release(self, page: int) -> None:
        self.refs[page] = self.refs.get(page, 1) - 1
        if self.refs[page] <= 0:
            self.refs.pop(page, None)
            self.pages.pop(page, None)
            self.pins.discard(page)
            self.free.append(page)


class PagedKV:
    def __init__(self, page_size: int = 64, store: "_Store | None" = None):
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page size must be a positive integer")
        self.page_size = page_size
        self._store = store or _Store()
        self._table: list[int] = []
        self._offset = 0
        self._last_used: dict[int, int] = {}
        self.sink_pages = 0
        self.recent_pages = 0
        self.cow_pages = 0

    # -- shape -------------------------------------------------------------------------------------------------
    @property
    def tokens(self) -> int:
        return (len(self._table) - 1) * self.page_size + self._offset if self._table else 0

    @property
    def pages(self) -> int:
        return len(self._table)

    def table(self) -> tuple[int, ...]:
        return tuple(self._table)

    def stats(self) -> dict:
        return {"tokens": self.tokens, "pages": self.pages, "page_size": self.page_size,
                "physical_pages": len(self._store.pages), "shared_pages": sum(
                    1 for page in self._table if self._store.refs.get(page, 0) > 1),
                "cow_pages": self.cow_pages, "sink_pages": self.sink_pages,
                "recent_pages": self.recent_pages, "pinned_pages": len(self._store.pins)}

    # -- write -------------------------------------------------------------------------------------------------
    def _alloc_page(self, keys, values) -> int:
        page = self._store.allocate()
        shape = list(keys.shape)
        shape[-2] = self.page_size
        self._store.pages[page] = {
            "keys": _zeros_like(keys, shape),
            "values": _zeros_like(values, shape),
            "length": 0,
        }
        self._store.refs[page] = 0
        return page

    def _cow(self, logical: int) -> int:
        """Replace a shared physical page with a private copy of its valid bytes."""
        old = self._table[logical]
        if self._store.refs.get(old, 0) <= 1:
            return old
        new = self._alloc_page(self._store.pages[old]["keys"], self._store.pages[old]["values"])
        source, target = self._store.pages[old], self._store.pages[new]
        valid = source["length"]
        target["keys"] = _copy_into(target["keys"], source["keys"], valid)
        target["values"] = _copy_into(target["values"], source["values"], valid)
        target["length"] = valid
        self._store.refs[old] -= 1
        self._store.refs[new] = 1
        self._table[logical] = new
        self.cow_pages += 1
        return new

    def append(self, keys, values) -> "PagedKV":
        """Append ``keys``/``values`` (time on axis ``-2``) into pages, copying pages that are shared."""
        length = keys.shape[-2]
        if values.shape[-2] != length:
            raise ValueError("keys and values must have the same token count")
        consumed = 0
        while consumed < length:
            if not self._table or self._offset == self.page_size:
                page = self._alloc_page(keys, values)
                self._store.refs[page] = 1
                self._table.append(page)
                self._offset = 0
            page = self._cow(len(self._table) - 1)
            take = min(length - consumed, self.page_size - self._offset)
            entry = self._store.pages[page]
            source = _slice_axis(-2, consumed, consumed + take, keys.ndim)
            target = _slice_axis(-2, self._offset, self._offset + take, keys.ndim)
            _assign(entry["keys"], target, keys[source])
            _assign(entry["values"], target, values[source])
            entry["length"] = self._offset + take
            self._offset += take
            consumed += take
        self._touch()
        return self

    # -- read --------------------------------------------------------------------------------------------------
    def _touch(self) -> None:
        self._store.epoch += 1
        for logical in range(max(0, len(self._table) - 2), len(self._table)):
            self._last_used[self._table[logical]] = self._store.epoch

    def gather(self):
        """The visible K/V concatenated in logical order (a test and fallback path, not the hot path)."""
        if not self._table:
            return None, None
        if any(page < 0 for page in self._table):
            raise ValueError("spilled pages must be restored before a dense gather")
        keys = [_page_values(self._store.pages[page], "keys", self._visible(index))
                for index, page in enumerate(self._table)]
        values = [_page_values(self._store.pages[page], "values", self._visible(index))
                  for index, page in enumerate(self._table)]
        return _concat(keys, -2), _concat(values, -2)

    def _visible(self, logical: int) -> int:
        return max(0, min(self.page_size, self.tokens - logical * self.page_size))

    def page_arrays(self, logical: int):
        page = self._store.pages[self._table[logical]]
        visible = self._visible(logical)
        return _page_values(page, "keys", visible), _page_values(page, "values", visible)

    # -- sharing -----------------------------------------------------------------------------------------------
    def fork(self, tokens: int | None = None) -> "PagedKV":
        """A new cache sharing complete pages up to ``tokens``; appends copy a shared page first."""
        limit = self.tokens if tokens is None else max(0, min(int(tokens), self.tokens))
        full, remainder = divmod(limit, self.page_size)
        child = PagedKV(self.page_size, self._store)
        count = full + (1 if remainder else 0)
        child._table = list(self._table[:count])
        child._offset = remainder if remainder else (self.page_size if count else 0)
        for page in child._table:
            self._store.refs[page] = self._store.refs.get(page, 0) + 1
        if remainder:
            child._cow(count - 1)
        child.sink_pages, child.recent_pages = self.sink_pages, self.recent_pages
        return child

    def pin_sink(self, pages: int) -> int:
        """Pin the first ``pages`` pages (attention sinks) so eviction cannot pick them."""
        self.sink_pages = max(0, min(int(pages), len(self._table)))
        return self._pin(range(self.sink_pages))

    def pin_recent(self, pages: int) -> int:
        """Pin the last ``pages`` pages (the active window) so eviction cannot pick them."""
        self.recent_pages = max(0, min(int(pages), len(self._table)))
        start = len(self._table) - self.recent_pages
        return self._pin(range(start, len(self._table)))

    def _pin(self, logical_pages) -> int:
        pinned = 0
        for logical in logical_pages:
            page = self._table[logical]
            if page not in self._store.pins:
                self._store.pins.add(page)
                pinned += 1
        return pinned

    # -- eviction (feed for an SSD cold tier) ------------------------------------------------------------------
    def victims(self) -> list[int]:
        """Unpinned physical pages, least recently used first, with refcounts in this table."""
        counts = {}
        for page in self._table:
            counts[page] = counts.get(page, 0) + 1
        candidates = [page for page in self._table if page not in self._store.pins]
        return sorted(candidates, key=lambda page: self._last_used.get(page, 0))

    def evict(self, logical: int) -> int:
        """Spill one complete page out of RAM; its slot stays in the table for ``restore``.

        The active (last) page is never evicted, because it holds the write cursor.
        """
        if not 0 <= logical < len(self._table) - 1:
            raise ValueError("only complete, non-active pages can be evicted")
        page = self._table[logical]
        if page < 0:
            return logical
        if page in self._store.pins:
            raise ValueError("pinned page cannot be evicted")
        self._table[logical] = -1
        self._store.release(page)
        self._last_used.pop(page, None)
        return logical

    def restore(self, logical: int, keys, values) -> int:
        """Put spilled bytes back into the slot that ``evict`` left behind."""
        if not 0 <= logical < len(self._table) or self._table[logical] >= 0:
            raise ValueError("restore needs a spilled page slot")
        page = self._alloc_page(keys, values)
        entry = self._store.pages[page]
        entry["keys"] = keys
        entry["values"] = values
        entry["length"] = keys.shape[-2]
        self._store.refs[page] = 1
        self._table[logical] = page
        self._last_used[page] = self._store.epoch
        return page


# -- array helpers: MLX and NumPy both accept these forms -------------------------------------------------------

def _zeros_like(template, shape):
    try:
        import mlx.core as mx
    except Exception:                     # pragma: no cover - MLX is a project dependency
        mx = None
    if mx is not None and isinstance(template, mx.array):
        return mx.zeros(shape, dtype=template.dtype)
    import numpy as np
    return np.zeros(shape, dtype=template.dtype)


def _assign(target, selection, source) -> None:
    target[selection] = source


def _copy_into(target, source, valid: int):
    selection = _slice_axis(-2, 0, valid, source.ndim)
    _assign(target, selection, source[selection])
    return target


def _page_values(page: dict, name: str, valid: int | None = None):
    count = page["length"] if valid is None else min(valid, page["length"])
    return page[name][_slice_axis(-2, 0, count, page[name].ndim)]


def _concat(arrays, axis: int):
    try:
        import mlx.core as mx
    except Exception:                     # pragma: no cover
        mx = None
    if mx is not None and isinstance(arrays[0], mx.array):
        return mx.concatenate(arrays, axis=axis)
    import numpy as np
    return np.concatenate(arrays, axis=axis)

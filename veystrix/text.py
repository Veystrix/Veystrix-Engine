"""Turning user text into tokens without letting it become chat structure."""
from __future__ import annotations

import re


# Include CJK / fullwidth punctuation so Chinese and Japanese text with no
# spaces still streams phrase by phrase instead of stalling to the end.
_TEXT_BOUNDARY = re.compile(
    r".*?(?:\s+|[.!?,;:]+|[。！？；：，、！？]+)", re.DOTALL
)
# Hard cap: flush a long run of unbroken text (e.g. CJK without punctuation)
# so streaming never stalls for a whole generation (Sonnet finding 13).
_MAX_PENDING = 16


def build_user_encoder(tokenizer):
    """Encode user text as content. Control markers stay literal text.

    A pasted transcript often carries <think>, </think> or <|im_end|>. The
    tokenizer turns those strings into control tokens, and the model then
    reads the paste as chat structure and drops most of it from the turn.

    Returns (pattern, encode). `pattern` finds markers so a caller can tell
    whether the slow path is needed; text without markers should be encoded
    whole, which keeps its tokenization byte-identical to the plain call.
    """
    markers = sorted(
        {token.content for token in tokenizer.added_tokens_decoder.values()},
        key=len,
        reverse=True,
    )
    pattern = re.compile(
        "|".join(re.escape(marker) for marker in markers) if markers else r"(?!)"
    )

    def plain(chunk: str) -> list[int]:
        if not chunk:
            return []
        return tokenizer(chunk, add_special_tokens=False)["input_ids"]

    def encode(text: str) -> list[int]:
        ids: list[int] = []
        position = 0
        for match in pattern.finditer(text):
            marker = match.group(0)
            ids.extend(plain(text[position:match.start()]))
            ids.extend(plain(marker[:1]))
            ids.extend(plain(marker[1:]))
            position = match.end()
        ids.extend(plain(text[position:]))
        return ids

    return pattern, encode


def stream_decode(tokenizer, partial: list[int], value: int, limit: int = 8):
    """Decode a streamed token, holding back incomplete characters.

    A character whose UTF-8 bytes span several tokens decodes to U+FFFD when
    each token is decoded alone. U+1F604 is two tokens and printed as two
    filled boxes. Returns the text to emit, or "" while more tokens are
    needed. `partial` is the caller's buffer and is cleared on emit.
    """
    partial.append(value)
    piece = tokenizer.decode(partial)
    if "\ufffd" in piece and len(partial) < limit:
        return ""
    partial.clear()
    return piece


# Inkling 2 Mini delimits reasoning, answer and tool calls with SPECIAL TOKENS,
# not with literal <think>/</think> text. The checkpoint's `response_template`
# (tokenizer_config.json) declares exactly these fields. Everything in Veystrix
# used to look only for <think>, so with thinking on the whole generation --
# including the final answer -- landed in reasoning_content and `content` was
# empty. These helpers make that format a first-class citizen.
INKLING_THINK_OPEN = "<|content_thinking|>"
INKLING_TEXT_OPENS = ("<|content_text|>", "<|content_invoke_tool_text|>", "<|content_tool_error|>")
INKLING_TOOL_OPEN = "<|content_invoke_tool_json|>"
INKLING_CLOSE = "<|end_message|>"
INKLING_ASSISTANT = "<|message_model|>"
INKLING_EOS = "<|content_model_end_sampling|>"
INKLING_MARKERS = (
    INKLING_ASSISTANT,
    INKLING_THINK_OPEN,
    *INKLING_TEXT_OPENS,
    INKLING_TOOL_OPEN,
    INKLING_CLOSE,
    INKLING_EOS,
)


def response_format_for(model: str | None = None) -> str:
    """Return "inkling" for models whose response uses special-token fields."""
    if not model:
        return "think"
    from veystrix import model_identity

    family = model_identity.for_family(model).family
    if family == "unknown":
        for key, identity in model_identity.IDENTITIES.items():
            if model in (identity.served_id, identity.checkpoint_alias, key) or model.startswith(key):
                family = key
                break
    identity = model_identity.IDENTITIES.get(family)
    return "inkling" if identity is not None and identity.tool_format == "inkling" else "think"


def _held_marker(text: str, markers) -> int:
    """Length of the longest suffix of `text` that could start a marker."""
    held = 0
    for marker in markers:
        for size in range(min(len(text), len(marker) - 1), held, -1):
            if text.endswith(marker[:size]):
                held = size
                break
    return held


def inkling_initial_kind(rendered: str | None = None) -> str:
    """Determine Inkling's initial segment kind from the rendered prompt tail.

    Inkling 2 Mini emits its own block headers (``<|content_thinking|>``,
    ``<|content_text|>``, ``<|content_invoke_tool_json|>``) after
    ``<|message_model|>``. Only when the prompt itself ends with
    ``<|content_thinking|>`` (or ``<|content_invoke_tool_json|>``) does
    generation start inside that block; otherwise unprefixed output is
    visible content, never trapped in reasoning.
    """
    if not rendered:
        return "content"
    tail = rendered.rstrip()
    if tail.endswith(INKLING_THINK_OPEN):
        return "thinking"
    if tail.endswith(INKLING_TOOL_OPEN):
        return "tool"
    return "content"


class InklingSegmentFilter:
    """Split a token stream into (kind, text) using Inkling's special tokens.

    Drop-in engine for `ReasoningStreamFilter` / `ThinkingStreamFilter`: it
    handles markers split across streamed pieces and never leaks a marker or a
    partial marker into visible text. Tool-call JSON blocks are tagged as
    ``"tool"`` so raw tool payloads never stream into content or reasoning.
    """

    def __init__(self, thinking_enabled: bool = True, initial_kind: str | None = None):
        self.thinking_enabled = bool(thinking_enabled)
        self.kind = initial_kind if initial_kind is not None else "content"
        self.pending = ""
        self._trim = True

    @property
    def inside(self) -> bool:
        return self.kind == "thinking"

    def feed(self, piece: str) -> list[tuple[str, str]]:
        self.pending += piece
        items: list[tuple[str, str]] = []
        while self.pending:
            if self._trim:
                self.pending = self.pending.lstrip("\r\n")
                if not self.pending:
                    break
                self._trim = False
            found = [(self.pending.find(marker), marker) for marker in INKLING_MARKERS]
            found = [(index, marker) for index, marker in found if index >= 0]
            if found:
                index, marker = min(found, key=lambda item: item[0])
                before = self.pending[:index]
                if before:
                    items.append((self.kind, before))
                self.pending = self.pending[index + len(marker):]
                if marker == INKLING_THINK_OPEN:
                    self.kind = "thinking"
                elif marker == INKLING_TOOL_OPEN:
                    self.kind = "tool"
                elif marker in INKLING_TEXT_OPENS:
                    self.kind = "content"
                elif marker in (INKLING_CLOSE, INKLING_ASSISTANT, INKLING_EOS):
                    self.kind = "content"
                self._trim = True
                continue
            held = _held_marker(self.pending, INKLING_MARKERS)
            safe = self.pending[:-held] if held else self.pending
            self.pending = self.pending[-held:] if held else ""
            if safe:
                items.append((self.kind, safe))
            break
        return items

    def finish(self) -> list[tuple[str, str]]:
        text, self.pending = self.pending, ""
        return [(self.kind, text)] if text else []


def extract_inkling_parts(
    full_text: str,
    thinking_enabled: bool = True,
    initial_kind: str = "content",
) -> tuple[str, str, list[str]]:
    """Split a finished Inkling generation into (reasoning, content, tool_blocks)."""
    reasoning: list[str] = []
    content: list[str] = []
    tool_blocks: list[str] = []
    current_tool: list[str] = []

    def flush_tool() -> None:
        if current_tool:
            joined = "".join(current_tool).strip()
            if joined:
                tool_blocks.append(joined)
            current_tool.clear()

    def add(text: str) -> None:
        if not text:
            return
        if kind == "thinking":
            reasoning.append(text)
        elif kind == "tool":
            current_tool.append(text)
        elif kind == "content":
            content.append(text)

    kind: str | None = initial_kind
    position = 0
    while position < len(full_text):
        found = [(full_text.find(marker, position), marker) for marker in INKLING_MARKERS]
        found = [(index, marker) for index, marker in found if index >= 0]
        if not found:
            add(full_text[position:])
            break
        index, marker = min(found, key=lambda item: item[0])
        add(full_text[position:index])
        position = index + len(marker)
        if kind == "tool":
            flush_tool()
        if marker == INKLING_THINK_OPEN:
            kind = "thinking"
        elif marker == INKLING_TOOL_OPEN:
            kind = "tool"
        elif marker in INKLING_TEXT_OPENS:
            kind = "content"
        elif marker in (INKLING_CLOSE, INKLING_ASSISTANT, INKLING_EOS):
            kind = None
    if kind == "tool":
        flush_tool()
    return (
        "".join(reasoning).strip() if thinking_enabled else "",
        "".join(content).lstrip("\r\n"),
        tool_blocks,
    )


def extract_inkling_reasoning_and_content(
    full_text: str,
    thinking_enabled: bool = True,
    initial_kind: str = "content",
) -> tuple[str, str]:
    """Split a finished Inkling generation into (reasoning_content, content)."""
    reasoning, content, _tools = extract_inkling_parts(
        full_text, thinking_enabled=thinking_enabled, initial_kind=initial_kind
    )
    return reasoning, content


class ThinkingStreamFilter:
    """Hide thinking without leaking tags split across streamed pieces."""

    def __init__(
        self,
        inside: bool,
        show: bool,
        format: str = "think",
        initial_kind: str | None = None,
    ):
        self.inside = inside
        self.show = show
        self.format = format
        self.pending = ""
        self._trim_leading = True
        self._inkling = (
            InklingSegmentFilter(inside, initial_kind=initial_kind)
            if format == "inkling"
            else None
        )
        self._inkling_saw_thinking = False
        self._inkling_sep_pending = False

    @staticmethod
    def _partial_marker(text: str, marker: str) -> int:
        maximum = min(len(text), len(marker) - 1)
        for size in range(maximum, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def _collect_inkling(self, items: list[tuple[str, str]]) -> str:
        visible: list[str] = []
        for kind, text in items:
            if kind == "thinking":
                if self.show and text:
                    self._inkling_saw_thinking = True
                    self._inkling_sep_pending = True
                    visible.append(text)
            elif kind == "content":
                if text:
                    if self.show and self._inkling_saw_thinking and self._inkling_sep_pending:
                        visible.append("\n\n")
                        self._inkling_sep_pending = False
                    visible.append(text)
        self.inside = self._inkling.inside if self._inkling is not None else False
        return "".join(visible)

    def feed(self, piece: str) -> str:
        if self._inkling is not None:
            return self._collect_inkling(self._inkling.feed(piece))
        self.pending += piece
        visible = []
        while self.pending:
            if self._trim_leading:
                self.pending = self.pending.lstrip("\r\n")
                if not self.pending:
                    break
                self._trim_leading = False
            marker = "</think>" if self.inside else "<think>"
            index = self.pending.find(marker)
            if index >= 0:
                before = self.pending[:index]
                if self.inside and self.show:
                    visible.append(before.rstrip("\r\n"))
                    visible.append("\n\n")
                elif not self.inside:
                    visible.append(before)
                self.pending = self.pending[index + len(marker):]
                self.inside = not self.inside
                self._trim_leading = True
                continue
            keep = self._partial_marker(self.pending, marker)
            safe = self.pending[:-keep] if keep else self.pending
            remainder = self.pending[-keep:] if keep else ""
            if self.inside and self.show:
                stripped = safe.rstrip("\r\n")
                remainder = safe[len(stripped):] + remainder
                safe = stripped
            if self.show or not self.inside:
                visible.append(safe)
            self.pending = remainder
            break
        return "".join(visible)

    def finish(self) -> str:
        if self._inkling is not None:
            return self._collect_inkling(self._inkling.finish())
        text = self.pending if self.show or not self.inside else ""
        self.pending = ""
        return text


class ToolCallStreamFilter:
    """Hide model protocol markup while preserving text before a tool call."""

    MARKER = "<tool_call>"
    FUNCTION = re.compile(r"<function=([^>\s]+)>")
    JSON_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')

    def __init__(self, format: str = "think"):
        self.format = format
        self.marker = INKLING_TOOL_OPEN if format == "inkling" else self.MARKER
        self.pending = ""
        self.hidden = False
        self.hidden_text = ""
        self._started_event = False
        self._name_event = None
        self._name_found = False

    def feed(self, piece: str) -> str:
        if self.hidden:
            self._inspect_hidden(piece)
            return ""
        self.pending += piece
        index = self.pending.find(self.marker)
        if index >= 0:
            visible = self.pending[:index]
            tail = self.pending[index + len(self.marker):]
            self.pending = ""
            self.hidden = True
            self._started_event = True
            self._inspect_hidden(tail)
            return visible
        keep = ThinkingStreamFilter._partial_marker(self.pending, self.marker)
        visible = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        return visible

    def _inspect_hidden(self, piece: str) -> None:
        if self._name_found:
            return
        self.hidden_text = (self.hidden_text + piece)[-512:]
        pattern = self.JSON_NAME if self.format == "inkling" else self.FUNCTION
        match = pattern.search(self.hidden_text)
        if match:
            self._name_found = True
            self._name_event = match.group(1)

    def take_events(self) -> tuple[bool, str | None]:
        """Return and clear pending protocol-start and function-name events."""
        events = self._started_event, self._name_event
        self._started_event = False
        self._name_event = None
        return events

    def finish(self) -> str:
        """Release buffered normal text and discard buffered tool protocol."""
        visible = "" if self.hidden else self.pending
        self.pending = ""
        return visible


class CompletedTextBuffer:
    """Release complete words or phrases and hold an unfinished word."""

    def __init__(self):
        self.pending = ""

    def feed(self, piece: str) -> list[str]:
        self.pending += piece
        complete = []
        position = 0
        for match in _TEXT_BOUNDARY.finditer(self.pending):
            complete.append(match.group(0))
            position = match.end()
        if position:
            self.pending = self.pending[position:]
        if not complete and len(self.pending) >= _MAX_PENDING:
            complete.append(self.pending)
            self.pending = ""
        return complete

    def finish(self) -> list[str]:
        if not self.pending:
            return []
        final = self.pending
        self.pending = ""
        return [final]


class ToolCallTextGate:
    """Streams a reply's visible text while the request declares tools, without ever showing the start of a tool call.

    A model that may call tools writes ``text <tool_call>...</tool_call>``. The server used to hold every word until the
    turn ended, because the text could turn into a tool call at any point, so a client saw nothing for the whole turn.
    The gate releases text as it arrives and stops at the first ``<tool_call>`` marker. It also holds back an unfinished
    marker (``<tool_`` at the end of a piece) and trailing whitespace, because the finished reply is stripped of both.
    ``remainder(final_text)`` is whatever the finished reply has beyond what was already released (text after a tool
    call, a held tail, a stray marker), so the client's concatenated text equals the finished text exactly.
    """

    MARKER = "<tool_call>"

    def __init__(self):
        self.pending = ""
        self.sent = ""
        self.blocked = False
        self.mismatched = False

    def feed(self, piece: str) -> str:
        if self.blocked or not piece:
            return ""
        self.pending += piece
        cut = self.pending.find(self.MARKER)
        if cut >= 0:
            self.blocked = True
            ready, self.pending = self.pending[:cut], ""
            ready = ready.rstrip()
        else:
            held = 0
            for size in range(min(len(self.MARKER) - 1, len(self.pending)), 0, -1):
                if self.MARKER.startswith(self.pending[-size:]):
                    held = size
                    break
            ready = self.pending[:len(self.pending) - held].rstrip()
            self.pending = self.pending[len(ready):]
        if not self.sent:
            ready = ready.lstrip()
        self.sent += ready
        return ready

    def remainder(self, final_text: str) -> str:
        """The part of the finished text the client has not been sent."""
        if final_text.startswith(self.sent):
            return final_text[len(self.sent):]
        self.mismatched = True              # should not happen: the stream and the parsed reply disagree
        common = 0
        for left, right in zip(self.sent, final_text):
            if left != right:
                break
            common += 1
        return final_text[common:]


class ReasoningStreamFilter:
    """Separates thinking and visible content without leaking tags split across streamed pieces.

    Yields items as (kind, text) where kind is "thinking", "content", or "tool".
    """

    def __init__(
        self,
        thinking_enabled: bool = True,
        format: str = "think",
        initial_kind: str | None = None,
    ):
        self.inside = bool(thinking_enabled)
        self.pending = ""
        self._trim_leading = True
        self._inkling = (
            InklingSegmentFilter(thinking_enabled, initial_kind=initial_kind)
            if format == "inkling"
            else None
        )

    @staticmethod
    def _partial_marker(text: str, marker: str) -> int:
        maximum = min(len(text), len(marker) - 1)
        for size in range(maximum, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def feed(self, piece: str) -> list[tuple[str, str]]:
        if self._inkling is not None:
            items = self._inkling.feed(piece)
            self.inside = self._inkling.inside
            return items
        self.pending += piece
        items: list[tuple[str, str]] = []
        while self.pending:
            if self._trim_leading:
                self.pending = self.pending.lstrip("\r\n")
                if not self.pending:
                    break
                self._trim_leading = False
            marker = "</think>" if self.inside else "<think>"
            index = self.pending.find(marker)
            if index >= 0:
                before = self.pending[:index]
                if before:
                    kind = "thinking" if self.inside else "content"
                    items.append((kind, before))
                self.pending = self.pending[index + len(marker):]
                self.inside = not self.inside
                self._trim_leading = True
                continue
            keep = self._partial_marker(self.pending, marker)
            safe = self.pending[:-keep] if keep else self.pending
            remainder = self.pending[-keep:] if keep else ""
            if safe:
                kind = "thinking" if self.inside else "content"
                items.append((kind, safe))
            self.pending = remainder
            break
        return items

    def finish(self) -> list[tuple[str, str]]:
        if self._inkling is not None:
            items = self._inkling.finish()
            self.inside = self._inkling.inside
            return items
        items = []
        if self.pending:
            kind = "thinking" if self.inside else "content"
            items.append((kind, self.pending))
            self.pending = ""
        return items


def extract_reasoning_and_content(
    full_text: str,
    thinking_enabled: bool = True,
    format: str = "think",
    initial_kind: str | None = None,
) -> tuple[str, str]:
    """Split raw generated text into (reasoning_content, visible_content)."""
    if format == "inkling":
        return extract_inkling_reasoning_and_content(
            full_text,
            thinking_enabled,
            initial_kind=initial_kind if initial_kind is not None else "content",
        )
    if not thinking_enabled:
        return "", full_text
    if "</think>" in full_text:
        think_part, _, content_part = full_text.partition("</think>")
        if think_part.startswith("<think>"):
            think_part = think_part[len("<think>"):]
        return think_part.strip(), content_part.lstrip("\r\n")
    if full_text.startswith("<think>"):
        return full_text[len("<think>"):].strip(), ""
    return full_text.strip(), ""

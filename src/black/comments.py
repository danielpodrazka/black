import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, List, Optional, Union
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
from typing import Iterator, Union, List
from typing import Optional


COMMENT_EXCEPTIONS = ["!", ":", "#"]
NON_BREAKING_SPACE = " "


STANDALONE_COMMENT = token.ST_COMMENT

# Define a custom typing for LN, which is a union of Leaf and Node
LN = Union[Leaf, Node]

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

from black.nodes import (
    CLOSING_BRACKETS,
    STANDALONE_COMMENT,
    WHITESPACE,
    container_of,
    first_leaf_of,
    preceding_leaf,
    syms,
)

# types
LN = Union[Leaf, Node]

FMT_OFF: Final = {"# fmt: off", "# fmt:off", "# yapf: disable"}
FMT_SKIP: Final = {"# fmt: skip", "# fmt:skip"}
FMT_PASS: Final = {*FMT_OFF, *FMT_SKIP}
FMT_ON: Final = {"# fmt: on", "# fmt:on", "# yapf: enable"}


def normalize_fmt_off(node: Node) -> None:
    """
    Convert content between `# fmt: off`/`# fmt: on` into standalone comments.

    This function modifies the code tree, replacing code nodes between
    `# fmt: off` and `# fmt: on` with standalone comment nodes. The
    process repeats until no more `# fmt: off`/`# fmt: on` pairs are found.

    Args:
        node (Node): The root node of the code tree to be normalized.
    """
    while convert_one_fmt_off_pair(node):
        pass


def find_fmt_off_on_pair(node: Node) -> Optional[List[Node]]:
    """
    Find a pair of `# fmt: off` and `# fmt: on` in the code tree.

    This helper function searches for a pair of nodes with `# fmt: off`
    and `# fmt: on` comments in the given code tree.

    Args:
        node (Node): The root node of the code tree to search.

    Returns:
        Optional[List[Node]]: A list of nodes between the pair, or None if not found.
    """
    pair = (None, None)  # (fmt: off index, fmt: on index)
    for i, child in enumerate(node.children):
        if not isinstance(child, Leaf):
            continue

        if child.type == token.COMMENT and child.value.strip() == "# fmt: off":
            if pair[0] is None:
                pair = (i, None)
            else:
                # Found another # fmt: off; reset the search
                pair = (None, None)
        elif child.type == token.COMMENT and child.value.strip() == "# fmt: on":
            if pair[0] is not None:
                pair = (pair[0], i)
                return node.children[pair[0] + 1 : pair[1]]

    return None


def convert_one_fmt_off_pair(node: Node) -> bool:
    """
    Convert one pair of `# fmt: off`/`# fmt: on` into standalone comments.

    This helper function modifies the code tree, removing code nodes
    between `# fmt: off` and `# fmt: on` and reinserting them as
    standalone comment nodes.

    Args:
        node (Node): The root node of the code tree to be normalized.

    Returns:
        bool: Returns True if a pair of `# fmt: off`/`# fmt: on` was found
              and converted, False otherwise.
    """
    fmt_off_on_pair = find_fmt_off_on_pair(node)
    if fmt_off_on_pair is None:
        return False

    for child in fmt_off_on_pair:
        child.type = token.COMMENT
        child.value = f"# {child.value}"
    return True

COMMENT_EXCEPTIONS = " !:#'"


@dataclass
class ProtoComment:
    """Describes a piece of syntax that is a comment.

    It's not a :class:`blib2to3.pytree.Leaf` so that:

    * it can be cached (`Leaf` objects should not be reused more than once as
      they store their lineno, column, prefix, and parent information);
    * `newlines` and `consumed` fields are kept separate from the `value`. This
      simplifies handling of special marker comments like ``# fmt: off/on``.
    """

    type: int  # token.COMMENT or STANDALONE_COMMENT
    value: str  # content of the comment
    newlines: int  # how many newlines before the comment
    consumed: int  # how many characters of the original leaf's prefix did we consume


def make_comment(content: str) -> str:
    """
    Format a given comment string consistently.

    All comments (except for "##", "#!", "#:", '#'") should have a single
    space between the hash sign and the content. If the content didn't start
    with a hash sign, one is provided.

    Args:
        content (str): The content to be formatted as a comment.

    Returns:
        str: The formatted comment string.
    """

    def strip_trailing_spaces(content: str) -> str:
        return content.rstrip()

    def remove_initial_hash(content: str) -> str:
        if content and content[0] == "#":
            return content[1:]
        return content

    def replace_nbsp_by_space(content: str) -> str:
        if (
            content
            and content[0] == NON_BREAKING_SPACE
            and not content.lstrip().startswith("type:")
        ):
            return " " + content[1:]
        return content

    def add_space_if_needed(content: str) -> str:
        if content and content[0] not in COMMENT_EXCEPTIONS:
            return " " + content
        return content

    content = strip_trailing_spaces(content)
    if not content:
        return "#"

    content = remove_initial_hash(content)
    content = replace_nbsp_by_space(content)
    content = add_space_if_needed(content)

    return "#" + content


def is_escaped_newline(
    line: str, index: int, ignored_lines: int, is_endmarker: bool
) -> bool:
    return index == ignored_lines and not is_endmarker


def create_proto_comment(
    comment_type: int, line: str, nlines: int, consumed: int
) -> ProtoComment:
    comment = make_comment(line)
    return ProtoComment(
        type=comment_type, value=comment, newlines=nlines, consumed=consumed
    )


@lru_cache(maxsize=4096)
def list_comments(prefix: str, *, is_endmarker: bool) -> List[ProtoComment]:
    """Return a list of ProtoComment objects parsed from the given prefix."""
    result: List[ProtoComment] = []
    if not prefix or "#" not in prefix:
        return result

    consumed = 0
    nlines = 0
    ignored_lines = 0
    for index, line in enumerate(re.split("\r?\n", prefix)):
        consumed += len(line) + 1
        line = line.lstrip()
        if not line:
            nlines += 1
        if not line.startswith("#"):
            if line.endswith("\\"):
                ignored_lines += 1
            continue

        comment_type = (
            token.COMMENT
            if is_escaped_newline(line, index, ignored_lines, is_endmarker)
            else STANDALONE_COMMENT
        )

        result.append(create_proto_comment(comment_type, line, nlines, consumed))
        nlines = 0
    return result


def create_leaf(pc: ParsedComment) -> Leaf:
    """Create a Leaf object for the given ParsedComment.

    Args:
        pc: The ParsedComment object.

    Returns:
        Leaf: The created Leaf object.
    """
    return Leaf(pc.type, pc.value, prefix="\n" * pc.newlines)


def generate_comments(leaf: LN) -> Iterator[Leaf]:
    """Clean the prefix of the `leaf` and generate comments from it, if any.

    Comments in lib2to3 are shoved into the whitespace prefix. This happens
    in `pgen2/driver.py:Driver.parse_tokens()`. This was a brilliant implementation
    move because it does away with modifying the grammar to include all the
    possible places in which comments can be placed.

    The sad consequence for us though is that comments don't "belong" anywhere.
    This is why this function generates simple parentless Leaf objects for
    comments. We simply don't know what the correct parent should be.

    No matter though, we can live without this. We really only need to
    differentiate between inline and standalone comments. The latter don't
    share the line with any code.

    Inline comments are emitted as regular token.COMMENT leaves. Standalone
    are emitted with a fake STANDALONE_COMMENT token identifier.

    Args:
        leaf: An LN (Leaf or Node) object.

    Yields:
        Leaf: One Leaf object per generated comment.
    """
    is_endmarker = leaf.type == token.ENDMARKER
    parsed_comments = list_comments(leaf.prefix, is_endmarker)
    for pc in parsed_comments:
        yield create_leaf(pc)


@lru_cache(maxsize=4096)
def list_comments(prefix: str, *, is_endmarker: bool) -> List[ProtoComment]:
    """Return a list of :class:`ProtoComment` objects parsed from the given `prefix`."""
    result: List[ProtoComment] = []
    if not prefix or "#" not in prefix:
        return result

    consumed = 0
    nlines = 0
    ignored_lines = 0
    for index, line in enumerate(re.split("\r?\n", prefix)):
        consumed += len(line) + 1  # adding the length of the split '\n'
        line = line.lstrip()
        if not line:
            nlines += 1
        if not line.startswith("#"):
            # Escaped newlines outside of a comment are not really newlines at
            # all. We treat a single-line comment following an escaped newline
            # as a simple trailing comment.
            if line.endswith("\\"):
                ignored_lines += 1
            continue

        if index == ignored_lines and not is_endmarker:
            comment_type = token.COMMENT  # simple trailing comment
        else:
            comment_type = STANDALONE_COMMENT
        comment = make_comment(line)
        result.append(
            ProtoComment(
                type=comment_type, value=comment, newlines=nlines, consumed=consumed
            )
        )
        nlines = 0
    return result


def make_comment(content: str) -> str:
    """Return a consistently formatted comment from the given `content` string.

    All comments (except for "##", "#!", "#:", '#'") should have a single
    space between the hash sign and the content.

    If `content` didn't start with a hash sign, one is provided.
    """
    content = content.rstrip()
    if not content:
        return "#"

    if content[0] == "#":
        content = content[1:]
    NON_BREAKING_SPACE = " "
    if (
        content
        and content[0] == NON_BREAKING_SPACE
        and not content.lstrip().startswith("type:")
    ):
        content = " " + content[1:]  # Replace NBSP by a simple space
    if content and content[0] not in COMMENT_EXCEPTIONS:
        content = " " + content
    return "#" + content


def convert_one_fmt_off_pair(node: Node) -> bool:
    """Convert content of a single `# fmt: off`/`# fmt: on` into a standalone comment.

    Returns True if a pair was converted.
    """
    for leaf in node.leaves():
        previous_consumed = 0
        for comment in list_comments(leaf.prefix, is_endmarker=False):
            if comment.value not in FMT_PASS:
                previous_consumed = comment.consumed
                continue
            # We only want standalone comments. If there's no previous leaf or
            # the previous leaf is indentation, it's a standalone comment in
            # disguise.
            if comment.value in FMT_PASS and comment.type != STANDALONE_COMMENT:
                prev = preceding_leaf(leaf)
                if prev:
                    if comment.value in FMT_OFF and prev.type not in WHITESPACE:
                        continue
                    if comment.value in FMT_SKIP and prev.type in WHITESPACE:
                        continue

            ignored_nodes = list(generate_ignored_nodes(leaf, comment))
            if not ignored_nodes:
                continue

            first = ignored_nodes[0]  # Can be a container node with the `leaf`.
            parent = first.parent
            prefix = first.prefix
            if comment.value in FMT_OFF:
                first.prefix = prefix[comment.consumed :]
            if comment.value in FMT_SKIP:
                first.prefix = ""
                standalone_comment_prefix = prefix
            else:
                standalone_comment_prefix = (
                    prefix[:previous_consumed] + "\n" * comment.newlines
                )
            hidden_value = "".join(str(n) for n in ignored_nodes)
            if comment.value in FMT_OFF:
                hidden_value = comment.value + "\n" + hidden_value
            if comment.value in FMT_SKIP:
                hidden_value += "  " + comment.value
            if hidden_value.endswith("\n"):
                # That happens when one of the `ignored_nodes` ended with a NEWLINE
                # leaf (possibly followed by a DEDENT).
                hidden_value = hidden_value[:-1]
            first_idx: Optional[int] = None
            for ignored in ignored_nodes:
                index = ignored.remove()
                if first_idx is None:
                    first_idx = index
            assert parent is not None, "INTERNAL ERROR: fmt: on/off handling (1)"
            assert first_idx is not None, "INTERNAL ERROR: fmt: on/off handling (2)"
            parent.insert_child(
                first_idx,
                Leaf(
                    STANDALONE_COMMENT,
                    hidden_value,
                    prefix=standalone_comment_prefix,
                    fmt_pass_converted_first_leaf=first_leaf_of(first),
                ),
            )
            return True

    return False


def generate_ignored_nodes(leaf: Leaf, comment: ProtoComment) -> Iterator[LN]:
    """Starting from the container of `leaf`, generate all leaves until `# fmt: on`.

    If comment is skip, returns leaf only.
    Stops at the end of the block.
    """
    if comment.value in FMT_SKIP:
        yield from _generate_ignored_nodes_from_fmt_skip(leaf, comment)
        return
    container: Optional[LN] = container_of(leaf)
    while container is not None and container.type != token.ENDMARKER:
        if is_fmt_on(container):
            return

        # fix for fmt: on in children
        if children_contains_fmt_on(container):
            for index, child in enumerate(container.children):
                if isinstance(child, Leaf) and is_fmt_on(child):
                    if child.type in CLOSING_BRACKETS:
                        # This means `# fmt: on` is placed at a different bracket level
                        # than `# fmt: off`. This is an invalid use, but as a courtesy,
                        # we include this closing bracket in the ignored nodes.
                        # The alternative is to fail the formatting.
                        yield child
                    return
                if (
                    child.type == token.INDENT
                    and index < len(container.children) - 1
                    and children_contains_fmt_on(container.children[index + 1])
                ):
                    # This means `# fmt: on` is placed right after an indentation
                    # level, and we shouldn't swallow the previous INDENT token.
                    return
                if children_contains_fmt_on(child):
                    return
                yield child
        else:
            if container.type == token.DEDENT and container.next_sibling is None:
                # This can happen when there is no matching `# fmt: on` comment at the
                # same level as `# fmt: on`. We need to keep this DEDENT.
                return
            yield container
            container = container.next_sibling


def _generate_ignored_nodes_from_fmt_skip(
    leaf: Leaf, comment: ProtoComment
) -> Iterator[LN]:
    """Generate all leaves that should be ignored by the `# fmt: skip` from `leaf`."""
    prev_sibling = leaf.prev_sibling
    parent = leaf.parent
    # Need to properly format the leaf prefix to compare it to comment.value,
    # which is also formatted
    comments = list_comments(leaf.prefix, is_endmarker=False)
    if not comments or comment.value != comments[0].value:
        return
    if prev_sibling is not None:
        leaf.prefix = ""
        siblings = [prev_sibling]
        while "\n" not in prev_sibling.prefix and prev_sibling.prev_sibling is not None:
            prev_sibling = prev_sibling.prev_sibling
            siblings.insert(0, prev_sibling)
        yield from siblings
    elif (
        parent is not None and parent.type == syms.suite and leaf.type == token.NEWLINE
    ):
        # The `# fmt: skip` is on the colon line of the if/while/def/class/...
        # statements. The ignored nodes should be previous siblings of the
        # parent suite node.
        leaf.prefix = ""
        ignored_nodes: List[LN] = []
        parent_sibling = parent.prev_sibling
        while parent_sibling is not None and parent_sibling.type != syms.suite:
            ignored_nodes.insert(0, parent_sibling)
            parent_sibling = parent_sibling.prev_sibling
        # Special case for `async_stmt` where the ASYNC token is on the
        # grandparent node.
        grandparent = parent.parent
        if (
            grandparent is not None
            and grandparent.prev_sibling is not None
            and grandparent.prev_sibling.type == token.ASYNC
        ):
            ignored_nodes.insert(0, grandparent.prev_sibling)
        yield from iter(ignored_nodes)


def is_fmt_on(container: LN) -> bool:
    """Determine whether formatting is switched on within a container.
    Determined by whether the last `# fmt:` comment is `on` or `off`.
    """
    fmt_on = False
    for comment in list_comments(container.prefix, is_endmarker=False):
        if comment.value in FMT_ON:
            fmt_on = True
        elif comment.value in FMT_OFF:
            fmt_on = False
    return fmt_on


def children_contains_fmt_on(container: LN) -> bool:
    """Determine if children have formatting switched on."""
    for child in container.children:
        leaf = first_leaf_of(child)
        if leaf is not None and is_fmt_on(leaf):
            return True

    return False


def contains_pragma_comment(comment_list: List[Leaf]) -> bool:
    """
    Returns:
        True iff one of the comments in @comment_list is a pragma used by one
        of the more common static analysis tools for python (e.g. mypy, flake8,
        pylint).
    """
    for comment in comment_list:
        if comment.value.startswith(("# type:", "# noqa", "# pylint:")):
            return True

    return False

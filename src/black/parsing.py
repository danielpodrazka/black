"""
Parse Python code and perform AST validation.
"""
import ast
import platform
import sys
from typing import Any, Iterable, Iterator, List, Set, Tuple, Type, Union

from black.mode import VERSION_TO_FEATURES, Feature, TargetVersion, supports_feature
from black.nodes import syms
from blib2to3 import pygram
from blib2to3.pgen2 import driver
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.parse import ParseError
from blib2to3.pgen2.tokenize import TokenError
from blib2to3.pytree import Leaf, Node
from typing import Set, List

from black.mode import TargetVersion, Feature, supports_feature

if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final

ast3: Any

_IS_PYPY = platform.python_implementation() == "PyPy"

try:
    from typed_ast import ast3
except ImportError:
    if sys.version_info < (3, 8) and not _IS_PYPY:
        print(
            (
                "The typed_ast package is required but not installed.\n"
                "You can upgrade to Python 3.8+ or install typed_ast with\n"
                "`python3 -m pip install typed-ast`."
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        ast3 = ast


PY2_HINT: Final = "Python 2 support was removed in version 22.0."


class InvalidInput(ValueError):
    """Raised when input source code fails all parse attempts."""


def get_errors(
    src_txt: str, grammars: List[Grammar]
) -> Tuple[Node, Union[None, InvalidInput]]:
    errors = {}
    for grammar in grammars:
        drv = driver.Driver(grammar)
        try:
            result = drv.parse_string(src_txt, True)
            return result, None

        except ParseError as pe:
            lineno, column = pe.context[1]
            lines = src_txt.splitlines()
            try:
                faulty_line = lines[lineno - 1]
            except IndexError:
                faulty_line = "<line number missing in source>"
            errors[grammar.version] = InvalidInput(
                f"Cannot parse: {lineno}:{column}: {faulty_line}"
            )

        except TokenError as te:
            lineno, column = te.args[1]
            errors[grammar.version] = InvalidInput(
                f"Cannot parse: {lineno}:{column}: {te.args[0]}"
            )
    return None, errors


def parse_source_with_driver(src_txt: str, drv: driver.Driver) -> bool:
    """
    Parse the source code with the given driver.

    Args:
        src_txt: The source code to be parsed as a string.
        drv: The blib2to3.pgen2.driver.Driver instance to use for parsing.

    Returns:
        True if parsing was successful, False if there was a parsing error.
    """
    try:
        drv.parse_string(src_txt, True)
        return True
    except (ParseError, TokenError, IndentationError):
        return False


def matches_grammar(src_txt: str, grammar: Grammar) -> bool:
    """
    Check if the given source text matches the provided grammar.

    Args:
        src_txt: The source code to be checked as a string.
        grammar: The Grammar instance to be checked against.

    Returns:
        True if source code matches the grammar, False otherwise.
    """
    drv = driver.Driver(grammar)
    return parse_source_with_driver(src_txt, drv)


def parse_single_version(
    src: str, version: Tuple[int, int], *, type_comments: bool
) -> Union[ast.AST, ast3.AST]:
    """
    Parse a single version of Python source code.

    Args:
        src: The source code string to parse.
        version: A tuple of the Python version to parse the source code for.
        type_comments: A flag indicating whether to include type comments in the parsed AST.

    Returns:
        An AST of the parsed source code in the specified Python version format.

    Raises:
        ParseError: If there is an error during parsing.
    """
    filename = "<unknown>"

    if sys.version_info >= (3, 8) and version >= (3,):
        return parse_python_38_or_higher(src, version, type_comments, filename)

    return parse_python_using_ast3(src, version, type_comments, filename)


def parse_ast(src: str) -> Union[ast.AST, ast.mod]:
    """
    Parse a Python source code string into an Abstract Syntax Tree (AST).

    Tries most recent Python versions first, and falls back to previous versions if necessary.
    Raises a SyntaxError only after trying all supported Python versions, and includes the
    first encountered error message.

    Args:
        src (str): The source code to parse into an AST.

    Returns:
        Union[ast.AST, ast.mod]: The parsed AST. If parsed successfully, returns a tree
        structure describing the abstract syntax of the Python source code string.

    Raises:
        SyntaxError: If unable to parse the source code into an AST using all supported
        Python versions.
    """
    versions = _generate_python_versions()
    first_error = ""

    for version in versions:
        try:
            return ast.parse(src, type_comments=True)
        except SyntaxError as e:
            first_error = first_error or str(e)

    return _attempt_parse_without_type_comments(src, versions, first_error)


def _generate_python_versions() -> List[Tuple[int, int]]:
    """Generates a list of Python versions to attempt parsing with."""
    return [(3, minor) for minor in range(3, sys.version_info[1] + 1)]


def _attempt_parse_without_type_comments(
    src: str, versions: List[Tuple[int, int]], first_error: str
) -> Union[ast.AST, ast.mod]:
    """Attempts to parse the source code without type comments."""
    for version in versions:
        try:
            return ast.parse(src, type_comments=False)
        except SyntaxError:
            pass

    raise SyntaxError(first_error)


def parse_python_38_or_higher(
    src: str, version: Tuple[int, int], type_comments: bool, filename: str
) -> ast.AST:
    """
    Parse Python source code for Python 3.8 or higher using built-in ast.

    Args:
        src: The source code string to parse.
        version: A tuple of the Python version to parse the source code for.
        type_comments: A flag indicating whether to include type comments in the parsed AST.
        filename: The filename of the source code.

    Returns:
        An AST of the parsed source code in the specified Python version format.
    """
    return ast.parse(
        src, filename, feature_version=version, type_comments=type_comments
    )


def parse_python_using_ast3(
    src: str, version: Tuple[int, int], type_comments: bool, filename: str
) -> ast3.AST:
    """
    Parse Python source code using typed-ast (ast3) or default ast in PyPy.

    Args:
        src: The source code string to parse.
        version: A tuple of the Python version to parse the source code for.
        type_comments: A flag indicating whether to include type comments in the parsed AST.
        filename: The filename of the source code.

    Returns:
        An AST of the parsed source code in the specified Python version format.
    """
    if _IS_PYPY:
        return parse_python_pypy(src, version, type_comments, filename)

    return parse_python_cpython(src, version, type_comments, filename)


def parse_python_pypy(
    src: str, version: Tuple[int, int], type_comments: bool, filename: str
) -> ast.AST:
    """
    Parse Python source code in PyPy.

    Args:
        src: The source code string to parse.
        version: A tuple of the Python version to parse the source code for.
        type_comments: A flag indicating whether to include type comments in the parsed AST.
        filename: The filename of the source code.

    Returns:
        An AST of the parsed source code in the specified Python version format.
    """
    if sys.version_info >= (3, 8):
        return ast3.parse(src, filename, type_comments=type_comments)
    else:
        return ast3.parse(src, filename)


def parse_python_cpython(
    src: str, version: Tuple[int, int], type_comments: bool, filename: str
) -> ast3.AST:
    """
    Parse Python source code in CPython using typed-ast (ast3).

    Args:
        src: The source code string to parse.
        version: A tuple of the Python version to parse the source code for.
        type_comments: A flag indicating whether to include type comments in the parsed AST.
        filename: The filename of the source code.

    Returns:
        An AST of the parsed source code in the specified Python version format.
    """
    if type_comments:
        return ast3.parse(src, filename, feature_version=version[1])

    return ast.parse(src, filename)


def ensure_parsed_src(src_txt: str) -> str:
    if not src_txt.endswith("\n"):
        src_txt += "\n"
    return src_txt


def lib2to3_parse(src_txt: str, target_versions: Iterable[TargetVersion] = ()) -> Node:
    """
    Given a string with source code, return the lib2to3 Node.

    Args:
        src_txt (str): The source code string to parse.
        target_versions (Iterable[TargetVersion]): A set of target Python versions.
            Defaults to an empty set.

    Returns:
        Node: The lib2to3 Node representing the parsed source code.

    Raises:
        InvalidInput: If the source code cannot be parsed.
    """
    src_txt = ensure_parsed_src(src_txt)
    grammars = get_grammars(set(target_versions))
    result, errors = get_errors(src_txt, grammars)

    if result is None:
        assert len(errors) >= 1
        exc = errors[max(errors)]

        if matches_grammar(src_txt, pygram.python_grammar) or matches_grammar(
            src_txt, pygram.python_grammar_no_print_statement
        ):
            original_msg = exc.args[0]
            msg = f"{original_msg}\n{PY2_HINT}"
            raise InvalidInput(msg) from None

        raise exc from None

    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])
    return result


def get_python37_39_grammar(target_versions: Set[TargetVersion]) -> List[Grammar]:
    """Get the Python 3.7-3.9 grammar."""
    if supports_feature(target_versions, Feature.ASYNC_IDENTIFIERS):
        return []
    return [pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords]


def get_python30_36_grammar(target_versions: Set[TargetVersion]) -> List[Grammar]:
    """Get the Python 3.0-3.6 grammar."""
    if supports_feature(target_versions, Feature.ASYNC_KEYWORDS):
        return []
    return [pygram.python_grammar_no_print_statement_no_exec_statement]


def get_python310_grammar(target_versions: Set[TargetVersion]) -> List[Grammar]:
    """Get the Python 3.10+ grammar."""
    if not any(
        Feature.PATTERN_MATCHING in VERSION_TO_FEATURES[v] for v in target_versions
    ):
        return []
    return [pygram.python_grammar_soft_keywords]


def get_grammars(target_versions: Set[TargetVersion]) -> List[Grammar]:
    """
    Get a list of suitable grammars based on the given target versions.

    Args:
        target_versions (Set[TargetVersion]): A set of target Python versions.

    Returns:
        List[Grammar]: A list of suitable grammars.

    If no target_versions are passed, all possible grammars will be returned.
    """
    if not target_versions:
        return get_grammars({*TargetVersion})
    grammars = (
        get_python37_39_grammar(target_versions)
        + get_python30_36_grammar(target_versions)
        + get_python310_grammar(target_versions)
    )

    return grammars


def lib2to3_unparse(node: Node) -> str:
    """Given a lib2to3 node, return its string representation."""
    code = str(node)
    return code


ast3_AST: Final[Type[ast3.AST]] = ast3.AST


def _normalize(lineend: str, value: str) -> str:
    # To normalize, we strip any leading and trailing space from
    # each line...
    stripped: List[str] = [i.strip() for i in value.splitlines()]
    normalized = lineend.join(stripped)
    # ...and remove any blank lines at the beginning and end of
    # the whole string
    return normalized.strip()


def stringify_ast(node: Union[ast.AST, ast3.AST], depth: int = 0) -> Iterator[str]:
    """Simple visitor generating strings to compare ASTs by content."""

    node = fixup_ast_constants(node)

    yield f"{'  ' * depth}{node.__class__.__name__}("

    type_ignore_classes: Tuple[Type[Any], ...]
    for field in sorted(node._fields):  # noqa: F402
        # TypeIgnore will not be present using pypy < 3.8, so need for this
        if not (_IS_PYPY and sys.version_info < (3, 8)):
            # TypeIgnore has only one field 'lineno' which breaks this comparison
            type_ignore_classes = (ast3.TypeIgnore,)
            if sys.version_info >= (3, 8):
                type_ignore_classes += (ast.TypeIgnore,)
            if isinstance(node, type_ignore_classes):
                break

        try:
            value: object = getattr(node, field)
        except AttributeError:
            continue

        yield f"{'  ' * (depth+1)}{field}="

        if isinstance(value, list):
            for item in value:
                # Ignore nested tuples within del statements, because we may insert
                # parentheses and they change the AST.
                if (
                    field == "targets"
                    and isinstance(node, (ast.Delete, ast3.Delete))
                    and isinstance(item, (ast.Tuple, ast3.Tuple))
                ):
                    for elt in item.elts:
                        yield from stringify_ast(elt, depth + 2)

                elif isinstance(item, (ast.AST, ast3.AST)):
                    yield from stringify_ast(item, depth + 2)

        # Note that we are referencing the typed-ast ASTs via global variables and not
        # direct module attribute accesses because that breaks mypyc. It's probably
        # something to do with the ast3 variables being marked as Any leading
        # mypy to think this branch is always taken, leaving the rest of the code
        # unanalyzed. Tighting up the types for the typed-ast AST types avoids the
        # mypyc crash.
        elif isinstance(value, (ast.AST, ast3_AST)):
            yield from stringify_ast(value, depth + 2)

        else:
            normalized: object
            # Constant strings may be indented across newlines, if they are
            # docstrings; fold spaces after newlines when comparing. Similarly,
            # trailing and leading space may be removed.
            if (
                isinstance(node, ast.Constant)
                and field == "value"
                and isinstance(value, str)
            ):
                normalized = _normalize("\n", value)
            else:
                normalized = value
            yield f"{'  ' * (depth+2)}{normalized!r},  # {value.__class__.__name__}"

    yield f"{'  ' * depth})  # /{node.__class__.__name__}"


def fixup_ast_constants(node: Union[ast.AST, ast3.AST]) -> Union[ast.AST, ast3.AST]:
    """Map ast nodes deprecated in 3.8 to Constant."""
    if isinstance(node, (ast.Str, ast3.Str, ast.Bytes, ast3.Bytes)):
        return ast.Constant(value=node.s)

    if isinstance(node, (ast.Num, ast3.Num)):
        return ast.Constant(value=node.n)

    if isinstance(node, (ast.NameConstant, ast3.NameConstant)):
        return ast.Constant(value=node.value)

    return node

"""对话区 Markdown 渲染：标题、粗体、斜体、列表、代码块、引用、分隔线、行内代码、链接、表格。"""

import re

from prompt_toolkit.formatted_text import StyleAndTextTuples
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from pygments.util import ClassNotFound
from wcwidth import wcswidth

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_HR_RE = re.compile(r"^\s*([-*_])\s*\1\s*\1\s*$")
_LIST_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
_QUOTE_RE = re.compile(r"^>\s?(.*)$")
_INLINE_RE = re.compile(
    r"(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|\*[^*]+\*)"
)
_LINK_RE = re.compile(r"^\[([^\]]+)\]\([^)]+\)$")
_CODE_BLOCK_DELIMITER = "```"
_TABLE_CELL_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^[\s:|-]+$")


def render_markdown(text: str, *, streaming: bool = False) -> StyleAndTextTuples:
    """把 Markdown 文本转换为带样式的片段列表，行与行之间插入换行。

    流式输出时文本可能不完整，未闭合代码块使用基础样式，避免反复进行全量语法高亮。
    """

    fragments: StyleAndTextTuples = []
    in_code_block = False
    code_language = ""
    code_lines: list[str] = []
    table_rows: list[list[str]] | None = None
    lines = text.split("\n")
    for line_index, line in enumerate(lines):
        is_last = line_index == len(lines) - 1
        add_newline = not is_last
        if in_code_block:
            if line.strip().startswith(_CODE_BLOCK_DELIMITER):
                fragments.extend(_render_code_block(code_language, code_lines))
                code_lines = []
                code_language = ""
                in_code_block = False
                # 结束围栏只负责把代码块与后续正文隔开
                add_newline = not is_last
            else:
                code_lines.append(line)
                # 代码行由 _render_code_block 统一输出换行
                add_newline = False
        elif line.strip().startswith(_CODE_BLOCK_DELIMITER):
            in_code_block = True
            code_language = line.strip()[len(_CODE_BLOCK_DELIMITER) :].strip()
            # 开始围栏不产生可见内容或额外换行
            add_newline = False
        elif _TABLE_CELL_RE.match(line):
            # 表格行由整表渲染统一换行，不在行内单独插入
            add_newline = False
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if table_rows is None:
                table_rows = [cells]
            else:
                table_rows.append(cells)
        else:
            if table_rows is not None:
                fragments.extend(_render_table(table_rows))
                fragments.append(("", "\n"))
                table_rows = None
            heading = _HEADING_RE.match(line)
            if heading:
                fragments.append(("class:md-heading", heading.group(2)))
            elif _HR_RE.match(line):
                fragments.append(("class:md-hr", "─" * 20))
            else:
                quote = _QUOTE_RE.match(line)
                if quote:
                    fragments.extend(
                        [("class:md-quote", "▍ "), *render_inline(quote.group(1))]
                    )
                else:
                    list_match = _LIST_RE.match(line)
                    if list_match:
                        fragments.extend([("", "• "), *render_inline(list_match.group(1))])
                    else:
                        ordered_match = _ORDERED_LIST_RE.match(line)
                        if ordered_match:
                            fragments.extend(
                                [("", "· "), *render_inline(ordered_match.group(1))]
                            )
                        else:
                            fragments.extend(render_inline(line))
        if add_newline:
            fragments.append(("", "\n"))
    if table_rows is not None:
        fragments.extend(_render_table(table_rows))
    if in_code_block:
        # 流式输出中途：未闭合代码块按已收集的行渲染
        fragments.extend(
            _render_code_block(
                code_language,
                code_lines,
                highlight=not streaming,
            )
        )
    return fragments


# 代码语法高亮的 token 到样式类映射（顺序敏感：更具体的 token 先匹配）
_TOKEN_STYLE_CLASSES = [
    ("md-tok-comment", Token.Comment),
    ("md-tok-string", Token.String),
    ("md-tok-number", Token.Number),
    ("md-tok-decorator", Token.Name.Decorator),
    ("md-tok-function", Token.Name.Function),
    ("md-tok-class", Token.Name.Class),
    ("md-tok-builtin", Token.Name.Builtin),
    ("md-tok-keyword", Token.Keyword),
    ("md-tok-operator", Token.Operator),
]


# 语言级 token 覆盖：新增语言定制只需在这里加一行（默认为空，全部走 Pygments 自动识别）
_LANGUAGE_TOKEN_OVERRIDES: dict[str, list[tuple[str, type]]] = {}


def _render_code_block(
    language: str,
    lines: list[str],
    *,
    highlight: bool = True,
) -> StyleAndTextTuples:
    """渲染代码块：首行语言名 + Pygments token 高亮，语言不支持时整块基础样式。"""

    fragments: StyleAndTextTuples = []
    if language:
        fragments.append(("class:md-code-lang", f"{language}\n"))
    code = "\n".join(lines)
    tokens = _highlight_tokens(code, language) if highlight else None
    if tokens is None:
        for line_index, line in enumerate(lines):
            fragments.append(("class:md-code-block", line))
            if line_index < len(lines) - 1:
                fragments.append(("", "\n"))
        return fragments
    for token_type, token_text in tokens:
        style_class = _token_style(token_type, language)
        fragments.append((style_class, token_text))
    if (
        fragments
        and not code.endswith("\n")
        and fragments[-1][1].endswith("\n")
    ):
        # 部分 Pygments lexer 会补一个源码中不存在的末尾换行
        style, token_text = fragments[-1]
        fragments[-1] = (style, token_text[:-1])
    return fragments


def _highlight_tokens(code: str, language: str):
    """用 Pygments 对代码分词，语言不支持时返回 None。"""

    if not language or not code:
        return None
    try:
        lexer = get_lexer_by_name(language)
    except ClassNotFound:
        return None
    return list(lexer.get_tokens(code))


def _token_style(token_type, language: str) -> str:
    """把 Pygments token 映射到样式类（带 class: 前缀），语言覆盖优先。"""

    overrides = _LANGUAGE_TOKEN_OVERRIDES.get(language)
    if overrides:
        for style_class, token_kind in overrides:
            if token_type in token_kind:
                return f"class:{style_class}"
    for style_class, token_kind in _TOKEN_STYLE_CLASSES:
        if token_type in token_kind:
            return f"class:{style_class}"
    return "class:md-code-block"


def render_inline(text: str) -> StyleAndTextTuples:
    """把一行内的行内代码、链接、粗体和斜体标记转换为带样式的片段。"""

    fragments: StyleAndTextTuples = []
    for part in _INLINE_RE.split(text):
        if part.startswith("`") and part.endswith("`") and len(part) > 2:
            fragments.append(("class:md-code", part[1:-1]))
        elif part.startswith("[") and _LINK_RE.match(part):
            fragments.append(("class:md-link", _LINK_RE.match(part).group(1)))
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            fragments.append(("class:md-bold", part[2:-2]))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            fragments.append(("class:md-italic", part[1:-1]))
        elif part:
            fragments.append(("", _strip_unclosed_markers(part)))
    return fragments


def _strip_unclosed_markers(text: str) -> str:
    """移除未闭合的加粗（**）与行内代码（`）标记符号，只保留文字。

    单个星号 * 保留（避免误伤乘法等普通用法）；只有成对标记出现奇数个
    时才认定存在未闭合标记，去掉最后一个符号本身。
    """

    for marker in ("**", "`"):
        if text.count(marker) % 2 == 1:
            index = text.rfind(marker)
            text = text[:index] + text[index + len(marker) :]
    return text


def _render_table(rows: list[list[str]]) -> StyleAndTextTuples:
    """把表格行渲染为 │ 边框对齐的片段，分隔行跳过。"""

    data_rows = [row for row in rows if not _is_separator_row(row)]
    if not data_rows:
        return []
    # 计算每列宽度（CJK 按显示宽度）
    widths: list[int] = []
    for row in data_rows:
        for index, cell in enumerate(row):
            cell_width = wcswidth(cell)
            if index >= len(widths):
                widths.append(cell_width)
            else:
                widths[index] = max(widths[index], cell_width)
    fragments: StyleAndTextTuples = []
    for row_index, row in enumerate(data_rows):
        padded = [
            cell + " " * (widths[index] - wcswidth(cell))
            for index, cell in enumerate(row)
        ]
        line = "│ " + " │ ".join(padded) + " │"
        if row_index > 0:
            fragments.append(("", "\n"))
        if row_index == 0:
            fragments.append(("class:md-table-header", line))
        else:
            fragments.append(("", line))
    return fragments


def _is_separator_row(cells: list[str]) -> bool:
    """判断一行是否为表格分隔行（由 - : | 空格组成）。"""

    return bool(cells) and all(
        _TABLE_SEPARATOR_RE.fullmatch(cell) for cell in cells
    )

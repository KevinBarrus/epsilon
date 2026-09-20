"""测试工具输出原文落盘、占位符与按需取回工具。"""

import re
from pathlib import Path

import pytest

from core.artifacts import (
    FIREWALL_THRESHOLD_CHARS,
    ArtifactStore,
    apply_output_firewall,
    artifact_placeholder,
    create_read_artifact_tool,
    is_artifact_placeholder,
)
from core.model import ToolCall
from core.tools.output_limits import TRUNCATION_NOTICE

ARTIFACT_ID_PATTERN = re.compile(r"\[artifact ([0-9a-f]+)\]")


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def _artifact_id(text: str) -> str:
    match = ARTIFACT_ID_PATTERN.search(text)
    assert match is not None
    return match.group(1)


def test_artifact_store_round_trip(tmp_path: Path) -> None:
    """测试原文落盘后可按 id 完整取回。"""

    store = _store(tmp_path)
    content = "运行输出\n" * 100

    artifact_id = store.save(content, session_id="s-1", source_tool="run_command")
    loaded = store.load(artifact_id)

    assert loaded is not None
    assert loaded.content == content
    assert loaded.session_id == "s-1"
    assert loaded.source_tool == "run_command"


def test_artifact_store_reuses_id_for_identical_content(tmp_path: Path) -> None:
    """测试相同原文复用同一 id 且只落盘一份。"""

    store = _store(tmp_path)
    content = "相同内容"

    first = store.save(content, session_id="s-1", source_tool="run_command")
    second = store.save(content, session_id="s-2", source_tool="read_file")

    assert first == second
    assert len(list((tmp_path / "artifacts").glob("*.json"))) == 1


def test_artifact_store_returns_none_for_corrupted_file(tmp_path: Path) -> None:
    """测试原文文件损坏时返回 None 而不是抛异常。"""

    store = _store(tmp_path)
    artifact_id = store.save("原文", session_id="s-1", source_tool="run_command")
    (tmp_path / "artifacts" / f"{artifact_id}.json").write_text(
        "{不是合法 JSON", encoding="utf-8"
    )

    assert store.load(artifact_id) is None
    assert store.load("0" * 16) is None


def test_artifact_placeholder_keeps_head_tail_and_hint(tmp_path: Path) -> None:
    """测试占位符保留头尾预览并给出取回指引。"""

    content = "\n".join(f"line-{index}" for index in range(30))

    placeholder = artifact_placeholder("abc123", len(content), content)

    assert "[artifact abc123]" in placeholder
    assert "line-0" in placeholder
    assert "line-29" in placeholder
    assert "line-15" not in placeholder
    assert "Use read_artifact" in placeholder
    assert is_artifact_placeholder(placeholder)


def test_is_artifact_placeholder_detects_marker() -> None:
    """测试占位符识别只认 artifact 标记。"""

    assert is_artifact_placeholder("prefix [artifact abc] tail") is True
    assert is_artifact_placeholder("普通工具输出") is False


def test_apply_output_firewall_stores_and_returns_placeholder(tmp_path: Path) -> None:
    """测试超阈值输出落盘并定型为有界占位符。"""

    store = _store(tmp_path)
    content = "a" * (FIREWALL_THRESHOLD_CHARS + 1)

    result = apply_output_firewall(
        content, store, session_id="s-1", source_tool="run_command"
    )

    assert is_artifact_placeholder(result)
    artifact_id = _artifact_id(result)
    loaded = store.load(artifact_id)
    assert loaded is not None
    assert loaded.content == content


def test_apply_output_firewall_without_store_uses_hard_truncation() -> None:
    """测试缺少 store 时退化为原有字节/行数硬截断。"""

    content = "a" * 20_000

    result = apply_output_firewall(
        content, None, session_id="s-1", source_tool="run_command"
    )

    assert result.endswith(TRUNCATION_NOTICE)
    assert is_artifact_placeholder(result) is False


@pytest.mark.asyncio
async def test_read_artifact_returns_requested_line_range(tmp_path: Path) -> None:
    """测试 read_artifact 按行范围取回并提示后续偏移。"""

    store = _store(tmp_path)
    artifact_id = store.save(
        "l1\nl2\nl3\n", session_id="s-1", source_tool="run_command"
    )
    _, handler = create_read_artifact_tool(store)

    result = await handler(
        ToolCall("c1", "read_artifact", {"artifact_id": artifact_id, "offset": 2, "limit": 1})
    )

    assert "l2" in result.content
    assert "l1" not in result.content
    assert "offset=3" in result.content


@pytest.mark.asyncio
async def test_read_artifact_rejects_out_of_range_offset(tmp_path: Path) -> None:
    """测试越界 offset 明确报错。"""

    store = _store(tmp_path)
    artifact_id = store.save("l1\nl2\n", session_id="s-1", source_tool="run_command")
    _, handler = create_read_artifact_tool(store)

    with pytest.raises(ValueError):
        await handler(
            ToolCall("c1", "read_artifact", {"artifact_id": artifact_id, "offset": 10})
        )


@pytest.mark.asyncio
async def test_read_artifact_rejects_unknown_id(tmp_path: Path) -> None:
    """测试不存在的 artifact id 明确报错。"""

    store = _store(tmp_path)
    _, handler = create_read_artifact_tool(store)

    with pytest.raises(ValueError):
        await handler(ToolCall("c1", "read_artifact", {"artifact_id": "0" * 16}))

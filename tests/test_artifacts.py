"""测试工具输出纯文本落盘、占位符与 artifact:// 引用。"""

from pathlib import Path

from core.artifacts import (
    ARTIFACT_URL_PREFIX,
    FIREWALL_THRESHOLD_CHARS,
    ArtifactStore,
    apply_output_firewall,
    artifact_placeholder,
    is_artifact_placeholder,
)
from core.tools.output_limits import TRUNCATION_NOTICE


def _store(tmp_path: Path, session_id: str = "s-1") -> ArtifactStore:
    """创建已绑定会话的 store，便于 load/resolve 默认定位。"""

    store = ArtifactStore(tmp_path / "artifacts")
    store.set_session_id(session_id)
    return store


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


def test_artifact_store_saves_plain_text_in_session_directory(tmp_path: Path) -> None:
    """测试落盘为会话子目录下的纯文本文件。"""

    store = _store(tmp_path)
    content = "line-1\nline-2\n"

    artifact_id = store.save(content, session_id="s-1", source_tool="run_command")

    path = tmp_path / "artifacts" / "s-1" / f"{artifact_id}.run_command.txt"
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == content


def test_artifact_store_uses_incrementing_ids(tmp_path: Path) -> None:
    """测试同一会话内 id 按落盘顺序递增。"""

    store = _store(tmp_path)

    first = store.save("a", session_id="s-1", source_tool="run_command")
    second = store.save("b", session_id="s-1", source_tool="run_command")

    assert first == "1"
    assert second == "2"


def test_artifact_store_isolates_sessions(tmp_path: Path) -> None:
    """测试不同会话目录互不干扰。"""

    store = _store(tmp_path)
    store.save("a", session_id="s-1", source_tool="run_command")
    store.save("b", session_id="s-2", source_tool="run_command")

    assert store.load("1", session_id="s-1").content == "a"
    assert store.load("1", session_id="s-2").content == "b"


def test_artifact_store_returns_none_for_unknown_or_invalid_id(tmp_path: Path) -> None:
    """测试未知或非法 id 解析为 None。"""

    store = _store(tmp_path)

    assert store.load("99") is None
    assert store.resolve("99") is None
    assert store.resolve("not-a-number") is None


def test_artifact_placeholder_keeps_head_tail_and_url() -> None:
    """测试占位符保留头尾预览并给出 artifact:// 取回地址。"""

    content = "\n".join(f"line-{index}" for index in range(30))

    placeholder = artifact_placeholder("7", content)

    assert "line-0" in placeholder
    assert "line-29" in placeholder
    assert "line-15" not in placeholder
    assert f"Full output: {ARTIFACT_URL_PREFIX}7" in placeholder
    assert is_artifact_placeholder(placeholder)


def test_is_artifact_placeholder_recognizes_url() -> None:
    """测试占位符识别只认 artifact:// 引用。"""

    assert is_artifact_placeholder(f"prefix {ARTIFACT_URL_PREFIX}3 tail") is True
    assert is_artifact_placeholder("普通工具输出") is False


def test_apply_output_firewall_stores_plain_text_and_returns_url(tmp_path: Path) -> None:
    """测试超阈值输出落盘为纯文本并注入 artifact:// 占位符。"""

    store = _store(tmp_path)
    content = "a" * (FIREWALL_THRESHOLD_CHARS + 1)

    result = apply_output_firewall(
        content, store, session_id="s-1", source_tool="run_command"
    )

    assert is_artifact_placeholder(result)
    assert f"{ARTIFACT_URL_PREFIX}1" in result
    assert store.load("1").content == content
    assert (
        tmp_path / "artifacts" / "s-1" / "1.run_command.txt"
    ).read_text(encoding="utf-8") == content


def test_apply_output_firewall_without_store_uses_hard_truncation() -> None:
    """测试缺少 store 时退化为原有字节/行数硬截断。"""

    content = "a" * 20_000

    result = apply_output_firewall(
        content, None, session_id="s-1", source_tool="run_command"
    )

    assert result.endswith(TRUNCATION_NOTICE)
    assert is_artifact_placeholder(result) is False

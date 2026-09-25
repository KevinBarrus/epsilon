"""大语料只读实验（codex）harness 的范围与哈希口径测试。"""

from pathlib import Path

from evaluation.big_repo_read_compare import (
    EXCLUDED_DIRS,
    MAX_FILE_BYTES,
    codex_scope,
)
from evaluation.read_summary_compare import copy_repository, repository_hash


def _build_source(root: Path) -> Path:
    """建一棵最小仓库：生产代码 / 测试代码 / docs / 生成物。"""

    source = root / "source"
    files = {
        "codex-rs/core/src/client.rs": "fn main() {}\n",
        "codex-rs/core/src/client_tests.rs": "// test\n",
        "codex-rs/core/tests/suite.rs": "// test\n",
        "codex-rs/core/README.md": "# crate\n",
        "codex-rs/core/Cargo.lock": "lock\n",
        "docs/index.md": "# docs\n",
        "scripts/build.sh": "echo build\n",
        "README.md": "# top\n",
        "target/debug/artifact.rs": "// build output\n",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return source


def test_codex_scope_selects_production_code_and_docs(tmp_path: Path) -> None:
    """范围只收生产 Rust、docs/、scripts/ 与顶层 Markdown。"""

    in_scope = [
        "codex-rs/core/src/client.rs",
        "docs/index.md",
        "scripts/build.sh",
        "README.md",
    ]
    out_of_scope = [
        "codex-rs/core/src/client_tests.rs",
        "codex-rs/core/tests/suite.rs",
        "codex-rs/core/README.md",
        "codex-rs/core/Cargo.lock",
        "target/debug/artifact.rs",
    ]

    assert all(codex_scope(Path(p)) for p in in_scope)
    assert not any(codex_scope(Path(p)) for p in out_of_scope)


def test_copy_repository_honours_scope_and_hash_matches(tmp_path: Path) -> None:
    """按范围复制后，原仓库与副本在同一口径下哈希一致。"""

    source = _build_source(tmp_path)
    target = tmp_path / "copy"
    stats = copy_repository(
        source,
        target,
        max_file_bytes=MAX_FILE_BYTES,
        excluded_dirs=EXCLUDED_DIRS,
        skip_tests=True,
        include=codex_scope,
    )

    assert stats["files"] == 4  # client.rs / docs / scripts / 顶层 README
    assert not (target / "codex-rs/core/src/client_tests.rs").exists()
    assert not (target / "target").exists()

    def hash_with_scope(root: Path) -> str:
        return repository_hash(
            root,
            MAX_FILE_BYTES,
            EXCLUDED_DIRS,
            skip_tests=True,
            include=codex_scope,
        )

    assert hash_with_scope(source) == hash_with_scope(target)


def test_hash_differs_when_scope_differs(tmp_path: Path) -> None:
    """换口径会让哈希变化——这正是"原仓库零改动"校验曾经误报的原因。"""

    source = _build_source(tmp_path)

    scoped = repository_hash(
        source, MAX_FILE_BYTES, EXCLUDED_DIRS, skip_tests=True, include=codex_scope
    )
    unscoped = repository_hash(source, MAX_FILE_BYTES, EXCLUDED_DIRS)

    assert scoped != unscoped

"""终端剪贴板输出测试。"""

import base64

from core.clipboard import Osc52Clipboard, copy_text_to_clipboard


def test_osc52_clipboard_keeps_memory_and_encodes_base64(monkeypatch) -> None:
    """测试写入系统剪贴板时保存内存副本并编码为 OSC52 负载。"""

    written: list[str] = []
    monkeypatch.setattr("core.clipboard._write_osc52", written.append)

    clipboard = Osc52Clipboard()
    clipboard.set_text("你好")

    assert clipboard.get_data().text == "你好"
    assert written == [base64.b64encode("你好".encode("utf-8")).decode("ascii")]


def test_copy_text_to_clipboard_encodes_base64(monkeypatch) -> None:
    """测试复制入口使用 OSC52 编码。"""

    written: list[str] = []
    monkeypatch.setattr("core.clipboard._write_osc52", written.append)

    copy_text_to_clipboard("abc")

    assert written == [base64.b64encode(b"abc").decode("ascii")]

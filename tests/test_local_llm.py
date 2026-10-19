"""只测试 HTTP 协议与错误处理，不下载模型或把模拟回答当成推理验证。"""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

MODULE = Path(__file__).resolve().parents[1] / "deploy" / "local-llm" / "chat.py"
SPEC = importlib.util.spec_from_file_location("local_chat", MODULE)
local_chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(local_chat)


def response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


class LocalLLMTests(unittest.TestCase):
    def test_request_and_record(self):
        payload = {"choices": [{"message": {"content": "演示回复"},
                                "finish_reason": "stop"}], "usage": {"total_tokens": 30}}
        with patch.object(local_chat, "urlopen", return_value=response(payload)) as opened:
            record = local_chat.chat("你好")
        request = opened.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(body["model"], "deepseek-r1:1.5b")
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"], [{"role": "user", "content": "你好"}])
        self.assertEqual(record["response"], payload)
        self.assertEqual(record["request"], body)
        self.assertGreaterEqual(record["elapsed_seconds"], 0)

    def test_custom_port_and_budget(self):
        with patch.object(local_chat, "urlopen", return_value=response(
                {"choices": [{"message": {"content": "ok"}}]})) as opened:
            local_chat.chat("hello", base_url="http://127.0.0.1:11435/",
                            timeout=600, max_tokens=4096)
        self.assertEqual(opened.call_args.kwargs["timeout"], 600)
        self.assertEqual(json.loads(opened.call_args.args[0].data)["max_tokens"], 4096)
        self.assertEqual(opened.call_args.args[0].full_url,
                         "http://127.0.0.1:11435/v1/chat/completions")

    def test_http_and_connection_errors(self):
        errors = [HTTPError("http://localhost", 404, "missing", {},
                            io.BytesIO(b"model not found")),
                  URLError("connection refused"), TimeoutError("timed out")]
        for error in errors:
            with self.subTest(error=error):
                with patch.object(local_chat, "urlopen", side_effect=error):
                    with self.assertRaises(RuntimeError):
                        local_chat.chat("test")

    def test_empty_or_invalid_arguments(self):
        for prompt, kwargs in [(" ", {}), ("x", {"timeout": 0}),
                               ("x", {"timeout": float("nan")}),
                               ("x", {"max_tokens": -1}), ("x", {"max_tokens": 1.5})]:
            with self.subTest(kwargs=kwargs):
                with patch.object(local_chat, "urlopen") as opened:
                    with self.assertRaises(ValueError):
                        local_chat.chat(prompt, **kwargs)
                    opened.assert_not_called()

    def test_invalid_response(self):
        for payload in ([], {}, {"choices": []}, {"choices": "bad"},
                        {"choices": [None]}, {"choices": [{}]},
                        {"choices": [{"message": {"content": 7}}]}):
            with patch.object(local_chat, "urlopen", return_value=response(payload)):
                with self.assertRaises(ValueError):
                    local_chat.chat("test")

    def test_existing_output_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "existing.json"
            target.write_text("keep", encoding="utf-8")
            with patch.object(sys, "argv", ["chat.py", "hello", "--output", str(target)]):
                with patch.object(local_chat, "chat") as called:
                    with patch("sys.stderr", new_callable=io.StringIO):
                        with self.assertRaises(SystemExit) as exited:
                            local_chat.main()
                    self.assertEqual(exited.exception.code, 2)
                    called.assert_not_called()
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_reasoning_only_response_is_saved_and_explained(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "record.json"
            payload = {"choices": [{"message": {"content": None, "reasoning": "analysis"},
                                    "finish_reason": "length"}]}
            with patch.object(sys, "argv", ["chat.py", "hello", "--output", str(target)]):
                with patch.object(local_chat, "urlopen", return_value=response(payload)):
                    with patch("sys.stdout", new_callable=io.StringIO) as output:
                        local_chat.main()
            self.assertIn("原始响应", output.getvalue())
            self.assertIn("token", output.getvalue())
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["response"], payload)

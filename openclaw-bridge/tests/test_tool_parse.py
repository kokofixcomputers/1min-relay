"""Тесты парсинга tool_calls из текста модели."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tool_parse import maybe_extract_tool_calls_from_text  # noqa: E402


class TestToolParse(unittest.TestCase):
    def test_plain_json_tool_calls(self):
        payload = {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]}
        text = json.dumps(payload, ensure_ascii=False)
        clean, calls = maybe_extract_tool_calls_from_text(text)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")

    def test_no_tools_plain_text(self):
        clean, calls = maybe_extract_tool_calls_from_text("Hello world")
        self.assertIsNone(calls)
        self.assertIn("Hello", clean)


if __name__ == "__main__":
    unittest.main()

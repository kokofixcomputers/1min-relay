import json
import unittest
from unittest.mock import patch


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class TestMagicArtVariation(unittest.TestCase):
    def test_square_bracket_v4_triggers_variation_for_magic_art(self):
        # Import here so patched symbols bind correctly.
        import app as relay_app

        flask_app = relay_app.app
        client = flask_app.test_client()

        # A prior assistant message containing 4 generated images.
        assistant_content = "\n".join(
            [
                "![Image 1](https://asset.1min.ai/images/a.png) `[_V1_]`",
                "![Image 2](https://asset.1min.ai/images/b.png) `[_V2_]`",
                "![Image 3](https://asset.1min.ai/images/c.png) `[_V3_]`",
                "![Image 4](https://asset.1min.ai/images/2026_04_01_09_41_14_107_520624.png) `[_V4_]`",
            ]
        )

        body = {
            "model": "magic-art_6_1",
            "messages": [
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": "[_V4_]"},
            ],
            "stream": False,
            "n": 1,
        }

        fake_upstream = _FakeResponse(
            200,
            {
                "aiRecord": {
                    "aiRecordDetail": {
                        "resultObject": [
                            "images/var1.png",
                            "images/var2.png",
                            "images/var3.png",
                            "images/var4.png",
                        ]
                    }
                }
            },
        )

        with patch("routes.text.api_request", return_value=fake_upstream) as mock_api:
            resp = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer test-key"},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, dict)
        content = data["choices"][0]["message"]["content"]
        self.assertIn("Variation", content)

        # Ensure we called upstream as IMAGE_VARIATOR (not IMAGE_GENERATOR redirect).
        self.assertTrue(mock_api.called)
        _args, kwargs = mock_api.call_args
        self.assertEqual(kwargs["json"]["type"], "IMAGE_VARIATOR")

    def test_v_tag_without_history_uses_last_images_cache(self):
        import app as relay_app

        flask_app = relay_app.app
        client = flask_app.test_client()

        body = {
            "model": "magic-art_6_1",
            # No assistant history, only the tag
            "messages": [{"role": "user", "content": "[_V1_]"}],
            "stream": False,
            "n": 1,
        }

        cached = {
            "urls": [
                "https://asset.1min.ai/images/2026_04_01_11_01_14_544_702899.png",
                "https://asset.1min.ai/images/2026_04_01_11_01_14_532_723034.png",
                "https://asset.1min.ai/images/2026_04_01_11_01_14_597_908046.png",
                "https://asset.1min.ai/images/2026_04_01_11_01_14_582_309205.png",
            ],
            "created": 0,
        }

        fake_upstream = _FakeResponse(
            200,
            {
                "aiRecord": {
                    "aiRecordDetail": {
                        "resultObject": [
                            "images/var1.png",
                            "images/var2.png",
                            "images/var3.png",
                            "images/var4.png",
                        ]
                    }
                }
            },
        )

        def fake_memcache_get(op, key, value=None, expiry=None):
            if op != "get":
                return None
            if key.startswith("last_images:"):
                return cached
            return None

        with patch("routes.text.safe_memcached_operation", side_effect=fake_memcache_get), patch(
            "routes.text.api_request", return_value=fake_upstream
        ) as mock_api:
            resp = client.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer test-key"},
            )

        self.assertEqual(resp.status_code, 200)
        _args, kwargs = mock_api.call_args
        self.assertEqual(kwargs["json"]["type"], "IMAGE_VARIATOR")


if __name__ == "__main__":
    unittest.main()


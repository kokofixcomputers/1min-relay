import unittest

from utils.model_mapping import (
    canonicalize_image_model_name,
    choose_variator_model,
    DEFAULT_VARIATOR_FALLBACK,
)


class TestImageModelMapping(unittest.TestCase):
    def test_canonicalize_ui_flux_names(self):
        self.assertEqual(
            canonicalize_image_model_name("Flux Schnell - Black Forest Labs"),
            "black-forest-labs/flux-schnell",
        )
        self.assertEqual(
            canonicalize_image_model_name("Flux Redux Schnell - Black Forest Labs"),
            "flux-redux-schnell",
        )

    def test_canonicalize_magic_art_names(self):
        self.assertEqual(canonicalize_image_model_name("Magic Art 6.1"), "magic-art_6_1")
        self.assertEqual(canonicalize_image_model_name("Magic Art 7.0"), "magic-art_7_0")

    def test_choose_variator_model_maps_generator_to_redux(self):
        supported = {"flux-redux-schnell", "flux-redux-dev", "magic-art_6_1"}
        self.assertEqual(
            choose_variator_model("Flux Schnell - Black Forest Labs", supported_variators=supported),
            "flux-redux-schnell",
        )

    def test_choose_variator_model_openai_images_fallback_to_dalle2(self):
        supported = {"dall-e-2", "flux-redux-schnell"}
        self.assertEqual(
            choose_variator_model("dall-e-3", supported_variators=supported),
            "dall-e-2",
        )
        self.assertEqual(
            choose_variator_model("gpt-image-1", supported_variators=supported),
            "dall-e-2",
        )

    def test_choose_variator_model_magic_art_keeps_version(self):
        supported = {"magic-art", "magic-art_6_1", "magic-art_7_0", "flux-redux-schnell"}
        self.assertEqual(
            choose_variator_model("Magic Art 6.1", supported_variators=supported),
            "magic-art_6_1",
        )
        self.assertEqual(
            choose_variator_model("midjourney_7_0", supported_variators=supported),
            "magic-art_7_0",
        )

    def test_choose_variator_model_fallback_when_unsupported(self):
        supported = {DEFAULT_VARIATOR_FALLBACK}
        self.assertEqual(
            choose_variator_model("Some Unknown Image Model", supported_variators=supported),
            DEFAULT_VARIATOR_FALLBACK,
        )


if __name__ == "__main__":
    unittest.main()


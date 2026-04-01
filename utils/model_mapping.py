import re
from typing import Optional


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"[-/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# UI-facing names can differ between Image Generator and Image Variator in 1min.ai.
# This mapping makes the relay resilient to those naming differences.
_UI_TO_CANONICAL = {
    # Magic Art (generator & variator)
    _norm("Magic Art 5.2"): "magic-art",
    _norm("Magic Art"): "magic-art",
    _norm("Magic Art 6.1"): "magic-art_6_1",
    _norm("Magic Art 7.0"): "magic-art_7_0",
    # Midjourney legacy names (mapped to Magic Art in this relay)
    _norm("Midjourney"): "midjourney",
    _norm("Midjourney 6.1"): "midjourney_6_1",
    _norm("Midjourney 5.2"): "midjourney_5_2",
    _norm("Midjourney 7.0"): "midjourney_7_0",
    # Flux generator names
    _norm("Flux Schnell - Black Forest Labs"): "black-forest-labs/flux-schnell",
    _norm("Flux Dev - Black Forest Labs"): "black-forest-labs/flux-dev",
    _norm("Flux Pro - Black Forest Labs"): "black-forest-labs/flux-pro",
    _norm("Flux 1.1 Pro - Black Forest Labs"): "black-forest-labs/flux-1.1-pro",
    _norm("Flux 1.1 Pro Ultra - Black Forest Labs"): "black-forest-labs/flux-1.1-pro-ultra",
    # Flux variator names (Redux)
    _norm("Flux Redux Schnell - Black Forest Labs"): "flux-redux-schnell",
    _norm("Flux Redux Dev - Black Forest Labs"): "flux-redux-dev",
    # Common shortened forms people copy/paste
    _norm("flux schnell"): "black-forest-labs/flux-schnell",
    _norm("flux dev"): "black-forest-labs/flux-dev",
    _norm("flux pro"): "black-forest-labs/flux-pro",
    _norm("flux redux schnell"): "flux-redux-schnell",
    _norm("flux redux dev"): "flux-redux-dev",
}


GENERATOR_TO_VARIATOR = {
    # Flux generator -> Flux Redux variator
    "flux-schnell": "flux-redux-schnell",
    "black-forest-labs/flux-schnell": "flux-redux-schnell",
    "flux-dev": "flux-redux-dev",
    "black-forest-labs/flux-dev": "flux-redux-dev",
    "black-forest-labs/flux-pro": "flux-redux-dev",
    "flux-pro": "flux-redux-dev",
    "black-forest-labs/flux-1.1-pro": "flux-redux-dev",
    "flux-1.1-pro": "flux-redux-dev",
    "black-forest-labs/flux-1.1-pro-ultra": "flux-redux-dev",
    "flux-1.1-pro-ultra": "flux-redux-dev",
}


DEFAULT_VARIATOR_FALLBACK = "flux-redux-schnell"

_OPENAI_IMAGE_GENERATORS = {
    "dall-e-3",
    "dall-e-2",
    "gpt-image-1",
    "gpt-image-1-mini",
}


def _best_effort_vendor_variator(model: str, supported_variators: set[str]) -> Optional[str]:
    """
    Pick the closest variator model by family/vendor, even if names differ.
    Priority:
    - Flux family -> Flux Redux (Black Forest Labs)
    - Magic Art / Midjourney family -> Magic Art
    - OpenAI image family -> dall-e-2 (only OpenAI variator we support)
    - Models that already have a variator (dzine/recraft) are handled earlier
    - Otherwise: DEFAULT_VARIATOR_FALLBACK
    """
    if not model:
        return None

    ml = str(model).strip().lower()

    # --------------------
    # Black Forest Labs / Flux family
    # --------------------
    if "flux" in ml:
        mapped = GENERATOR_TO_VARIATOR.get(model) or GENERATOR_TO_VARIATOR.get(canonicalize_image_model_name(model))
        if mapped and mapped in supported_variators:
            return mapped
        # Prefer Schnell for "schnell" generators, else Dev.
        if "schnell" in ml and "flux-redux-schnell" in supported_variators:
            return "flux-redux-schnell"
        if "flux-redux-dev" in supported_variators:
            return "flux-redux-dev"
        if "flux-redux-schnell" in supported_variators:
            return "flux-redux-schnell"

    # --------------------
    # Magic Art / Midjourney family (1min.ai migrated MJ -> Magic Art)
    # --------------------
    if "magic art" in ml or "magic-art" in ml or "midjourney" in ml:
        # Preserve version when possible.
        if ("6.1" in ml or "_6_1" in ml) and "magic-art_6_1" in supported_variators:
            return "magic-art_6_1"
        if ("7.0" in ml or "_7_0" in ml) and "magic-art_7_0" in supported_variators:
            return "magic-art_7_0"
        if "magic-art" in supported_variators:
            return "magic-art"

    # --------------------
    # OpenAI image family: only dall-e-2 variations supported
    # --------------------
    if model in _OPENAI_IMAGE_GENERATORS or "dall-e" in ml or "gpt-image" in ml:
        if "dall-e-2" in supported_variators:
            return "dall-e-2"

    # --------------------
    # Explicit known families that have variators
    # --------------------
    if "dzine" in ml and "dzine" in supported_variators:
        return "dzine"
    if "recraft" in ml and "recraft" in supported_variators:
        return "recraft"

    # Otherwise: no close match
    return None


def canonicalize_image_model_name(name: str) -> str:
    """
    Best-effort canonicalization of user-provided model strings.
    Returns the original string if we can't improve it.
    """
    if not isinstance(name, str):
        return ""
    raw = name.strip()
    if not raw:
        return ""
    n = _norm(raw)
    mapped = _UI_TO_CANONICAL.get(n)
    return mapped or raw


def choose_variator_model(generator_or_user_model: str, *, supported_variators: set[str]) -> Optional[str]:
    """
    Pick a safe variator model given a requested model name.
    - Normalizes UI names
    - Applies generator->variator mapping (e.g. Flux Schnell -> Flux Redux Schnell)
    - Falls back to DEFAULT_VARIATOR_FALLBACK if unsupported
    """
    m = canonicalize_image_model_name(generator_or_user_model)
    # direct support
    if m in supported_variators:
        return m
    # vendor/family closest match
    vendor_pick = _best_effort_vendor_variator(m, supported_variators)
    if vendor_pick and vendor_pick in supported_variators:
        return vendor_pick
    # try generator->variator
    mapped = GENERATOR_TO_VARIATOR.get(m)
    if mapped and mapped in supported_variators:
        return mapped
    # also try canonicalized mapping key
    mapped = GENERATOR_TO_VARIATOR.get(canonicalize_image_model_name(m))
    if mapped and mapped in supported_variators:
        return mapped
    # fallback
    if DEFAULT_VARIATOR_FALLBACK in supported_variators:
        return DEFAULT_VARIATOR_FALLBACK
    return None


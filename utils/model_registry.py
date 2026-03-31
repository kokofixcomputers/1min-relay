import json
import time
from dataclasses import dataclass

from utils.logger import logger
from utils.common import api_request
from utils.constants import ONE_MIN_API_URL
from utils.memcached import safe_memcached_operation


@dataclass(frozen=True)
class ModelRegistryData:
    model_ids: list[str]
    vision_model_ids: list[str]
    image_model_ids: list[str]
    speech_model_ids: list[str]
    code_interpreter_model_ids: list[str]
    fetched_at: float
    source: str  # "upstream" | "cache" | "fallback"


_MEM_TTL_SECONDS = 300  # 5 minutes
_MEM_CACHE: ModelRegistryData | None = None
_MEM_CACHE_EXPIRES_AT: float = 0.0

_MEMCACHED_KEY = "model_registry:v1"
_MEMCACHED_TTL_SECONDS = 3600  # 1 hour


def _now() -> float:
    return time.time()


def _get_upstream_models(feature: str, api_key: str, request_id: str) -> list[dict]:
    # 1min.ai doesn't expose a stable public models endpoint in this project;
    # we query the features endpoint with a feature filter similar to worker implementations.
    url = f"{ONE_MIN_API_URL}?feature={feature}"
    resp = api_request(
        "GET",
        url,
        headers={"API-KEY": api_key},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"[{request_id}] Upstream models fetch failed: {resp.status_code}")
    data = resp.json()
    models = data.get("models")
    if not isinstance(models, list):
        raise RuntimeError(f"[{request_id}] Upstream models response missing 'models' array")
    return models


def _extract_model_ids(models: list[dict]) -> list[str]:
    ids: list[str] = []
    for m in models:
        if isinstance(m, dict):
            mid = m.get("modelId") or m.get("id") or m.get("model")
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
    # preserve order but deduplicate
    seen: set[str] = set()
    out: list[str] = []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def _extract_featured_model_ids(models: list[dict], feature: str) -> list[str]:
    out: list[str] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        feats = m.get("features")
        if isinstance(feats, list) and feature in feats:
            mid = m.get("modelId") or m.get("id") or m.get("model")
            if isinstance(mid, str) and mid.strip():
                out.append(mid.strip())
    # dedupe
    seen: set[str] = set()
    deduped: list[str] = []
    for mid in out:
        if mid not in seen:
            seen.add(mid)
            deduped.append(mid)
    return deduped


def _cache_set_mem(data: ModelRegistryData) -> None:
    global _MEM_CACHE, _MEM_CACHE_EXPIRES_AT
    _MEM_CACHE = data
    _MEM_CACHE_EXPIRES_AT = _now() + _MEM_TTL_SECONDS


def _cache_get_mem() -> ModelRegistryData | None:
    if _MEM_CACHE and _now() < _MEM_CACHE_EXPIRES_AT:
        return _MEM_CACHE
    return None


def _cache_get_memcached() -> ModelRegistryData | None:
    raw = safe_memcached_operation("get", _MEMCACHED_KEY)
    if not raw:
        return None
    try:
        # safe_memcached_operation('get') already tries to JSON-decode bytes.
        obj = raw
        if not isinstance(obj, dict):
            return None
        return ModelRegistryData(
            model_ids=list(obj.get("model_ids") or []),
            vision_model_ids=list(obj.get("vision_model_ids") or []),
            image_model_ids=list(obj.get("image_model_ids") or []),
            speech_model_ids=list(obj.get("speech_model_ids") or []),
            code_interpreter_model_ids=list(obj.get("code_interpreter_model_ids") or []),
            fetched_at=float(obj.get("fetched_at") or 0.0),
            source="cache",
        )
    except Exception:
        return None


def _cache_set_memcached(data: ModelRegistryData) -> None:
    try:
        payload = {
            "model_ids": data.model_ids,
            "vision_model_ids": data.vision_model_ids,
            "image_model_ids": data.image_model_ids,
            "speech_model_ids": data.speech_model_ids,
            "code_interpreter_model_ids": data.code_interpreter_model_ids,
            "fetched_at": data.fetched_at,
        }
        safe_memcached_operation("set", _MEMCACHED_KEY, payload, expiry=_MEMCACHED_TTL_SECONDS)
    except Exception:
        pass


def get_model_registry_data(*, api_key: str, request_id: str, fallback_model_ids: list[str]) -> ModelRegistryData:
    """
    Fetch model list dynamically from upstream when possible.
    Caching order: in-memory (5m) -> memcached (1h) -> upstream -> fallback.
    """
    mem = _cache_get_mem()
    if mem:
        return mem

    cached = _cache_get_memcached()
    if cached and cached.model_ids:
        _cache_set_mem(cached)
        return cached

    try:
        chat_models = _get_upstream_models("UNIFY_CHAT_WITH_AI", api_key, request_id)
        image_models = _get_upstream_models("IMAGE_GENERATOR", api_key, request_id)
        speech_models = _get_upstream_models("SPEECH_TO_TEXT", api_key, request_id)

        chat_ids = _extract_model_ids(chat_models)
        image_ids = _extract_model_ids(image_models)
        speech_ids = _extract_model_ids(speech_models)

        vision_ids = _extract_featured_model_ids(chat_models, "CHAT_WITH_IMAGE")
        code_ids = _extract_featured_model_ids(chat_models, "CODE_GENERATOR")

        # Union of all model IDs (chat first, then image, then speech)
        all_ids: list[str] = []
        for bucket in (chat_ids, image_ids, speech_ids):
            for mid in bucket:
                if mid not in all_ids:
                    all_ids.append(mid)

        data = ModelRegistryData(
            model_ids=all_ids,
            vision_model_ids=vision_ids,
            image_model_ids=image_ids,
            speech_model_ids=speech_ids,
            code_interpreter_model_ids=code_ids,
            fetched_at=_now(),
            source="upstream",
        )
        _cache_set_mem(data)
        _cache_set_memcached(data)
        return data
    except Exception as e:
        logger.warning("[%s] Dynamic model registry failed, using fallback: %s", request_id, str(e))
        data = ModelRegistryData(
            model_ids=fallback_model_ids,
            vision_model_ids=[],
            image_model_ids=[],
            speech_model_ids=[],
            code_interpreter_model_ids=[],
            fetched_at=_now(),
            source="fallback",
        )
        _cache_set_mem(data)
        return data


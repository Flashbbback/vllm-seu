# SPDX-License-Identifier: Apache-2.0
"""Cache for transformers' per-call kwargs TypedDict validation.

transformers 5.x validates processor kwargs on every ``ProcessorMixin``
call via ``huggingface_hub.dataclasses.validate_typed_dict``. The schema
is often a *dynamically merged* TypedDict built per call, which defeats
huggingface_hub's identity-keyed ``lru_cache`` and costs ~5ms per request
on weak CPUs (``make_dataclass`` + ``strict`` rebuild). This module patches
``_build_strict_cls_from_typed_dict`` with a fingerprint-keyed cache
(schema name + ``__total__`` + annotations), which is semantics-preserving
because TypedDict structure fully determines the built dataclass.
"""

from __future__ import annotations

_installed = False


def install_kwargs_validation_cache() -> None:
    """Patch huggingface_hub's strict-class builder with a fingerprint cache.

    Idempotent and best-effort: if huggingface_hub internals change, the
    patch is skipped silently and behavior falls back to upstream.
    """
    global _installed
    if _installed:
        return
    _installed = True
    try:
        import huggingface_hub.dataclasses as hdc

        impl = hdc._build_strict_cls_from_typed_dict
        if getattr(impl, "_vllm_fp_patched", False):
            return

        cache: dict = {}

        def fingerprint_build(schema):
            try:
                key = (
                    schema.__name__,
                    getattr(schema, "__total__", True),
                    tuple(
                        sorted((k, str(v)) for k, v in schema.__annotations__.items())
                    ),
                )
            except Exception:
                return impl(schema)
            if key not in cache:
                cache[key] = impl(schema)
            return cache[key]

        fingerprint_build._vllm_fp_patched = True  # type: ignore[attr-defined]
        hdc._build_strict_cls_from_typed_dict = fingerprint_build
    except Exception:
        # huggingface_hub missing or internals changed: keep upstream behavior
        pass

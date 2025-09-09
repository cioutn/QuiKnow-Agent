"""轻量级统计工具：字符与 token 估算。

优先使用 tiktoken；缺失则采用 (len(chars)/3.7) 近似。
"""
from __future__ import annotations
from typing import Optional

_ENC_CACHE = {}

def _get_encoder(model: str):  # pragma: no cover (动态依赖)
    if model in _ENC_CACHE:
        return _ENC_CACHE[model]
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = None
    _ENC_CACHE[model] = enc
    return enc

def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:  # pragma: no cover
            pass
    # 经验近似：英文 ~4chars/中文 ~2chars；混合场景用 3.7
    return int(len(text) / 3.7) + 1

def summarize_blocks(blocks: list[str], model: str) -> dict:
    total_chars = sum(len(b) for b in blocks)
    total_tokens = sum(estimate_tokens(b, model) for b in blocks)
    return {"chars": total_chars, "tokens": total_tokens, "blocks": len(blocks)}

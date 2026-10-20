"""Groq chat call that returns parsed JSON. Kept small on purpose — swapping
providers should never be more than this file."""

import json
import re
import time

from specrag.config import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RATE_LIMIT_WAIT,
    LLM_TEMPERATURE,
)

_CLIENT = None
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
# "Please try again in 967.5ms" / "in 12.3s" / "in 1m2.5s"
# The minutes group needs a digit after the "m", or "300ms" reads as 300 minutes.
_RETRY_IN = re.compile(r"try again in (?:(\d+)m(?=\d))?(?:([\d.]+)s)?(?:([\d.]+)ms)?")


def get_client():
    global _CLIENT
    if _CLIENT is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env, or use --mode rules.")
        from groq import Groq

        _CLIENT = Groq(api_key=GROQ_API_KEY)
    return _CLIENT


def _parse(text: str) -> dict:
    """JSON mode should return pure JSON; this also survives a model that
    wraps it in prose or a ```json fence anyway."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if match:
            return json.loads(match.group(0))
        raise


def _retry_after(exc) -> float:
    """Seconds Groq asked us to wait, from the error message; 5 s if absent."""
    match = _RETRY_IN.search(str(exc))
    if not match or not any(match.groups()):
        return 5.0
    minutes, seconds, millis = match.groups()
    return int(minutes or 0) * 60 + float(seconds or 0) + float(millis or 0) / 1000


def _human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def _create(client, kwargs):
    """One request, waiting out rate limits.

    Groq's free tier allows 8,000 tokens per minute for gpt-oss-120b and one
    extraction call is ~2,500 tokens, so only about three calls fit in a
    minute. The SDK's own retry gives up after two quick attempts — that is
    what crashed the first full eval run. Here we sleep for as long as Groq
    says, plus a margin, until LLM_RATE_LIMIT_WAIT seconds have been spent.
    """
    from groq import RateLimitError

    waited = 0.0
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            delay = _retry_after(exc) + 1.0
            if waited + delay > LLM_RATE_LIMIT_WAIT:
                # Too long to wait out — almost always the DAILY token cap
                # (free tier: 200k tokens/day for gpt-oss-120b, ~70 queries).
                # Raised as RuntimeError so the CLI prints one clear line and
                # the API answers 503 with this message, not a traceback.
                per_day = "per day" in str(exc) or "TPD" in str(exc)
                limit = "daily token limit" if per_day else "rate limit"
                raise RuntimeError(
                    f"Groq {limit} reached for {kwargs['model']}. It frees up in about "
                    f"{_human(delay)}. Use --mode rules meanwhile, or set SPECRAG_LLM_MODEL "
                    "to another Groq model (each model has its own quota)."
                ) from exc
            time.sleep(delay)
            waited += delay


def chat_json(system: str, user: str, model: str = LLM_MODEL) -> dict:
    """One completion in JSON mode -> parsed dict."""
    from groq import BadRequestError

    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    # gpt-oss is a reasoning model. Extraction from a handful of excerpts does
    # not need long deliberation, and "low" cuts latency a lot.
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"

    client = get_client()
    try:
        response = _create(client, kwargs)
    except (BadRequestError, TypeError):
        # Some models reject JSON mode or reasoning_effort (BadRequestError),
        # and older groq SDKs do not know the reasoning_effort kwarg at all
        # (TypeError). Retry plain and rely on _parse; the prompt still
        # demands JSON.
        kwargs.pop("response_format", None)
        kwargs.pop("reasoning_effort", None)
        response = _create(client, kwargs)

    return _parse(response.choices[0].message.content)

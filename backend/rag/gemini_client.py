"""Direct HTTP client for the Gemini API (no SDK), with retries and model fallback."""
import os
import time

import requests

from rag.tracing import _get_langfuse_client

# Google periodically deprecates/renames Gemini models. Instead of hardcoding
# a single name, we try several candidates and cache whichever one works for
# subsequent calls.
GEMINI_MODEL_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest",
]
_working_model_name = None  # whichever model succeeds gets cached here

GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_OUTPUT_TOKENS = 4096  # generous ceiling so real answers don't get cut off,
                                 # but still a real, reportable model parameter


class GeminiResponseError(Exception):
    """Raised when Gemini returns a 200 OK but the response has no usable
    answer text (e.g. blocked by a safety filter, or hit MAX_TOKENS before
    producing any content). Caught in _call_gemini() so it's treated the same
    as a failed model candidate -- the next candidate model is tried instead
    of crashing the request with a raw KeyError."""
    pass


def _post_to_gemini(model_name, api_key, prompt, max_retries=2):
    """
    Calls Gemini with one specific model name, retrying on temporary errors.
    Returns (answer_text, usage_dict) where usage_dict has input/output/total tokens.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": GEMINI_TEMPERATURE,
                        "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            candidates = data.get("candidates") or []
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason")
                raise GeminiResponseError(
                    f"no candidates in response (blockReason={block_reason}, raw={str(data)[:300]})"
                )

            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts")
            if not parts:
                finish_reason = candidate.get("finishReason")
                raise GeminiResponseError(
                    f"no content parts in response (finishReason={finish_reason}, raw={str(candidate)[:300]})"
                )

            answer_text = parts[0].get("text", "")
            if not answer_text:
                raise GeminiResponseError(f"empty text in response part (raw={str(parts[0])[:300]})")

            usage = data.get("usageMetadata", {})
            token_usage = {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            }
            return answer_text, token_usage
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_error = e
            if status in (429, 500, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise  # 404, 401, 400 etc. -> raise immediately (retrying won't help)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # Network slow or connection dropped, wait a bit and retry
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue
        except GeminiResponseError:
            raise  # malformed/blocked response -- retrying the same model won't help
    raise last_error


def _call_gemini(prompt, trace_name="gemini-call", metadata=None, langfuse_prompt=None):
    """
    Calls the Gemini REST API directly over HTTP (no SDK, to avoid
    protobuf/tensorflow version conflicts).
    Returns None if GEMINI_API_KEY is not set.
    Model names change over time (Google deprecates them), so we try
    several candidates until one works.
    Logs the call (model, prompt, answer, token usage, temperature,
    max_completion_tokens, and any extra `metadata` passed in -- e.g. mode,
    selected_sources) to Langfuse if LANGFUSE_PUBLIC_KEY is configured in
    .env; otherwise this is skipped. This is what populates the Preview
    panel's metadata table in the Langfuse UI.
    `langfuse_prompt` (optional): the prompt object returned by
    prompt_registry.get_prompt(). When provided, it's linked to this
    generation so the Langfuse UI shows which prompt version produced it.
    """
    global _working_model_name
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    langfuse = _get_langfuse_client()

    # Try whichever model worked last time first (if any)
    models_to_try = ([_working_model_name] if _working_model_name else []) + \
        [m for m in GEMINI_MODEL_CANDIDATES if m != _working_model_name]

    last_error = None
    for model_name in models_to_try:
        try:
            if langfuse:
                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name=trace_name,
                    model=model_name,
                    input=prompt,
                    model_parameters={
                        "temperature": GEMINI_TEMPERATURE,
                        "max_completion_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                    },
                    metadata=metadata or {},
                    prompt=langfuse_prompt,
                ) as generation:
                    answer, token_usage = _post_to_gemini(model_name, api_key, prompt)
                    generation.update(output=answer, usage_details=token_usage)
                langfuse.flush()  # send the trace to Langfuse right away
            else:
                answer, token_usage = _post_to_gemini(model_name, api_key, prompt)

            _working_model_name = model_name  # cache this model for next time
            return answer
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else None
            body = e.response.text[:300] if e.response is not None else str(e)
            print(f"[Gemini] Model '{model_name}' failed (status={status}): {body}")
            if status in (401, 403, 400):
                raise  # auth/bad-request issues, trying another model won't help
            continue  # 404 (model not found) or 429/500/503 (overloaded) -> try next model
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            print(f"[Gemini] Model '{model_name}' timeout/connection error: {e}")
            continue  # network slow/unstable -> try next model
        except GeminiResponseError as e:
            last_error = e
            print(f"[Gemini] Model '{model_name}' returned an unusable response: {e}")
            continue  # blocked/empty response -> try next model

    # None of the candidate models worked
    raise last_error


def _safe_call_gemini(prompt, trace_name, metadata, langfuse_prompt):
    """
    Wraps _call_gemini() so a total failure (all candidate models timed out,
    or all returned unusable/blocked responses) becomes a friendly message
    instead of an unhandled exception crashing the Flask request with a 500.
    Returns None only when GEMINI_API_KEY isn't set at all (same as before).
    """
    try:
        return _call_gemini(prompt, trace_name=trace_name, metadata=metadata, langfuse_prompt=langfuse_prompt)
    except Exception as e:
        print(f"[Gemini] All candidate models failed for '{trace_name}': {e}")
        return ("Sorry, I couldn't reach the AI model right now (it may be slow or "
                "temporarily unavailable). Please try again in a moment.")

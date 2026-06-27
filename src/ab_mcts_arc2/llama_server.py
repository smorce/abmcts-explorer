from __future__ import annotations

import asyncio
import os
import random
import threading
from dataclasses import dataclass, replace
from typing import Any, Optional


def _ensure_openai_v1_base_url(raw: str) -> str:
    value = (raw or "").strip() or "http://127.0.0.1:1067"
    if "://" not in value:
        value = f"http://{value}"
    value = value.rstrip("/")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def _without_v1_suffix(base_url: str) -> str:
    value = _ensure_openai_v1_base_url(base_url)
    return value[:-3] if value.endswith("/v1") else value


def _get_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _get_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _should_retry(exc: Exception) -> bool:
    try:
        from openai import APIConnectionError, APIStatusError, RateLimitError
    except Exception:
        APIConnectionError = APIStatusError = RateLimitError = None  # type: ignore[assignment]

    if APIConnectionError and isinstance(exc, APIConnectionError):
        return True
    if RateLimitError and isinstance(exc, RateLimitError):
        return True
    if APIStatusError and isinstance(exc, APIStatusError):
        return bool(getattr(exc, "status_code", 0) >= 500)
    if isinstance(exc, ConnectionError):
        return True
    return False


def messages_to_prompt(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


async def call_llama_server(
    client: Any,
    model: str,
    prompt: str,
    *,
    enable_thinking: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    min_p: Optional[float],
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    retry_base_delay_seconds: float,
    retry_max_delay_seconds: float,
) -> str:
    def _run() -> str:
        if enable_thinking:
            defaults = {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        else:
            defaults = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        opts = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }
        extra_body = {key: value if value is not None else defaults[key] for key, value in opts.items()}
        extra_body["chat_template_kwargs"] = defaults["chat_template_kwargs"]

        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=extra_body["temperature"],
            top_p=extra_body["top_p"],
            max_output_tokens=max_tokens,
            extra_body={
                "top_k": extra_body["top_k"],
                "min_p": extra_body["min_p"],
                "chat_template_kwargs": extra_body["chat_template_kwargs"],
            },
        )

        return response.output_text or ""

    attempts = max(1, max_retries)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            last_exc = RuntimeError("llama-server request timeout")
            retryable = True
        except Exception as exc:
            last_exc = exc
            retryable = _should_retry(exc)

        if not retryable or attempt >= attempts:
            break

        exp_backoff = retry_base_delay_seconds * (2 ** (attempt - 1))
        sleep_seconds = min(exp_backoff, retry_max_delay_seconds) + random.uniform(0, 0.2)
        await asyncio.sleep(sleep_seconds)

    if last_exc is None:
        raise RuntimeError("llama-server request failed")
    raise RuntimeError(f"llama-server request failed after retries: {last_exc}") from last_exc


@dataclass(frozen=True)
class LlamaServerEnvConfig:
    base_url_no_v1: str
    api_key: str
    model: str
    enable_thinking: bool
    temperature: float
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    retry_base_delay_seconds: float
    retry_max_delay_seconds: float

    @classmethod
    def from_env(cls) -> LlamaServerEnvConfig:
        resolved = os.getenv("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:1067")
        base_no_v1 = _without_v1_suffix(resolved)
        return cls(
            base_url_no_v1=base_no_v1,
            api_key=os.getenv("LLAMA_SERVER_API_KEY", "sk-local-no-key-required"),
            model=os.getenv("LLM_MODEL", "unsloth/Qwen3.6-27B-MTP-GGUF-UD-Q4_K_XL"),
            enable_thinking=os.getenv("LLAMA_SERVER_ENABLE_THINKING", "false").lower()
            in ("true", "1", "yes"),
            temperature=float(os.getenv("LLAMA_SERVER_TEMPERATURE", "0.3")),
            top_p=_get_optional_float(os.getenv("LLAMA_SERVER_TOP_P")),
            top_k=_get_optional_int(os.getenv("LLAMA_SERVER_TOP_K")),
            min_p=_get_optional_float(os.getenv("LLAMA_SERVER_MIN_P")),
            max_tokens=int(os.getenv("LLAMA_SERVER_MAX_TOKENS", "1000")),
            timeout_seconds=float(os.getenv("LLAMA_SERVER_TIMEOUT_SECONDS", "250")),
            max_retries=int(os.getenv("LLAMA_SERVER_MAX_RETRIES", "5")),
            retry_base_delay_seconds=float(os.getenv("LLAMA_SERVER_RETRY_BASE_SECONDS", "0.8")),
            retry_max_delay_seconds=float(os.getenv("LLAMA_SERVER_RETRY_MAX_SECONDS", "8.0")),
        )

    def build_openai_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            base_url=_ensure_openai_v1_base_url(self.base_url_no_v1),
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    async def complete(self, prompt: str) -> str:
        client = self.build_openai_client()
        return await call_llama_server(
            client,
            self.model,
            prompt,
            enable_thinking=self.enable_thinking,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_base_delay_seconds=self.retry_base_delay_seconds,
            retry_max_delay_seconds=self.retry_max_delay_seconds,
        )

    def with_overrides(
        self,
        *,
        model: str | None = None,
        base_url_no_v1: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        enable_thinking: bool | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> LlamaServerEnvConfig:
        return replace(
            self,
            model=model or self.model,
            base_url_no_v1=(
                _without_v1_suffix(base_url_no_v1)
                if base_url_no_v1 is not None
                else self.base_url_no_v1
            ),
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            timeout_seconds=(
                timeout_seconds if timeout_seconds is not None else self.timeout_seconds
            ),
            enable_thinking=(
                enable_thinking if enable_thinking is not None else self.enable_thinking
            ),
            top_p=top_p if top_p is not None else self.top_p,
            top_k=top_k if top_k is not None else self.top_k,
            min_p=min_p if min_p is not None else self.min_p,
        )


def run_llama_server_completion_sync(config: LlamaServerEnvConfig, prompt: str) -> str:
    async def _complete() -> str:
        return await config.complete(prompt)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_complete())

    result: str | None = None
    error: BaseException | None = None

    def _worker() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(_complete())
        except BaseException as exc:
            error = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result or ""

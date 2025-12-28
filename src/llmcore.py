from __future__ import annotations
"""
narremgen.core
=============
Central LLM routing layer across local and remote providers.

Defines the `LLMConnect` router, which normalizes chat calls across providers,
handles client construction, retries with backoff, provider guessing, token
estimation, and prompt flattening for OpenAI-compatible and HTTP-style APIs.
"""

from typing import Any, Dict, List, Optional, Tuple
import os, time, requests, threading, re
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
logger_ = logger.info

try:
    # OpenAI serves a client to other providers (changing base_url)
    # Add here other client package if required
    from openai import OpenAIError
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None
    OpenAIError = Exception

try:
    # SDK Officiel Google 2025
    from google import genai as GoogleGenAI
    from google.genai import types
except ImportError:
    GoogleGenAI = None
    types = None

Message = Dict[str, str]  # {"role": "...", "content": "..."}


class LLMCancellation(RuntimeError):
    """Raised when an LLM call is cancelled mid-flight."""


@dataclass
class DryTestResult:
    provider_model: str
    provider: str
    model: str
    ok: bool
    message: str
    status_code: Optional[int] = None
    error_code: Optional[str] = None

class LLMConnect:
    """
    Thin provider-agnostic wrapper for chat-based LLM calls.

    The class normalizes chat calls across providers, handles retries, and offers
    a small set of helper methods intended for both CLI and GUI usage.

    Example
    -------
    llm = LLMConnect()
    reply = llm.safe_chat_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    Notes
    -----
    Providers and authentication are configured elsewhere; this wrapper focuses on
    runtime call behavior (formatting, retries, and error handling).
    """

    _global_instance: "LLMConnect | None" = None
    _cancel_event: Optional[threading.Event] = None

    def __init__(
        self,
        llmodels: Optional[Dict[str, str]] = None,
        default_model: str = "openai\\gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        request_timeout: int = 60,
        backoff_factor: float = 1.0,
    ) -> None:
        self.default_model = default_model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.request_timeout = int(request_timeout)
        self.backoff_factor = float(backoff_factor)
        if self.backoff_factor<0.0: self.backoff_factor=0.5
        if self.backoff_factor>3.0: self.backoff_factor=3.0

        self.llmodels: Dict[str, str] = dict(llmodels) if llmodels is not None else {}

        self._openai_client = self._build_openai_client()
        self._openrouter_client = self._build_openrouter_client() #use for gemini for instance
        self._grok_client = self._build_grok_client()
        self._mistral_client = self._build_mistral_client()
        self._gemini_client = self._build_gemini_client()
        self._gemini_openai_client = self._build_gemini_openai_compat_client()

    @classmethod
    def set_cancel_event(cls, event: Optional[threading.Event]) -> None:
        cls._cancel_event = event

    @classmethod
    def _raise_if_cancelled(cls) -> None:
        event = cls._cancel_event
        if event and event.is_set():
            raise LLMCancellation("LLM request cancelled by user.")
    
    def chat(
        self,
        provider_model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Send a single chat request to a specific provider/model pair.

        This method expects a *provider-qualified* model string using a single
        backslash separator:

            "provider\\model"

        Examples
        --------
        llm.chat("openai\\gpt-4o-mini", messages)
        llm.chat("openrouter\\x-ai/grok-4.1-mini", messages)
        llm.chat("mistral\\mistral-small-latest", messages)
        llm.chat("gemini\\gemini-1.5-pro", messages)
        llm.chat("ollama\\llama3.2:3b", messages)
        llm.chat("huggingface\\meta-llama/Llama-2-7b-chat-hf", messages)

        If you only have a raw model name (e.g. "gpt-4o-mini") and want automatic
        provider inference, prefer `safe_chat_completion()`.

        Parameters
        ----------
        provider_model : str
            Provider-qualified model string in the form "provider\\model".
            The provider part is lowercased internally (e.g. "openai", "gemini",
            "openrouter", "mistral", "grok"/"xai", "huggingface", "ollama"/"local").
        messages : list[dict]
            Chat messages in OpenAI-style format, e.g.
            {"role": "user", "content": "Hello"}.
        **kwargs
            Provider-specific keyword arguments forwarded to the underlying
            provider call. Common options:
            - temperature: float
            - max_tokens: int
            Some providers accept extra options (e.g. OpenRouter `extra_headers`,
            Gemini `thinking_level` / `thinking_budget`).

        Returns
        -------
        str
            Assistant response text (empty string if missing).

        Raises
        ------
        LLMCancellation
            If a cancellation event was set before dispatch.
        ValueError
            If `provider_model` does not contain a backslash separator, or if the
            provider is unknown.
        RuntimeError / provider SDK errors
            If the underlying provider call fails (depends on provider).
        """
        provider, model = self._split_provider_model(provider_model)
        self._raise_if_cancelled()

        if provider == "openai":
            return self._chat_openai(model, messages, **kwargs)
        if provider == "gemini":
            return self._chat_gemini(model, messages, **kwargs)
        if provider == "openrouter":
            return self._chat_openrouter(model, messages, **kwargs)
        if provider in ("grok", "xai"):
            return self._chat_grok(model, messages, **kwargs)
        if provider == "huggingface":
            return self._chat_huggingface(model, messages, **kwargs)
        if provider in ("ollama", "local"):
            return self._chat_ollama(model, messages, **kwargs)
        if provider in ("mistral"):
            return self._chat_mistral(model, messages, **kwargs)

        raise ValueError(f"Unknown provider: {provider!r} in {provider_model!r}")

    def safe_chat_completion(
        self,
        model: str,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
        retries: int = 3,
        backoff_seconds: float = 1.5,
        **kwargs: Any,
    ) -> str | None:
        """
        Call `chat` with retries and structured error handling.

        This method retries transient failures (network errors, rate limits, temporary
        provider issues) and returns the first successful completion.

        Parameters
        ----------
        model : str
            Model name (provider-specific).
        messages : list[dict]
            Chat messages, each item being a dict like
            `{"role": "user", "content": "Hello"}`.
        max_tokens : int, default 1200
            Maximum tokens for the response.
        temperature : float, default 0.7
            Sampling temperature.
        provider : str | None, default None
            Optional provider hint such as "openai", "openrouter", "mistral", or "ollama".
        retries : int, default 3
            Number of attempts for transient failures.
        backoff_seconds : float, default 1.0
            Base backoff used between retries.
        **kwargs
            Extra provider-specific parameters forwarded to `chat`.

        Returns
        -------
        str
            Assistant response text.

        Raises
        ------
        RuntimeError
            If all retries fail.
        """

        self._raise_if_cancelled()
        if "\\" in model:
            provider_model = model
        else:
            raw_model = model
            prov = provider

            KNOWN_PROVIDERS = {"openai","openrouter","gemini","mistral","ollama","local","huggingface","grok","xai"}
            if prov is None:
                for sep in ("/", ":"):
                    if sep in raw_model:
                        maybe, rest = raw_model.split(sep, 1)
                        maybe = maybe.strip().lower()
                        if maybe in KNOWN_PROVIDERS:
                            prov, raw_model = maybe, rest.strip()
                            break
            if prov is None:
                prov = self._guess_provider_from_model(raw_model)
            provider_model = f"{prov}\\{raw_model}"
        
        eff_max_tokens = max_tokens or self.max_tokens
        eff_temperature = temperature if temperature is not None else self.temperature

        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            self._raise_if_cancelled()
            try:
                return self.chat(
                    provider_model,
                    messages,
                    max_tokens=eff_max_tokens,
                    temperature=eff_temperature,
                    **kwargs,
                )
            except Exception as e:
                if isinstance(e, LLMCancellation):
                    raise
                last_err = e
                logging.warning(
                    "[LLMConnect] chat error (attempt %d/%d) for %s: %s",
                    attempt,
                    retries,
                    provider_model,
                    e,
                )
                if attempt < retries:
                    if self.backoff_factor == 1.0:
                        delay = backoff_seconds * attempt
                    else:
                        delay = backoff_seconds * (self.backoff_factor ** (attempt - 1))
                    
                    time.sleep(delay)
                    self._raise_if_cancelled()

        logging.error("[LLMConnect] all retries failed for %s: %s", provider_model, last_err)
        return None

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Very rough token count estimate, assuming ≈4 characters per token.

        Designed only for lightweight checks (e.g. "warn if prompt is huge"),
        not for billing or exact context-window management.

        Parameters
        ----------
        text : str
            Input text; empty or None-like values yield 0.

        Returns
        -------
        int
            Estimated number of tokens (at least 1 for non-empty text).
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _split_provider_model(provider_model: str) -> Tuple[str, str]:
        """
        Split a "provider\\model" string into its two components.

        Parameters
        ----------
        provider_model : str
            String containing exactly one backslash between provider and model.

        Returns
        -------
        tuple[str, str]
            `(provider, model)` where provider is lowercased and stripped.

        Raises
        ------
        ValueError
            If the input does not contain a backslash.
        """
        if "\\" not in provider_model:
            raise ValueError(
                f"Expected 'provider\\\\model', got {provider_model!r}"
            )
        provider, model = provider_model.split("\\", 1)
        provider = provider.strip().lower()
        model = model.strip().lstrip("\\/")  # removes accidental leading "\" or "/"
        return provider, model

    @staticmethod
    def _guess_provider_from_model(model: str) -> str:
        """
        Extended 2025 heuristic to infer a provider if absent from the model name.

        This version handles modern model naming conventions (Gemini 3, OpenRouter, Mistral)
        and avoids conflicts between local tags (Ollama) and organization-style slashes.

        Applied Rules:
        - Prefixes "gemini" or "google/" → "gemini"
        - Prefixes "gpt-", "o1-", "o3-" → "openai"
        - Prefixes "mistral-", "pixtral-", "codestral" → "mistral"
        - Presence of a slash "/" (org/model format) → "openrouter"
        - Presence of a colon ":" or keywords (llama, phi, gemma...) → "ollama"
        - Fallback default → "openai"

        Parameters
        ----------
        model : str
            The raw model name (e.g., "gpt-4o", "gemini-3-flash-preview", "mistral:7b").

        Returns
        -------
        str
            The guessed provider name ("gemini", "openai", "mistral", "openrouter", or "ollama").
        """
        m = model.lower().strip()
        if m.startswith("gemini") or m.startswith("google/"):
            return "gemini"
        if any(m.startswith(p) for p in ["gpt-", "o1-", "o3-"]):
            return "openai"
        if m.startswith("mistral-") or m.startswith("pixtral-") or m.startswith("codestral"):
            return "mistral"
        if "/" in m:
            return "openrouter"
        if ":" in m or any(kw in m for kw in ["llama", "phi3", "phi4", "gemma"]):
            return "ollama"
        return "openai"

    @staticmethod
    def _messages_to_prompt(messages: List[Message]) -> str:
        """
        Flatten chat messages into a single text prompt for non-chat APIs.

        The format is:

            [ROLE] content

        separated by blank lines, e.g.:

            [SYSTEM] You are helpful.

            [USER] Explain gravity.

        Parameters
        ----------
        messages : list[dict]
            Chat messages in OpenAI format.

        Returns
        -------
        str
            Concatenated prompt string.
        """
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role.upper()}] {content}")
        return "\n\n".join(parts)

    def _build_openai_client(self) -> Optional[Any]:    
        """
        Build an OpenAI-compatible client using the OPENAI_API_KEY env var.

        Returns
        -------
        object or None
            An `OpenAIClient` instance if the key is available and the
            library is installed, otherwise None (disables OpenAI calls).
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAIClient is None:
            return None
        return OpenAIClient(api_key=api_key)

    def _build_gemini_client(self) -> Optional[Any]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or GoogleGenAI is None:
            return None
        return GoogleGenAI.Client(api_key=api_key)

    def _build_gemini_openai_compat_client(self) -> Optional[Any]:
        """
        Gemini via endpoint OpenAI-compatible (sans google-genai).
        Base URL officielle: https://generativelanguage.googleapis.com/v1beta/openai/
        """
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or OpenAIClient is None:
            return None

        base_url = os.getenv("GEMINI_OPENAI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
        return OpenAIClient(api_key=api_key, base_url=base_url)

    def _build_mistral_client(self) -> Optional[Any]:
        """
        Build an OpenAI-compatible client for Mistral.ai chat completions.

        Uses the MISTRAL_API_KEY environment variable and the
        https://api.mistral.ai/v1 base URL.

        Returns
        -------
        object or None
            An `OpenAIClient` instance configured for Mistral, or None if
            the key or OpenAI client is missing.
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key or OpenAIClient is None:
            return None
        return OpenAIClient(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
        )

    def _build_openrouter_client(self) -> Optional[Any]:
        """
        Build an OpenAI-compatible client for OpenRouter.

        Reads OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY (fallback).
        Targets https://openrouter.ai/api/v1.

        Returns
        -------
        object or None
            Configured OpenAIClient or None.
        """
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAIClient is None:
            return None
        return OpenAIClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _build_grok_client(self) -> Optional[Any]:
        """
        Build an OpenAI-compatible client for xAI Grok.

        API key is taken from XAI_API_KEY or GROK_API_KEY and the base URL is
        https://api.x.ai/v1.

        Returns
        -------
        object or None
            An `OpenAIClient` instance configured for xAI, or None if the
            key or client is missing.
        """
        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key or OpenAIClient is None:
            return None
        return OpenAIClient(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def _chat_openai(
        self,
        model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Call OpenAI chat.completions with compatibility handling.

        The method:
        - uses the global OpenAI-style client built from OPENAI_API_KEY,
        - transparently switches between `max_completion_tokens` and
          `max_tokens` depending on API support,
        - disables temperature for o1*/o3* models, keeps it otherwise.

        Parameters
        ----------
        model : str
            OpenAI model name (e.g. "gpt-4o-mini", "gpt-4.1").
        messages : list[dict]
            Chat messages in OpenAI format.
        **kwargs :
            Extra parameters forwarded as-is to the API.

        Returns
        -------
        str
            First candidate's message content (empty string if None).

        Raises
        ------
        RuntimeError
            If the OpenAI client is not available.
        OpenAIError
            If the underlying API call fails with a non-recoverable error.
        """
        if self._openai_client is None:
            raise RuntimeError("OPENAI_API_KEY not set or openai client not available")

        client = self._openai_client
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        m_lower = model.lower()

        def call_with(param_name: str) -> str:
            token_kwargs: Dict[str, Any] = {param_name: int(max_tokens)}
            extra_kwargs = {k: v for k, v in kwargs.items() 
                            if k not in ("max_tokens", "max_completion_tokens","temperature")}
            token_kwargs.update(extra_kwargs)

            temp_for_api: Optional[float] = None
            if m_lower.startswith("o1") or m_lower.startswith("o3"):
                temp_for_api = None
            else:
                if temperature is not None:
                    temp_for_api = float(temperature)

            call_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                **token_kwargs,
            }
            if temp_for_api is not None:
                call_kwargs["temperature"] = temp_for_api

            resp = client.chat.completions.create(**call_kwargs)
            content = resp.choices[0].message.content
            return content or ""

        try:
            return call_with("max_completion_tokens")
        except OpenAIError as e:
            txt = str(e).lower()
            if "unsupported parameter" in txt and "max_completion_tokens" in txt:
                return call_with("max_tokens")
            raise

    def _chat_gemini_openai_compat(self, model: str, messages: List[Message], **kwargs: Any) -> str:
        if self._gemini_openai_client is None:
            return ""

        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        completion = self._gemini_openai_client.chat.completions.create(**params)

        if not completion.choices:
            return ""
        content = completion.choices[0].message.content
        return content or ""

    def _chat_gemini(self, model: str, messages: List[Message], **kwargs: Any) -> str:
        """
        Gemini:
        1) google-genai (si dispo)
        2) endpoint OpenAI-compatible (si dispo)
        3) fallback OpenAI (si dispo)
        Jamais d'exception ici: on retourne "" si tout échoue.
        """

        if self._gemini_client is not None and types is not None:
            try:
                system_instr = None
                contents = []
                for m in messages:
                    role = m["role"].lower()
                    if role == "system":
                        system_instr = m["content"]
                    else:
                        gen_role = "user" if role == "user" else "model"
                        contents.append({"role": gen_role, "parts": [{"text": m["content"]}]})

                thinking_cfg = None
                if "thinking_level" in kwargs:
                    thinking_cfg = types.ThinkingConfig(thinking_level=kwargs["thinking_level"])
                elif "thinking_budget" in kwargs:
                    thinking_cfg = types.ThinkingConfig(thinking_budget=int(kwargs["thinking_budget"]))

                eff_temp = kwargs.get("temperature", self.temperature)
                if thinking_cfg:
                    eff_temp = 1.0

                config = types.GenerateContentConfig(
                    system_instruction=system_instr,
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=eff_temp,
                    thinking_config=thinking_cfg,
                )

                response = self._gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                logger_("[LLMConnect] Gemini SDK failed")

        txt = self._chat_gemini_openai_compat(model, messages, **kwargs)
        if txt:
            return txt

        try:
            fallback_pm = self.default_model
            if "\\" not in fallback_pm:
                fallback_pm = f"openai\\{fallback_pm}"
            fb_provider, fb_model = self._split_provider_model(fallback_pm)

            if fb_provider == "gemini":
                fb_provider, fb_model = "openai", "gpt-4o-mini"

            drop = {"thinking_level", "thinking_budget"}
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in drop}

            if fb_provider == "openai":
                return self._chat_openai(fb_model, messages, **clean_kwargs)
            if fb_provider == "openrouter":
                return self._chat_openrouter(fb_model, messages, **clean_kwargs)
            if fb_provider == "mistral":
                return self._chat_mistral(fb_model, messages, **clean_kwargs)
            if fb_provider in ("grok", "xai"):
                return self._chat_grok(fb_model, messages, **clean_kwargs)

            logger_("[LLMConnect] Gemini fallback provider unsupported")
            return ""
        except Exception as e:
            logger_("[LLMConnect] Gemini fallback OpenAI failed")
            return ""

    def _chat_mistral(
        self,
        model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Call Mistral.ai chat completions through the OpenAI-compatible client.

        Parameters
        ----------
        model : str
            Mistral model name (e.g. "mistral-small-latest").
        messages : list[dict]
            Chat messages in OpenAI format.
        **kwargs :
            Extra parameters; `temperature` and `max_tokens` override the
            instance defaults.

        Returns
        -------
        str
            First candidate's message content, or empty string if missing.

        Raises
        ------
        RuntimeError
            If the Mistral client is not available.
        """
        if self._mistral_client is None:
            raise RuntimeError("MISTRAL_API_KEY not set or mistral client not available")

        params: Dict[str, Any] = {
            "model": model,   # example: "mistral-small-latest", "open-mistral-7b"
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        completion = self._mistral_client.chat.completions.create(**params)
        return completion.choices[0].message.content or ""

    @staticmethod
    def _scrub_secret(text: str) -> str:
        if not text:
            return ""
        s = str(text)

        s = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)([A-Za-z0-9._\-=/+]{16,})",
            r"\1***REDACTED***",
            s,
        )
        s = re.sub(
            r"(?i)(\bbearer\s+)([A-Za-z0-9._\-=/+]{16,})",
            r"\1***REDACTED***",
            s,
        )

        s = re.sub(r"\bsk-[A-Za-z0-9]{4,}\b", "sk-***REDACTED***", s)                         # OpenAI-like
        s = re.sub(r"\bsk-ant-[A-Za-z0-9\-_]{10,}\b", "sk-ant-***REDACTED***", s)             # Anthropic
        s = re.sub(r"\bsk-or-v1-[A-Za-z0-9\-_]{10,}\b", "sk-or-v1-***REDACTED***", s)         # OpenRouter
        s = re.sub(r"\bAIza[0-9A-Za-z\-_]{30,}\b", "AIza***REDACTED***", s)                   # Google/Gemini

        s = re.sub(
            r"(?i)\b(api[_-]?key|x-api-key|token|secret)\b(\s*[:=]\s*)(['\"]?)([^\s'\",]{12,})",
            r"\1\2\3***REDACTED***",
            s,
        )
        s = re.sub(
            r"(?i)\b((?:OPENAI|ANTHROPIC|OPENROUTER|GOOGLE|GEMINI|MISTRAL|XAI|HUGGINGFACE|HF)_[A-Z0-9_]*KEY)\b(\s*[:=]\s*)(['\"]?)([^\s'\",]{12,})",
            r"\1\2\3***REDACTED***",
            s,
        )

        return s

    def _chat_openrouter(
        self,
        model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Call OpenRouter via its OpenAI-compatible API.

        Extra HTTP headers can be injected temporarily through the
        `extra_headers` kwarg; existing headers are restored afterwards.

        Parameters
        ----------
        model : str
            OpenRouter model name (e.g. "x-ai/grok-2-mini").
        messages : list[dict]
            Chat messages in OpenAI format.
        **kwargs :
            Extra parameters, including optional `extra_headers`.

        Returns
        -------
        str
            First candidate's message content, or empty string if missing.

        Raises
        ------
        RuntimeError
            If the OpenRouter client is not available.
        """
        if self._openrouter_client is None:
            raise RuntimeError("OPENROUTER_API_KEY not set or client not available")

        # example : model = "x-ai/grok-2-mini" ou "meta-llama/llama-3.1-8b-instruct"
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        extra_headers = kwargs.get("extra_headers") or {}
        hdrs = getattr(self._openrouter_client, "headers", None)
        old_headers_copy = dict(hdrs) if hdrs is not None else None
        if hdrs is not None and extra_headers:
            hdrs.update(extra_headers)
        try:
            completion = self._openrouter_client.chat.completions.create(**params)
        finally:
            if hdrs is not None and old_headers_copy is not None:
                hdrs.clear()
                hdrs.update(old_headers_copy)
        return completion.choices[0].message.content or ""

    def _chat_grok(
        self,
        model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Call xAI Grok through the OpenAI-compatible client.

        Parameters
        ----------
        model : str
            Grok model name as understood by the xAI API.
        messages : list[dict]
            Chat messages in OpenAI format.
        **kwargs :
            Extra parameters; `temperature` and `max_tokens` override defaults.

        Returns
        -------
        str
            First candidate's message content, or empty string if missing.

        Raises
        ------
        RuntimeError
            If the Grok client is not available.
        """
        if self._grok_client is None:
            raise RuntimeError("XAI_API_KEY / GROK_API_KEY not set or client not available")

        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        completion = self._grok_client.chat.completions.create(**params)
        return completion.choices[0].message.content or ""


    def _chat_huggingface(
        self,
        model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Send a chat request through the Hugging Face provider backend.

        Parameters
        ----------
        model : str
            Model repo id on Hugging Face (for example: "gpt2" or
            "meta-llama/Llama-2-7b-chat-hf").
        messages : list[dict]
            Chat messages to be flattened into a single prompt string.
        **kwargs
            Provider-specific keyword arguments.

        Returns
        -------
        str
            Assistant response text.
        """
        api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY")
        if not api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY / HF_API_KEY not set")

        prompt = self._messages_to_prompt(messages)
        max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data and isinstance(data[0], dict):
            txt = data[0].get("generated_text")
            if txt:
                if txt.startswith(prompt):
                    return txt[len(prompt) :].strip()
                return txt.strip()

            generated = data[0].get("generated_text") or data[0].get("text")
            if generated:
                return generated.strip()

        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()

        raise RuntimeError(
            f"Unexpected HF response: {self._scrub_secret(repr(data))}"
        )

    def _chat_ollama(
        self,
        model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Call a local Ollama server via its /api/chat endpoint.

        Parameters
        ----------
        model : str
            Ollama model name (e.g. "llama3.2:3b").
        messages : list[dict]
            Chat messages forwarded directly to the Ollama API.
        **kwargs :
            Extra options; `max_tokens` and `temperature` are translated into
            `num_predict` and `temperature` inside the `options` payload.

        Returns
        -------
        str
            Content of the "message" field returned by Ollama, stripped.

        Raises
        ------
        RuntimeError
            If the HTTP request fails or returns an invalid JSON payload.
        """
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        url = f"{base_url.rstrip('/')}/api/chat"

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = requests.post(
            url,
            json=payload,
            timeout=self.request_timeout,
            #timeout=max(600,self.request_timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        # format: {"message": {"role": "...", "content": "..."}, ...}
        msg = data.get("message", {})
        content = msg.get("content", "")
        return str(content).strip()

    def chatco(
        self,
        provider_model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Instance-level alias for `chat`, kept for backward compatibility.

        Example
        -------
        llm.chatco("openrouter\\x-ai/grok-4.1-mini", messages)
        """
        return self.chat(provider_model, messages, **kwargs)

    
    def dry_test_models(self, provider_models: List[str], verbose: bool = False) -> List[DryTestResult]:
        """
        Worker-friendly dry test for a subset of provider models.
        """
        results: List[DryTestResult] = []
        test_message = [{"role": "user", "content": "Respond exactly with 'OK'."}]

        for provider_model in provider_models:
            self._raise_if_cancelled()
            provider, model = self._split_provider_model(provider_model)
            if not self._is_provider_available_for_dry_test(provider):
                msg = f"SKIPPED (no API key or client for provider '{provider}')"
                if verbose:
                    logger_(f"[AUTO-DRY] {provider_model}: {msg}")
                results.append(
                    DryTestResult(
                        provider_model=provider_model,
                        provider=provider,
                        model=model,
                        ok=False,
                        message=msg,
                    )
                )
                continue

            try:
                resp = self.chat(
                    provider_model,
                    test_message,
                    max_tokens=4,
                    temperature=0.0,
                )
                if resp and "OK" in resp.upper():
                    result = DryTestResult(
                        provider_model=provider_model,
                        provider=provider,
                        model=model,
                        ok=True,
                        message="OK",
                    )
                else:
                    safe = str(self._scrub_secret(str(resp)))
                    if len(safe) > 500:
                        safe = safe[:500] + "…"
                    #message=f"FAILED (unexpected response type={type(resp).__name__}: {safe})"
                    result = DryTestResult(
                        provider_model=provider_model,
                        provider=provider,
                        model=model,
                        ok=False,
                        #message=f"FAILED (unexpected response: {self._scrub_secret(resp)!r})",
                        message=f"FAILED (unexpected response type={type(resp).__name__}: {safe})",
                    )
            except Exception as exc:
                status_code = getattr(exc, "http_status", None) or getattr(exc, "status_code", None)
                error_code = getattr(exc, "code", None) or getattr(getattr(exc, "error", None), "code", None)
                message = self._scrub_secret(str(exc))
                result = DryTestResult(
                    provider_model=provider_model,
                    provider=provider,
                    model=model,
                    ok=False,
                    message=f"FAILED ({message})",
                    status_code=status_code,
                    error_code=error_code,
                )

            if verbose:
                logger_(f"[AUTO-DRY] {provider_model}: {result.message}")
            results.append(result)

        return results

    def _is_provider_available_for_dry_test(self, provider: str) -> bool:
        provider = (provider or "").lower().strip()
        if not provider:
            return False
        if provider in ("grok", "xai"):
            return self._grok_client is not None
        if provider == "openai":
            return self._openai_client is not None
        if provider == "gemini":
            return ((self._gemini_client is not None)
                    or (getattr(self, "_gemini_openai_client", None) is not None)
                    or (self._openai_client is not None) 
                   )
        if provider == "openrouter":
            return self._openrouter_client is not None
        if provider == "mistral":
            return self._mistral_client is not None
        if provider == "ollama":
            return True
        if provider == "huggingface":
            return bool(os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY"))
        return getattr(self, f"_{provider}_client", None) is not None

    def auto_dry_run(self, verbose: bool = False, models: Optional[List[str]] = None) -> dict[str, str]:
        """
        Backward-compatible wrapper returning legacy status strings.
        """
        if models is None:
            provider_models = list(self.llmodels.values())
        else:
            provider_models = list(models)

        dry_results = self.dry_test_models(provider_models, verbose=verbose)
        legacy: dict[str, str] = {}
        for res in dry_results:
            legacy[res.provider_model] = "OK" if res.ok else res.message
        return legacy


    @classmethod
    def init_global(
        cls,
        llmodels: Optional[Dict[str, str]] = None,
        default_model: str = "openai\\gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        request_timeout: int = 60,
        **kwargs: Any,
    ) -> LLMConnect:
        """
        Create and register the global `LLMConnect` instance.

        Parameters
        ----------
        llmodels : dict, optional
            Mapping from logical roles (e.g. "ADVICE", "NARRATIVE") to
            provider\\model strings.
        default_model : str, default "openai\\gpt-4o-mini"
            Fallback model used when no mapping exists for a role.
        temperature : float, default 0.7
            Default sampling temperature for all calls.
        max_tokens : int, default 1024
            Default max completion tokens for all calls.
        request_timeout : int, default 60
            Timeout in seconds for HTTP-based providers.
        **kwargs :
            Reserved for future extensions.

        Returns
        -------
        LLMConnect
            The created global instance.
        """
        cls._global_instance = cls(
            llmodels=llmodels,
            default_model=default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            **kwargs,
        )
        return cls._global_instance

    @classmethod
    def get_model(cls, key: str) -> str:
        """
        Resolve a logical role name to an actual model string.

        Parameters
        ----------
        key : str
            Logical role identifier, e.g. "ADVICE", "MAPPING", "CONTEXT",
            "NARRATIVE", "THEME_ANALYSIS", "VARIANTS_GENERATION", etc.

        Returns
        -------
        str
            Provider\\model string for the role, or the current default
            model if no mapping exists.
        """
        inst = cls.get_global()
        if getattr(inst, "llmodels", None):
            return inst.llmodels.get(key, inst.default_model)
        return inst.default_model

    @classmethod
    def get_role_models(cls) -> Dict[str, str]:
        """
        Return a copy of the role→model mapping of the global instance.

        Returns
        -------
        dict
            Shallow copy of `llmodels` from the global `LLMConnect`
            instance. May be empty if no mapping was configured.
        """
        inst = cls.get_global()
        return dict(getattr(inst, "llmodels", {}))

    @classmethod
    def get_global(cls) -> LLMConnect:
        """
        Retrieve the global `LLMConnect` instance, creating a default one if needed.

        If `init_global` was never called, this method instantiates a new
        `LLMConnect` with default parameters and stores it in `_global_instance`.

        Returns
        -------
        LLMConnect
            The global router instance.
        """
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def chatco_global(
        cls,
        provider_model: str,
        messages: List[Message],
        **kwargs: Any,
    ) -> str:
        """
        Classmethod variant of `chatco` using the global instance.

        Example
        -------
        LLMConnect.chatco_global("openrouter\\x-ai/grok-4.1-mini", messages)
        """
        return cls.get_global().chat(provider_model, messages, **kwargs)


def dry_run_model(llm: LLMConnect, 
                  model: str, 
                  verbose: bool = False) -> bool:
    """
    Minimal smoke test for a single model with a given `LLMConnect` instance.

    Sends a short prompt asking the model to answer exactly "OK" and checks
    whether this token appears (case-insensitive) in the response.

    Parameters
    ----------
    llm : LLMConnect
        Router instance used to perform the test call.
    model : str
        Model identifier, passed directly to `llm.safe_chat_completion`.
    verbose : bool, default False
        If True, prints the raw response or the caught exception.

    Returns
    -------
    bool
        True if the model appears to work (response contains "OK"),
        False otherwise.
    """
    test_message = [
        {"role": "user", "content": "Respond exactly with 'OK'."}
    ]

    try:
        response = llm.safe_chat_completion(
            model=model,
            messages=test_message,
            max_tokens=4,
            temperature=0.0,
        )
        if verbose:
            f"[SMOKE] Raw response from {model!r}: {LLMConnect._scrub_secret(response)!r}"

        return bool(response) and ("OK" in response.upper())

    except Exception as e:
        if verbose:
            logger_(f"[SMOKE] Error testing model {model!r}")
        return False

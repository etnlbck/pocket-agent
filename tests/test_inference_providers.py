"""Tests for multi-provider inference backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from palmtop.inference.base import Message
from palmtop.inference.cloud import (
    PROVIDER_DEFAULTS,
    PROVIDER_ENV_KEYS,
    AnthropicBackend,
    CohereBackend,
    GeminiBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    create_cloud_backend,
)

# ── Factory tests ────────────────────────────────────────────────────────


class TestCreateCloudBackend:
    def test_anthropic(self):
        backend = create_cloud_backend("anthropic", "sk-test")
        assert isinstance(backend, AnthropicBackend)
        assert backend._model == "claude-haiku-4-5-20251001"

    def test_google(self):
        backend = create_cloud_backend("google", "goog-test")
        assert isinstance(backend, GeminiBackend)
        assert backend._model == "gemini-2.5-flash"

    def test_openai(self):
        backend = create_cloud_backend("openai", "sk-test")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend._model == "gpt-4o-mini"
        assert "openai.com" in backend._base_url

    def test_groq(self):
        backend = create_cloud_backend("groq", "gsk-test")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend._model == "llama-3.3-70b-versatile"
        assert "groq.com" in backend._base_url

    def test_together(self):
        backend = create_cloud_backend("together", "tog-test")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert "together.xyz" in backend._base_url

    def test_fireworks(self):
        backend = create_cloud_backend("fireworks", "fw-test")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert "fireworks.ai" in backend._base_url

    def test_deepseek(self):
        backend = create_cloud_backend("deepseek", "ds-test")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend._model == "deepseek-chat"
        assert "deepseek.com" in backend._base_url

    def test_mistral(self):
        backend = create_cloud_backend("mistral", "ms-test")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend._model == "mistral-large-latest"
        assert "mistral.ai" in backend._base_url

    def test_openrouter(self):
        backend = create_cloud_backend("openrouter", "or-test")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert "openrouter.ai" in backend._base_url
        # Should have extra headers for OpenRouter
        assert "X-Title" in backend._extra_headers

    def test_ollama(self):
        backend = create_cloud_backend("ollama", "")
        assert isinstance(backend, OllamaBackend)
        assert backend._model == "llama3.2"
        assert "localhost:11434" in backend._base_url

    def test_ollama_custom_url(self):
        backend = create_cloud_backend("ollama", "", base_url="http://gpu-box:11434")
        assert isinstance(backend, OllamaBackend)
        assert "gpu-box" in backend._base_url

    def test_cohere(self):
        backend = create_cloud_backend("cohere", "co-test")
        assert isinstance(backend, CohereBackend)
        assert backend._model == "command-r-plus"

    def test_custom_model_override(self):
        backend = create_cloud_backend("openai", "sk-test", model="gpt-4o")
        assert backend._model == "gpt-4o"

    def test_custom_base_url_override(self):
        backend = create_cloud_backend("openai", "sk-test", base_url="https://my-proxy.com/v1")
        assert backend._base_url == "https://my-proxy.com/v1"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown cloud provider"):
            create_cloud_backend("not-real", "key")

    def test_error_lists_supported_providers(self):
        with pytest.raises(ValueError, match="anthropic"):
            create_cloud_backend("typo", "key")


class TestProviderRegistry:
    def test_all_providers_have_defaults(self):
        expected = {
            "anthropic",
            "google",
            "openai",
            "groq",
            "together",
            "fireworks",
            "deepseek",
            "mistral",
            "openrouter",
            "ollama",
            "cohere",
        }
        assert set(PROVIDER_DEFAULTS.keys()) == expected

    def test_all_keyed_providers_have_env_vars(self):
        # Every provider except ollama should have an env var mapping
        for provider in PROVIDER_DEFAULTS:
            if provider == "ollama":
                assert provider not in PROVIDER_ENV_KEYS
            else:
                assert provider in PROVIDER_ENV_KEYS

    def test_all_defaults_have_model(self):
        for provider, defaults in PROVIDER_DEFAULTS.items():
            assert "model" in defaults, f"{provider} missing default model"


# ── OpenAICompatibleBackend tests ────────────────────────────────────────


class TestOpenAICompatibleBackend:
    @pytest.fixture
    def backend(self):
        return OpenAICompatibleBackend("sk-test", model="gpt-4o-mini")

    def test_headers(self, backend):
        headers = backend._headers()
        assert headers["authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"

    def test_extra_headers_merged(self):
        backend = OpenAICompatibleBackend(
            "sk-test",
            extra_headers={"X-Custom": "value"},
        )
        headers = backend._headers()
        assert headers["X-Custom"] == "value"
        assert headers["authorization"] == "Bearer sk-test"

    @pytest.mark.asyncio
    async def test_complete_success(self, backend):
        mock_response = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello!"}}],
            },
        )
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        result = await backend.complete(messages)
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_complete_error_raises(self, backend):
        mock_response = httpx.Response(429, text="Rate limited")
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        with pytest.raises(RuntimeError, match="429"):
            await backend.complete(messages)

    @pytest.mark.asyncio
    async def test_complete_no_choices_raises(self, backend):
        mock_response = httpx.Response(200, json={"choices": []})
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        with pytest.raises(RuntimeError, match="no choices"):
            await backend.complete(messages)

    @pytest.mark.asyncio
    async def test_system_message_passed_through(self, backend):
        """OpenAI-compatible APIs accept system role directly in messages."""
        mock_response = httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}}]},
        )
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hi"),
        ]
        await backend.complete(messages)

        # Verify system message was included in the request
        call_args = backend._client.post.call_args
        body = call_args.kwargs["json"]
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are helpful"


# ── OllamaBackend tests ──────────────────────────────────────────────────


class TestOllamaBackend:
    @pytest.fixture
    def backend(self):
        return OllamaBackend(model="llama3.2")

    def test_default_url(self, backend):
        assert backend._base_url == "http://localhost:11434"

    def test_custom_url(self):
        b = OllamaBackend(base_url="http://192.168.1.50:11434")
        assert b._base_url == "http://192.168.1.50:11434"

    @pytest.mark.asyncio
    async def test_complete_success(self, backend):
        mock_response = httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hi there!"}},
        )
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hello")]
        result = await backend.complete(messages)
        assert result == "Hi there!"

    @pytest.mark.asyncio
    async def test_complete_uses_chat_endpoint(self, backend):
        mock_response = httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "OK"}},
        )
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.complete([Message(role="user", content="test")])

        call_args = backend._client.post.call_args
        url = call_args.args[0]
        assert "/api/chat" in url

    @pytest.mark.asyncio
    async def test_complete_sends_options(self, backend):
        mock_response = httpx.Response(
            200,
            json={"message": {"content": "OK"}},
        )
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        await backend.complete([Message(role="user", content="test")], max_tokens=256)

        body = backend._client.post.call_args.kwargs["json"]
        assert body["options"]["num_predict"] == 256
        assert body["stream"] is False

    @pytest.mark.asyncio
    async def test_complete_error_raises(self, backend):
        mock_response = httpx.Response(500, text="Model not found")
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="500"):
            await backend.complete([Message(role="user", content="Hi")])


# ── CohereBackend tests ──────────────────────────────────────────────────


class TestCohereBackend:
    @pytest.fixture
    def backend(self):
        return CohereBackend("co-test", model="command-r-plus")

    def test_headers(self, backend):
        headers = backend._headers()
        assert headers["authorization"] == "Bearer co-test"

    @pytest.mark.asyncio
    async def test_complete_success(self, backend):
        mock_response = httpx.Response(
            200,
            json={
                "message": {"content": [{"type": "text", "text": "Hello from Cohere!"}]},
            },
        )
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        result = await backend.complete(messages)
        assert result == "Hello from Cohere!"

    @pytest.mark.asyncio
    async def test_complete_error_raises(self, backend):
        mock_response = httpx.Response(401, text="Invalid API key")
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="401"):
            await backend.complete([Message(role="user", content="Hi")])

    @pytest.mark.asyncio
    async def test_complete_empty_content(self, backend):
        mock_response = httpx.Response(
            200,
            json={"message": {"content": []}},
        )
        backend._client = AsyncMock()
        backend._client.post = AsyncMock(return_value=mock_response)

        result = await backend.complete([Message(role="user", content="Hi")])
        assert result == ""


# ── Config integration tests ─────────────────────────────────────────────


class TestConfigProviderResolution:
    def test_groq_provider_resolves_key(self, tmp_path):
        """Provider set in TOML + env var → key auto-filled."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[cloud.light]\nprovider = "groq"\n[cloud.heavy]\nprovider = "anthropic"\n')
        import os

        from palmtop.config.settings import Config

        env = {
            "GROQ_API_KEY": "gsk-test-123",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.load(config_file)

        assert cfg.cloud_light.provider == "groq"
        assert cfg.cloud_light.api_key == "gsk-test-123"
        assert cfg.cloud_heavy.provider == "anthropic"
        assert cfg.cloud_heavy.api_key == "sk-ant-test"

    def test_ollama_no_key_needed(self, tmp_path):
        """Ollama provider works without any API key."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[cloud.light]\nprovider = "ollama"\nmodel = "llama3.2"\n')
        import os

        from palmtop.config.settings import Config

        with patch.dict(os.environ, {}, clear=False):
            cfg = Config.load(config_file)

        assert cfg.cloud_light.provider == "ollama"
        # Ollama doesn't need a key — should still be empty
        assert cfg.cloud_light.api_key == ""

    def test_fallback_to_google_light_anthropic_heavy(self, tmp_path):
        """Default behavior: Google for light, Anthropic for heavy."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")  # empty config
        import os

        from palmtop.config.settings import Config

        env = {
            "GOOGLE_API_KEY": "goog-key",
            "ANTHROPIC_API_KEY": "ant-key",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.load(config_file)

        assert cfg.cloud_light.provider == "google"
        assert cfg.cloud_light.api_key == "goog-key"
        assert cfg.cloud_heavy.provider == "anthropic"
        assert cfg.cloud_heavy.api_key == "ant-key"

    def test_openrouter_both_tiers(self, tmp_path):
        """Can use OpenRouter for both tiers with different models."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[cloud.light]\nprovider = "openrouter"\nmodel = "google/gemini-2.5-flash"\n'
            '[cloud.heavy]\nprovider = "openrouter"\nmodel = "anthropic/claude-sonnet-4"\n'
        )
        import os

        from palmtop.config.settings import Config

        env = {"OPENROUTER_API_KEY": "or-key-123"}
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.load(config_file)

        assert cfg.cloud_light.provider == "openrouter"
        assert cfg.cloud_light.api_key == "or-key-123"
        assert cfg.cloud_light.model == "google/gemini-2.5-flash"
        assert cfg.cloud_heavy.provider == "openrouter"
        assert cfg.cloud_heavy.api_key == "or-key-123"
        assert cfg.cloud_heavy.model == "anthropic/claude-sonnet-4"


# ── Backend creation logic tests (mirrors _make_cloud in __main__) ───────


class TestMakeCloudLogic:
    """Tests for the provider creation logic used in __main__._make_cloud."""

    def test_ollama_no_key_creates_backend(self):
        """Ollama should be creatable without an API key."""
        backend = create_cloud_backend("ollama", "", model="llama3.2")
        assert isinstance(backend, OllamaBackend)
        assert backend._model == "llama3.2"

    def test_openai_with_key_creates_backend(self):
        """OpenAI with a key should create a valid backend."""
        backend = create_cloud_backend("openai", "sk-test-key", model="gpt-4o")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend._model == "gpt-4o"

    def test_provider_validation_clear_error(self):
        """Unknown provider gives a helpful error listing supported ones."""
        with pytest.raises(ValueError) as exc_info:
            create_cloud_backend("gpt4", "key")
        error_msg = str(exc_info.value)
        assert "openai" in error_msg
        assert "groq" in error_msg
        assert "ollama" in error_msg

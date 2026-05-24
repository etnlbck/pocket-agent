"""Tests for the Discord channel — discord.py integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

discord_mod = pytest.importorskip("discord", reason="discord.py not installed")

from palmtop.channels.discord import (  # noqa: E402
    MAX_MESSAGE_LENGTH,
    DiscordChannel,
    _split_message,
)


class TestDiscordChannelInit:
    def test_requires_bot_token(self):
        with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
            DiscordChannel(bot_token="")

    def test_basic_init(self):
        with patch("palmtop.channels.discord.discord.Client"):
            ch = DiscordChannel(bot_token="test-token-123")
            assert ch.name == "discord"
            assert ch._allowed_users is None
            assert ch._guild_id is None
            assert ch._channel_id is None

    def test_allowed_users(self):
        with patch("palmtop.channels.discord.discord.Client"):
            ch = DiscordChannel(
                bot_token="test-token",
                allowed_users=[123, 456, 789],
            )
            assert ch._allowed_users == {123, 456, 789}

    def test_guild_and_channel_filter(self):
        with patch("palmtop.channels.discord.discord.Client"):
            ch = DiscordChannel(
                bot_token="test-token",
                guild_id=111222333,
                channel_id=444555666,
            )
            assert ch._guild_id == 111222333
            assert ch._channel_id == 444555666


class TestDiscordChannelProtocol:
    def test_name_property(self):
        with patch("palmtop.channels.discord.discord.Client"):
            ch = DiscordChannel(bot_token="test-token")
            assert ch.name == "discord"


class TestOnMessage:
    @pytest.fixture
    def channel(self):
        with patch("palmtop.channels.discord.discord.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.user = MagicMock()
            mock_client.user.id = 99999
            mock_client_cls.return_value = mock_client
            ch = DiscordChannel(bot_token="test-token", allowed_users=[12345])
            ch._agent = AsyncMock()
            ch._agent.handle = AsyncMock(return_value="Hello from the agent!")
            # Override the client's user for comparison
            ch._client = mock_client
            return ch

    @pytest.mark.asyncio
    async def test_ignores_own_messages(self, channel):
        msg = MagicMock()
        msg.author = channel._client.user
        await channel._on_message(msg)
        channel._agent.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_bot_messages(self, channel):
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = True
        msg.author.id = 77777
        await channel._on_message(msg)
        channel._agent.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_non_allowed_user(self, channel):
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 99999  # Not in allowed list
        msg.author.name = "stranger"
        # Make it not equal to client.user
        msg.author.__eq__ = lambda self, other: False
        await channel._on_message(msg)
        channel._agent.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_processes_allowed_user(self, channel):
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 12345
        msg.author.name = "testuser"
        msg.author.__eq__ = lambda self, other: False
        msg.content = "What's on my calendar?"
        msg.guild = None  # DM
        msg.channel = MagicMock()
        # typing() returns an async context manager
        typing_ctx = MagicMock()
        typing_ctx.__aenter__ = AsyncMock(return_value=None)
        typing_ctx.__aexit__ = AsyncMock(return_value=None)
        msg.channel.typing = MagicMock(return_value=typing_ctx)
        msg.reply = AsyncMock()

        await channel._on_message(msg)

        channel._agent.handle.assert_called_once()
        call_args = channel._agent.handle.call_args
        assert call_args[0][0] == "What's on my calendar?"
        assert call_args[1]["user_id"] == "discord:12345"
        msg.reply.assert_called_once_with("Hello from the agent!")

    @pytest.mark.asyncio
    async def test_ignores_empty_message(self, channel):
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 12345
        msg.author.__eq__ = lambda self, other: False
        msg.content = "   "
        msg.guild = None
        await channel._on_message(msg)
        channel._agent.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_guild_filter(self, channel):
        channel._guild_id = 111222
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 12345
        msg.author.__eq__ = lambda self, other: False
        msg.content = "Hello"
        msg.guild = MagicMock()
        msg.guild.id = 999888  # Wrong guild
        await channel._on_message(msg)
        channel._agent.handle.assert_not_called()


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_sends_dm(self):
        with patch("palmtop.channels.discord.discord.Client"):
            ch = DiscordChannel(bot_token="test-token")
            mock_user = AsyncMock()
            mock_user.send = AsyncMock()
            ch._client = AsyncMock()
            ch._client.fetch_user = AsyncMock(return_value=mock_user)

            await ch.send_message("12345", "Your reminder: meeting in 5min")

            ch._client.fetch_user.assert_called_once_with(12345)
            mock_user.send.assert_called_once_with("Your reminder: meeting in 5min")


class TestSplitMessage:
    def test_short_message_unchanged(self):
        assert _split_message("Hello world") == ["Hello world"]

    def test_splits_at_newline(self):
        text = "A" * 1900 + "\n" + "B" * 200
        chunks = _split_message(text)
        assert len(chunks) == 2
        assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)

    def test_splits_at_space(self):
        # No newlines, but has spaces
        text = " ".join(["word"] * 500)
        chunks = _split_message(text)
        assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)

    def test_hard_split_no_whitespace(self):
        text = "A" * 3000  # No whitespace at all
        chunks = _split_message(text)
        assert len(chunks) == 2
        assert chunks[0] == "A" * MAX_MESSAGE_LENGTH
        assert chunks[1] == "A" * 1000

    def test_preserves_code_blocks(self):
        # Unclosed code block at split point — needs to exceed 2000 chars
        text = "```python\n" + "x = 1\n" * 400 + "```"
        assert len(text) > MAX_MESSAGE_LENGTH
        chunks = _split_message(text)
        assert len(chunks) >= 2
        # First chunk should close the code block
        assert chunks[0].endswith("```")
        # Second chunk should reopen it
        assert chunks[1].startswith("```\n")

    def test_exact_limit(self):
        text = "A" * MAX_MESSAGE_LENGTH
        assert _split_message(text) == [text]

    def test_empty_message(self):
        assert _split_message("") == [""]


class TestStopChannel:
    @pytest.mark.asyncio
    async def test_stop_sets_event_and_closes(self):
        with patch("palmtop.channels.discord.discord.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.is_closed.return_value = False
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client

            ch = DiscordChannel(bot_token="test-token")
            ch._client = mock_client

            await ch.stop()
            assert ch._stop_event.is_set()
            mock_client.close.assert_called_once()

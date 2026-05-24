"""Tests for the IRC channel — raw asyncio socket IRC protocol."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from palmtop.channels.irc import (
    MAX_IRC_LINE,
    IrcChannel,
    _split_message,
)


class TestIrcChannelInit:
    def test_requires_server(self):
        with pytest.raises(ValueError, match="IRC server"):
            IrcChannel(server="")

    def test_basic_init(self):
        ch = IrcChannel(server="irc.libera.chat")
        assert ch.name == "irc"
        assert ch._port == 6667
        assert ch._nick == "palmtop"
        assert ch._allowed_users is None
        assert ch._channels == []

    def test_custom_config(self):
        ch = IrcChannel(
            server="irc.example.org",
            port=6697,
            nick="mybot",
            channels=["#test", "#dev"],
            use_ssl=True,
            allowed_users=["Alice", "Bob"],
        )
        assert ch._port == 6697
        assert ch._nick == "mybot"
        assert ch._channels == ["#test", "#dev"]
        assert ch._use_ssl is True
        assert ch._allowed_users == {"alice", "bob"}  # lowercased


class TestIrcOnPrivmsg:
    @pytest.fixture
    def channel(self):
        ch = IrcChannel(
            server="irc.test",
            nick="palmtop",
            allowed_users=["alice"],
        )
        ch._agent = AsyncMock()
        ch._agent.handle = AsyncMock(return_value="Agent reply!")
        ch._writer = AsyncMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._connected = True
        return ch

    @pytest.mark.asyncio
    async def test_handles_dm(self, channel):
        await channel._on_privmsg(
            "alice!user@host",
            "palmtop :Hello bot",
        )
        channel._agent.handle.assert_called_once()
        call_args = channel._agent.handle.call_args
        assert call_args[0][0] == "Hello bot"
        assert call_args[1]["user_id"] == "irc:alice"

    @pytest.mark.asyncio
    async def test_handles_mention(self, channel):
        await channel._on_privmsg(
            "alice!user@host",
            "#channel :palmtop: what time is it?",
        )
        channel._agent.handle.assert_called_once()
        call_args = channel._agent.handle.call_args
        assert call_args[0][0] == "what time is it?"

    @pytest.mark.asyncio
    async def test_ignores_non_addressed_channel_msg(self, channel):
        await channel._on_privmsg(
            "alice!user@host",
            "#channel :just chatting with others",
        )
        channel._agent.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_non_allowed_user(self, channel):
        await channel._on_privmsg(
            "stranger!user@host",
            "palmtop :hack me",
        )
        channel._agent.handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_replies_in_dm(self, channel):
        await channel._on_privmsg(
            "alice!user@host",
            "palmtop :Hello",
        )
        # Should reply to alice directly (DM)
        written = channel._writer.write.call_args_list
        reply_line = written[-1][0][0].decode()
        assert "PRIVMSG alice" in reply_line
        assert "Agent reply!" in reply_line

    @pytest.mark.asyncio
    async def test_replies_in_channel(self, channel):
        await channel._on_privmsg(
            "alice!user@host",
            "#test :palmtop, help me",
        )
        # Should reply in #test with nick prefix
        written = channel._writer.write.call_args_list
        reply_line = written[-1][0][0].decode()
        assert "PRIVMSG #test" in reply_line
        assert "alice: Agent reply!" in reply_line


class TestHandleLine:
    @pytest.fixture
    def channel(self):
        ch = IrcChannel(server="irc.test", nick="palmtop", channels=["#dev"])
        ch._writer = AsyncMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._connected = True
        return ch

    @pytest.mark.asyncio
    async def test_responds_to_ping(self, channel):
        await channel._handle_line("PING :server.example.com")
        written = channel._writer.write.call_args[0][0].decode()
        assert written.startswith("PONG")

    @pytest.mark.asyncio
    async def test_joins_channels_on_welcome(self, channel):
        await channel._handle_line(":server 001 palmtop :Welcome to IRC")
        written = channel._writer.write.call_args[0][0].decode()
        assert "JOIN #dev" in written

    @pytest.mark.asyncio
    async def test_handles_nick_collision(self, channel):
        await channel._handle_line(":server 433 * palmtop :Nickname already in use")
        assert channel._nick == "palmtop_"
        written = channel._writer.write.call_args[0][0].decode()
        assert "NICK palmtop_" in written


class TestSendMessage:
    @pytest.mark.asyncio
    async def test_sends_privmsg(self):
        ch = IrcChannel(server="irc.test")
        ch._writer = AsyncMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._connected = True

        await ch.send_message("alice", "Hello there!")
        written = ch._writer.write.call_args[0][0].decode()
        assert "PRIVMSG alice :Hello there!\r\n" == written

    @pytest.mark.asyncio
    async def test_send_when_not_connected(self):
        ch = IrcChannel(server="irc.test")
        # No writer, not connected
        await ch.send_message("alice", "hi")
        # Should not raise


class TestStopChannel:
    @pytest.mark.asyncio
    async def test_stop_sends_quit(self):
        ch = IrcChannel(server="irc.test")
        ch._writer = AsyncMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._writer.close = MagicMock()
        ch._connected = True

        await ch.stop()
        assert ch._stop_event.is_set()
        written = ch._writer.write.call_args[0][0].decode()
        assert "QUIT" in written


class TestSplitMessage:
    def test_short_single_line(self):
        assert _split_message("Hello") == ["Hello"]

    def test_multiline_splits(self):
        text = "Line 1\nLine 2\nLine 3"
        chunks = _split_message(text)
        assert chunks == ["Line 1", "Line 2", "Line 3"]

    def test_long_line_splits_at_space(self):
        text = " ".join(["word"] * 100)
        chunks = _split_message(text)
        assert all(len(c) <= MAX_IRC_LINE for c in chunks)

    def test_hard_split_no_space(self):
        text = "A" * 600
        chunks = _split_message(text)
        assert len(chunks) == 2
        assert chunks[0] == "A" * MAX_IRC_LINE
        assert chunks[1] == "A" * 150

    def test_empty_message(self):
        assert _split_message("") == [""]

    def test_blank_lines_skipped(self):
        text = "Hello\n\n\nWorld"
        chunks = _split_message(text)
        assert chunks == ["Hello", "World"]

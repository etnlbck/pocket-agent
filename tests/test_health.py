"""Tests for admin/health.py — health endpoint and admin stats."""

import asyncio
import json
import time

import pytest

from palmtop.admin.health import HealthServer, HealthState, health_json, stats_json


class TestHealthJson:
    def test_returns_ok_status(self):
        state = HealthState()
        result = health_json(state)
        assert result["status"] == "ok"
        assert "uptime" in result
        assert result["uptime_seconds"] >= 0

    def test_includes_channels(self):
        state = HealthState(channels_active=["telegram", "sms"])
        result = health_json(state)
        assert result["channels"] == ["telegram", "sms"]

    def test_includes_message_stats(self):
        state = HealthState(messages_handled=42, last_message_at=time.time())
        result = health_json(state)
        assert result["messages_handled"] == 42
        assert result["last_message_at"] is not None

    def test_null_last_message_when_none(self):
        state = HealthState()
        result = health_json(state)
        assert result["last_message_at"] is None

    def test_database_sizes(self, tmp_path):
        # Create a fake DB file
        db_file = tmp_path / "conversations.db"
        db_file.write_text("fake")
        state = HealthState(data_dir=tmp_path)
        result = health_json(state)
        assert result["databases"]["conversations"] == 4
        assert result["databases"]["memories"] is None  # doesn't exist


class TestStatsJson:
    def test_extends_health(self):
        state = HealthState()
        result = stats_json(state)
        assert result["status"] == "ok"
        assert result["admin"] is True


@pytest.mark.asyncio
async def test_health_server_responds_200():
    """Health server should respond to GET / with 200 JSON."""
    state = HealthState(channels_active=["telegram"])
    server = HealthServer(state, host="127.0.0.1", port=0)  # port 0 = random

    await server.start()
    # Get the actual port
    actual_port = server._server.sockets[0].getsockname()[1]

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", actual_port)
        writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()

        response = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        response_str = response.decode()

        assert "200 OK" in response_str
        # Extract JSON body
        body = response_str.split("\r\n\r\n", 1)[1]
        data = json.loads(body)
        assert data["status"] == "ok"
        assert data["channels"] == ["telegram"]

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_admin_requires_auth():
    """Admin stats should return 401 without valid token."""
    state = HealthState(admin_token="secret123")
    server = HealthServer(state, host="127.0.0.1", port=0)

    await server.start()
    actual_port = server._server.sockets[0].getsockname()[1]

    try:
        # Request without auth
        reader, writer = await asyncio.open_connection("127.0.0.1", actual_port)
        writer.write(b"GET /admin/stats HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        assert "401" in response.decode()
        writer.close()
        await writer.wait_closed()

        # Request with valid auth
        reader, writer = await asyncio.open_connection("127.0.0.1", actual_port)
        writer.write(b"GET /admin/stats HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret123\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        response_str = response.decode()
        assert "200 OK" in response_str
        body = response_str.split("\r\n\r\n", 1)[1]
        data = json.loads(body)
        assert data["admin"] is True
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_admin_disabled_without_token():
    """Without admin_token configured, admin routes return 401."""
    state = HealthState(admin_token="")
    server = HealthServer(state, host="127.0.0.1", port=0)

    await server.start()
    actual_port = server._server.sockets[0].getsockname()[1]

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", actual_port)
        writer.write(b"GET /admin/stats HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer anything\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        assert "401" in response.decode()
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()

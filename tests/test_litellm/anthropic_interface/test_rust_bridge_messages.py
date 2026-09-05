from __future__ import annotations

from collections.abc import Awaitable, Generator
from types import ModuleType
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages_handler,
)
from litellm.rust_bridge import configuration, loader
from litellm.rust_bridge import messages as bridge
from litellm.rust_bridge.callback_adapters import ProviderLoggingAdapter


@pytest.fixture(autouse=True)
def reset_bridge() -> Generator[None]:
    bridge.set_rust_messages(sync=None, asynchronous=None)
    configuration.reset_rust_configuration()
    loader.reset_native_bridge_cache()
    yield
    bridge.set_rust_messages(sync=None, asynchronous=None)
    configuration.reset_rust_configuration()
    loader.reset_native_bridge_cache()


@pytest.mark.asyncio
async def test_public_messages_passes_shared_callback_adapter_to_native() -> None:
    calls: list[dict[str, object]] = []

    async def native(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hello from rust"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    bridge.set_rust_messages(asynchronous=native)
    result: Final = await anthropic_messages_handler(
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-sonnet-4-5",
        is_async=True,
        rust=True,
        litellm_logging_obj=MagicMock(),
    )

    assert result["content"][0]["text"] == "hello from rust"
    assert len(calls) == 1
    assert isinstance(calls[0]["callback_adapter"], ProviderLoggingAdapter)


@pytest.mark.asyncio
async def test_unready_messages_route_uses_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    native_calls: list[str] = []
    native: Final = ModuleType("litellm.rust_bridge._native")

    async def native_messages(**_kwargs: object) -> dict[str, object]:
        native_calls.append("native")
        return {}

    native.amessages = native_messages
    native.ready_endpoints = {"ocr": frozenset({"callbacks"})}
    monkeypatch.setattr(loader, "get_native_bridge", lambda: native)
    monkeypatch.setattr(bridge, "get_native_bridge", lambda: native, raising=False)
    configuration.rust(True)

    async def fallback() -> object:
        return "python"

    pending: Final[Awaitable[object]] = bridge.adispatch_messages(
        prepare=lambda: pytest.fail("unready native route must not prepare a request"),
        fallback=fallback,
        model="claude-sonnet-4-5",
        provider="anthropic",
        request_override=True,
        eligible=True,
    )

    assert await pending == "python"
    assert native_calls == []

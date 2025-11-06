#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

import websockets
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.utils.network import exponential_backoff_time

class WebsocketService(ABC):
    """Base class for websocket-based services with reconnection logic."""

    def __init__(self):
        """Initialize websocket attributes."""
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def _verify_connection(self) -> bool:
        """Verify websocket connection is working.

        Returns:
            bool: True if connection is verified working, False otherwise
        """
        try:
            if not self._websocket:
                return False
            await self._websocket.ping()
            return True
        except Exception as e:
            logger.error(f"{self} connection verification failed: {e}")
            return False

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect the websocket.

        Args:
            attempt_number: Current retry attempt number

        Returns:
            bool: True if reconnection and verification successful, False otherwise
        """
        logger.warning(f"{self} reconnecting (attempt: {attempt_number})")
        await self._disconnect_websocket()
        await self._connect_websocket()
        return await self._verify_connection()

    async def _receive_task_handler(self, report_error: Callable[[ErrorFrame], Awaitable[None]]):
        """Handles WebSocket message receiving with automatic retry logic.

        Args:
            report_error: Callback to report errors
        """
        retry_count = 0
        MAX_RETRIES = 3

        while True:
            try:
                await self._receive_messages()
                retry_count = 0  # Reset counter on successful message receive
                if self._websocket and self._websocket.state == State.CLOSED:
                    raise websockets.ConnectionClosedOK(
                        self._websocket.close_rcvd,
                        self._websocket.close_sent,
                        self._websocket.close_rcvd_then_sent,
                    )
            except Exception as e:
                # Check if this is a recoverable WebSocket close code
                is_recoverable = False
                if isinstance(e, (websockets.ConnectionClosedOK, websockets.ConnectionClosedError)):
                    # Extract close code if available
                    close_code = getattr(e, 'rcvd_code', None) or getattr(e, 'code', None)
                    # 1012 = Service Restart (server is restarting, should reconnect)
                    # 1013 = Try Again Later (temporary issue)
                    recoverable_codes = {1012, 1013}
                    if close_code in recoverable_codes:
                        is_recoverable = True
                        logger.info(f"{self} received recoverable close code {close_code}, reconnecting without penalty")

                # Only increment retry count for non-recoverable errors
                if not is_recoverable:
                    retry_count += 1

                if retry_count >= MAX_RETRIES:
                    message = f"{self} error receiving messages: {e}"
                    logger.error(message)
                    await report_error(ErrorFrame(message, fatal=True))
                    break

                logger.exception(f"{self} connection error, will retry: {e}",)

                try:
                    if await self._reconnect_websocket(retry_count if not is_recoverable else 1):
                        retry_count = 0  # Reset counter on successful reconnection
                    wait_time = exponential_backoff_time(retry_count if not is_recoverable else 0)
                    await asyncio.sleep(wait_time)
                except Exception as reconnect_error:
                    logger.error(f"{self} reconnection failed: {reconnect_error}")
                    continue

    @abstractmethod
    async def _connect(self):
        """Implement service-specific connection logic. This function will
        connect to the websocket via _connect_websocket() among other connection
        logic."""
        pass

    @abstractmethod
    async def _disconnect(self):
        """Implement service-specific disconnection logic. This function will
        disconnect to the websocket via _connect_websocket() among other
        connection logic.

        """
        pass

    @abstractmethod
    async def _connect_websocket(self):
        """Implement service-specific websocket connection logic. This function
        should only connect to the websocket."""
        pass

    @abstractmethod
    async def _disconnect_websocket(self):
        """Implement service-specific websocket disconnection logic. This
        function should only disconnect from the websocket."""
        pass

    @abstractmethod
    async def _receive_messages(self):
        """Implement service-specific message receiving logic."""
        pass

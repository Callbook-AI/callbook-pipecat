"""Exercise the HTTP and binary WebSocket APIs used by voice services over loopback."""

import socket
import unittest

import aiohttp
from aiohttp import web

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.serializers.protobuf import ProtobufFrameSerializer


class TestHTTPMediaCompatibility(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = web.Application()
        app.router.add_get("/audio", self.audio)
        app.router.add_get("/websocket", self.websocket)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        await web.SockSite(self.runner, sock).start()
        self.url = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def audio(self, request):
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(b"\x01\x00\x02\x00")
        await response.write(b"\x03\x00\x04\x00")
        await response.write_eof()
        return response

    async def websocket(self, request):
        response = web.WebSocketResponse()
        await response.prepare(request)
        async for message in response:
            if message.type == aiohttp.WSMsgType.BINARY:
                await response.send_bytes(message.data)
        return response

    async def test_streamed_audio_bytes_are_preserved(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(self.url + "/audio") as response:
                response.raise_for_status()
                data = b"".join([part async for part in response.content.iter_chunked(3)])
        self.assertEqual(data, b"\x01\x00\x02\x00\x03\x00\x04\x00")

    async def test_binary_websocket_preserves_audio_frame(self):
        serializer = ProtobufFrameSerializer()
        original = OutputAudioRawFrame(audio=b"\x01\x00\x02\x00", sample_rate=16000, num_channels=1)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.ws_connect(self.url + "/websocket") as websocket:
                await websocket.send_bytes(await serializer.serialize(original))
                message = await websocket.receive(timeout=5)
                self.assertEqual(message.type, aiohttp.WSMsgType.BINARY)
                restored = await serializer.deserialize(message.data)
        self.assertEqual(restored.audio, original.audio)
        self.assertEqual(restored.sample_rate, original.sample_rate)
        self.assertEqual(restored.num_channels, original.num_channels)

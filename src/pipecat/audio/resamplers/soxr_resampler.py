#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import time

import numpy as np
import soxr

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler

_PROFILING = os.environ.get("CALLBOOK_PROFILING", "0") == "1"
_profiler = None

def _get_profiler():
    global _profiler
    if _profiler is None:
        try:
            from profiling.call_profiler import CallProfiler
            _profiler = CallProfiler.get()
        except Exception:
            _profiler = False
    return _profiler if _profiler else None


class SOXRAudioResampler(BaseAudioResampler):
    """Audio resampler implementation using the SoX resampler library.

    Quality levels (CPU cost high → low): VHQ > HQ > MQ > LQ > QQ
    For telephony (8kHz), MQ or LQ is more than sufficient.
    """

    def __init__(self, quality="MQ", **kwargs):
        self._quality = quality

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        if in_rate == out_rate:
            return audio
        if _PROFILING:
            t0 = time.monotonic()
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = soxr.resample(audio_data, in_rate, out_rate, quality=self._quality)
        result = resampled_audio.astype(np.int16).tobytes()
        if _PROFILING:
            p = _get_profiler()
            if p:
                p.record(f"soxr_resample_async_{in_rate}>{out_rate}", (time.monotonic() - t0) * 1000)
        return result

    def resample_sync(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """
        Synchronous resample method for use in non‐async callbacks.
        """
        if in_rate == out_rate:
            return audio
        if _PROFILING:
            t0 = time.monotonic()
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled = soxr.resample(audio_data, in_rate, out_rate, quality=self._quality)
        result = resampled.astype(np.int16).tobytes()
        if _PROFILING:
            p = _get_profiler()
            if p:
                p.record(f"soxr_resample_sync_{in_rate}>{out_rate}", (time.monotonic() - t0) * 1000)
        return result

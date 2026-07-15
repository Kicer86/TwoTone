"""Track-timeline knowledge: where a stream's content really starts and how to
decode it identically across ffmpeg builds.

This module owns the AAC encoder-delay (priming) story, told once:

AAC and similar lapped-transform codecs prepend ``initial_padding`` priming
samples that must be skipped on decode.  Containers signal that delay in two
ways: MP4/MOV carry it in an edit list that every ffmpeg build honours on
decode, while Matroska uses *CodecDelay* — and builds disagree about it.  Some
builds skip the priming automatically ("absorbing"), others decode it as real
leading samples and report the stream's ``start_time`` one encoder-delay frame
(~21 ms for AAC) too early ("exposing").  Every operation that reads, places or
re-encodes AAC audio must therefore be normalized, or the track shifts by one
frame depending on which build happens to run:

- ``_aac_priming_exposed`` detects the running build's convention once per run.
- ``_audio_content_start_ms`` returns a stream's true content start,
  independent of that convention.
- ``_decode_source_audio_to_flac`` decodes audio with the priming removed
  deterministically (via an ADTS remux that strips the CodecDelay signalling),
  so every downstream step works in a priming-free FLAC domain and AAC is
  encoded exactly once.
- ``_audio_needs_mkvmerge_normalization`` says when a raw mkvmerge remux would
  re-introduce the build dependency and the FLAC-domain flow is required.

The mixin also owns the plain stream-timeline helpers (ffprobe start/end
offsets and the mkvmerge ``--sync`` computation) that the priming logic builds
on.
"""

import logging
import os

from typing import Any

from ..utils import files_utils, generic_utils, process_utils, video_utils
from .melt_common import AudioStreamRef


class TrackTimelineMixin:
    """Stream-timeline and AAC-priming helpers for :class:`MeltPerformer`.

    Requires the host class to provide ``workspace`` and ``logger``.
    """

    workspace: files_utils.Workspace
    logger: logging.Logger
    _media_info_cache: dict[str, dict[str, Any]]
    _stream_info_cache: dict[tuple[str, str, int], dict[str, Any] | None]

    _aac_priming_exposed_cache: bool | None = None

    # Containers whose stream start offsets mkvmerge preserves on remux.
    _MKVMERGE_PRESERVES_START_EXTENSIONS = frozenset({".mkv", ".mk3d", ".mka", ".webm"})

    # --- plain stream-timeline probing -----------------------------------

    def _media_info(self, path: str) -> dict[str, Any]:
        info = self._media_info_cache.get(path)
        if info is None:
            info = video_utils.get_video_full_info(path, logger=self.logger)
            self._media_info_cache[path] = info
        return info

    def _stream_info(self, path: str, stream_type: str, stream_index: int) -> dict[str, Any] | None:
        cache_key = (path, stream_type, stream_index)
        if cache_key in self._stream_info_cache:
            return self._stream_info_cache[cache_key]

        info = self._media_info(path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") != stream_type:
                continue
            try:
                probed_index = int(stream.get("index", -1))
            except (TypeError, ValueError):
                continue
            if probed_index == stream_index:
                self._stream_info_cache[cache_key] = stream
                return stream
        self._stream_info_cache[cache_key] = None
        return None

    def _audio_stream_info(self, stream: AudioStreamRef) -> dict[str, Any] | None:
        return self._stream_info(stream.path, "audio", stream.stream_index)

    @staticmethod
    def _stream_start_offset_ms(stream: dict[str, Any] | None) -> int:
        if stream is None:
            return 0
        try:
            start_time = float(stream.get("start_time") or 0.0)
        except (TypeError, ValueError):
            return 0
        return max(0, round(start_time * 1000))

    @staticmethod
    def _stream_end_offset_ms(stream: dict[str, Any] | None) -> int | None:
        if stream is None:
            return None

        duration = stream.get("duration")
        if duration is not None:
            try:
                return TrackTimelineMixin._stream_start_offset_ms(stream) + round(float(duration) * 1000)
            except (TypeError, ValueError):
                pass

        tag_duration = stream.get("tags", {}).get("DURATION")
        if tag_duration is not None:
            return generic_utils.time_to_ms(tag_duration)

        return None

    def _source_stream_start_offset_ms(self, path: str, stream_type: str, stream_index: int) -> int:
        return self._stream_start_offset_ms(self._stream_info(path, stream_type, stream_index))

    def _source_stream_end_offset_ms(self, path: str, stream_type: str, stream_index: int) -> int | None:
        return self._stream_end_offset_ms(self._stream_info(path, stream_type, stream_index))

    def _mkvmerge_input_start_offset_ms(self, path: str, stream_type: str, stream_index: int) -> int:
        extension = os.path.splitext(path)[1].lower()
        if extension in self._MKVMERGE_PRESERVES_START_EXTENSIONS:
            return self._source_stream_start_offset_ms(path, stream_type, stream_index)
        return 0

    def _track_sync_offset_ms(
        self,
        path: str,
        stream_type: str,
        stream_index: int,
        desired_start_ms: int | None,
    ) -> int | None:
        if desired_start_ms is None:
            return None
        input_start_ms = self._mkvmerge_input_start_offset_ms(path, stream_type, stream_index)
        sync_offset_ms = desired_start_ms - input_start_ms
        return sync_offset_ms if sync_offset_ms else None

    # --- AAC encoder-delay (priming) handling ----------------------------

    def _source_audio_priming(
        self,
        video_path: str,
        logger: logging.Logger | None = None,
        *,
        audio_stream_index: int | None = None,
    ) -> tuple[str | None, int, int, dict[str, Any]]:
        """Return (codec_name, initial_padding_samples, sample_rate, stream).

        ``initial_padding`` is the lossy-codec encoder-delay (priming) sample count the
        container signals via Matroska *CodecDelay*; it is 0 for MP4/MOV (where the same
        delay is carried by an edit list that every ffmpeg build honours on decode).
        ``audio_stream_index`` selects a specific stream by its absolute container
        index; when omitted the first audio stream is used.
        """
        stream: dict[str, Any]
        if audio_stream_index is not None:
            stream = self._stream_info(video_path, "audio", audio_stream_index) or {}
        else:
            info = video_utils.get_video_full_info(video_path, logger=logger)
            streams = info.get("streams", [])
            stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
        codec = stream.get("codec_name")
        try:
            init_pad = int(stream.get("initial_padding") or 0)
        except (TypeError, ValueError):
            init_pad = 0
        try:
            sample_rate = int(stream.get("sample_rate") or 0)
        except (TypeError, ValueError):
            sample_rate = 0
        return codec, init_pad, sample_rate, stream

    def _aac_priming_exposed(self) -> bool:
        """Whether this ffmpeg build exposes AAC encoder-delay priming as a negative
        container start time instead of absorbing it.

        Builds disagree: some report a freshly encoded AAC stream's ``start_time`` as
        ``-encoder_delay`` (priming exposed, kept in the decoded samples), others as 0
        (priming absorbed).  The same convention applies when *reading* a source AAC
        stream, so on exposing builds a positive-offset (asO) audio stream's reported
        ``start_time`` is short by one frame (~21 ms) — which otherwise mis-places the
        patched track by exactly that frame.  Detected once and cached per run.
        """
        if self._aac_priming_exposed_cache is None:
            exposed = False
            probe = self.workspace.unique_file("priming_probe", "mka")
            try:
                process_utils.raise_on_error(
                    process_utils.start_process("ffmpeg", [
                        "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                        "-t", "0.5", "-c:a", "aac", probe,
                    ], logger=self.logger)
                )
                info = video_utils.get_video_full_info(probe, logger=self.logger)
                stream: dict[str, Any] = next(
                    (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
                    {},
                )
                exposed = float(stream.get("start_time") or 0.0) < -0.001
            except Exception:  # noqa: BLE001 - detection failure falls back to "absorbed"
                exposed = False
            finally:
                if not self.workspace.keep and os.path.exists(probe):
                    os.remove(probe)
            self._aac_priming_exposed_cache = exposed
        return self._aac_priming_exposed_cache

    def _audio_content_start_ms(self, stream: dict[str, Any] | None) -> int:
        """Audio stream's true content start (ms), priming-convention independent.

        On builds that expose AAC priming, the reported ``start_time`` is short by the
        encoder delay; add it back so positive-offset placement matches absorbing
        builds.  ``_stream_start_offset_ms`` clamps negatives to 0, so asR streams are
        unaffected; only positive (asO) offsets carry the leaked frame.
        """
        base = self._stream_start_offset_ms(stream)
        if not stream or stream.get("codec_name") != "aac" or not self._aac_priming_exposed():
            return base
        try:
            pad = int(stream.get("initial_padding") or 0)
            sample_rate = int(stream.get("sample_rate") or 0)
            raw_start = float(stream.get("start_time") or 0.0)
        except (TypeError, ValueError):
            return base
        if pad <= 0 or sample_rate <= 0:
            return base
        return max(0, round((raw_start + pad / sample_rate) * 1000))

    def _decode_source_audio_to_flac(
        self,
        source_video: str,
        output_path: str,
        *,
        sample_fmt: str = "s32",
        trim_start_ms: int | None = None,
        trim_end_ms: int | None = None,
        audio_stream_index: int | None = None,
        output_channels: int | None = None,
        output_sample_rate: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Decode the first audio stream of *source_video* to FLAC with the lossy-codec
        encoder-delay priming removed deterministically across all ffmpeg builds.

        When the source carries CodecDelay priming we strip that signalling by
        remuxing the AAC bitstream to ADTS (which has no priming metadata), decode
        the now-always-present priming, and trim exactly ``initial_padding`` samples
        ourselves — identical output on exposing and absorbing builds.

        ``trim_start_ms`` / ``trim_end_ms`` select a window on the source stream's
        timeline.  Callers that start from video-frame timestamps must first transform
        those timestamps into this timeline.  When the priming strip drops container
        timestamps, the window is re-anchored to the priming-independent content start
        so exposing and absorbing ffmpeg builds select identical samples.  The emitted
        FLAC is always rebased to its first selected sample.
        """
        codec, init_pad, sample_rate, stream = self._source_audio_priming(
            source_video, logger=logger, audio_stream_index=audio_stream_index,
        )
        source_map = f"0:{audio_stream_index}" if audio_stream_index is not None else "0:a:0"

        prime_offset_s = 0.0
        window_anchor_ms = 0
        input_path = source_video
        input_map = source_map
        adts_path: str | None = None
        if codec == "aac" and init_pad > 0 and sample_rate > 0:
            adts_path = f"{output_path}.priming.aac"
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", source_video, "-map", source_map,
                    "-c:a", "copy", "-f", "adts", adts_path,
                ], logger=logger)
            )
            input_path = adts_path
            input_map = "0:a:0"  # the remuxed ADTS file carries only the selected stream
            prime_offset_s = init_pad / sample_rate
            # ADTS carries no container timestamps, so the decoded frames start at 0
            # instead of the source's content start.  The raw start_time on builds
            # exposing AAC priming is one encoder-delay frame too early, so use the
            # same priming-independent content start as final track placement.
            window_anchor_ms = self._audio_content_start_ms(stream)

        filters: list[str] = []
        if prime_offset_s or trim_start_ms is not None or trim_end_ms is not None:
            start_s = prime_offset_s + max(0, (trim_start_ms or 0) - window_anchor_ms) / 1000
            atrim = f"atrim=start={start_s:.6f}"
            if trim_end_ms is not None:
                end_s = prime_offset_s + max(0, trim_end_ms - window_anchor_ms) / 1000
                atrim += f":end={end_s:.6f}"
            filters.append(atrim)
        # All downstream patching works in a priming-free, zero-based FLAC
        # timeline.  Rebase even a whole-stream decode so segment cutters never
        # accidentally compare a source-video timestamp with a preserved audio PTS.
        filters.append("asetpts=PTS-STARTPTS")

        args = ["-y", "-i", input_path, "-map", input_map]
        args += ["-filter:a", ",".join(filters)]
        if output_channels is not None:
            args += ["-ac", str(output_channels)]
        if output_sample_rate is not None:
            args += ["-ar", str(output_sample_rate)]
        args += ["-sample_fmt", sample_fmt, "-c:a", "flac", output_path]
        try:
            process_utils.raise_on_error(process_utils.start_process("ffmpeg", args, logger=logger))
        finally:
            if adts_path and not self.workspace.keep and os.path.exists(adts_path):
                os.remove(adts_path)

    def _extract_audio_to_flac(self, video_path: str, output_path: str, logger: logging.Logger | None = None) -> None:
        self._decode_source_audio_to_flac(video_path, output_path, sample_fmt="s32", logger=logger)

    def _decode_audio_stream_to_flac(
        self,
        stream: AudioStreamRef,
        output_path: str,
        *,
        sample_fmt: str = "s32",
        trim_start_ms: int | None = None,
        trim_end_ms: int | None = None,
        normalize_to: tuple[int, int, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Decode one explicitly selected audio stream to priming-free FLAC."""
        output_channels: int | None = None
        output_sample_rate: int | None = None
        if normalize_to is not None:
            output_channels, output_sample_rate, sample_fmt = normalize_to
        self._decode_source_audio_to_flac(
            stream.path,
            output_path,
            sample_fmt=sample_fmt,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            audio_stream_index=stream.stream_index,
            output_channels=output_channels,
            output_sample_rate=output_sample_rate,
            logger=logger,
        )

    def _extract_selected_audio_to_flac(
        self,
        stream: AudioStreamRef,
        output_path: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._decode_audio_stream_to_flac(stream, output_path, sample_fmt="s32", logger=logger)

    def _audio_needs_mkvmerge_normalization(self, stream_ref: AudioStreamRef) -> bool:
        """Return True when direct mkvmerge remux can shift decoded AAC timing.

        Non-Matroska AAC always needs a priming-free remux: mkvmerge would otherwise
        drop the container start offset.  Matroska normally preserves start, so raw
        passthrough is fine -- *except* when the AAC stream carries encoder-delay
        priming (CodecDelay / ``initial_padding``).  Remuxed raw, such a stream is
        placed build-dependently: builds that expose priming shift its decoded
        content by one frame (~21 ms) relative to a priming-stripped base track.
        Route those through the FLAC-domain flow too so the priming is removed
        deterministically, exactly as for non-Matroska inputs.
        """
        stream = self._audio_stream_info(stream_ref)
        if stream is None or stream.get("codec_name") != "aac":
            return False
        extension = os.path.splitext(stream_ref.path)[1].lower()
        if extension not in self._MKVMERGE_PRESERVES_START_EXTENSIONS:
            return True
        try:
            init_pad = int(stream.get("initial_padding") or 0)
        except (TypeError, ValueError):
            init_pad = 0
        return init_pad > 0

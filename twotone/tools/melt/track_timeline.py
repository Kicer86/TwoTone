"""Track-timeline knowledge: where a stream's content really starts and how to
decode it identically across ffmpeg builds.

This module owns the AAC encoder-delay (priming) story, told once:

AAC and similar lapped-transform codecs prepend ``initial_padding`` priming
samples that must be skipped on decode.  Containers signal that delay in two
ways: MP4/MOV carry it in an edit list that every ffmpeg build honours on
decode, while Matroska uses *CodecDelay* — and builds disagree about it.  Some
builds skip the priming automatically ("absorbing"), others decode it as real
leading samples and report the stream's ``start_time`` one encoder-delay frame
(~21 ms for AAC) too early ("exposing").  Every operation that decodes, places
or re-encodes AAC audio must therefore be normalized, or the track shifts by
one frame depending on which build happens to run.  Untouched audio remuxed
together with the base video from the same Matroska container keeps its
original timestamps and CodecDelay and does not enter this flow:

- ``aac_priming_exposed`` detects the running build's convention once per run.
- ``audio_content_start_ms`` returns a stream's true content start,
  independent of that convention.
- ``decode_audio_stream_to_flac`` decodes audio with the priming removed
  deterministically (via an ADTS remux that strips the CodecDelay signalling),
  so every downstream step works in a priming-free FLAC domain and AAC is
  encoded exactly once.
- ``audio_mkvmerge_normalization_reason`` says when a raw mkvmerge remux would
  re-introduce the build dependency and why the FLAC-domain flow is required.

The service also owns the plain stream-timeline helpers (ffprobe start/end
offsets and the mkvmerge ``--sync`` computation) that the priming logic builds
on.
"""

import logging
import os
from typing import Any

from ..utils import files_utils, generic_utils, process_utils, video_utils
from .melt_common import AudioStreamRef

# Containers whose stream start offsets mkvmerge preserves on remux.
_MKVMERGE_PRESERVES_START_EXTENSIONS = frozenset({".mkv", ".mk3d", ".mka", ".webm"})


def mkvmerge_preserves_start(path: str) -> bool:
    """Whether mkvmerge keeps the stream start offsets of *path* on remux."""
    return os.path.splitext(path)[1].lower() in _MKVMERGE_PRESERVES_START_EXTENSIONS


class TrackTimeline:
    """Run-scoped stream-timeline and AAC-priming service."""

    def __init__(self, workspace: files_utils.Workspace, logger: logging.Logger) -> None:
        self.workspace = workspace
        self.logger = logger
        self._media_info_cache: dict[str, dict[str, Any]] = {}
        self._stream_info_cache: dict[tuple[str, str, int], dict[str, Any] | None] = {}
        self._input_file_ids: dict[str, int] = {}
        self._aac_priming_exposed_cache: bool | None = None

    def set_input_file_ids(self, file_ids: dict[str, int]) -> None:
        self._input_file_ids = dict(file_ids)

    def _input_file_description(self, path: str) -> str:
        file_id = self._input_file_ids.get(path)
        return f"file #{file_id}" if file_id is not None else "input file"

    def audio_source_description(self, stream: AudioStreamRef) -> str:
        return (
            f"{self._input_file_description(stream.path)}, "
            f"audio track #{stream.mkvmerge_track_id}"
        )

    # --- plain stream-timeline probing -----------------------------------

    def media_info(self, path: str) -> dict[str, Any]:
        info = self._media_info_cache.get(path)
        if info is None:
            info = video_utils.get_video_full_info(path, logger=self.logger)
            self._media_info_cache[path] = info
        return info

    def stream_info(self, path: str, stream_type: str, ffprobe_stream_index: int) -> dict[str, Any] | None:
        cache_key = (path, stream_type, ffprobe_stream_index)
        if cache_key in self._stream_info_cache:
            return self._stream_info_cache[cache_key]

        info = self.media_info(path)
        for stream in info.get("streams", []):
            if stream.get("codec_type") != stream_type:
                continue
            try:
                probed_index = int(stream.get("index", -1))
            except (TypeError, ValueError):
                continue
            if probed_index == ffprobe_stream_index:
                self._stream_info_cache[cache_key] = stream
                return stream
        self._stream_info_cache[cache_key] = None
        return None

    def audio_stream_info(self, stream: AudioStreamRef) -> dict[str, Any] | None:
        return self.stream_info(stream.path, "audio", stream.ffprobe_stream_index)

    @staticmethod
    def stream_start_offset_ms(stream: dict[str, Any] | None) -> int:
        if stream is None:
            return 0
        try:
            start_time = float(stream.get("start_time") or 0.0)
        except (TypeError, ValueError):
            return 0
        return max(0, round(start_time * 1000))

    @staticmethod
    def stream_end_offset_ms(stream: dict[str, Any] | None) -> int | None:
        if stream is None:
            return None

        duration = stream.get("duration")
        if duration is not None:
            try:
                return TrackTimeline.stream_start_offset_ms(stream) + round(float(duration) * 1000)
            except (TypeError, ValueError):
                pass

        tag_duration = stream.get("tags", {}).get("DURATION")
        if tag_duration is not None:
            return generic_utils.time_to_ms(tag_duration)

        return None

    def source_stream_start_offset_ms(self, path: str, stream_type: str, ffprobe_stream_index: int) -> int:
        return self.stream_start_offset_ms(self.stream_info(path, stream_type, ffprobe_stream_index))

    def source_stream_end_offset_ms(self, path: str, stream_type: str, ffprobe_stream_index: int) -> int | None:
        return self.stream_end_offset_ms(self.stream_info(path, stream_type, ffprobe_stream_index))

    def mkvmerge_input_start_offset_ms(
        self,
        path: str,
        stream_type: str,
        ffprobe_stream_index: int,
    ) -> int:
        if mkvmerge_preserves_start(path):
            return self.source_stream_start_offset_ms(path, stream_type, ffprobe_stream_index)
        return 0

    def track_sync_offset_ms(
        self,
        path: str,
        stream_type: str,
        ffprobe_stream_index: int,
        desired_start_ms: int | None,
    ) -> int | None:
        if desired_start_ms is None:
            return None
        input_start_ms = self.mkvmerge_input_start_offset_ms(path, stream_type, ffprobe_stream_index)
        sync_offset_ms = desired_start_ms - input_start_ms
        return sync_offset_ms if sync_offset_ms else None

    # --- AAC encoder-delay (priming) handling ----------------------------

    def aac_priming_exposed(self) -> bool:
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

    def audio_content_start_ms(self, stream: dict[str, Any] | None) -> int:
        """Audio stream's true content start (ms), priming-convention independent.

        On builds that expose AAC priming, the reported ``start_time`` is short by the
        encoder delay; add it back so positive-offset placement matches absorbing
        builds.  ``stream_start_offset_ms`` clamps negatives to 0, so asR streams are
        unaffected; only positive (asO) offsets carry the leaked frame.
        """
        base = self.stream_start_offset_ms(stream)
        if not stream or stream.get("codec_name") != "aac":
            return base
        try:
            pad = int(stream.get("initial_padding") or 0)
            sample_rate = int(stream.get("sample_rate") or 0)
            raw_start = float(stream.get("start_time") or 0.0)
        except (TypeError, ValueError):
            return base
        if pad <= 0 or sample_rate <= 0:
            return base
        if not self.aac_priming_exposed():
            return base
        return max(0, round((raw_start + pad / sample_rate) * 1000))

    def decode_audio_stream_to_flac(
        self,
        stream_ref: AudioStreamRef,
        output_path: str,
        *,
        sample_fmt: str = "s32",
        trim_start_ms: int | None = None,
        trim_end_ms: int | None = None,
        normalize_to: tuple[int, int, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Decode one selected audio stream to FLAC with the lossy-codec
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

        ``normalize_to`` optionally forces ``(channels, sample_rate, sample_fmt)``.
        """
        stream = self.audio_stream_info(stream_ref) or {}
        codec = stream.get("codec_name")
        try:
            init_pad = int(stream.get("initial_padding") or 0)
        except (TypeError, ValueError):
            init_pad = 0
        try:
            sample_rate = int(stream.get("sample_rate") or 0)
        except (TypeError, ValueError):
            sample_rate = 0
        progress_source = self.audio_source_description(stream_ref)
        source_codec = codec.upper() if codec else "audio"
        source_map = f"0:{stream_ref.ffprobe_stream_index}"

        prime_offset_s = 0.0
        # Keep the public trim arguments on the source-stream timeline, then
        # move them into the decoded-content timeline here.  This must apply
        # to every codec, not only the AAC-to-ADTS priming path below.
        window_anchor_ms = self.audio_content_start_ms(stream)
        input_path = stream_ref.path
        input_map = source_map
        adts_path: str | None = None
        if codec == "aac" and init_pad > 0 and sample_rate > 0:
            adts_path = f"{output_path}.priming.aac"
            process_utils.raise_on_error(
                process_utils.start_process("ffmpeg", [
                    "-y", "-i", stream_ref.path, "-map", source_map,
                    "-c:a", "copy", "-f", "adts", adts_path,
                ], show_progress=True,
                    progress_description=f"Extracting AAC bitstream from {progress_source}",
                    logger=logger)
            )
            input_path = adts_path
            input_map = "0:a:0"  # the remuxed ADTS file carries only the selected stream
            prime_offset_s = init_pad / sample_rate
            # ADTS carries no container timestamps, so the decoded frames start at 0
            # instead of the source's content start.  ``window_anchor_ms`` above
            # already uses that same priming-independent content start for every
            # decode path.

        # Containers such as MP4/MOV can preserve a positive audio-frame PTS
        # even though the trim window above is relative to content.  Rebase
        # before cutting so ``atrim`` always sees that same zero-based content
        # timeline.  Rebase again afterwards because a selected window must
        # itself start at zero for downstream patching.
        filters = ["asetpts=PTS-STARTPTS"]
        if prime_offset_s or trim_start_ms is not None or trim_end_ms is not None:
            start_s = prime_offset_s + max(0, (trim_start_ms or 0) - window_anchor_ms) / 1000
            atrim = f"atrim=start={start_s:.6f}"
            if trim_end_ms is not None:
                end_s = prime_offset_s + max(0, trim_end_ms - window_anchor_ms) / 1000
                atrim += f":end={end_s:.6f}"
            filters.append(atrim)
            filters.append("asetpts=PTS-STARTPTS")

        args = ["-y", "-i", input_path, "-map", input_map]
        args += ["-filter:a", ",".join(filters)]
        if normalize_to is not None:
            output_channels, output_sample_rate, sample_fmt = normalize_to
            args += ["-ac", str(output_channels), "-ar", str(output_sample_rate)]
        args += ["-sample_fmt", sample_fmt, "-c:a", "flac", output_path]
        try:
            process_utils.raise_on_error(process_utils.start_process(
                "ffmpeg", args, show_progress=True,
                progress_description=f"Decoding {progress_source}: {source_codec} -> FLAC",
                logger=logger,
            ))
        finally:
            if adts_path and not self.workspace.keep and os.path.exists(adts_path):
                os.remove(adts_path)

    def audio_mkvmerge_normalization_reason(self, stream_ref: AudioStreamRef) -> str | None:
        """Describe why direct mkvmerge passthrough would change AAC timing.

        Non-Matroska AAC needs a priming-free remux because mkvmerge would
        otherwise drop the container start offset. Matroska normally preserves
        start, except when AAC carries CodecDelay priming. The caller may keep
        an untouched base-container track to preserve its original timing.
        """
        stream = self.audio_stream_info(stream_ref)
        if stream is None or stream.get("codec_name") != "aac":
            return None
        try:
            init_pad = int(stream.get("initial_padding") or 0)
        except (TypeError, ValueError):
            init_pad = 0
        if init_pad > 0:
            return f"removing AAC encoder delay ({init_pad} priming samples)"
        if not mkvmerge_preserves_start(stream_ref.path):
            return "preserving the AAC start time across container conversion"
        return None

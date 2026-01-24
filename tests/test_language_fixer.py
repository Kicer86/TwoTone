import os
import unittest

from twotone.tools.utils import process_utils, subtitles_utils, video_utils
from common import TwoToneTestCase, build_test_video, get_audio, get_video, run_twotone, write_subtitle


class LanguageFixerTests(TwoToneTestCase):
    def test_subtitle_language_detection(self):
        video_input = get_video("moon.mp4")
        subtitle_path = write_subtitle(
            os.path.join(self.wd.path, "sub.srt"),
            [
                "1",
                "00:00:00,000 --> 00:00:02,000",
                "Hello world.",
                "",
                "2",
                "00:00:02,500 --> 00:00:04,000",
                "This is a sample subtitle in English.",
                "",
            ],
        )

        subtitle = subtitles_utils.SubtitleFile(path=subtitle_path, language=None, encoding="utf-8")
        output_video = os.path.join(self.wd.path, "missing_subtitle_lang.mkv")
        video_utils.generate_mkv(input_video=video_input, output_path=output_video, subtitles=[subtitle])

        run_twotone("language_fix", [self.wd.path], ["--no-dry-run", "--audio"])

        info = video_utils.get_video_data_mkvmerge(output_video)
        subtitles = info["tracks"].get("subtitle", [])
        self.assertEqual(len(subtitles), 1)
        self.assertEqual(subtitles[0]["language"], "eng")

    def test_audio_language_detection_from_track_name(self):
        audio_path = get_audio("807184__logicmoon__mirrors.wav")

        base_video = os.path.join(self.wd.path, "base_no_audio.mkv")
        build_test_video(base_video, self.wd.path, "moon.mp4")

        output_video = os.path.join(self.wd.path, "missing_audio_lang.mkv")
        audio = {"path": audio_path, "language": None, "name": "English"}
        video_utils.generate_mkv(input_video=base_video, output_path=output_video, audios=[audio])

        run_twotone("language_fix", [self.wd.path], ["--no-dry-run"])

        info = video_utils.get_video_data_mkvmerge(output_video)
        audios = info["tracks"].get("audio", [])
        self.assertEqual(len(audios), 1)
        self.assertEqual(audios[0]["language"], "eng")

    def test_webvtt_extraction_fails(self):
        vtt_path = write_subtitle(
            os.path.join(self.wd.path, "sub.vtt"),
            [
                "WEBVTT",
                "",
                "00:00:00.000 --> 00:00:00.500",
                "Hello",
            ],
        )
        output_video = os.path.join(self.wd.path, "webvtt_subs.mkv")
        status = process_utils.start_process(
            "ffmpeg",
            [
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x240:d=1",
                "-i",
                vtt_path,
                "-map",
                "0:v:0",
                "-map",
                "1:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:s",
                "copy",
                output_video,
            ],
        )
        self.assertEqual(status.returncode, 0, status.stderr)

        info = video_utils.get_video_full_info_mkvmerge(output_video)
        subtitle_tracks = [track for track in info.get("tracks", []) if track.get("type") in ("subtitle", "subtitles")]
        self.assertTrue(subtitle_tracks, "Expected at least one subtitle track in generated file.")

        track = None
        for candidate in subtitle_tracks:
            props = candidate.get("properties", {})
            codec_id = (props.get("codec_id") or "").lower()
            if codec_id.startswith("d_webvtt"):
                track = candidate
                break

        self.assertIsNotNone(track, "Expected a D_WEBVTT subtitle track in generated file.")
        props = track.get("properties", {})
        codec_id = (props.get("codec_id") or "").lower()
        self.assertIn("webvtt", codec_id or track.get("codec", "").lower())
        self.assertTrue(
            codec_id.startswith("d_webvtt"),
            f"Expected D_WEBVTT subtitle track, got codec_id={codec_id!r}",
        )

        extracted = os.path.join(self.wd.path, "extracted.vtt")
        status = process_utils.start_process("mkvextract", ["tracks", output_video, f"{track['id']}:{extracted}"])
        self.assertNotEqual(status.returncode, 0)


if __name__ == "__main__":
    unittest.main()

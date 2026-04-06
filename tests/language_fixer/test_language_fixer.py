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

        run_twotone("language_fix", [self.wd.path], ["--no-dry-run"])

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

        run_twotone("language_fix", [self.wd.path, "--audio"], ["--no-dry-run"])

        info = video_utils.get_video_data_mkvmerge(output_video)
        audios = info["tracks"].get("audio", [])
        self.assertEqual(len(audios), 1)
        self.assertEqual(audios[0]["language"], "eng")

    def test_audio_language_detection_from_filename(self):
        audio_path = get_audio("807184__logicmoon__mirrors.wav")
        video_input = get_video("moon.mp4")

        output_video = os.path.join(self.wd.path, "sample_endub.mp4")
        status = process_utils.start_process(
            "ffmpeg",
            [
                "-y",
                "-i",
                video_input,
                "-i",
                audio_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                output_video,
            ],
        )
        self.assertEqual(status.returncode, 0, status.stderr)

        run_twotone("language_fix", [self.wd.path, "--audio"], ["--no-dry-run"])

        # language_fix converts non-MKV files to MKV
        converted_video = os.path.splitext(output_video)[0] + ".mkv"
        info = video_utils.get_video_data(converted_video)
        audios = info.get("audio", [])
        self.assertEqual(len(audios), 1)
        self.assertEqual(audios[0]["language"], "eng")

    def test_webvtt_subtitle_handled_gracefully(self):
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
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
                "-i", vtt_path,
                "-map", "0:v:0", "-map", "1:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:s", "copy",
                output_video,
            ],
        )
        self.assertEqual(status.returncode, 0, status.stderr)

        # language_fix should handle unextractable WebVTT subtitles without crashing
        run_twotone("language_fix", [self.wd.path], ["--no-dry-run"])


if __name__ == "__main__":
    unittest.main()


import json
import os
import subprocess
import sys
import tempfile

import mod.vof_algo as vof_algo
import mod.video_probing as video_probing


def load_recipe(path: str) -> dict:
    data = dict()

    with open(path, 'r') as f:
        data = json.load(f)

    return data


def extract_audio(path: str, scale: float, wd: str) -> ():
    audio_codec_type = video_probing.audio_codec(path)
    audio_codec_ext = audio_codec_type
    audio_codec = "copy"

    # TODO: I couldn't figure out how to simply copy this codec, so for now it is being transformed into ogg
    if audio_codec_type == "cook":
        audio_codec_ext = "flac"
        audio_codec = "flac"

    ffmpeg_exec = ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-acodec", audio_codec]

    # check if we need to scale audio to match desired length
    if abs(scale - 1.0) > 0.001:
        ffmpeg_exec.append("-filter:a")
        ffmpeg_exec.append("atempo=" + str(scale))

    tmp_file=tempfile.NamedTemporaryFile(dir=wd)
    output_file = tmp_file.name + "." + audio_codec_ext
    ffmpeg_exec.append(output_file)

    process = subprocess.Popen(ffmpeg_exec)
    return (output_file, process)



def process_recipe(recipe: dict):
    try:
        files = recipe.get("files")
        assert len(files) == 2              # TODO: handling many files at once would be nice

        # when vof.py is read, "common timestamps" will be replaced with "overlapping segments"
        # so no calcualtions are needed here
        common_timestamps = recipe.get("common timestamps")
        assert len(common_timestamps) > 1

        file1_timestamps = []
        file2_timestamps = []
        for timestamp_pair in common_timestamps:
            file1_timestamps.append(timestamp_pair.get("#1"))
            file2_timestamps.append(timestamp_pair.get("#2"))

        file1_fps = video_probing.fps(files[0])
        file2_fps = video_probing.fps(files[1])
        file1_len = video_probing.length(files[0])
        file2_len = video_probing.length(files[1])
        adjustments = vof_algo.adjust_videos(file1_timestamps, file2_timestamps,
                                             file1_fps, file2_fps,
                                             file1_len, file2_len)

        segments = adjustments.get("segments")
        assert len(segments) == 1           # TODO: handle more

        segment = segments[0]
        file1_segment = segment.get("#1")
        file2_segment = segment.get("#2")
        file1_begin = file1_segment.get("begin")
        file1_end = file1_segment.get("end")
        file2_begin = file2_segment.get("begin")
        file2_end = file2_segment.get("end")

        file2_len_scaled = file2_end - file2_begin

        temp_location = tempfile.gettempdir() + "/MKVAssembler/" + str(os.getpid()) + "/"
        os.makedirs(name = temp_location)

        # extract audio from files
        file2_audio, file2_audio_process = extract_audio(files[1], file2_len / file2_len_scaled, temp_location)

        # if second file's audio track is shorter, embed it in first file's audio
        if file2_begin > file1_begin or file2_end < file1_end:
            file1_audio, file1_audio_process = extract_audio(files[0], 1.0, temp_location)
            file2_audio_process.wait()
            file1_audio_process.wait()

            ffmpeg_exec = ["ffmpeg", "-i", file1_audio, "-i", file2_audio]
            ffmpeg_exec.append("-filter_complex")

            filter_complex = str()
            audio_inputs = ""
            audio_parts = 1
            if file2_begin > file1_begin:
                # take intro part from first file
                audio_input = "[audio" + str(audio_parts) + "]"
                filter_complex += "[0:0]atrim=end=" + str(file2_begin) + audio_input + ";"
                audio_parts += 1
                audio_inputs += audio_input

            audio_inputs += "[1:0]"

            if file2_end < file1_end:
                # take outro part
                audio_input = "[audio" + str(audio_parts) + "]"
                filter_complex += "[0:0]atrim=start=" + str(file2_end) + audio_input + ";"
                audio_parts += 1
                audio_inputs += audio_input

            filter_complex += audio_inputs + "concat=n=" + str(audio_parts) + ":v=0:a=1[output]"
            #filter_complex += "\""

            ffmpeg_exec.append(filter_complex)

            ffmpeg_exec.append("-map")
            ffmpeg_exec.append("[output]")

            output_file = temp_location + "mixed_audio.flac"
            ffmpeg_exec.append(output_file)

            subprocess.Popen(ffmpeg_exec).wait()
            file2_audio = output_file
        else:
            file2_audio_process.wait()

        # build mkv file
        subprocess.Popen(["mkvmerge", "-o", temp_location + "output.mkv", files[0], file2_audio]).wait()

    except:
        print("Invalid recipe structure")
        exit(1)


if len(sys.argv) != 2:
    print(f"python {sys.argv[0]} recipe.json")
    exit(1)

recipe = load_recipe(sys.argv[1])
process_recipe(recipe)


# recipe structure:
#
# {
#   "files": ["file1,avi", "file2.mp4", "file3.mp4"],
#   "common timestamps": [
#     { "#1": 12.34, "#2": 12.34 },
#     { "#1": 25.7,  "#2": 25.6 }
#   ]
# }
#

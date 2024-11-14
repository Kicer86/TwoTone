
import cv2 as cv
import json
import numpy as np
import os
from PIL import Image
import re
import sys
import subprocess
from scipy.stats import entropy
import shutil
import tempfile

import mod.video_probing as video_probing
import mod.vof_algo as vof_algo


def process_video(path: str, wd: str) -> []:
    os.makedirs(name = wd)
    process = subprocess.Popen(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-filter:v", "scale=-1:240,select=gt(scene\,0.3),showinfo",
                               "-fps_mode", "passthrough", wd + "/%05d.png"],
                              stderr=subprocess.PIPE)

    result = {}

    while True:
        line_raw = process.stderr.readline()
        if not line_raw:
            break

        line = line_raw.decode("utf-8")
        if line[1:17] == "Parsed_showinfo_":
            matched = re.search("^\[Parsed_showinfo_.+ n: *([0-9]+) .+ pts_time:([0-9\.]+).+", line)

            if matched:
                frame_id = int(matched.group(1)) + 1
                time_sig = float(matched.group(2))

                result[frame_id] = { "time": time_sig,
                                     "path": os.path.join(wd, "{:05}.png".format(frame_id)) }

    return result


def generate_hashes( scenes: {}) -> []:
    for scene, params in scenes.items():
        frame_path = params["path"]
        image = cv.imread(frame_path)
        img_hash = cv.img_hash.blockMeanHash(image)
        # img_hash = int.from_bytes(img_hash_raw.tobytes(), byteorder='big', signed=False)
        params["hash"] = img_hash


def frame_entropy(path: str) -> float:
    pil_image = Image.open(path)
    image = np.array(pil_image.convert("L"))
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram = histogram / float(np.sum(histogram))
    e = entropy(histogram)
    return e;


def filter_low_detailed(scenes: {}):
    valuable_scenes = { scene: params for scene, params in scenes.items() if frame_entropy(params["path"]) > 4}
    return valuable_scenes

output = {}

if len(sys.argv) < 3 or len(sys.argv) > 5:
    print(f"python {sys.argv[0]} video1 video2 [output.json] [matching-timestamps.csv]")
    exit(1)

else:
    video1 = sys.argv[1]
    video2 = sys.argv[2]
    output_json = sys.argv[3]
    timestamps_csv = sys.argv[4] if len(sys.argv) == 5 else None

    temp_location = tempfile.gettempdir() + "/VOF/" + str(os.getpid()) + "/"
    video1_scenes_location = temp_location + "1"
    video2_scenes_location = temp_location + "2"

    # filters to be considered: atadenoise,hue=s=0,scdet=s=1:t=10
    video1_scenes = process_video(video1, video1_scenes_location)
    video2_scenes = process_video(video2, video2_scenes_location)

    print(f"Scene changes for video #1: {len(video1_scenes)}")
    print(f"Scene changes for video #2: {len(video2_scenes)}")

    video1_fps = video_probing.fps(video1)
    video2_fps = video_probing.fps(video2)

    video1_len = video_probing.length(video1)
    video2_len = video_probing.length(video2)

    video1_scenes = filter_low_detailed(video1_scenes)
    video2_scenes = filter_low_detailed(video2_scenes)

    print(f"Scenes for video #1 after filtration: {len(video1_scenes)}")
    print(f"Scenes for video #2 after filtration: {len(video2_scenes)}")

    # perform matching
    generate_hashes(video1_scenes)
    generate_hashes(video2_scenes)

    # find corresponding scenes
    hash_algo = cv.img_hash.BlockMeanHash().create()
    matching_frames = vof_algo.match_scenes(video1_scenes, video2_scenes,
                                            lambda l, r: hash_algo.compare(l, r) < 10)

    if len(matching_frames) > 1:
        print("first matching pair: {}. last matching pair {}".format(matching_frames[0], matching_frames[-1]))

        matching_timestamps1 = []
        matching_timestamps2 = []

        for (match1, match2) in matching_frames:
            matching_timestamps1.append(video1_scenes[match1]["time"])
            matching_timestamps2.append(video2_scenes[match2]["time"])

        output = vof_algo.adjust_videos(matching_timestamps1, matching_timestamps2,
                                        video1_fps, video2_fps,
                                        video1_len, video2_len)

        #generate csv file
        if timestamps_csv:
            timestamps_file = open(timestamps_csv, "w")

            for (timestamp1, timestamp2) in zip(matching_timestamps1, matching_timestamps2):
                timestamps_file.write(f"{timestamp1}, {timestamp2}\n")

            timestamps_file.close()

    else:
        print(f"Found: {len(matching_frames)} matching frames. At least two are necessary")

    shutil.rmtree(temp_location)

# generate json file
if output_json:
    output_file = open(output_json, "w")
    output_file.write(json.dumps(output))
    output_file.close()
else:
    print(json.dumps(output))

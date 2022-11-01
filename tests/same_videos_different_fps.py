
import sys
sys.path.append("..")

import json
import subprocess
import sys

import video_probing
import unit_tests.utils


vof_script = sys.argv[1]
test_videos_dir = sys.argv[2]

video_pairs=[("movie-2160p.mp4", "movie-360p.mp4"),
             ("movie-720p-ntsc_fps.mp4", "movie-360p.mp4"),
             ("movie-720p-ntsc_film_fps.mp4", "movie-720p-120_fps.mp4"),
             ("movie-720p-film_fps.mp4", "movie-720p-20_fps.mp4")]

passes = 0

for (video1, video2) in video_pairs:

    video1_path = test_videos_dir + "/" + video1
    video2_path = test_videos_dir + "/" + video2

    print(f"Starting VOF for:\n{video1_path}\n{video2_path}")
    status = subprocess.run(["python", vof_script, video1_path, video2_path],
                            capture_output=True)

    if status.returncode != 0:
        print("VOF exited with error")
        exit(1)

    output = json.loads(status.stdout)

    video1_len = video_probing.length(video1_path)
    video2_len = video_probing.length(video2_path)
    expected_result = {
        "segments": [
            {
                "#1": {
                    "begin": unit_tests.utils.Around(0.0),
                    "end": unit_tests.utils.Around(video1_len)
                },
                "#2": {
                    "begin": unit_tests.utils.Around(0.0),
                    "end": unit_tests.utils.Around(video2_len)
                }
            }
        ]
    }

    if output != expected_result:
        print("No valid segments data")
        exit(1)

    passes = passes + 1

if passes == len(video_pairs):
    print("All cases passed")
else:
    print("Some tests failed")
    exit(1)

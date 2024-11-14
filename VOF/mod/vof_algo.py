
import cv2 as cv
import numpy as np


def match_scenes(video1_scenes: {}, video2_scenes: {}, comp) -> []:
    # O^2 solution, but it should do
    matches = []

    for scene1, params1 in video1_scenes.items():
        for scene2, params2 in video2_scenes.items():
            if comp(params1["hash"], params2["hash"]):
                matches.append((scene1, scene2))
                break

    return matches


def adjust_videos(video1_keyframes: [], video2_keyframes: [],
                  video1_fps: float, video2_fps: float,
                  video1_length: float, video2_length: float) -> {}:
    assert len(video1_keyframes) == len(video2_keyframes)
    assert len(video1_keyframes) > 1


    # calculate % of diff for first and last frame
    video1_last_first_diff = video1_keyframes[-1] - video1_keyframes[0]
    video2_last_first_diff = video2_keyframes[-1] - video2_keyframes[0]

    assert video1_last_first_diff > 0
    assert video2_last_first_diff > 0

    scale_factor = video1_last_first_diff / video2_last_first_diff

    # apply scaling and movement:
    # use first keyframe of first video as a origin and bas for all other points
    video1_begin = 0.0
    video2_begin = video1_keyframes[0] - video2_keyframes[0] * scale_factor
    video2_length = video1_keyframes[0] + (video2_length - video2_keyframes[0]) * scale_factor

    video1_segment = {
        "begin": video1_begin,
        "end": video1_length
    }

    video2_segment = {
        "begin": video2_begin,
        "end": video2_length
    }

    segment_scope = {
        "#1": video1_segment,
        "#2": video2_segment
    }

    segments = [segment_scope]

    return {"segments": segments}


def are_timestamps_equal(timestamp1: float, timestamp2: float, video1_fps: float, video2_fps: float) -> bool:
    min_fps = min([video1_fps, video2_fps])
    frame_duration = 1 / min_fps

    return abs(timestamp1 - timestamp2) < frame_duration

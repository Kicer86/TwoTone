
import cv2
import imagehash
import logging
import unittest
import numpy as np
import os

from PIL import Image
from functools import partial
from overrides import override
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List

import twotone.tools.utils as utils
from twotone.tools.melt import Melter, DuplicatesSource
from twotone.tools.utils2 import process
from common import WorkingDirectoryForTest, FileCache, add_test_media, add_to_test_dir, current_path, get_video, hashes, list_files


class Duplicates(DuplicatesSource):
    def __init__(self, interruption: utils.InterruptibleProcess):
        super().__init__(interruption)
        self.duplicates = {}

    def setDuplicates(self, duplicates: Dict):
        self.duplicates = duplicates

    @override
    def collect_duplicates(self) -> Dict[str, List[str]]:
        return self.duplicates


class MeltingTest(unittest.TestCase):

    def setUp(self):
        logging.getLogger("Melter").setLevel(logging.DEBUG)

    def test_simple_duplicate_detection(self):
        return
        with WorkingDirectoryForTest() as td:
            file1 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v1"])
            file2 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v2"])
            files = [*file1, *file2]

            interruption = utils.InterruptibleProcess()
            duplicates = Duplicates(interruption)
            duplicates.setDuplicates({"Grass": files})

            files_before = hashes(td.path)
            self.assertEqual(len(files_before), 2)

            melter = Melter(interruption, duplicates, live_run = True)
            melter.melt()

            # expect second file to be removed
            files_after = hashes(td.path)
            self.assertEqual(len(files_after), 1)

            # check if file was not altered
            self.assertTrue(files_after.items() < files_before.items())


    def test_dry_run_is_being_respected(self):
        return
        with WorkingDirectoryForTest() as td:
            file1 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v1"])
            file2 = add_test_media("Grass - 66810.mp4", td.path, suffixes = ["v2"])
            files = [*file1, *file2]

            interruption = utils.InterruptibleProcess()
            duplicates = Duplicates(interruption)
            duplicates.setDuplicates({"Grass": files})

            files_before = hashes(td.path)
            self.assertEqual(len(files_before), 2)

            melter = Melter(interruption, duplicates, live_run = False)
            melter.melt()

            # expect no changes in files
            files_after = hashes(td.path)
            self.assertEqual(files_before, files_after)


    def test_same_multiscene_video_duplicate_detection(self):
        file_cache = FileCache("TwoToneTests")

        with WorkingDirectoryForTest() as td:
            def gen_sample(out_path: Path):
                files = ["Atoms - 8579.mp4", "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov", "Frog - 113403.mp4", "sea-waves-crashing-on-beach-shore-4793288.mp4", "Woman - 58142.mp4"]

                #unify fps
                output_files = []
                for file in files:
                    input_path = get_video(file)
                    output_path = os.path.join(td.path, file + ".mp4")
                    process.start_process("ffmpeg", ["-i", input_path, "-r", "25", "-vf", "fps=25", "-c:v", "libx265", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", output_path])
                    output_files.append(output_path)

                # concatenate
                files_list_path = os.path.join(td.path, "filelist.txt")
                with open(files_list_path, "w", encoding="utf-8") as f:
                    for path in output_files:
                        # Escape single quotes if needed
                        safe_path = path.replace("'", "'\\''")
                        f.write(f"file '{safe_path}'\n")

                process.start_process("ffmpeg", ["-f", "concat", "-safe", "0", "-i", files_list_path, "-c", "copy", str(out_path)])

            def gen_vhs(out_path: Path, input: str):
                """
                    Process input file and worse its quality
                """
                duration=88                                                                                 # duration of original video

                vf = ",".join([
                    #"fps=26.5",                                                                             # use non standard fps
                    #"setpts=PTS/1.05",                                                                      # speed it up by 5%
                    "boxblur=enable='between(t,5,10)':lr=2",                                                # add some blur
                    f"crop=w=iw*0.9:h=ih*0.9:x='(iw-iw*0.9)*t/{duration}':y='(ih-ih*0.9)*t/{duration}'",    # add a crop (90% H and W) which moves from top left corner to bottom right
                    f"scale={960}:{720}"                                                                    # scale to 4:3
                ])

                af = "" #"atempo=1.05"

                args = [
                    "-i", input,
                    "-filter_complex", vf,
                    "-filter:a", af,
                    "-c:v", "libx265",
                    "-crf", "40",                                                                           # use low quality
                    "-preset", "slow",
                    "-pix_fmt", "yuv420p",
                    str(out_path)
                ]

                process.start_process("ffmpeg", args)

            file1 = file_cache.get_or_generate("melter_tests_sample", "1", "mp4", gen_sample)
            file1 = add_to_test_dir(td.path, str(file1))
            file2 = file_cache.get_or_generate("melter_tests_vhs", "1", "mp4", partial(gen_vhs, input=file1))
            file2 = add_to_test_dir(td.path, str(file2))

            files = ["/media/nfs/raspberry/storage/Wideo/Seriale/Brygada RR/season 1/01. Podwodni piraci.avi",
                     "/media/nfs/raspberry/storage/seed/Chip n Dales Rescue Rangers Season 1 Complete 720p WEBRip x264 [i_c]/Chip 'n Dale Rescue Rangers S01E01 Piratsy Under the Seas.mkv"]

            files = [file1, file2]

            interruption = utils.InterruptibleProcess()
            duplicates = Duplicates(interruption)
            duplicates.setDuplicates({"video": files})

            files_before = hashes(td.path)
            self.assertEqual(len(files_before), 2)

            melter = Melter(logging.getLogger("Melter"), interruption, duplicates, live_run = True)
            melter.melt()

            # expect second file to be removed
            files_after = hashes(td.path)
            self.assertEqual(len(files_after), 1)

            # check if file was not altered
            self.assertTrue(files_after.items() < files_before.items())


    def test_images_comparison(self):
        return
        images_dir = os.path.join(current_path, "images")
        images = sorted(os.listdir(images_dir))

        def phash(path):
            img = Image.open(path).convert('L').resize((256, 256))
            img_hash = imagehash.phash(img, hash_size=16)
            return img_hash

        def phashdiff(h1, h2):
            return abs(h1 - h2)

        def compare_ssim(img1_path, img2_path):
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))
            similarity, _ = ssim(img1, img2, full=True)
            return similarity

        def orb_similarity(img1_path, img2_path):
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)

            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]  # lower distance = better match

            return len(good_matches) / max(len(kp1), len(kp2))


        for image1 in images:
            image1_path = os.path.join(images_dir, image1)
            p1 = phash(image1_path)

            for image2 in images:
                image2_path = os.path.join(images_dir, image2)
                p2 = phash(image2_path)

                pdiff = phashdiff(p1, p2)
                ssim_diff = compare_ssim(image1_path, image2_path)
                orb_diff = orb_similarity(image1_path, image2_path)

                print(f"P1: {image1} P2: {image2}, pdiff = {pdiff}, ssim = {ssim_diff}, orb = {orb_diff}")


    def test_similarities2(self):
        return
        def align_images(im1, im2):
            # Convert images to grayscale
            gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

            # Use ORB detector
            orb = cv2.ORB_create(500)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)

            # Match features
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)

            if len(matches) < 4:
                print("Not enough matches to align images.")
                return None, None

            matches = sorted(matches, key=lambda x: x.distance)
            points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography
            h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

            height, width = gray1.shape
            aligned_im2 = cv2.warpPerspective(im2, h, (width, height))

            return im1, aligned_im2

        def compute_ssim(im1, im2):
            gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

            score, _ = ssim(gray1, gray2, full=True)
            return score

        def compare_all_images_in_dir(directory):
            image_files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]
            images = {file: cv2.imread(file) for file in image_files}

            print("Pairwise SSIM similarities after alignment:\n")
            for file1, img1 in images.items():
                for file2, img2 in images.items():
                    if img1 is None or img2 is None:
                        print(f"Could not read {file1} or {file2}. Skipping.")
                        continue
                    aligned_pair = align_images(img1, img2)
                    if aligned_pair[0] is None:
                        print(f"Alignment failed for {file1} and {file2}.")
                        continue

                    similarity = compute_ssim(aligned_pair[0], aligned_pair[1])
                    print(f"{os.path.basename(file1)} vs {os.path.basename(file2)}: SSIM = {similarity:.4f}")


        compare_all_images_in_dir(os.path.join(current_path, "images"))



if __name__ == '__main__':
    unittest.main()

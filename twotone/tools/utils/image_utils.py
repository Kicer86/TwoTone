
import cv2 as cv
import numpy as np

from PIL import Image
from scipy.stats import entropy


def are_images_similar(lhs_path: str, rhs_path: str, threshold = 10) -> bool:
    img1 = cv.imread(lhs_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(rhs_path, cv.IMREAD_GRAYSCALE)

    orb = cv.ORB_create()  # type: ignore[attr-defined]
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    return len(matches) >= threshold


def image_entropy(path: str) -> float:
    pil_image = Image.open(path)
    image = np.array(pil_image.convert("L"))
    histogram, _ = np.histogram(image, bins = 256, range=(0, 256))
    histogram = histogram / float(np.sum(histogram))
    e = entropy(histogram)
    return float(e)

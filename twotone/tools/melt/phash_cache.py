
import imagehash

from PIL import Image


def compute_phash(image_path: str, hash_size: int = 16) -> imagehash.ImageHash:
    """Hash the image as it is on disk right now, without caching.

    Use this when a path-keyed cache would add nothing (the file is hashed
    once) or cannot be trusted (the content at the path may change).
    """
    with Image.open(image_path) as img:
        return imagehash.phash(img, hash_size=hash_size)


class PhashCache:
    def __init__(self, hash_size: int = 16):
        self.hash_size = hash_size
        self._memory_cache: dict[str, imagehash.ImageHash] = {}

    def get(self, image_path: str) -> imagehash.ImageHash:
        if image_path in self._memory_cache:
            return self._memory_cache[image_path]

        phash = compute_phash(image_path, hash_size=self.hash_size)

        self._memory_cache[image_path] = phash
        return phash



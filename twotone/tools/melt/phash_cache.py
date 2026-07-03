
import imagehash

from PIL import Image


def compute_phash(image_path: str, hash_size: int = 16) -> imagehash.ImageHash:
    """Hash the image as it is on disk right now, without caching.

    Use this for files whose path may be re-used with different content —
    e.g. frames re-extracted into the same ``frame_%08d`` name — where a
    path-keyed cache would return the hash of the previous content.
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



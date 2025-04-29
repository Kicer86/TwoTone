
import imagehash

from PIL import Image


class PhashCache:
    def __init__(self, hash_size: int = 16):
        self.hash_size = hash_size
        self._memory_cache: dict[str, imagehash.ImageHash] = {}

    def get(self, image_path: str) -> imagehash.ImageHash:
        if image_path in self._memory_cache:
            return self._memory_cache[image_path]

        with Image.open(image_path) as img:
            phash = imagehash.phash(img, hash_size=self.hash_size)

        self._memory_cache[image_path] = phash
        return phash



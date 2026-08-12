import tempfile
import unittest
from pathlib import Path

from PIL import Image


def images_match(paths, tolerance=1):
    images = [Image.open(path).convert("RGB") for path in paths]
    reference = images[0]
    for image in images[1:]:
        extrema = __import__("PIL.ImageChops").ImageChops.difference(
            reference, image
        ).getextrema()
        if any(high > tolerance for _, high in extrema):
            return False
    return True


class ModelInitImageToleranceTests(unittest.TestCase):
    def test_one_lsb_passes_and_two_lsb_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, value in enumerate((100, 101, 102)):
                path = root / f"{index}.png"
                Image.new("RGB", (4, 4), (value, value, value)).save(path)
                paths.append(path)
            self.assertTrue(images_match(paths[:2]))
            self.assertFalse(images_match((paths[0], paths[2])))


import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import read_rgb224


class PreprocessingTests(unittest.TestCase):
    def test_rgb_channels_uint8_and_square_output_for_both_orientations(self):
        with tempfile.TemporaryDirectory() as folder:
            for height, width in [(30, 500), (500, 30)]:
                image = np.full((height, width, 3), (11, 53, 207), dtype=np.uint8)
                path = Path(folder) / f'{height}_{width}.png'
                cv2.imwrite(str(path), image)
                output = read_rgb224(path)
                self.assertEqual(output.shape, (224, 224, 3))
                self.assertEqual(output.dtype, np.dtype('uint8'))
                np.testing.assert_array_equal(output[112, 112], [207, 53, 11])

    def test_reads_without_modifying_source_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            cv2.imwrite(str(path), np.arange(12 * 25 * 3, dtype=np.uint8).reshape(12, 25, 3))
            before = path.read_bytes()
            read_rgb224(path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from src.augmentation import augment_sample, augment_sample_indexed


class AugmentationCharacterizationTests(unittest.TestCase):
    def test_flip_reverses_columns_and_steering(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        augmented, steering = augment_sample(image, 0.4, flip=True)

        np.testing.assert_array_equal(augmented, image[:, ::-1, :])
        self.assertEqual(steering, -0.4)

    def test_no_flip_copies_image_and_preserves_label(self):
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)

        augmented, steering = augment_sample(image, -0.2)

        np.testing.assert_array_equal(augmented, image)
        self.assertIsNot(augmented, image)
        self.assertEqual(steering, -0.2)

    def test_empty_batch_dimension_is_supported(self):
        image = np.empty((0, 3, 1), dtype=np.uint8)

        augmented, steering = augment_sample(image, 0.0, flip=True)

        self.assertEqual(augmented.shape, image.shape)
        self.assertEqual(steering, 0.0)

    def test_indexed_implementation_matches_flip_implementation(self):
        image = np.arange(24, dtype=np.int16).reshape(2, 4, 3)

        reference = augment_sample(image, 0.6, flip=True)
        alternative = augment_sample_indexed(image, 0.6, flip=True)

        np.testing.assert_array_equal(alternative[0], reference[0])
        self.assertEqual(alternative[1], reference[1])


if __name__ == "__main__":
    unittest.main()

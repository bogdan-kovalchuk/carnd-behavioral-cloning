import unittest

import numpy as np

from src.data_generator import (
    batch_generator,
    expand_camera_samples,
    indexed_batch_generator,
)


class BatchGeneratorCharacterizationTests(unittest.TestCase):
    def test_batches_are_reproducible_and_do_not_mutate_input(self):
        samples = ["a", "b", "c", "d", "e"]
        before = list(samples)

        first = list(batch_generator(samples, batch_size=2, shuffle=True, seed=7))
        second = list(batch_generator(samples, batch_size=2, shuffle=True, seed=7))

        self.assertEqual(first, second)
        self.assertEqual(samples, before)
        self.assertEqual([len(batch) for batch in first], [2, 2, 1])

    def test_empty_input_has_no_batches(self):
        self.assertEqual(list(batch_generator([], batch_size=4)), [])

    def test_camera_expansion_preserves_order_and_signs(self):
        rows = [("center.jpg", "left.jpg", "right.jpg", "0.1")]

        self.assertEqual(
            expand_camera_samples(rows),
            (
                ("center.jpg", 0.1, "center"),
                ("left.jpg", 0.30000000000000004, "left"),
                ("right.jpg", -0.1, "right"),
            ),
        )

    def test_indexed_implementation_is_deterministic_and_complete(self):
        samples = tuple(range(9))

        first = list(indexed_batch_generator(samples, 4, shuffle=True, seed=11))
        second = list(indexed_batch_generator(samples, 4, shuffle=True, seed=11))

        self.assertEqual(first, second)
        self.assertEqual(sorted(value for batch in first for value in batch), list(samples))

    def test_indexed_implementation_drops_only_the_short_batch(self):
        samples = tuple(range(7))

        batches = list(indexed_batch_generator(samples, 3, drop_last=True))

        self.assertEqual(batches, [(0, 1, 2), (3, 4, 5)])

    def test_indexed_shuffle_does_not_change_global_numpy_state(self):
        np.random.seed(17)
        expected = np.random.random(3)
        np.random.seed(17)

        list(indexed_batch_generator(tuple(range(8)), 2, shuffle=True, seed=5))

        np.testing.assert_array_equal(np.random.random(3), expected)

    def test_two_implementations_have_the_same_batch_shape(self):
        samples = tuple("abcdefg")
        reference = list(batch_generator(samples, 3, shuffle=False))
        alternative = list(indexed_batch_generator(samples, 3, shuffle=False))

        self.assertEqual(alternative, reference)

    def test_invalid_batch_sizes_and_seeds_are_rejected(self):
        for bad_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=bad_size):
                with self.assertRaises((TypeError, ValueError)):
                    list(batch_generator([1], bad_size))
        with self.assertRaises(TypeError):
            list(indexed_batch_generator([1], seed=True))

    def test_invalid_rows_and_non_finite_numbers_are_rejected(self):
        with self.assertRaises(ValueError):
            expand_camera_samples([("only", "three", "fields")])
        with self.assertRaises(ValueError):
            expand_camera_samples([("c", "l", "r", "nan")])
        with self.assertRaises(ValueError):
            expand_camera_samples([("c", "l", "r", "0")], correction=float("inf"))


if __name__ == "__main__":
    unittest.main()

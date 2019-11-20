import unittest

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

    def test_two_implementations_have_the_same_batch_shape(self):
        samples = tuple("abcdefg")
        reference = list(batch_generator(samples, 3, shuffle=False))
        alternative = list(indexed_batch_generator(samples, 3, shuffle=False))

        self.assertEqual(alternative, reference)


if __name__ == "__main__":
    unittest.main()

import unittest

from src.camera_samples import (
    CameraSample,
    correct_steering,
    correct_steering_lookup,
    make_camera_samples,
)


class CameraCorrectionCharacterizationTests(unittest.TestCase):
    def test_center_left_and_right_corrections(self):
        self.assertEqual(correct_steering(0.1, "center"), 0.1)
        self.assertAlmostEqual(correct_steering(0.1, "left"), 0.3)
        self.assertAlmostEqual(correct_steering(0.1, "right"), -0.1)

    def test_samples_are_immutable_and_ordered(self):
        samples = make_camera_samples("c", "l", "r", -0.25, correction=0.15)

        self.assertEqual(
            samples,
            (
                CameraSample("c", -0.25, "center"),
                CameraSample("l", -0.1, "left"),
                CameraSample("r", -0.4, "right"),
            ),
        )
        with self.assertRaises((AttributeError, TypeError)):
            samples[0].steering = 1.0

    def test_custom_correction_zero_keeps_all_labels_equal(self):
        self.assertEqual(correct_steering(-0.7, "left", 0), -0.7)
        self.assertEqual(correct_steering(-0.7, "right", 0), -0.7)

    def test_lookup_implementation_matches_branch_implementation(self):
        for camera in ("center", "left", "right"):
            self.assertAlmostEqual(
                correct_steering_lookup(-0.35, camera, 0.27),
                correct_steering(-0.35, camera, 0.27),
            )

    def test_invalid_and_non_finite_inputs_are_rejected_by_both(self):
        for implementation in (correct_steering, correct_steering_lookup):
            with self.subTest(implementation=implementation.__name__):
                with self.assertRaises(ValueError):
                    implementation(float("nan"), "center")
                with self.assertRaises(ValueError):
                    implementation(0.0, "center", -0.1)
                with self.assertRaises(ValueError):
                    implementation(0.0, "rear")
                with self.assertRaises(TypeError):
                    implementation(True, "center")


if __name__ == "__main__":
    unittest.main()

import unittest

from src.training_config import TrainingConfig


class TrainingConfigCharacterizationTests(unittest.TestCase):
    def test_defaults_capture_existing_training_shape(self):
        config = TrainingConfig()

        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.validation_split, 0.2)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.steering_correction, 0.2)

    def test_config_is_immutable(self):
        config = TrainingConfig()

        with self.assertRaises((AttributeError, TypeError)):
            config.epochs = 10

    def test_basic_bounds_are_enforced(self):
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0)
        with self.assertRaises(ValueError):
            TrainingConfig(validation_split=1.0)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.0)


if __name__ == "__main__":
    unittest.main()

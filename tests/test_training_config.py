import unittest

from src.training_config import (
    MappingTrainingConfig,
    TrainingConfig,
    configuration_signature,
    to_training_config,
)


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

    def test_non_finite_values_and_boolean_numbers_are_rejected(self):
        for field in ("validation_split", "learning_rate", "steering_correction"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    TrainingConfig(**{field: float("nan")})
        with self.assertRaises(TypeError):
            TrainingConfig(batch_size=True)
        with self.assertRaises(TypeError):
            TrainingConfig(seed=False)

    def test_mapping_implementation_matches_values_and_is_immutable(self):
        config = MappingTrainingConfig({"batch_size": 8, "epochs": 2})

        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config["epochs"], 2)
        self.assertEqual(config.validation_split, 0.2)
        with self.assertRaises(AttributeError):
            config.epochs = 3
        with self.assertRaises(TypeError):
            config.as_mapping()["epochs"] = 3

    def test_representations_round_trip_through_adapter(self):
        values = {"batch_size": 8, "epochs": 2, "seed": 4}
        mapping_config = MappingTrainingConfig(values)
        canonical = mapping_config.as_training_config()

        self.assertEqual(TrainingConfig.from_mapping(values), to_training_config(mapping_config))
        self.assertIs(to_training_config(canonical), canonical)
        self.assertEqual(to_training_config(values).batch_size, 8)

    def test_signature_compares_both_representations(self):
        dataclass_config = TrainingConfig(batch_size=8, epochs=2)
        mapping_config = MappingTrainingConfig({"batch_size": 8, "epochs": 2})

        self.assertEqual(configuration_signature(dataclass_config), configuration_signature(mapping_config))
        self.assertEqual(hash(configuration_signature(mapping_config)), hash(configuration_signature(dataclass_config)))


if __name__ == "__main__":
    unittest.main()

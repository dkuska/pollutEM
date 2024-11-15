import unittest
import pandas as pd
import numpy as np

from pollutEM.polluters.error_polluter import GaussianNoisePolluter, UniformNoisePolluter
from pollutEM.polluters.mathematical_polluter import (
    ScalingPolluter,
    ShiftingPolluter,
    LogTransformationPolluter,
    ReciprocalPolluter,
)
from pollutEM.polluters.precision_polluter import RoundingPolluter
from pollutEM.polluters.representation_polluter import (
    SeparatorConversionPolluter,
    ScientificNotationPolluter,
)


class TestPolluters(unittest.TestCase):
    def setUp(self):
        """Create test data with different types."""
        self.df = pd.DataFrame(
            {
                "integers": [1000, 2000, 3000, 4000, 5000],
                "floats": [1111.11, 2222.22, 3333.33, 4444.44, 5555.55],
                "strings": ["1.1", "2.2", "3.3", "4.4", "5.5"],
                "text": ["a", "b", "c", "d", "e"],
            }
        )

    # TODO: Digit Polluters
    # TODO: Domain Specific Polluters
    # TODO: Error Polluters
    def test_gaussian_noise_polluter(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Parameters for the Gaussian noise
        mean = 0.0
        std_dev = 1.0
        polluter = GaussianNoisePolluter(mean=mean, std_dev=std_dev)

        # Level 1: Column Transformations
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        for column in ["integers", "floats"]:
            # Calculate the differences (noise)
            noise = result[column] - self.df[column]

            # Statistical tests
            # 1. Check if mean is close to expected
            self.assertAlmostEqual(
                noise.mean(),
                mean,
                places=0,
                msg=f"Mean of noise in {column} is not close to expected {mean}",
            )

            # 2. Check if standard deviation is close to expected
            self.assertAlmostEqual(
                noise.std(),
                std_dev,
                places=0,
                msg=f"Std dev of noise in {column} is not close to expected {std_dev}",
            )
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")

    def test_uniform_noise_polluter(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Parameters for the uniform noise
        low = -1.0
        high = 1.0
        polluter = UniformNoisePolluter(low=low, high=high)

        # Apply pollution
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])

        for column in ["integers", "floats"]:
            # Calculate the differences (noise)
            noise = result[column] - self.df[column]

            # Statistical tests
            # 1. Check if values are within bounds
            self.assertTrue(
                noise.min() >= low, msg=f"Minimum noise in {column} is less than {low}"
            )
            self.assertTrue(
                noise.max() <= high, msg=f"Maximum noise in {column} is greater than {high}"
            )
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")

    # TODO: Mathematical Polluters
    def test_scaling_polluter(self):
        polluter = ScalingPolluter(multiplier=2)

        # Level 1: Column Transformations
        # Test integers
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        self.assertEqual(result["integers"][0], 2000)
        self.assertIsInstance(result["integers"][0], np.int64)
        # Test floats
        result = polluter.apply(self.df, "floats")
        self.assertEqual(result["floats"][0], 2222.22)
        self.assertIsInstance(result["floats"][0], float)
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")

    def test_shifting_polluter(self):
        polluter = ShiftingPolluter(shift_amount=2)

        # Level 1: Column Transformations
        # Test integers & floats
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        self.assertEqual(result["integers"][0], 1002)
        self.assertIsInstance(result["integers"][0], np.int64)
        self.assertEqual(result["floats"][0], 1113.11)
        self.assertIsInstance(result["floats"][0], float)
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")

    def test_log_transform_polluter(self):
        polluter = LogTransformationPolluter(base=10)

        # Level 1: Column Transformations
        # Test integers & floats
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        self.assertAlmostEqual(result["integers"][0], 3.0, places=1)
        self.assertIsInstance(result["integers"][0], float)
        self.assertAlmostEqual(result["floats"][0], 3.0, places=1)
        self.assertIsInstance(result["floats"][0], float)
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")

    def test_reciprocal_polluter(self):
        polluter = ReciprocalPolluter()

        # Level 1: Column Transformations
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        self.assertAlmostEqual(result["integers"][0], 0.001)
        self.assertIsInstance(result["integers"][0], float)
        self.assertAlmostEqual(result["floats"][0], 0.001, places=1)
        self.assertIsInstance(result["floats"][0], float)
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")

    # Precision Polluters
    def test_rounding_polluter(self):
        polluter = RoundingPolluter(decimal_places=1)

        # Level 1: Column Transformations
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        self.assertEqual(result["integers"][0], 1000)
        self.assertIsInstance(result["integers"][0], np.int64)
        self.assertEqual(result["floats"][0], 1111.1)
        self.assertIsInstance(result["floats"][0], float)
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")

    # Representation Polluters
    def test_separator_polluter(self):
        polluter = SeparatorConversionPolluter(decimal_sep=",", thousands_sep=".")

        # Level 1: Column Transformations
        # Test numeric inputs
        result = polluter.apply(
            self.df, level="column", target_columns=["strings", "text", "floats"]
        )
        self.assertEqual(result["floats"][0], "1.111,11")
        self.assertIsInstance(result["floats"][0], str)
        # Test string inputs
        self.assertEqual(result["strings"][0], "1,10")
        self.assertIsInstance(result["strings"][0], str)
        # Test invalid inputs
        self.assertEqual(result["text"][0], "a")
        self.assertIsInstance(result["text"][0], str)
        # Test error cases
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "non_existent_column")

    def test_scientific_notation_polluter(self):
        polluter = ScientificNotationPolluter()

        # Level 1: Column Transformations
        # Test numeric inputs
        result = polluter.apply(
            self.df, level="column", target_columns=["integers", "floats", "text"]
        )
        self.assertEqual(result["floats"][0], "1.11e+03")
        self.assertIsInstance(result["floats"][0], str)
        self.assertEqual(result["integers"][0], "1.00e+03")
        self.assertIsInstance(result["integers"][0], str)
        self.assertEqual(result["text"][0], "a")
        self.assertIsInstance(result["text"][0], str)
        # Test error cases
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "non_existent_column")

    # TODO: Statistical Polluters


if __name__ == "__main__":
    unittest.main()

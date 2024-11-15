import unittest
import pandas as pd
import numpy as np

from pollutEM.polluters.mathematical_polluter import (
    ScalingPolluter,
    ShiftingPolluter,
    LogTransformationPolluter,
    ReciprocalPolluter,
)
from pollutEM.polluters.representation_polluter import SeparatorConversionPolluter


class TestPolluters(unittest.TestCase):
    def setUp(self):
        """Create test data with different types."""
        self.df = pd.DataFrame(
            {
                "integers": [1, 2, 3, 4, 5],
                "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
                "strings": ["1.1", "2.2", "3.3", "4.4", "5.5"],
                "text": ["a", "b", "c", "d", "e"],
            }
        )

    def test_separator_conversion(self):
        polluter = SeparatorConversionPolluter(decimal_sep=",", thousands_sep=".")

        # Level 1: Column Transformations
        # Test numeric inputs
        result = polluter.apply(self.df, "floats")
        self.assertEqual(result["floats"][0], "1,10")
        self.assertIsInstance(result["floats"][0], str)
        # Test string inputs
        result = polluter.apply(self.df, "strings")
        self.assertEqual(result["strings"][0], "1,10")
        self.assertIsInstance(result["strings"][0], str)
        # Test invalid inputs
        result = polluter.apply(self.df, "text")
        self.assertEqual(result["text"][0], "a")
        self.assertIsInstance(result["text"][0], str)
        # Test error cases
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "non_existent_column")
        # TODO: Add tests for Row and Cell Transformations!

    def test_linear_transformation(self):
        polluter = ScalingPolluter(multiplier=2)

        # Level 1: Column Transformations
        # Test integers
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        self.assertEqual(result["integers"][0], 2)
        self.assertIsInstance(result["integers"][0], np.int64)
        # Test floats
        result = polluter.apply(self.df, "floats")
        self.assertEqual(result["floats"][0], 2.2)
        self.assertIsInstance(result["floats"][0], float)
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")
        # TODO: Add tests for Row and Cell Transformations!

    def test_shifting_transformation(self):
        polluter = ShiftingPolluter(shift_amount=2)

        # Level 1: Column Transformations
        # Test integers & floats
        result = polluter.apply(self.df, level="column", target_columns=["integers", "floats"])
        self.assertEqual(result["integers"][0], 3)
        self.assertIsInstance(result["integers"][0], np.int64)
        self.assertEqual(result["floats"][0], 3.1)
        self.assertIsInstance(result["floats"][0], float)
        # Test type validation
        with self.assertRaises(ValueError):
            polluter.apply(self.df, "strings")
        # TODO: Add tests for Row and Cell Transformations!

    LogTransformationPolluter


if __name__ == "__main__":
    unittest.main()

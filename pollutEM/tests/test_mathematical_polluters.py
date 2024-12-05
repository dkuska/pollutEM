import pytest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing

from pollutEM.polluters import ScalingPolluter, ShiftingPolluter, ScalingShiftingPolluter


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "str_col": ["a", "b", "c", "d", "e"],
        }
    )


class TestScalingPolluter:
    def test_column_level_scaling(self, sample_df):
        polluter = ScalingPolluter(multiplier=2.0, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)

        expected = sample_df["float_col"] * 2.0
        pd_testing.assert_series_equal(expected, result["float_col"])
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])

    def test_row_level_scaling(self, sample_df):
        polluter = ScalingPolluter(multiplier=3.0, level="row", indices=[0, 2])
        result = polluter.apply(sample_df)

        # Check scaled rows
        assert result.loc[0, "float_col"] == sample_df.loc[0, "float_col"] * 3.0
        assert result.loc[2, "float_col"] == sample_df.loc[2, "float_col"] * 3.0

        # Check unaffected rows
        assert result.loc[1, "float_col"] == sample_df.loc[1, "float_col"]
        assert result.loc[3, "float_col"] == sample_df.loc[3, "float_col"]

    def test_cell_level_scaling(self, sample_df):
        np.random.seed(42)
        polluter = ScalingPolluter(multiplier=2.0, level="cell", probability=1.0)
        result = polluter.apply(sample_df)

        assert all(result["float_col"] == sample_df["float_col"] * 2.0)
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_integer_preservation(self, sample_df):
        polluter = ScalingPolluter(multiplier=2, level="column", indices=["int_col"])
        result = polluter.apply(sample_df)

        assert result["int_col"].dtype == np.int64
        pd_testing.assert_series_equal(sample_df["int_col"] * 2, result["int_col"])


class TestShiftingPolluter:
    def test_column_level_shifting(self, sample_df):
        polluter = ShiftingPolluter(shift_amount=10, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)

        expected = sample_df["float_col"] + 10
        pd_testing.assert_series_equal(expected, result["float_col"])
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])

    def test_row_level_shifting(self, sample_df):
        polluter = ShiftingPolluter(shift_amount=-5, level="row", indices=[1, 3])
        result = polluter.apply(sample_df)

        # Check shifted rows
        assert result.loc[1, "float_col"] == sample_df.loc[1, "float_col"] - 5
        assert result.loc[3, "float_col"] == sample_df.loc[3, "float_col"] - 5

        # Check unaffected rows
        assert result.loc[0, "float_col"] == sample_df.loc[0, "float_col"]
        assert result.loc[2, "float_col"] == sample_df.loc[2, "float_col"]

    def test_mixed_types(self, sample_df):
        polluter = ShiftingPolluter(shift_amount=1.5, level="column", indices=["int_col"])
        result = polluter.apply(sample_df)

        # Integer column should remain integer
        assert result["int_col"].dtype == np.int64
        pd_testing.assert_series_equal(sample_df["int_col"] + 1, result["int_col"])

    def test_input_validation(self, sample_df):
        with pytest.raises(ValueError):
            ShiftingPolluter(shift_amount=1, probability=2.0)

        with pytest.raises(ValueError):
            ShiftingPolluter(shift_amount=1, level="invalid")

        with pytest.raises(ValueError):
            ShiftingPolluter(shift_amount=1, probability=0.5, indices=["col1"])


class TestScalingShiftingPolluter:
    def test_column_level_transformation(self, sample_df):
        polluter = ScalingShiftingPolluter(
            scale_amount=2.0, shift_amount=1.0, level="column", indices=["float_col"]
        )
        result = polluter.apply(sample_df)

        # Check transformation: value * 2 + 1
        expected = sample_df["float_col"] * 2.0 + 1.0
        pd_testing.assert_series_equal(expected, result["float_col"])

        # Check other columns are unchanged
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])

    def test_row_level_transformation(self, sample_df):
        polluter = ScalingShiftingPolluter(
            scale_amount=3.0, shift_amount=2.0, level="row", indices=[0, 2]
        )
        result = polluter.apply(sample_df)

        # Check transformed rows
        assert result.loc[0, "float_col"] == sample_df.loc[0, "float_col"] * 3.0 + 2.0
        assert result.loc[0, "int_col"] == sample_df.loc[0, "int_col"] * 3.0 + 2.0
        assert result.loc[2, "float_col"] == sample_df.loc[2, "float_col"] * 3.0 + 2.0
        assert result.loc[2, "int_col"] == sample_df.loc[2, "int_col"] * 3.0 + 2.0

        # Check unaffected rows
        assert result.loc[1, "float_col"] == sample_df.loc[1, "float_col"]
        assert result.loc[3, "float_col"] == sample_df.loc[3, "float_col"]

        # String column should be unchanged
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_integer_preservation(self, sample_df):
        polluter = ScalingShiftingPolluter(
            scale_amount=2, shift_amount=1, level="column", indices=["int_col"]
        )
        result = polluter.apply(sample_df)

        # Check type preservation
        assert result["int_col"].dtype == np.int64
        # Check transformation
        expected = sample_df["int_col"] * 2 + 1
        pd_testing.assert_series_equal(expected, result["int_col"])

    def test_negative_scale_amount(self, sample_df):
        polluter = ScalingShiftingPolluter(
            scale_amount=-2.0, shift_amount=1.0, level="column", indices=["float_col"]
        )
        result = polluter.apply(sample_df)

        # Values should be negatively scaled and shifted
        expected = sample_df["float_col"] * -2.0 + 1.0
        pd_testing.assert_series_equal(expected, result["float_col"])

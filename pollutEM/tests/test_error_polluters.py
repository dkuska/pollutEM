import pytest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing

from pollutEM.polluters import GaussianNoisePolluter, UniformNoisePolluter, MissingValuePolluter


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "str_col": ["a", "b", "c", "d", "e"],
        }
    )


class TestGaussianNoisePolluter:
    def test_column_level_pollution(self, sample_df):
        polluter = GaussianNoisePolluter(mean=0, std_dev=1, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)

        # Check only specified column changed
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])
        assert not pd_testing.assert_series_equal(
            sample_df["float_col"], result["float_col"], check_exact=True
        )

    def test_row_level_pollution(self, sample_df):
        polluter = GaussianNoisePolluter(mean=0, std_dev=1, level="row", indices=[0, 2])
        result = polluter.apply(sample_df)

        # Check only specified rows changed for numeric columns
        assert result.loc[1, "float_col"] == sample_df.loc[1, "float_col"]
        assert result.loc[3, "float_col"] == sample_df.loc[3, "float_col"]
        assert result.loc[0, "float_col"] != sample_df.loc[0, "float_col"]
        assert result.loc[2, "float_col"] != sample_df.loc[2, "float_col"]

        # String column should remain unchanged
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_cell_level_pollution(self, sample_df):
        np.random.seed(42)  # For reproducibility
        polluter = GaussianNoisePolluter(mean=0, std_dev=1, level="cell", probability=0.5)
        result = polluter.apply(sample_df)

        # Some cells should change, others shouldn't
        changes = (result != sample_df).sum().sum()
        assert changes > 0 and changes < len(sample_df) * 2  # Only numeric columns can change

        # String column should remain unchanged
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])


class TestUniformNoisePolluter:
    def test_column_level_pollution(self, sample_df):
        polluter = UniformNoisePolluter(low=-1, high=1, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)

        # Check noise bounds
        differences = result["float_col"] - sample_df["float_col"]
        assert all(-1 <= diff <= 1 for diff in differences)

        # Check other columns unchanged
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])

    def test_type_conversion(self, sample_df):
        polluter = UniformNoisePolluter(low=-1, high=1, level="column", indices=["int_col"])
        result = polluter.apply(sample_df)

        # Check int column converted to float
        assert result["int_col"].dtype == np.float64

    def test_invalid_inputs(self, sample_df):
        with pytest.raises(ValueError):
            UniformNoisePolluter(low=-1, high=1, probability=1.5)

        with pytest.raises(ValueError):
            UniformNoisePolluter(low=-1, high=1, probability=0.5, indices=["col1"])

        with pytest.raises(ValueError):
            UniformNoisePolluter(low=-1, high=1, level="invalid")


class TestMissingValuePolluter:
    def test_column_level_missing(self, sample_df):
        # Test with 100% missing ratio to verify column selection
        polluter = MissingValuePolluter(
            missing_value=np.nan, missing_ratio=1.0, level="column", indices=["float_col"]
        )
        result = polluter.apply(sample_df)

        # Check that specified column is all NaN
        assert result["float_col"].isna().all()
        # Check other columns unchanged
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])

    def test_partial_column_missing(self, sample_df):
        np.random.seed(42)
        polluter = MissingValuePolluter(
            missing_value=np.nan,
            missing_ratio=0.4,  # 40% missing values
            level="column",
            indices=["float_col"],
        )
        result = polluter.apply(sample_df)

        # Check that exactly 40% of values are NaN
        assert result["float_col"].isna().mean() == 0.4
        # Verify non-NaN values are unchanged
        original_values = set(sample_df["float_col"])
        result_values = set(result["float_col"].dropna())
        assert result_values.issubset(original_values)

    def test_row_level_missing(self, sample_df):
        polluter = MissingValuePolluter(
            missing_value=np.nan, missing_ratio=1.0, level="row", indices=[0, 2]
        )
        result = polluter.apply(sample_df)

        # Check specified rows have NaN in numeric columns
        assert pd.isna(result.loc[0, "float_col"])
        assert pd.isna(result.loc[2, "float_col"])
        assert pd.isna(result.loc[0, "int_col"])
        assert pd.isna(result.loc[2, "int_col"])

        # Check unaffected rows
        assert result.loc[1, "float_col"] == sample_df.loc[1, "float_col"]
        assert result.loc[3, "float_col"] == sample_df.loc[3, "float_col"]

        # String column should be unchanged
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_custom_missing_value(self, sample_df):
        polluter = MissingValuePolluter(
            missing_value=-999, missing_ratio=1.0, level="column", indices=["int_col"]
        )
        result = polluter.apply(sample_df)

        # Check that missing values are -999
        assert (result["int_col"] == -999).all()

    def test_type_preservation(self, sample_df):
        polluter = MissingValuePolluter(
            missing_value=np.nan,
            missing_ratio=0.5,
            level="column",
            indices=["float_col", "int_col"],
        )
        result = polluter.apply(sample_df)

        # Integer column should be converted to float64 to accommodate NaN
        assert result["float_col"].dtype == np.float64
        assert result["int_col"].dtype == np.float64
        # String column should maintain its type
        assert result["str_col"].dtype == sample_df["str_col"].dtype

    def test_invalid_missing_ratio(self, sample_df):
        with pytest.raises(ValueError, match="missing_ratio must be between 0 and 1"):
            MissingValuePolluter(missing_ratio=1.5)
        with pytest.raises(ValueError, match="missing_ratio must be between 0 and 1"):
            MissingValuePolluter(missing_ratio=-0.1)

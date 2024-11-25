import pytest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing

from pollutEM.polluters import RoundingPolluter


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.23456, 2.34567, 3.45678, 4.56789, 5.67890],
            "str_col": ["a", "b", "c", "d", "e"],
        }
    )


class TestRoundingPolluter:
    def test_column_level_rounding(self, sample_df):
        polluter = RoundingPolluter(decimal_places=2, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)

        expected = sample_df["float_col"].round(2)
        pd_testing.assert_series_equal(expected, result["float_col"])
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])

    def test_row_level_rounding(self, sample_df):
        polluter = RoundingPolluter(decimal_places=1, level="row", indices=[0, 2])
        result = polluter.apply(sample_df)

        assert result.loc[0, "float_col"] == round(sample_df.loc[0, "float_col"], 1)
        assert result.loc[2, "float_col"] == round(sample_df.loc[2, "float_col"], 1)
        assert result.loc[1, "float_col"] == sample_df.loc[1, "float_col"]
        assert result.loc[3, "float_col"] == sample_df.loc[3, "float_col"]

    def test_cell_level_rounding(self, sample_df):
        np.random.seed(42)
        polluter = RoundingPolluter(decimal_places=0, level="cell", probability=1.0)
        result = polluter.apply(sample_df)

        assert all(result["float_col"] == sample_df["float_col"].round(0))
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_type_preservation(self, sample_df):
        polluter = RoundingPolluter(
            decimal_places=0, level="column", indices=["float_col", "int_col"]
        )
        result = polluter.apply(sample_df)

        assert result["float_col"].dtype == np.float64
        assert result["int_col"].dtype == np.int64

    def test_negative_decimal_places(self, sample_df):
        polluter = RoundingPolluter(decimal_places=-1, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)

        expected = sample_df["float_col"].round(-1)
        pd_testing.assert_series_equal(expected, result["float_col"])

    def test_probability_based_selection(self, sample_df):
        np.random.seed(42)
        polluter = RoundingPolluter(decimal_places=2, level="column", probability=0.5)
        result = polluter.apply(sample_df)

        cols_changed = sum(
            not pd_testing.assert_series_equal(sample_df[col], result[col], check_exact=True)
            for col in ["int_col", "float_col"]
        )
        assert 0 < cols_changed <= 2

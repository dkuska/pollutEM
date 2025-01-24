import pytest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing

from pollutEM.polluters import (
    BitShiftPolluter,
    DigitPermutationPolluter,
    IntegerOverflowPolluter,
    ModuloPolluter,
    SignFlipPolluter,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "int_col": [1000, 2000, 3000, 4000, 5000],
            "float_col": [1.23, 2.34, 3.45, 4.56, 5.67],
            "str_col": ["a", "b", "c", "d", "e"],
        }
    )


class TestIntegerOverflowPolluter:
    def test_column_level_pollution(self, sample_df):
        polluter = IntegerOverflowPolluter(threshold=3000, level="column", indices=["int_col"])
        result = polluter.apply(sample_df)
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["float_col"], result["float_col"])
        assert result.loc[3, "int_col"] < 0  # Should overflow
        assert result.loc[4, "int_col"] < 0  # Should overflow
        assert result.loc[0, "int_col"] == 1000  # Should not overflow

    def test_row_level_pollution(self, sample_df):
        polluter = IntegerOverflowPolluter(threshold=3000, level="row", indices=[3, 4])
        result = polluter.apply(sample_df)
        assert result.loc[3, "int_col"] < 0  # Should overflow
        assert result.loc[4, "int_col"] < 0  # Should overflow
        assert result.loc[0, "int_col"] == 1000  # Should not overflow
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_cell_level_pollution(self, sample_df):
        np.random.seed(42)
        polluter = IntegerOverflowPolluter(threshold=3000, level="cell", probability=1.0)
        result = polluter.apply(sample_df)
        assert result.loc[3, "int_col"] < 0  # Should overflow
        assert result.loc[4, "int_col"] < 0  # Should overflow
        assert result.loc[0, "int_col"] == 1000  # Should not overflow
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])


class TestDigitPermutationPolluter:
    def test_column_level_pollution(self, sample_df):
        np.random.seed(42)
        polluter = DigitPermutationPolluter(num_swaps=1, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["int_col"], result["int_col"])
        assert not pd_testing.assert_series_equal(
            sample_df["float_col"], result["float_col"], check_exact=True
        )
        # Check digits are permuted, not just changed
        assert set(str(result.loc[0, "float_col"]).replace(".", "")) == set(
            str(sample_df.loc[0, "float_col"]).replace(".", "")
        )

    def test_row_level_pollution(self, sample_df):
        np.random.seed(42)
        polluter = DigitPermutationPolluter(num_swaps=1, level="row", indices=[0, 2])
        result = polluter.apply(sample_df)
        assert result.loc[1, "float_col"] == sample_df.loc[1, "float_col"]
        assert result.loc[1, "int_col"] == sample_df.loc[1, "int_col"]
        assert set(str(result.loc[0, "float_col"]).replace(".", "")) == set(
            str(sample_df.loc[0, "float_col"]).replace(".", "")
        )
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_cell_level_pollution(self, sample_df):
        np.random.seed(42)
        polluter = DigitPermutationPolluter(num_swaps=1, level="cell", probability=0.5)
        result = polluter.apply(sample_df)
        changes = (result != sample_df).sum().sum()
        assert changes > 0 and changes < len(sample_df) * 2
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        # Check changed values maintain same digits
        changed_mask = result != sample_df
        for col in ["int_col", "float_col"]:
            for idx in changed_mask[col][changed_mask[col]].index:
                assert set(str(result.loc[idx, col]).replace(".", "")) == set(
                    str(sample_df.loc[idx, col]).replace(".", "")
                )


class TestBitShiftPolluter:
    def test_column_level_pollution(self, sample_df):
        polluter = BitShiftPolluter(
            direction="left", num_bits=1, level="column", indices=["int_col"]
        )
        result = polluter.apply(sample_df)

        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])
        pd_testing.assert_series_equal(sample_df["float_col"], result["float_col"])
        assert result.loc[0, "int_col"] == sample_df.loc[0, "int_col"] << 1
        assert result.loc[1, "int_col"] == sample_df.loc[1, "int_col"] << 1

    def test_row_level_pollution(self, sample_df):
        polluter = BitShiftPolluter(direction="left", num_bits=1, level="row", indices=[0, 2])
        result = polluter.apply(sample_df)

        assert result.loc[1, "int_col"] == sample_df.loc[1, "int_col"]
        assert result.loc[0, "int_col"] == sample_df.loc[0, "int_col"] << 1
        assert result.loc[2, "int_col"] == sample_df.loc[2, "int_col"] << 1

    def test_cell_level_pollution(self, sample_df):
        np.random.seed(42)
        polluter = BitShiftPolluter(direction="right", num_bits=1, level="cell", probability=0.5)
        result = polluter.apply(sample_df)

        changes = (result != sample_df).sum().sum()
        assert 0 < changes < len(sample_df) * 2
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            BitShiftPolluter(direction="invalid")

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            BitShiftPolluter(num_bits=0)


class TestSignFlipPolluter:
    def test_column_level_pollution(self, sample_df):
        polluter = SignFlipPolluter(level="column", indices=["int_col"])
        result = polluter.apply(sample_df)

        assert result.loc[0, "int_col"] == -1000
        assert result.loc[1, "int_col"] == -2000
        pd_testing.assert_series_equal(sample_df["float_col"], result["float_col"])
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_row_level_pollution(self, sample_df):
        polluter = SignFlipPolluter(level="row", indices=[0, 2])
        result = polluter.apply(sample_df)

        assert result.loc[0, "int_col"] == -1000
        assert result.loc[0, "float_col"] == -1.23
        assert result.loc[1, "int_col"] == 2000  # Unchanged
        assert result.loc[2, "int_col"] == -3000

    def test_cell_level_pollution(self, sample_df):
        np.random.seed(42)
        polluter = SignFlipPolluter(level="cell", probability=1.0)
        result = polluter.apply(sample_df)

        assert result.loc[0, "int_col"] == -1000
        assert result.loc[0, "float_col"] == -1.23
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])


class TestModuloPolluter:
    def test_column_level_pollution(self, sample_df):
        polluter = ModuloPolluter(modulo=1000, level="column", indices=["int_col"])
        result = polluter.apply(sample_df)

        assert result.loc[0, "int_col"] == 0  # 1000 % 1000
        assert result.loc[1, "int_col"] == 0  # 2000 % 1000
        assert result.loc[2, "int_col"] == 0  # 3000 % 1000
        pd_testing.assert_series_equal(sample_df["float_col"], result["float_col"])
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_row_level_pollution(self, sample_df):
        polluter = ModuloPolluter(modulo=1000, level="row", indices=[0, 2])
        result = polluter.apply(sample_df)

        assert result.loc[0, "int_col"] == 0  # 1000 % 1000
        assert result.loc[1, "int_col"] == 2000  # Unchanged
        assert result.loc[2, "int_col"] == 0  # 3000 % 1000

    def test_float_modulo(self, sample_df):
        polluter = ModuloPolluter(modulo=2, level="column", indices=["float_col"])
        result = polluter.apply(sample_df)

        assert result.loc[0, "float_col"] == 1.0  # 1.23 % 2
        assert result.loc[1, "float_col"] == 0.0  # 2.34 % 2
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

    def test_cell_level_pollution(self, sample_df):
        np.random.seed(42)
        polluter = ModuloPolluter(modulo=100, level="cell", probability=1.0)
        result = polluter.apply(sample_df)

        assert result.loc[0, "int_col"] == 0  # 1000 % 100
        assert result.loc[1, "int_col"] == 0  # 2000 % 100
        pd_testing.assert_series_equal(sample_df["str_col"], result["str_col"])

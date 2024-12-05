import pytest
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
from datetime import datetime, timedelta

from pollutEM.polluters import TimestampPolluter, CoordinatePolluter


@pytest.fixture
def timestamp_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2023-04-05T06:28:04.640Z",
                    "2023-04-05T07:28:04.640Z",
                    "2023-04-05T08:28:04.640Z",
                ]
            ),
            "other_col": [1, 2, 3],
        }
    )


@pytest.fixture
def coordinate_df():
    return pd.DataFrame(
        {
            "latitude": [40.7128, 51.5074, 35.6762],
            "longitude": [-74.0060, -0.1278, 139.6503],
            "other_col": [1, 2, 3],
        }
    )


class TestTimestampPolluter:
    def test_forward_shift(self, timestamp_df):
        np.random.seed(42)
        polluter = TimestampPolluter(
            max_time_shift=3600, direction="forward", indices=["timestamp"]  # 1 hour
        )
        result = polluter.apply(timestamp_df)

        # Check all timestamps are shifted forward
        assert all(result["timestamp"] > timestamp_df["timestamp"])
        # Check max shift is within bounds
        max_shift = (result["timestamp"] - timestamp_df["timestamp"]).max()
        assert max_shift <= timedelta(seconds=3600)
        # Check other column unchanged
        pd_testing.assert_series_equal(timestamp_df["other_col"], result["other_col"])

    def test_backward_shift(self, timestamp_df):
        np.random.seed(42)
        polluter = TimestampPolluter(
            max_time_shift=3600, direction="backward", indices=["timestamp"]
        )
        result = polluter.apply(timestamp_df)

        assert all(result["timestamp"] < timestamp_df["timestamp"])
        max_shift = (timestamp_df["timestamp"] - result["timestamp"]).max()
        assert max_shift <= timedelta(seconds=3600)

    def test_bidirectional_shift(self, timestamp_df):
        np.random.seed(42)
        polluter = TimestampPolluter(max_time_shift=3600, direction="both", indices=["timestamp"])
        result = polluter.apply(timestamp_df)

        max_abs_shift = abs((result["timestamp"] - timestamp_df["timestamp"]).max())
        assert max_abs_shift <= timedelta(seconds=3600)


class TestCoordinatePolluter:
    def test_latitude_bounds(self, coordinate_df):
        np.random.seed(42)
        polluter = CoordinatePolluter(max_deviation=10.0, indices=["latitude"])
        result = polluter.apply(coordinate_df)

        # Check latitude stays within bounds
        assert all(result["latitude"] >= -90)
        assert all(result["latitude"] <= 90)
        # Values should be different
        assert not result["latitude"].equals(coordinate_df["latitude"])
        # Other columns unchanged
        pd_testing.assert_series_equal(coordinate_df["other_col"], result["other_col"])

    def test_longitude_pollution(self, coordinate_df):
        np.random.seed(42)
        polluter = CoordinatePolluter(max_deviation=0.001, indices=["longitude"])
        result = polluter.apply(coordinate_df)

        # Check deviation is within bounds
        max_change = abs(result["longitude"] - coordinate_df["longitude"]).max()
        assert max_change <= 0.001

        # Values should be different
        assert not result["longitude"].equals(coordinate_df["longitude"])

    def test_both_coordinates(self, coordinate_df):
        np.random.seed(42)
        polluter = CoordinatePolluter(max_deviation=0.001, indices=["latitude", "longitude"])
        result = polluter.apply(coordinate_df)

        assert not result["latitude"].equals(coordinate_df["latitude"])
        assert not result["longitude"].equals(coordinate_df["longitude"])
        assert all(result["latitude"] >= -90) and all(result["latitude"] <= 90)

    def test_invalid_level(self, coordinate_df):
        with pytest.raises(
            ValueError, match="This object does not allow for manipulation at row level"
        ):
            CoordinatePolluter(level="row")

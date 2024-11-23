from abc import ABC, abstractmethod
from typing import Callable, Type, Union, Optional
import pandas as pd
import numpy as np


class BasePolluter(ABC):
    def __init__(
        self,
        transformation: Callable,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Base class for data polluters.
        Args:
            transformation: A callable that applies a transformation to a value.
        """
        self.transformation = transformation
        self.level = level
        self.probability = probability
        self.indices = indices
        self.seed = seed

        self.type_mapping = self._get_type_mapping()
        self.allowed_levels = self._get_allowed_levels()

        if self.level not in self.allowed_levels:
            raise ValueError(f"This object does not allow for manipulation at {level} level.")

        if self.probability is not None and not (0 <= self.probability <= 1):
            raise ValueError("Probability must be between 0 and 1")

        if self.probability is None and self.indices is None:
            raise ValueError("Must specify either probability or indices. Use one or the other.")

        if self.probability is not None and self.indices is not None:
            raise ValueError("Cannot specify both probability and indices. Use one or the other.")

        if self.indices is not None:
            if self.level == "column" and not all(isinstance(idx, str) for idx in self.indices):
                raise ValueError("Column indices must be strings (column names)")
            elif self.level == "row" and not all(isinstance(idx, int) for idx in self.indices):
                raise ValueError("Row indices must be integers")

    @abstractmethod
    def _get_type_mapping(self) -> dict[Type, Type]:
        """
        Define the mapping of input types to output types for this polluter.
        Must be implemented by subclasses.
        Returns:
            dict mapping input Pandas datatypes to output Pandas datatypes
        """
        pass

    @abstractmethod
    def _get_allowed_levels(self) -> list[str]:
        """
        Define what levels are allowed for a Polluter.
        Not all Polluters work on Cell, Row and Column-Level.

        Returns:
            list[str]: Subset of ["cell", "column", "row"]
        """
        pass

    def _get_target_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Get columns to transform based on type and indices/probability.
        """
        valid_columns = [col for col in df.columns if self._is_valid_column_type(df[col].dtype)]
        if self.indices is not None:
            # Validate specified columns exist and are of valid type
            invalid = set(self.indices) - set(valid_columns)
            if invalid:
                raise ValueError(f"Invalid columns for transformation: {invalid}")
            return self.indices
        else:
            # Sample columns from valid columns
            mask = np.random.random(len(valid_columns)) < self.probability
            return list(pd.Index(valid_columns)[mask])

    def _get_target_rows(self, df: pd.DataFrame) -> list[int]:
        """
        Get rows to transform based on indices/probability.
        """
        if self.indices is not None:
            # Validate that specified row indices exist
            invalid = set(self.indices) - set(df.index)
            if invalid:
                raise ValueError(f"Rows not found in DataFrame: {invalid}")
            return self.indices
        else:
            # Sample rows based on probability
            mask = np.random.random(len(df)) < self.probability
            return list(df.index[mask])

    def _is_valid_column_type(self, input_type: Type) -> bool:
        """Check if the column type is supported."""
        return any(
            issubclass(input_type.type, supported_type)
            for supported_type in self.type_mapping.keys()
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the pollution to the data.
        Args:
            df: Input DataFrame
        Returns:
            Modified DataFrame
        """
        df = df.copy()

        # Set random seed if specified
        if self.seed is not None:
            np.random.seed(self.seed)

        # Get columns to transform
        target_columns = self._get_target_columns(df)
        if not target_columns:
            return df  # No columns to transform

        # Apply transformation based on level
        if self.level == "column":
            df = self._apply_column(df, target_columns)
        elif self.level == "row":
            target_rows = self._get_target_rows(df)
            df = self._apply_row(df, target_rows, target_columns)
        elif self.level == "cell":
            target_rows = self._get_target_rows(df)
            df = self._apply_cell(df, target_rows, target_columns)

        return df

    def _validate_column_type(self, input_type: Type) -> Type:
        """Get corresponding output type for supported input type."""
        for supported_input_type, output_type in self.type_mapping.items():
            if issubclass(input_type.type, supported_input_type):
                return output_type
        raise ValueError(
            f"Input type {input_type} not supported. "
            f"Supported types: {list(self.type_mapping.keys())}"
        )

    def _apply_column(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Apply transformation to selected columns."""
        for column in columns:
            output_type = self._validate_column_type(df[column].dtype)
            try:
                # Try column-wise transformation
                transformed_values = self.transformation(df[column])
            except TypeError:
                # Fall back to element-wise
                transformed_values = df[column].apply(self.transformation)
            df[column] = transformed_values.astype(output_type)
        return df

    def _apply_row(self, df: pd.DataFrame, rows: list[int], columns: list[str]) -> pd.DataFrame:
        """Apply transformation to selected rows."""
        for column in columns:
            output_type = self._validate_column_type(df[column].dtype)
            transformed_values = df.loc[rows, column].apply(self.transformation)
            df.loc[rows, column] = transformed_values.astype(output_type)
        return df

    def _apply_cell(self, df: pd.DataFrame, rows: list[int], columns: list[str]) -> pd.DataFrame:
        """Apply transformation to selected cells."""
        for column in columns:
            for row in rows:
                if np.random.random() < self.probability:
                    output_type = self._validate_column_type(df[column].dtype)
                    df.at[row, column] = self.transformation(df.at[row, column]).astype(
                        output_type
                    )
        return df

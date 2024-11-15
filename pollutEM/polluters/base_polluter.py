from abc import ABC, abstractmethod
from typing import Callable, Type, Union, List
import pandas as pd
import numpy as np


class BasePolluter(ABC):
    def __init__(self, transformation: Callable):
        """
        Base class for data polluters.
        Args:
            transformation: A callable that applies a transformation to a value.
        """
        self.transformation = transformation
        self.type_mapping = self._get_type_mapping()
        self.allowed_levels = self._get_allowed_levels()

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

    def _validate_column_type(self, input_type: Type) -> Type:
        """
        Validate that the input type is supported and get its corresponding output type.
        Args:
            input_type: The Pandas dtype of the input column
        Returns:
            The expected output Pandas dtype
        Raises:
            ValueError: If the input type is not supported by this polluter
        """
        for supported_input_type, output_type in self.type_mapping.items():
            if issubclass(input_type.type, supported_input_type):
                return output_type
        raise ValueError(
            f"Input type {input_type} is not supported by this polluter. "
            f"Supported types: {list(self.type_mapping.keys())}"
        )

    def apply(
        self,
        df: pd.DataFrame,
        target_columns: Union[str, List[str]],
        level: str = "column",
    ) -> pd.DataFrame:
        """
        Applies the pollution to the data.
        Args:
            df: Input DataFrame
            target_columns: Column(s) to apply transformation to
            level: Level of application ('column', 'cell', or 'row')
        Returns:
            Modified DataFrame
        """
        df = df.copy()

        # Step 0: Check whether Polluter permits this level
        if level not in self.allowed_levels:
            raise ValueError(f"This object does not allow for manipulation at {level} level.")

        # Convert single column to list
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        # Step 1: Check if all target columns exist in the DataFrame
        missing_columns = [col for col in target_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following column(s) are missing from the DataFrame: {missing_columns}"
            )

        # Validate all column types before proceeding
        for column in target_columns:
            input_type = df[column].dtype
            self._validate_column_type(input_type)

        if level == "column":
            for column in target_columns:
                df = self._apply_column(df, column)
        elif level == "cell":
            for column in target_columns:
                df = self._apply_cell(df, column)
        elif level == "row":
            df = self._apply_row(df, target_columns)
        else:
            raise ValueError(f"Invalid level: {level}. Must be 'column', 'cell', or 'row'")
        return df

    def _apply_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply transformation to the entire column."""
        # TODO: Change this so that Polluters working on cell and column level both work!
        # If a function can't work with apply directly because it needs the whole column, like Z-Score Normalization, this causes issues!
        input_type = df[column].dtype
        output_type = self._validate_column_type(input_type)
        df[column] = df[column].apply(self.transformation).astype(output_type)
        return df

    def _apply_row(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply transformation to specific rows."""
        # Randomly select 10% of rows to transform
        mask = np.random.random(len(df)) < 0.1

        for column in columns:
            output_type = self._validate_column_type(df[column].dtype)
            transformed_values = df.loc[mask, column].apply(self.transformation)
            df.loc[mask, column] = transformed_values.astype(output_type)

        return df

    def _apply_cell(self, df: pd.DataFrame, column: str, ratio: float = 0.1) -> pd.DataFrame:
        """Apply transformation to random cells in the column."""
        # Create a mask for randomly selected cells
        mask = np.random.random(len(df)) < ratio

        output_type = self._validate_column_type(df[column].dtype)
        transformed_values = df.loc[mask, column].apply(self.transformation)
        df.loc[mask, column] = transformed_values.astype(output_type)

        return df

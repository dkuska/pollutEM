from abc import ABC, abstractmethod
from typing import Callable, Type, Union, List

import polars as pl


class BasePolluter(ABC):
    def __init__(self, transformation: Callable):
        """
        Base class for data polluters.

        Args:
            transformation: A callable that applies a transformation to a value.
        """
        self.transformation = transformation
        self.type_mapping = self._get_type_mapping()

    @abstractmethod
    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        """
        Define the mapping of input types to output types for this polluter.
        Must be implemented by subclasses.

        Returns:
            dict mapping input Polars datatypes to output Polars datatypes
        """
        pass

    def _validate_column_type(self, input_type: pl.DataType) -> pl.DataType:
        """
        Validate that the input type is supported and get its corresponding output type.

        Args:
            input_type: The Polars datatype of the input column

        Returns:
            The expected output Polars datatype

        Raises:
            ValueError: If the input type is not supported by this polluter
        """
        for supported_input_type, output_type in self.type_mapping.items():
            if isinstance(input_type, supported_input_type):
                return output_type
        raise ValueError(
            f"Input type {input_type} is not supported by this polluter. "
            f"Supported types: {list(self.type_mapping.keys())}"
        )

    def apply(
        self,
        df: pl.DataFrame,
        target_columns: Union[str, List[str]],
        level: str = "column",
    ) -> pl.DataFrame:
        """
        Applies the pollution to the data.

        Args:
            df: Input DataFrame
            target_columns: Column(s) to apply transformation to
            level: Level of application ('column', 'cell', or 'row')

        Returns:
            Modified DataFrame
        """
        df = df.clone()

        # Convert single column to list
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        # Validate all column types before proceeding
        for column in target_columns:
            input_type = df[column].dtype
            self._validate_column_type(input_type)

        if level == "column":
            for column in target_columns:
                df = self._apply_column(df, column)
        elif level == "cell":
            pass
        elif level == "row":
            pass
        else:
            raise ValueError(f"Invalid level: {level}. Must be 'column', 'cell', or 'row'")

        return df

    def _apply_column(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Apply transformation to the entire column."""
        input_type = df[column].dtype
        output_type = self._validate_column_type(input_type)

        return df.with_columns(
            pl.col(column)
            .map_elements(self.transformation, return_dtype=output_type)
            .alias(column)
        )

    def _apply_row(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Apply transformation to specific rows."""
        # Implementation will depend on specific needs
        pass

    def _apply_cell(self, df: pl.DataFrame, column: str, ratio: float = 0.1) -> pl.DataFrame:
        """Apply transformation to random cells in the column."""
        # Implementation will depend on specific needs
        pass

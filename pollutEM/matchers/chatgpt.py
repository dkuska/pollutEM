import time
from typing import List, Tuple, Set, Optional

import pandas as pd
import numpy as np
import openai


class ChatGPTMatcher:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the ChatGPT-based entity matcher.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            temperature: Temperature parameter for GPT responses (0.0 for most deterministic)
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.numeric_columns: Optional[Set[str]] = None
        openai.api_key = api_key

    def get_numeric_columns(self, data: pd.DataFrame) -> Set[str]:
        """Get columns with numeric data."""
        return set(data.select_dtypes(include=[np.number]).columns)

    def calculate_features(
        self, data1: pd.DataFrame, data2: pd.DataFrame, pairs_df: pd.DataFrame, mode: str = "train"
    ) -> list:
        """
        Calculate feature differences and format them as strings using the pair-based approach.
        Args:
            data1: First dataset (original/clean data)
            data2: Second dataset (can be same as data1 for training, or polluted for testing)
            pairs_df: DataFrame containing the pairs to compare (with p1, p2 columns)
            mode: Either "train" or "test" - affects how the data is merged
        Returns:
            List of tuples containing formatted string pairs
        """
        # Input validation
        if "id" not in data1.columns or "id" not in data2.columns:
            raise KeyError("Both datasets must contain an 'id' column")

        # Ensure IDs are integers
        data1 = data1.copy()
        data2 = data2.copy()
        pairs_df = pairs_df.copy()
        data1["id"] = data1["id"].astype(int)
        data2["id"] = data2["id"].astype(int)
        pairs_df["p1"] = pairs_df["p1"].astype(int)
        pairs_df["p2"] = pairs_df["p2"].astype(int)

        # Get numeric columns if not already set
        if self.numeric_columns is None:
            self.numeric_columns = set(self.get_numeric_columns(data1))

        # Create left and right dataframes
        if mode == "train":
            # For training, both merges use the same dataset
            left_features = data1.merge(pairs_df, left_on="id", right_on="p1")
            right_features = data1.merge(pairs_df, left_on="id", right_on="p2")
        else:
            # For testing, merge with respective datasets based on mode
            left_features = data1.merge(pairs_df, left_on="id", right_on="p1")
            right_features = data2.merge(pairs_df, left_on="id", right_on="p2")

        # Format the data using the same approach as pair_based_data_wrangling_formatter
        formatted_pairs = []

        for idx in range(len(left_features)):
            left_record = left_features.iloc[idx]
            right_record = right_features.iloc[idx]

            # Format left entity
            left_str = "Entity A is "
            for col in self.numeric_columns:
                if col != "id":
                    left_str += f"COL {col} VAL {left_record[col]} "

            # Format right entity
            right_str = "Entity B is "
            for col in self.numeric_columns:
                if col != "id":
                    right_str += f"COL {col} VAL {right_record[col]} "

            formatted_pairs.append((left_str.strip(), right_str.strip()))

        return formatted_pairs

    def create_prompt(self, entity_pair: Tuple[str, str]) -> str:
        """Create a prompt for ChatGPT to determine if two entities match."""
        return f"""Given the following two entities, determine if they represent the same entity.
Reply with a single word: either 'match' or 'non-match'.

{entity_pair[0]}
{entity_pair[1]}

Answer:"""

    def query_chatgpt(self, prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
        """Query ChatGPT with retry logic for API rate limits."""
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip().lower()
            except openai.error.RateLimitError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
            except Exception as e:
                print(f"Error querying ChatGPT: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def test(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        test_split: pd.DataFrame,
    ) -> List[bool]:
        """
        Perform entity matching using ChatGPT.

        Args:
            data1: First dataset
            data2: Second dataset
            pairs_df: DataFrame containing the pairs to compare
            mode: Either "train" or "test"
            batch_size: Number of pairs to process before sleeping to avoid rate limits

        Returns:
            List of boolean values indicating matches
        """
        formatted_pairs = self.calculate_features(data1, data2, test_split, mode="test")
        matches = []

        for i, pair in enumerate(formatted_pairs):
            prompt = self.create_prompt(pair)
            response = self.query_chatgpt(prompt)
            matches.append(response == "match")

        return matches

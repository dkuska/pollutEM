import time
from typing import Tuple, Set, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


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
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> list[str]:
        """Get numeric columns following original implementation."""
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        cols = list(df.select_dtypes(include=numerics).columns)
        if "id" not in cols:
            cols.append("id")

        # Return None if only ID column exists
        if len(cols) == 1 and cols[0] == "id":
            return None

        return cols

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
            self.numeric_columns = set(ChatGPTMatcher.get_numeric_columns(data1))

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
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                response_text = response.choices[0].message.content.strip().lower()
                return response_text
            except Exception as e:
                print(f"Error querying ChatGPT: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def test(
        self,
        original_features: pd.DataFrame,
        polluted_features: pd.DataFrame,
        test_split: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
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
        formatted_pairs = self.calculate_features(
            original_features, polluted_features, test_split, mode="test"
        )

        scores = []

        for pair in tqdm(
            formatted_pairs,
            desc="Processing entity pairs",
        ):
            prompt = self.create_prompt(pair)
            response = self.query_chatgpt(prompt)

            # Convert ChatGPT's response to a confidence score
            if response == "match":
                scores.append(1.0)
            elif response == "non-match":
                scores.append(0.0)
            else:
                print(f"Error for sample {pair} and resp {response}")

        # Format results similar to the test function
        result = pd.DataFrame({"score": scores})
        result["prediction"] = 0
        result.loc[result["score"] > threshold, "prediction"] = 1

        return result

from typing import List, Any, Generator, Optional
import random
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class PollutionConfigGenerator:
    def __init__(self, master_config: Optional[dict[str, Any]] = None):
        """
        Initialize the PollutionConfigGenerator.

        Args:
            master_config: Optional dictionary containing the master configuration.
                         Can be set later using set_master_config.
        """
        self.master_config = master_config
        self.templates: dict[str, dict] = {}
        if master_config:
            self.templates = self._extract_templates(master_config)

    def set_master_config(self, master_config: dict[str, Any]) -> None:
        """
        Set or update the master configuration.

        Args:
            master_config: dictionary containing the master configuration
        """
        self.master_config = master_config
        self.templates = self._extract_templates(master_config)

    def _extract_templates(self, master_config: dict[str, Any]) -> dict[str, dict]:
        """
        Extract pollution templates from master configuration.

        Args:
            master_config: dictionary containing the master configuration

        Returns:
            dictionary of pollution templates
        """
        templates = {}
        for pollution in master_config["pollutions"]:
            name = pollution["name"]
            params = pollution["params"].copy()
            params.pop("indices", None)
            params.pop("probability", None)
            params["level"] = "column"

            templates[name] = {
                "params": params,
                "applicable_columns": pollution.get("applicable_columns", []),
            }
        return templates

    def create_config(self, name: str, params: dict, columns: List[str]) -> dict[str, Any]:
        """
        Create a pollution configuration for given columns.

        Args:
            name: Name of the pollution type
            params: Parameters for the pollution
            columns: List of column names to apply pollution to

        Returns:
            Configuration dictionary
        """
        config = {"name": name, "params": params.copy()}
        config["params"]["indices"] = columns
        return {"pollutions": [config]}

    def generate_configs(
        self, name: str, template: dict, all_columns: List[str], samples_per_size: int = 5
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate configuration samples for different column combinations.

        Args:
            name: Name of the pollution type
            template: Template containing parameters and applicable columns
            all_columns: List of all available columns
            samples_per_size: Number of random samples to generate per combination size

        Yields:
            Configuration dictionaries
        """

        logger.info(f"Generating configurations for {name}")
        param_str = ", ".join(f"{k}={v}" for k, v in template["params"].items() if k != "level")
        logger.info(f"  With parameters: {param_str}")

        applicable_columns = [col for col in template["applicable_columns"] if col in all_columns]

        if not applicable_columns:
            logger.info(f"Warning: No applicable columns found for {name}")
            return

        logger.info(f"  Applicable columns: {', '.join(applicable_columns)}")

        for r in range(1, len(applicable_columns) + 1):
            all_combinations = list(combinations(applicable_columns, r))
            n_samples = min(samples_per_size, len(all_combinations))

            if n_samples > 0:
                selected = random.sample(all_combinations, n_samples)
                for cols in selected:
                    yield self.create_config(name, template["params"], list(cols))

    def get_all_configs(
        self, all_columns: List[str], samples_per_size: int = 5
    ) -> Generator[dict[str, Any], None, None]:
        """
        Generate configurations for all pollution types.

        Args:
            all_columns: List of all available columns
            samples_per_size: Number of random samples per combination size

        Yields:
            Configuration dictionaries for all pollution types
        """
        if not self.templates:
            raise ValueError("No templates available. Set master_config first.")

        for name, template in self.templates.items():
            yield from self.generate_configs(name, template, all_columns, samples_per_size)

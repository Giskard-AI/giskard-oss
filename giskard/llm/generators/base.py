from typing import Dict, Optional, Sequence

import json
import logging
from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel

from ...datasets.base import Dataset
from ..client import LLMClient, get_default_client
from ..client.base import ChatMessage
from ..errors import LLMGenerationError

logger = logging.getLogger("giskard.llm")


class BaseGenerator(ABC):
    @abstractmethod
    def generate_dataset(self, model, num_samples=10, column_types=None) -> Dataset:
        ...


class _BaseLLMGenerator(BaseGenerator, ABC):
    _default_temperature = 0.5
    _output_key = "inputs"

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        llm_temperature: Optional[float] = None,
        llm_client: LLMClient = None,
        llm_seed: int = 1729,
    ):
        self.languages = languages or ["en"]
        self.llm_temperature = llm_temperature if llm_temperature is not None else self._default_temperature
        self.llm_client = llm_client or get_default_client()
        self.llm_seed = llm_seed

    def generate_dataset(self, model: BaseModel, num_samples: int = 10, column_types: Dict = None) -> Dataset:
        """Generates a test dataset for the model.

        Parameters
        ----------
        model : BaseModel
            The model to generate a test dataset for.
        num_samples : int
            The number of samples to generate, by default 10.
        column_types : dict, optional
            The column types for the generated datasets. (Default value = None)

        Returns
        -------
        Dataset
            The generated dataset.

        Raises
        ------
        LLMGenerationError
            If the generation fails.
        """
        messages = self._format_messages(model, num_samples, column_types)

        out = self.llm_client.complete(
            messages=messages,
            temperature=self.llm_temperature,
            caller_id=self.__class__.__name__,
            seed=self.llm_seed,
            format="json_object",
        )

        generated = self._parse_output(out)

        df = pd.DataFrame(generated)

        # Ensure column names match the model's feature_names
        # This handles cases where LLM generates different column names than expected
        if model.feature_names:
            df = self._align_columns_to_feature_names(df, model.feature_names)

        dataset = Dataset(
            df=df,
            name=self._make_dataset_name(model),
            validation=False,
            column_types=column_types,
        )

        return dataset

    def _align_columns_to_feature_names(self, df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
        """Align DataFrame columns to match the model's feature_names.

        If the generated columns don't match feature_names, attempt to rename them.
        This handles the case where LLM generates data with different column names
        than specified in the model's feature_names.

        Parameters
        ----------
        df : pd.DataFrame
            The generated DataFrame.
        feature_names : Sequence[str]
            The expected feature names from the model.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns aligned to feature_names.
        """
        current_columns = list(df.columns)

        # If columns already match, return as-is
        if set(current_columns) == set(feature_names):
            return df[list(feature_names)]  # Reorder to match feature_names order

        # If column count matches, assume positional mapping
        if len(current_columns) == len(feature_names):
            rename_map = dict(zip(current_columns, feature_names))
            logger.debug(
                f"Renaming generated columns to match feature_names: {rename_map}"
            )
            return df.rename(columns=rename_map)

        # If we have fewer feature_names than columns, try to find matching columns
        # or use the first N columns
        if len(feature_names) < len(current_columns):
            # First try exact matches
            matching = [col for col in current_columns if col in feature_names]
            if set(matching) == set(feature_names):
                return df[list(feature_names)]
            # Otherwise, use first N columns and rename
            columns_to_use = current_columns[: len(feature_names)]
            rename_map = dict(zip(columns_to_use, feature_names))
            logger.debug(
                f"Using first {len(feature_names)} columns and renaming: {rename_map}"
            )
            return df[columns_to_use].rename(columns=rename_map)

        # If we have more feature_names than columns, log warning and return as-is
        logger.warning(
            f"Generated data has {len(current_columns)} columns but model expects "
            f"{len(feature_names)} features. Column names may not match."
        )
        return df

    def _parse_output(self, raw_output: ChatMessage):
        try:
            data = json.loads(raw_output.content)
            if self._output_key:
                data = data[self._output_key]
        except (json.JSONDecodeError, KeyError) as err:
            logger.error("Generator output parse error, got raw output: %s", raw_output.content)
            raise LLMGenerationError("Could not parse generated data") from err
        return data

    def _make_dataset_name(self, model: BaseModel):
        return f"Synthetic Test Dataset for {model.name}"

    @abstractmethod
    def _format_messages(self, model: BaseModel, num_samples: int):
        ...


class LLMBasedDataGenerator(_BaseLLMGenerator):
    def __init__(
        self,
        prompt: str,
        prefix_messages: Optional[Sequence[ChatMessage]] = None,
        languages: Optional[Sequence[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.prefix_messages = prefix_messages or []
        self.languages = languages or ["en"]

    def _format_messages(
        self, model: BaseModel, num_samples: int, column_types: Optional[Dict] = None
    ) -> Sequence[ChatMessage]:
        prompt = self.prompt.format(model=model, num_samples=num_samples, languages=", ".join(self.languages))
        return self.prefix_messages + [ChatMessage(role="user", content=prompt)]

from typing import Optional

import ast
from pathlib import Path, PurePath

import pandas as pd

from ...datasets.base import Dataset
from ..evaluators.string_matcher import StringMatcherConfig

INDIRECT_INJECTION_DATA_URL = "https://raw.githubusercontent.com/asamassekou10/giskard-indirect-injections/main/indirect_injections.csv"
INDIRECT_INJECTION_META_URL = "https://raw.githubusercontent.com/asamassekou10/giskard-indirect-injections/main/indirect_injections_meta.csv"

LOCAL_INDIRECT_INJECTION_DATA_PATH = Path(__file__).parent.joinpath("local", "indirect_injections.csv")
LOCAL_INDIRECT_INJECTION_META_PATH = Path(__file__).parent.joinpath("local", "indirect_injections_meta.csv")


def from_records_to_configs(records):
    """Convert metadata records to StringMatcherConfig objects."""
    configs = []
    for row in records:
        kwargs = {k: v for k, v in row.items() if k in list(StringMatcherConfig.__annotations__.keys())}
        configs.append(StringMatcherConfig(**kwargs))
    return configs


def _try_download(url: str, local_path: PurePath):
    """Try to download from URL, fall back to local path if unavailable."""
    try:
        return pd.read_csv(url, index_col=["index"])
    except:  # noqa NOSONAR
        return pd.read_csv(local_path, index_col=["index"])


class IndirectInjectionDataLoader:
    """
    Data loader for indirect prompt injection attack patterns.

    Loads attack patterns for:
    - RAG context poisoning (malicious documents in retrieval corpus)
    - Tool use manipulation (function calling hijacking)
    - External content injection (websites, emails, PDFs)
    - Multi-source context mixing (priority override attacks)

    Parameters
    ----------
    num_samples : Optional[int]
        Number of samples to load. If None, loads all available samples.
    injection_data_url : str
        URL to download indirect injection prompts CSV
    indirect_meta_url : str
        URL to download indirect injection metadata CSV

    Examples
    --------
    >>> loader = IndirectInjectionDataLoader(num_samples=50)
    >>> dataset = loader.load_dataset_from_group(features=["prompt"], group="RAG Poisoning")
    >>> configs = loader.configs_from_group("RAG Poisoning")
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        injection_data_url: str = INDIRECT_INJECTION_DATA_URL,
        indirect_meta_url: str = INDIRECT_INJECTION_META_URL,
    ):
        self.num_samples = num_samples
        self._df = None
        self.injection_data_url = injection_data_url
        self.indirect_meta_url = indirect_meta_url

    def load_dataset_from_group(self, features, group) -> Dataset:
        """
        Load dataset for a specific attack group.

        Parameters
        ----------
        features : list
            List of feature names for the dataset
        group : str
            Attack group name (e.g., "RAG Poisoning", "Tool Manipulation")

        Returns
        -------
        Dataset
            Giskard dataset containing prompts for the specified group
        """
        prompts = self.prompts_from_group(group)
        prompts = pd.DataFrame({feature: prompts for feature in features}, index=prompts.index)
        return Dataset(
            df=prompts,
            name=f"Indirect Injection Prompts - {group}",
            target=None,
            cat_columns=None,
            validation=False,
        )

    @property
    def df(self):
        """Load and cache the combined injection + metadata dataframe."""
        if self._df is None:
            # Load indirect injection prompts
            indirect_injections_df = _try_download(self.injection_data_url, LOCAL_INDIRECT_INJECTION_DATA_PATH)
            # Load metadata with evaluator configs
            meta_df = _try_download(self.indirect_meta_url, LOCAL_INDIRECT_INJECTION_META_PATH)
            # Parse expected_strings from string representation to list
            meta_df.expected_strings = meta_df.expected_strings.apply(ast.literal_eval)
            # Join datasets
            self._df = indirect_injections_df.join(meta_df)

            # Sample if requested
            if self.num_samples is not None:
                self._df = self._df.sample(min(self.num_samples, len(self._df)))

        return self._df

    @property
    def names(self):
        """Get list of all attack names."""
        return self.df.name.tolist()

    @property
    def groups(self):
        """Get list of unique attack groups."""
        return self.df.group_mapping.unique().tolist()

    def df_from_group(self, group):
        """Filter dataframe to specific attack group."""
        return self.df.loc[self.df["group_mapping"] == group]

    def prompts_from_group(self, group):
        """Get prompts for specific attack group."""
        return self.df_from_group(group).prompt

    def configs_from_group(self, group):
        """Get StringMatcherConfig objects for specific attack group."""
        configs_records = self.df_from_group(group).drop(["prompt"], axis=1).to_dict("records")
        return from_records_to_configs(configs_records)

    def group_description(self, group):
        """Get human-readable description for attack group."""
        group_description = self.df_from_group(group).description.to_list()
        return group_description[0] if group_description else "Indirect prompt injection attack"

    def group_deviation_description(self, group):
        """Get deviation description for attack group (used in issue metadata)."""
        group_deviation_description = self.df_from_group(group).deviation_description.to_list()
        return group_deviation_description[0] if group_deviation_description else "of the injected inputs successfully manipulated the model."

import typing
from typing import List, Optional

import inspect
import numbers
import re
import uuid
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2, ks_2samp, wasserstein_distance

from giskard.core.test_result import TestMessage, TestMessageLevel, TestResult
from giskard.datasets.base import Dataset
from giskard.models.base import BaseModel
from giskard.registry.decorators import test
from giskard.registry.slicing_function import SlicingFunction
from giskard.testing.utils.utils import check_slice_not_empty, validate_classification_label

from . import debug_description_prefix

other_modalities_pattern = "^other_modalities_[a-z0-9]{32}$"


def check_if_debuggable(actual_ds, reference_ds):
    if actual_ds.df.empty:
        raise ValueError("Your actual_dataset is empty. Debug is not defined for this case.")
    if reference_ds.df.empty:
        raise ValueError("Your reference_dataset is empty. Debug is not defined for this case.")
    if actual_ds.id == reference_ds.id:
        raise ValueError(
            "You passed the same dataset as actual_dataset and reference_dataset. "
            "Debug is not defined for this case."
        )
    if actual_ds.df.equals(reference_ds.df):
        raise ValueError(
            "Your actual_dataset is identical to your reference_dataset. " "Debug is not defined for this case."
        )


def _calculate_psi(category, actual_distribution, expected_distribution):
    # To use log and avoid zero distribution probability,
    # we bound distribution probability by min_distribution_probability
    min_distribution_probability = 0.0001

    expected_distribution_bounded = max(expected_distribution[category], min_distribution_probability)
    actual_distribution_bounded = max(actual_distribution[category], min_distribution_probability)
    modality_psi = (expected_distribution_bounded - actual_distribution_bounded) * np.log(
        expected_distribution_bounded / actual_distribution_bounded
    )
    return modality_psi


def _calculate_frequencies(actual_series, reference_series, max_categories=None):
    all_modalities = list(set(reference_series).union(set(actual_series)))
    if max_categories is not None and len(all_modalities) > max_categories:
        var_count_expected = dict(Counter(reference_series).most_common(max_categories))
        other_modalities_key = "other_modalities_" + uuid.uuid1().hex
        var_count_expected[other_modalities_key] = len(reference_series) - sum(var_count_expected.values())
        categories_list = list(var_count_expected.keys())

        var_count_actual = Counter(actual_series)
        # For test data, we take the same category names as expected_data
        var_count_actual = {i: var_count_actual[i] for i in categories_list}
        var_count_actual[other_modalities_key] = len(actual_series) - sum(var_count_actual.values())

        all_modalities = categories_list
    else:
        var_count_expected = Counter(reference_series)
        var_count_actual = Counter(actual_series)
    expected_frequencies = np.array([var_count_expected[i] for i in all_modalities])
    actual_frequencies = np.array([var_count_actual[i] for i in all_modalities])
    return all_modalities, actual_frequencies, expected_frequencies


def _calculate_drift_psi(actual_series, reference_series, max_categories):
    (
        all_modalities,
        actual_frequencies,
        expected_frequencies,
    ) = _calculate_frequencies(actual_series, reference_series, max_categories)
    expected_distribution = expected_frequencies / len(reference_series)
    actual_distribution = actual_frequencies / len(actual_series)
    total_psi = 0
    output_data = []
    for category, modality in enumerate(all_modalities):
        modality_psi = _calculate_psi(category, actual_distribution, expected_distribution)
        total_psi += modality_psi
        output_data.append(
            {
                "Modality": modality,
                "Reference_distribution": expected_distribution[category],
                "Actual_distribution": expected_distribution[category],
                "Psi": modality_psi,
            }
        )
    return total_psi, pd.DataFrame(output_data)


def _calculate_ks(actual_series, reference_series):
    return ks_2samp(reference_series, actual_series)


def _calculate_earth_movers_distance(actual_series, reference_series):
    unique_reference = np.unique(reference_series)
    unique_actual = np.unique(actual_series)
    sample_space = list(set(unique_reference).union(set(unique_actual)))
    val_max = max(sample_space)
    val_min = min(sample_space)
    if val_max == val_min:
        metric = 0
    else:
        # Normalizing reference_series and actual_series for comparison purposes
        reference_series = (reference_series - val_min) / (val_max - val_min)
        actual_series = (actual_series - val_min) / (val_max - val_min)

        metric = wasserstein_distance(reference_series, actual_series)
    return metric


def _calculate_chi_square(actual_series, reference_series, max_categories):
    (
        all_modalities,
        actual_frequencies,
        expected_frequencies,
    ) = _calculate_frequencies(actual_series, reference_series, max_categories)
    chi_square = 0
    # it's necessary for comparison purposes to normalize expected_frequencies
    # so that reference and actual has the same size
    # See https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_stats_py.py#L6787
    k_norm = actual_series.shape[0] / reference_series.shape[0]
    output_data = []
    for i, modality in enumerate(all_modalities):
        chi_square_value = (actual_frequencies[i] - expected_frequencies[i] * k_norm) ** 2 / (
            expected_frequencies[i] * k_norm
        )
        chi_square += chi_square_value

        output_data.append(
            {
                "Modality": modality,
                "Reference_frequencies": expected_frequencies[i],
                "Actual_frequencies": actual_frequencies[i],
                "Chi_square": chi_square_value,
            }
        )

    # if reference_series and actual_series has only one modality it turns nan (len(all_modalities)=1)
    if len(all_modalities) > 1:
        chi_cdf = chi2.cdf(chi_square, len(all_modalities) - 1)
        p_value = 1 - chi_cdf if chi_cdf != 0 else 0
    else:
        p_value = 0
    return chi_square, p_value, pd.DataFrame(output_data)


def _validate_feature_type(gsk_dataset, column_name, feature_type):
    assert (
        gsk_dataset.column_types[column_name] == feature_type
    ), f'Column "{column_name}" is not of type "{feature_type}"'


def _validate_column_name(actual_ds, reference_ds, column_name):
    assert (
        column_name in actual_ds.columns
    ), f'"{column_name}" is not a column of Actual Dataset Columns: {", ".join(actual_ds.columns)}'
    assert (
        column_name in reference_ds.columns
    ), f'"{column_name}" is not a column of Reference Dataset Columns: {", ".join(reference_ds.columns)}'


def _validate_series_notempty(actual_series, reference_series):
    if actual_series.empty:
        raise ValueError("Actual Series computed from the column is empty")
    if reference_series.empty:
        raise ValueError("Reference Series computed from the column is empty")


def _extract_series(actual_ds, reference_ds, column_name, feature_type):
    _validate_column_name(actual_ds, reference_ds, column_name)
    _validate_feature_type(actual_ds, column_name, feature_type)
    _validate_feature_type(reference_ds, column_name, feature_type)
    actual_series = actual_ds.df[column_name]
    reference_series = reference_ds.df[column_name]
    _validate_series_notempty(actual_series, reference_series)
    return actual_series, reference_series


@test(
    name="Categorical drift (PSI)",
    debug_description=debug_description_prefix + "with <b>the categories that have drifted "
    "the most from the 'actual_dataset'</b>.",
)
def test_drift_psi(
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    column_name: str,
    slicing_function: Optional[SlicingFunction] = None,
    threshold: float = 0.2,
    max_categories: int = 20,
    psi_contribution_percent: float = 0.2,
) -> TestResult:
    """Test categorical drift by PSI score.

    It tests that the PSI score between the actual and reference datasets is
    below the threshold for a given categorical feature.

    For example, the test is passed when the PSI score of gender between
    reference and actual sets is below threshold 0.2.

    Parameters
    ----------
    actual_dataset : Dataset
        Actual dataset used to compute the test
    reference_dataset : Dataset
        Reference dataset used to compute the test
    column_name : str
        Name of column with categorical feature
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets.
    threshold : float
        Threshold value for PSI. Default value = 0.2.
    max_categories : int
        The maximum categories to compute the PSI score.
    psi_contribution_percent : float
        The ratio between the PSI score of a given category over the total PSI
        score of the categorical variable. If there is a drift, the test
        provides all the categories that have a PSI contribution over than this
        ratio.
    debug : bool
        If True and the test fails, a dataset will be provided containing the
        actual_dataset rows with the categories that have drifted the most (more
        than psi_contribution_percent of the total PSI score). Default is False.

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_psi"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    actual_series, reference_series = _extract_series(actual_dataset, reference_dataset, column_name, "category")

    messages, passed, total_psi, output_data = _test_series_drift_psi(
        actual_series,
        reference_series,
        "data",
        max_categories,
        psi_contribution_percent,
        threshold,
    )

    # --- debug ---
    output_ds = list()
    if not passed:
        check_if_debuggable(actual_dataset, reference_dataset)
        main_drifting_modalities_bool = output_data["Psi"] > psi_contribution_percent * total_psi
        modalities_list = output_data[main_drifting_modalities_bool]["Modality"].tolist()
        if modalities_list:
            filtered_modalities = [w for w in modalities_list if not re.match(other_modalities_pattern, str(w))]
            output_ds.append(
                actual_dataset.slice(lambda df: df.loc[actual_series.isin(filtered_modalities)], row_level=False)
            )
            test_name = inspect.stack()[0][3]
            if len(output_ds[0].df.index) == 0:
                raise ValueError(
                    test_name
                    + f": the categories {filtered_modalities} completely drifted as they are not present in the "
                    f"'actual_dataset'"
                )
    # ---

    return TestResult(
        actual_slices_size=[len(actual_series)],
        reference_slices_size=[len(reference_series)],
        passed=passed,
        metric=total_psi,
        messages=messages,
        output_ds=output_ds,
    )


@test(
    name="Categorical drift (Chi-squared)",
    debug_description=debug_description_prefix + "with <b>the categories that have drifted "
    "the most from the 'actual_dataset'</b>.",
)
def test_drift_chi_square(
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    column_name: str,
    slicing_function: Optional[SlicingFunction] = None,
    threshold: float = 0.05,
    max_categories: int = 20,
    chi_square_contribution_percent: float = 0.2,
) -> TestResult:
    """Tests drift by chi-squared.

    The test checks if the p-value of the chi-square test between the actual and
    reference datasets is above a threshold for a given categorical feature.

    Parameters
    ----------
    actual_dataset : Dataset
        Actual dataset used to compute the test
    reference_dataset : Dataset
        Reference dataset used to compute the test
    column_name : str
        Name of column with categorical feature
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets
    threshold : float
        Threshold for p-value of chi-square. Default is 0.05.
    max_categories : int
        The maximum categories to compute the chi-square.
    chi_square_contribution_percent : float
        the ratio between the Chi-Square value of a given category over the
        total chi-square value of the categorical variable. If there is a drift,
        the test provides all the categories that have a PSI contribution over
        than this ratio.
    debug : bool
        If True and the test fails, a dataset will be provided containing the
        actual_dataset rows with the categories that have drifted the most
        (more than chi_square_contribution_percent of the total chi squared
        score).

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_chi_square"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    actual_series, reference_series = _extract_series(actual_dataset, reference_dataset, column_name, "category")

    messages, chi_square, p_value, passed, output_data = _test_series_drift_chi(
        actual_series,
        reference_series,
        "data",
        chi_square_contribution_percent,
        max_categories,
        threshold,
    )

    # --- debug ---
    output_ds = list()
    if not passed:
        check_if_debuggable(actual_dataset, reference_dataset)
        main_drifting_modalities_bool = output_data["Chi_square"] > chi_square_contribution_percent * chi_square
        modalities_list = output_data[main_drifting_modalities_bool]["Modality"].tolist()
        if modalities_list:
            filtered_modalities = [w for w in modalities_list if not re.match(other_modalities_pattern, str(w))]
            output_ds.append(
                actual_dataset.slice(lambda df: df.loc[actual_series.isin(filtered_modalities)], row_level=False)
            )
    # ---

    return TestResult(
        actual_slices_size=[len(actual_series)],
        reference_slices_size=[len(reference_series)],
        passed=passed,
        metric=p_value,
        messages=messages,
        output_ds=output_ds,
    )


@test(name="Numerical drift (Kolmogorov-Smirnov)")
def test_drift_ks(
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    column_name: str,
    slicing_function: Optional[SlicingFunction] = None,
    threshold: float = 0.05,
) -> TestResult:
    """Test drift with a Kolmogorov-Smirnov test.

    Test if the p-value of the Kolmogorov-Smirnov test between the actual and
    reference datasets is above a threshold for a given numerical feature.

    For example, if the threshold is set to 0.05, the test is passed when the
    p-value of the KS test of the numerical variable between the actual and
    reference datasets is higher than 0.05. It means that the null hypothesis
    (no drift) cannot be rejected at 5% confidence level and that we cannot
    assume drift for this variable.

    Parameters
    ----------
    actual_dataset : Dataset
        Actual dataset used to compute the test.
    reference_dataset : Dataset
        Reference dataset used to compute the test.
    column_name : str
        Name of column with numerical feature.
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets.
    threshold : float
        Threshold for p-value of Kolmogorov-Smirnov test. Default is = 0.05.

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_ks"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    actual_series, reference_series = _extract_series(actual_dataset, reference_dataset, column_name, "numeric")

    result = _calculate_ks(actual_series, reference_series)

    passed = bool(result.pvalue >= threshold)

    messages = _generate_message_ks(passed, result, threshold, "data")

    return TestResult(
        actual_slices_size=[len(actual_series)],
        reference_slices_size=[len(reference_series)],
        passed=passed,
        metric=result.pvalue,
        messages=messages,
    )


@test(name="Numerical drift (Earth mover's distance)")
def test_drift_earth_movers_distance(
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    column_name: str,
    slicing_function: Optional[SlicingFunction] = None,
    threshold: float = 0.2,
) -> TestResult:
    """Test drift by earth mover’s distance.

    Test if the earth movers distance between the actual and reference datasets
    is below a threshold for a given numerical feature.

    For example, if the threshold is set to 0.1, the test is passed when the
    earth movers distance of the numerical variable between the actual and
    reference datasets is lower than 0.1. It means that we cannot assume drift
    for this variable.

    Parameters
    ----------
    actual_dataset : Dataset
        Actual dataset used to compute the test
    reference_dataset : Dataset
        Reference dataset used to compute the test
    column_name : str
        Name of column with numerical feature
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets
    threshold : float
        Threshold for earth mover's distance. Default is 0.2.

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_earth_movers_distance"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    actual_series, reference_series = _extract_series(actual_dataset, reference_dataset, column_name, "numeric")

    metric = _calculate_earth_movers_distance(actual_series, reference_series)

    passed = bool(metric <= threshold)

    messages: Optional[List[TestMessage]] = None

    if not passed:
        messages = [
            TestMessage(
                type=TestMessageLevel.ERROR,
                text=f"The data is drifting (metric is equal to {np.round(metric, 9)} and is below the test risk level {threshold}) ",
            )
        ]
    return TestResult(
        actual_slices_size=[len(actual_series)],
        reference_slices_size=[len(reference_series)],
        passed=True if threshold is None else passed,
        metric=metric,
        messages=messages,
    )


@test(
    name="Label drift (PSI)",
    debug_description=debug_description_prefix + "with <b>the categories that have drifted "
    "the most from the 'actual_dataset'</b>.",
)
def test_drift_prediction_psi(
    model: BaseModel,
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    slicing_function: Optional[SlicingFunction] = None,
    max_categories: int = 10,
    threshold: float = 0.2,
    psi_contribution_percent: float = 0.2,
):
    """Tests drift of predictions by PSI score.

    Test if the PSI score between the reference and actual datasets is below the
    threshold for the classification labels predictions.

    Parameters
    ----------
    model : BaseModel
        Model used to compute the test
    actual_dataset : Dataset
        Actual dataset used to compute the test
    reference_dataset : Dataset
        Reference dataset used to compute the test
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets
    max_categories : int
        The maximum categories to compute the PSI score
    threshold : float
        Threshold value for PSI. Default is 0.2.
    psi_contribution_percent : float
        The ratio between the PSI score of a given category over the total PSI
        score of the categorical variable. If there is a drift, the test
        provides all the categories that have a PSI contribution over than this
        ratio.
    debug : bool
        If True and the test fails, a dataset will be provided containing the
        actual_dataset rows with the categories that have drifted the most (more
        than psi_contribution_percent of the total PSI score).

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_prediction_psi"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    prediction_reference = pd.Series(model.predict(reference_dataset).prediction)
    prediction_actual = pd.Series(model.predict(actual_dataset).prediction)
    messages, passed, total_psi, output_data = _test_series_drift_psi(
        prediction_actual,
        prediction_reference,
        "prediction",
        max_categories,
        psi_contribution_percent,
        threshold,
    )

    # --- debug ---
    output_ds = list()
    if not passed:
        check_if_debuggable(actual_dataset, reference_dataset)
        main_drifting_modalities_bool = output_data["Psi"] > psi_contribution_percent * total_psi
        modalities_list = output_data[main_drifting_modalities_bool]["Modality"].tolist()
        if modalities_list:
            filtered_modalities = [w for w in modalities_list if not re.match(other_modalities_pattern, str(w))]
            output_ds.append(
                actual_dataset.slice(
                    lambda df: df.loc[prediction_actual.isin(filtered_modalities).values], row_level=False
                )
            )

            test_name = inspect.stack()[0][3]
            if len(output_ds[0].df.index) == 0:
                raise ValueError(
                    test_name + f": the categories {filtered_modalities} completely drifted as they are not present "
                    f"in the 'actual_dataset'"
                )
    # ---

    return TestResult(
        actual_slices_size=[len(actual_dataset)],
        reference_slices_size=[len(reference_dataset)],
        passed=passed,
        metric=total_psi,
        messages=messages,
        output_ds=output_ds,
    )


def _test_series_drift_psi(
    actual_series,
    reference_series,
    test_data,
    max_categories,
    psi_contribution_percent,
    threshold,
):
    total_psi, output_data = _calculate_drift_psi(actual_series, reference_series, max_categories)
    passed = True if threshold is None else bool(total_psi <= threshold)
    main_drifting_modalities_bool = output_data["Psi"] > psi_contribution_percent * total_psi
    messages = _generate_message_modalities(main_drifting_modalities_bool, output_data, test_data)
    return messages, passed, total_psi, output_data


def _generate_message_modalities(main_drifting_modalities_bool, output_data, test_data):
    modalities_list = output_data[main_drifting_modalities_bool]["Modality"].tolist()
    filtered_modalities = [w for w in modalities_list if not re.match(other_modalities_pattern, str(w))]
    messages: Optional[List[TestMessage]] = None
    if filtered_modalities:
        messages = [
            TestMessage(
                type=TestMessageLevel.ERROR,
                text=f"The {test_data} is drifting for the following modalities: {','.join(str(filtered_modalities))}",
            )
        ]
    return messages


@test(
    name="Label drift (Chi-squared)",
    debug_description=debug_description_prefix + "with <b>the categories that have drifted "
    "the most from the 'actual_dataset'</b>.",
)
def test_drift_prediction_chi_square(
    model: BaseModel,
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    slicing_function: Optional[SlicingFunction] = None,
    max_categories: int = 10,
    threshold: float = 0.05,
    chi_square_contribution_percent: float = 0.2,
):
    """Tests drift of predictions by chi-squared test.

    Test if the chi-square p-value between the reference and actual datasets is
    below the threshold for the classification labels predictions for a given
    slice.

    Parameters
    ----------
    model : BaseModel
        Model used to compute the test
    actual_dataset : Dataset
        Actual dataset used to compute the test
    reference_dataset : Dataset
        Reference dataset used to compute the test
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets
    max_categories : int
        the maximum categories to compute the PSI score.
    threshold : float
        Threshold value of p-value of chi-square. Default is 0.05.
    chi_square_contribution_percent : float
        the ratio between the Chi-Square value of a given category over the
        total chi-square value of the categorical variable. If there is a drift,
        the test provides all the categories that have a PSI contribution over
        than this ratio.
    debug : bool
        If True and the test fails,
        a dataset will be provided containing the actual_dataset rows with the
        categories that have drifted the most (more than
        chi_square_contribution_percent of the total chi squared score).

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_prediction_chi_square"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    prediction_reference = pd.Series(model.predict(reference_dataset).prediction)
    prediction_actual = pd.Series(model.predict(actual_dataset).prediction)

    messages, chi_square, p_value, passed, output_data = _test_series_drift_chi(
        prediction_actual,
        prediction_reference,
        "prediction",
        chi_square_contribution_percent,
        max_categories,
        threshold,
    )

    # --- debug ---
    output_ds = list()
    if not passed:
        check_if_debuggable(actual_dataset, reference_dataset)
        main_drifting_modalities_bool = output_data["Chi_square"] > chi_square_contribution_percent * chi_square
        modalities_list = output_data[main_drifting_modalities_bool]["Modality"].tolist()
        if modalities_list:
            filtered_modalities = [w for w in modalities_list if not re.match(other_modalities_pattern, str(w))]
            output_ds.append(
                actual_dataset.slice(
                    lambda df: df.loc[prediction_actual.isin(filtered_modalities).values], row_level=False
                )
            )
    # ---

    return TestResult(
        actual_slices_size=[len(actual_dataset)],
        reference_slices_size=[len(reference_dataset)],
        passed=passed,
        metric=p_value,
        messages=messages,
        output_ds=output_ds,
    )


def _test_series_drift_chi(
    actual_series,
    reference_series,
    test_data,
    chi_square_contribution_percent,
    max_categories,
    threshold,
):
    chi_square, p_value, output_data = _calculate_chi_square(actual_series, reference_series, max_categories)
    passed = bool(p_value > threshold)
    main_drifting_modalities_bool = output_data["Chi_square"] > chi_square_contribution_percent * chi_square
    messages = _generate_message_modalities(main_drifting_modalities_bool, output_data, test_data)
    return messages, chi_square, p_value, passed, output_data


@test(name="Classification Probability drift (Kolmogorov-Smirnov)", tags=["classification"])
@validate_classification_label
def test_drift_prediction_ks(
    model: BaseModel,
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    slicing_function: Optional[SlicingFunction] = None,
    classification_label: Optional[str] = None,
    threshold: Optional[float] = None,
) -> TestResult:
    """Tests drift of predictions by Kolmogorov-Smirnov test.

    Test if the p-value of the KS test for prediction between the reference and
    actual datasets for a given subpopulation is above the threshold.

    Example: The test is passed when the p-value of the KS test for the
    prediction for females between reference and actual dataset is higher than
    0.05. It means that the null hypothesis cannot be rejected at 5% level and
    we cannot assume drift for this variable.

    Parameters
    ----------
    model : BaseModel
        Model used to compute the test
    actual_dataset : Dataset
        Actual dataset used to compute the test
    reference_dataset : Dataset
        Reference dataset used to compute the test
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets
    classification_label : Optional[str]
        One specific label value from the target column for classification model
    threshold : Optional[float]
        Threshold for p-value of Kolmogorov-Smirnov test

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_prediction_ks"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    # Try to automatically cast `classification_label` to the right type
    if classification_label is not None and isinstance(model.meta.classification_labels[0], numbers.Number):
        try:
            classification_label = int(classification_label)
        except ValueError:
            pass

    prediction_reference = (
        pd.Series(model.predict(reference_dataset).all_predictions[classification_label].values)
        if model.is_classification
        else pd.Series(model.predict(reference_dataset).prediction)
    )
    prediction_actual = (
        pd.Series(model.predict(actual_dataset).all_predictions[classification_label].values)
        if model.is_classification
        else pd.Series(model.predict(actual_dataset).prediction)
    )

    result = _calculate_ks(prediction_reference, prediction_actual)

    passed = True if threshold is None else bool(result.pvalue >= threshold)

    messages = _generate_message_ks(passed, result, threshold, "prediction")

    return TestResult(
        actual_slices_size=[len(actual_dataset)],
        reference_slices_size=[len(reference_dataset)],
        passed=passed,
        metric=result.pvalue,
        messages=messages,
    )


def _generate_message_ks(passed, result, threshold, data_type):
    messages: Optional[List[TestMessage]] = None
    if not passed:
        messages = [
            TestMessage(
                type=TestMessageLevel.ERROR,
                text=f"The {data_type} is drifting (p-value is equal to {np.round(result.pvalue, 9)} "
                f"and is below the test risk level {threshold}) ",
            )
        ]
    return messages


@test(name="Classification Probability drift (Earth mover's distance)", tags=["classification"])
@validate_classification_label
def test_drift_prediction_earth_movers_distance(
    model: BaseModel,
    actual_dataset: Dataset,
    reference_dataset: Dataset,
    slicing_function: Optional[SlicingFunction] = None,
    classification_label: Optional[str] = None,
    threshold: float = 0.2,
) -> TestResult:
    """Tests drift of predictions by earth mover’s distance.

    Test if the Earth Mover’s Distance value between the reference and actual
    datasets is below the threshold for the classification labels predictions
    for classification model and prediction for regression models.

    Examples
    For classification: the test is passed when the  Earth Mover’s Distance
    value of classification labels probabilities for females between reference
    and actual sets is below 0.2

    For regression: the test is passed when the  Earth Mover’s Distance value of
    prediction for females between reference and actual sets is below 0.2

    Parameters
    ----------
    model : BaseModel
        Model to test
    actual_dataset : Dataset
        Actual dataset used to compute the test
    reference_dataset : Dataset
        Reference dataset used to compute the test
    slicing_function : Optional[SlicingFunction]
        Slicing function to be applied on both actual and reference datasets
    classification_label : Optional[str]
        One specific label value from the target column for classification model
    threshold : float
        Threshold for earth mover's distance. Default is 0.2.

    Returns
    -------
    TestResult
        The test result.
    """
    if slicing_function:
        test_name = "test_drift_prediction_earth_movers_distance"
        actual_dataset = actual_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=actual_dataset, dataset_name="actual_dataset", test_name=test_name)
        reference_dataset = reference_dataset.slice(slicing_function)
        check_slice_not_empty(sliced_dataset=reference_dataset, dataset_name="reference_dataset", test_name=test_name)

    # Try to automatically cast `classification_label` to the right type
    if classification_label is not None and isinstance(model.meta.classification_labels[0], numbers.Number):
        try:
            classification_label = int(classification_label)
        except ValueError:
            pass

    prediction_reference = (
        model.predict(reference_dataset).all_predictions[classification_label].values
        if model.is_classification
        else model.predict(reference_dataset).prediction
    )
    prediction_actual = (
        model.predict(actual_dataset).all_predictions[classification_label].values
        if model.is_classification
        else model.predict(actual_dataset).prediction
    )

    metric = _calculate_earth_movers_distance(prediction_reference, prediction_actual)

    passed = True if threshold is None else bool(metric <= threshold)
    messages: Optional[typing.List[TestMessage]] = None

    if not passed:
        messages = [
            TestMessage(
                type=TestMessageLevel.ERROR,
                text=f"The prediction is drifting (metric is equal to {np.round(metric, 9)} "
                f"and is above the test risk level {threshold}) ",
            )
        ]

    return TestResult(
        actual_slices_size=[len(actual_dataset)],
        reference_slices_size=[len(reference_dataset)],
        passed=bool(True if threshold is None else metric <= threshold),
        metric=metric,
        messages=messages,
    )

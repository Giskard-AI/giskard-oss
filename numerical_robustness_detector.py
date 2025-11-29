

from giskard import Dataset, Model, Issue, IssueSeverity, IssueType
from typing import Optional, List, Tuple, Any
import numpy as np
import pandas as pd


class NumericalRobustnessScan:
    """
    Giskard Scan that detects minimal numerical perturbations capable of
    changing a model’s prediction or output significantly.
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        threshold: float = 0.1,
        max_steps: int = 100,
        verbose: bool = False
    ):
        """
        Initialize the scan.

        Args:
            model (Model): Giskard Model object.
            dataset (Dataset): Giskard Dataset object.
            threshold (float): Threshold change for regression predictions.
            max_steps (int): Steps between min/max to test perturbations.
            verbose (bool): If True, prints progress during scan.
        """
        self.model = model
        self.dataset = dataset
        self.threshold = threshold
        self.max_steps = max_steps
        self.verbose = verbose

        self.is_classification = self.model.is_classifier
        self.X = self.dataset.get_features_dataframe()
        self.feature_names = self.dataset.feature_names
        self.X_np = self.X.to_numpy()
        self.feature_bounds = self._get_feature_bounds()

    def _get_feature_bounds(self) -> List[Tuple[float, float]]:
        """Extract min/max bounds for each numerical feature."""
        bounds = []
        for feature in self.dataset.features:
            if feature.feature_type == "numerical":
                min_val = feature.min if feature.min is not None else self.X[feature.name].min()
                max_val = feature.max if feature.max is not None else self.X[feature.name].max()
                bounds.append((min_val, max_val))
            else:
                bounds.append((np.nan, np.nan)) 
        return bounds

    def _predict(self, sample: np.array) -> Any:
        """Run prediction for a single sample."""
        return self.model.predict(sample.reshape(1, -1))[0]

    def _build_issue(
        self,
        feature_index: int,
        perturb_value: float,
        original_pred: Any,
        new_pred: Any,
        sample_idx: int
    ) -> Issue:
        """Create a Giskard Issue object."""
        feature_name = self.feature_names[feature_index]
        description = (
            f"Perturbing '{feature_name}' by {abs(perturb_value):.4f} in sample {sample_idx} "
            f"changed prediction from {original_pred} to {new_pred}."
        )
        return Issue(
            type=IssueType.ROBUSTNESS,
            severity=IssueSeverity.MEDIUM,
            description=description,
            feature=feature_name,
            sample_index=sample_idx,
        )

    def _scan_feature(self, feature_index: int) -> Optional[Issue]:
        """Scan a single feature for robustness issues."""
        min_val, max_val = self.feature_bounds[feature_index]
        if np.isnan(min_val) or np.isnan(max_val):
            return None  

        step_size = (max_val - min_val) / self.max_steps

        for sample_idx in range(len(self.X_np)):
            original_sample = self.X_np[sample_idx].copy()
            original_pred = self._predict(original_sample)

            for step in range(1, self.max_steps + 1):
                for direction in [+1, -1]:
                    perturb = direction * step * step_size
                    new_val = original_sample[feature_index] + perturb

                    if not (min_val <= new_val <= max_val):
                        continue

                    perturbed_sample = original_sample.copy()
                    perturbed_sample[feature_index] = new_val
                    new_pred = self._predict(perturbed_sample)

                    if self.is_classification and new_pred != original_pred:
                        return self._build_issue(feature_index, perturb, original_pred, new_pred, sample_idx)
                    elif not self.is_classification and abs(new_pred - original_pred) > self.threshold:
                        return self._build_issue(feature_index, perturb, original_pred, new_pred, sample_idx)

        return None

    def run_scan(self) -> List[Issue]:
        """Run the full robustness scan across all numerical features."""
        issues = []
        for feature_index, feature_name in enumerate(self.feature_names):
            if self.verbose:
                print(f"Scanning feature: {feature_name} ({feature_index})")
            issue = self._scan_feature(feature_index)
            if issue:
                issues.append(issue)
                if self.verbose:
                    print(f"✔ Issue found on '{feature_name}'")
            elif self.verbose:
                print(f"✘ No issue on '{feature_name}'")
        return issues


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Numerical Robustness Scan.")
    parser.add_argument("--model_path", required=True, help="Path to Giskard model file (YAML)")
    parser.add_argument("--dataset_path", required=True, help="Path to Giskard dataset file (YAML)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for regression change")
    parser.add_argument("--max_steps", type=int, default=100, help="Steps to scan perturbations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    model = Model.load(args.model_path)
    dataset = Dataset.load(args.dataset_path)

    scan = NumericalRobustnessScan(
        model=model,
        dataset=dataset,
        threshold=args.threshold,
        max_steps=args.max_steps,
        verbose=args.verbose
    )

    issues = scan.run_scan()

    print(f"\nScan complete. Found {len(issues)} issue(s).")
    for issue in issues:
        print(f"- {issue.description}")

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Optional, List


class CategoricalRobustnessDetector:
    def __init__(
        self,
        model,
        task: str = "classification",
        regression_threshold: float = 0.1,
        prob_threshold: Optional[float] = None,
        n_jobs: int = -1,
    ):
        self.model = model
        self.task = task
        self.regression_threshold = regression_threshold
        self.prob_threshold = prob_threshold
        self.n_jobs = n_jobs
        self.categorical_columns = []
        self.issues = []

        self.has_predict_prob = hasattr(model, "predict_prob")

    def fit(self, X: pd.DataFrame, categorical_columns: Optional[List[str]] = None):
        if categorical_columns is not None:
            self.categorical_columns = categorical_columns
        else:
            self.categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

    def scan(self, X: pd.DataFrame) -> List[dict]:
        def process_row(idx, row):
            row_issues = []
            original_input = row.to_frame().T

            try:
                if self.task == "classification":
                    original_pred = self.model.predict(original_input)[0]
                    if self.has_predict_prob:
                        original_prob = self.model.predict_prob(original_input)[0]
                        original_conf = np.max(original_prob)
                    else:
                        original_conf = None
                else:
                    original_pred = self.model.predict(original_input)[0]
            except Exception as e:
                return [{"row": idx, "error": str(e), "change_type": "prediction_failed"}]

            for col in self.categorical_columns:
                original_value = row[col]
                unique_values = X[col].dropna().unique()

                for val in unique_values:
                    if val == original_value:
                        continue

                    perturbed_input = original_input.copy()
                    perturbed_input[col] = val

                    try:
                        if self.task == "classification":
                            new_pred = self.model.predict(perturbed_input)[0]
                            if self.has_predict_prob:
                                new_prob = self.model.predict_prob(perturbed_input)[0]
                                new_conf = np.max(new_prob)
                            else:
                                new_conf = None

                            if new_pred != original_pred:
                                row_issues.append({
                                    "row": idx,
                                    "feature": col,
                                    "from": original_value,
                                    "to": val,
                                    "original_pred": original_pred,
                                    "new_pred": new_pred,
                                    "original_conf": original_conf,
                                    "new_conf": new_conf,
                                    "confidence_drop": (original_conf - new_conf) if (original_conf is not None and new_conf is not None) else None,
                                    "change_type": "label_changed"
                                })

                            elif self.prob_threshold is not None and original_conf is not None and new_conf is not None and abs(original_conf - new_conf) >= self.prob_threshold:
                                row_issues.append({
                                    "row": idx,
                                    "feature": col,
                                    "from": original_value,
                                    "to": val,
                                    "original_pred": original_pred,
                                    "new_pred": new_pred,
                                    "original_conf": original_conf,
                                    "new_conf": new_conf,
                                    "confidence_drop": original_conf - new_conf,
                                    "change_type": "confidence_shift"
                                })

                        else:  # regression
                            new_pred = self.model.predict(perturbed_input)[0]
                            delta = abs(new_pred - original_pred)
                            if delta >= self.regression_threshold:
                                row_issues.append({
                                    "row": idx,
                                    "feature": col,
                                    "from": original_value,
                                    "to": val,
                                    "original_pred": original_pred,
                                    "new_pred": new_pred,
                                    "delta": delta,
                                    "change_type": "prediction_shift"
                                })
                    except Exception as e:
                        row_issues.append({
                            "row": idx,
                            "feature": col,
                            "from": original_value,
                            "to": val,
                            "error": str(e),
                            "change_type": "perturbation_failed"
                        })

            return row_issues

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_row)(idx, row) for idx, row in X.iterrows()
        )

        self.issues = [item for sublist in results for item in sublist]
        return self.issues

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues)

    def summary(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame()
        df = pd.DataFrame(self.issues)
        return df.groupby(["feature", "change_type"]).size().reset_index(name="count")

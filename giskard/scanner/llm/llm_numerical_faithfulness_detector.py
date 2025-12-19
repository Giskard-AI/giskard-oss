from typing import Sequence
import pandas as pd

from giskard.scanner import logger
from giskard.datasets.base import Dataset
from giskard.models.base.model import BaseModel
from giskard.rag.metrics.numerical_faithfulness import NumericalFaithfulnessMetric
from giskard.rag.base import AgentAnswer
from ..decorators import detector
from ..issues import Hallucination, Issue, IssueLevel
from ..registry import Detector

@detector(
    "llm_numerical_faithfulness",
    tags=["hallucination", "financial_risk", "numerics"],
)
class LLMNumericalFaithfulnessDetector(Detector):
    """
    Detects numerical hallucinations in RAG models by verifying if numbers in the answer are grounded in the retrieved context.
    
    This detector extracts numbers from both the answer and the context using a hybrid approach (Regex + extraction LLM)
    and checks for discrepancies.

    Parameters
    ----------
    num_samples : int, optional
        Number of samples to scan from the dataset, by default 100.
    tolerance : float, optional
        Relative tolerance for numerical comparison, by default 0.01 (1%).
    """
    def __init__(self, num_samples=100, tolerance=0.01):
        self.num_samples = num_samples
        self.tolerance = tolerance
        self.metric = NumericalFaithfulnessMetric(tolerance=tolerance, use_llm=True)

    def run(self, model: BaseModel, dataset: Dataset, features=None) -> Sequence[Issue]:
        logger.info(f"{self.__class__.__name__}: Running numerical faithfulness detection")
        
        # 1. Run Prediction
        # We might want to limit samples if dataset is huge, but let's assume it's manageable or sliced
        # For efficiency in a real scanner, we might sample.
        if len(dataset) > self.num_samples:
             logger.debug(f"{self.__class__.__name__}: Limiting to {self.num_samples} samples")
             dataset_to_scan = dataset.slice(lambda df: df.sample(self.num_samples, random_state=42))
        else:
             dataset_to_scan = dataset

        prediction_result = model.predict(dataset_to_scan)
        
        failures = []
        
        # 2. Iterate and Evaluate
        for idx, (prediction, example) in enumerate(zip(prediction_result.prediction, dataset_to_scan.df.to_dict('records'))):
            
            # We need to construct AgentAnswer with context.
            # If the model is a Giskard Model, it might not return documents in .prediction list directly unless it's a specific RAG model.
            # However, usually RAG answers in Giskard might be just strings.
            # We rely on specific columns in dataset for "reference_context" OR check if model returns specific metadata?
            # BUT: The prompt implies this is for RAG.
            # In Giskard RAG, usually we expect 'context' or 'documents' to be available.
            # If the model prediction doesn't return context, we can't check faithfulness against *retrieved* context,
            # but we can check failure against *reference* context if available in dataset.
            
            documents = []
            # Try to get Documents from prediction metadata if available?
            # Currently Giskard model.predict returns valid prediction object.
            
            # For this simplified implementation, we assume the model output might implicitly contain context (if it was a wrapper)
            # OR we rely on 'reference_context' in the dataset (Ground Truth checking).
            # OR we assume the user provided 'context' in the features.
            
            # Let's try to find context in the input example or prediction.
            if "reference_context" in example:
                context_source = example["reference_context"]
                documents = [context_source]
            elif "context" in example:
                context_source = example["context"]
                documents = [context_source]
            
            # Mocking the AgentAnswer structure expected by the metric
            agent_answer = AgentAnswer(
                message=prediction,
                documents=documents
            )
            
            # Construct question_sample for metric (expects dict with reference_context if needed)
            question_sample = example.copy()
            if documents:
                question_sample["reference_context"] = documents[0]

            score_result = self.metric(question_sample, agent_answer)
            score = score_result[self.metric.name]
            
            if score < 1.0:
                 reason = score_result.get(f"{self.metric.name}_reason", "Unknown numerical discrepancy")
                 failures.append({
                     "sample": example,
                     "prediction": prediction,
                     "reason": reason,
                     "score": score
                 })

        if not failures:
            return []
        
        # 3. Create Issue
        # Format examples for the report
        examples_df = pd.DataFrame([
            {
                "Question": f["sample"].get("question", str(f["sample"])),
                "Context": f["sample"].get("reference_context", f["sample"].get("context", "N/A")),
                "Agent Answer": f["prediction"],
                "Reason": f["reason"]
            }
            for f in failures
        ])

        # Create a new dataset for the failing samples
        failing_df = pd.DataFrame([f["sample"] for f in failures])
        # Ensure we preserve original indexing if possible, or just reset. 
        # For scanner reporting, just having the rows is usually sufficient.
        failing_dataset = Dataset(
            df=failing_df,
            name="Numerical Faithfulness Failures",
            target=dataset.target,
            column_types=dataset.column_types,
            validation=False # Skip validation for this transient dataset
        )

        return [
            Issue(
                model,
                failing_dataset,
                group=Hallucination,
                level=IssueLevel.MAJOR,
                description="The model generated numerical values that do not match the context information. This indicates a potential 'Financial Risk' or hallucination.",
                meta={
                    "metric": "Numerical Faithfulness Failure Rate",
                    "metric_value": len(failures) / len(dataset_to_scan),
                    "domain": "Financial / Numerical",
                    "tag": "Financial Risk"
                },
                examples=examples_df,
                detector_name=self.__class__.__name__,
            )
        ]

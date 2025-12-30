from typing import Dict, List, Optional, Sequence, Tuple

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ...core.test_result import TestResultStatus
from ...datasets.base import Dataset
from ...models.base.model import BaseModel
from ..client import LLMClient, get_default_client
from ..client.base import ChatMessage
from ..errors import LLMGenerationError
from .utils import format_conversation

logger = logging.getLogger("giskard.llm")


@dataclass
class EvaluationResultExample:
    # The status of the example
    status: TestResultStatus
    sample: Optional[Sequence[Dict]]
    # The reason why the example have given status
    reason: Optional[str] = None

    def to_example(self):
        if self.status == TestResultStatus.ERROR:
            return {"error": self.reason, "sample": self.sample}
        else:
            return {"reason": self.reason, "sample": self.sample}

    def to_dict(self):
        return {"reason": self.reason, "sample": self.sample, "status": self.status}


@dataclass
class EvaluationResult:
    results: List[EvaluationResultExample] = field(default_factory=list)

    @property
    def failure_examples(self):
        return [failed.to_example() for failed in self.results if failed.status == TestResultStatus.FAILED]

    @property
    def success_examples(self):
        return [passed.to_example() for passed in self.results if passed.status == TestResultStatus.PASSED]

    @property
    def errors(self):
        return [error.to_example() for error in self.results if error.status == TestResultStatus.ERROR]

    @property
    def passed(self):
        return len(self.failure_examples) == 0 and len(self.success_examples) > 0

    @property
    def failed(self):
        return not self.passed

    @property
    def has_errors(self):
        return len(self.errors) > 0

    @property
    def passed_ratio(self):
        return len(self.success_examples) / (len(self.success_examples) + len(self.failure_examples))

    def add_error(self, error: str, sample: Dict):
        self.results.append(EvaluationResultExample(reason=error, sample=sample, status=TestResultStatus.ERROR))

    def add_sample(self, eval_passed: bool, reason: Optional[str] = None, sample: Optional[Dict] = None):
        status = TestResultStatus.PASSED if eval_passed else TestResultStatus.FAILED
        self.results.append(EvaluationResultExample(reason=reason, sample=sample, status=status))


class BaseEvaluator(ABC):
    """Base interface for evaluators."""

    @abstractmethod
    def evaluate(self, model: BaseModel, dataset: Dataset):
        ...


class _BaseLLMEvaluator(BaseEvaluator):
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_temperature: float = 0.1,
        llm_seed: int = 42,
        llm_output_format="json_object",
        num_retries: int = 1,
    ):
        """Initialize the LLM evaluator.

        Parameters
        ----------
        llm_client : Optional[LLMClient]
            The LLM client to use for evaluation. If None, uses the default client.
        llm_temperature : float
            Temperature for LLM generation. Default is 0.1.
        llm_seed : int
            Seed for reproducibility. Default is 42.
        llm_output_format : str
            Output format for LLM. Default is "json_object".
        num_retries : int
            Number of times to retry each input with the model. This is useful for
            testing jailbreak scenarios where multiple attempts may be needed to
            trigger a vulnerability. If any retry fails the evaluation, the sample
            is marked as failed. Default is 1 (no retries).
        """
        self.llm_client = llm_client if llm_client is not None else get_default_client()
        self.llm_temperature = llm_temperature
        self.llm_seed = llm_seed
        self.llm_output_format = llm_output_format
        self.num_retries = max(1, num_retries)

    @abstractmethod
    def _format_messages(
        self, model: BaseModel, conversation: Sequence[Dict], meta: Optional[Dict] = None
    ) -> Sequence[ChatMessage]:
        ...

    def evaluate(self, model: BaseModel, dataset: Dataset):
        result = EvaluationResult()

        for row_id, row in dataset.df.iterrows():
            input_vars = {k: v for k, v in row.items() if k in model.feature_names}
            if len(input_vars) == 1:
                input_vars = list(input_vars.values())[0]
            input_meta = {k: v for k, v in row.items() if k not in model.feature_names}
            input_meta["__sample_id"] = row_id

            # Track results across retries
            sample_failed = False
            failure_reason = None
            last_sample = None
            error_occurred = False
            error_message = None

            for retry_idx in range(self.num_retries):
                # Get model output for this retry
                single_row_dataset = Dataset(
                    df=dataset.df.iloc[[dataset.df.index.get_loc(row_id)]],
                    name=dataset.name,
                    target=dataset.target,
                    column_types=dataset.column_types,
                    validation=False,
                )
                model_output = model.predict(single_row_dataset).prediction[0]

                conversation = [{"role": "user", "content": input_vars}, {"role": "agent", "content": model_output}]
                sample = {
                    "conversation": conversation,
                    "meta": {**input_meta, "__retry_idx": retry_idx},
                }
                last_sample = sample

                logger.debug(f"{self.__class__.__name__}: evaluating sample {sample} (retry {retry_idx + 1}/{self.num_retries})")

                try:
                    eval_passed, reason = self._evaluate_sample(model, sample)
                except LLMGenerationError as err:
                    logger.debug(f"{self.__class__.__name__} evaluation error on retry {retry_idx + 1}: {err}")
                    error_occurred = True
                    error_message = str(err)
                    continue

                logger.debug(
                    f"{self.__class__.__name__} evaluation result (retry {retry_idx + 1}): "
                    f"eval_passed={eval_passed}, reason={reason}"
                )

                # If any retry fails, mark the sample as failed
                if not eval_passed:
                    sample_failed = True
                    failure_reason = reason
                    if self.num_retries > 1:
                        failure_reason = f"[Failed on retry {retry_idx + 1}/{self.num_retries}] {reason}"
                    break

            # Add final result for this sample
            if sample_failed:
                result.add_sample(False, failure_reason, last_sample)
            elif error_occurred and last_sample is not None:
                result.add_error(error_message, last_sample)
            elif last_sample is not None:
                result.add_sample(True, None, last_sample)

        return result

    def _evaluate_sample(self, model: BaseModel, sample: Dict) -> Tuple[bool, Optional[str]]:
        messages = self._format_messages(model, sample["conversation"], meta=sample.get("meta"))
        raw_eval = self.llm_client.complete(
            messages,
            temperature=self.llm_temperature,
            caller_id=self.__class__.__name__,
            seed=self.llm_seed,
            format=self.llm_output_format,
        )
        eval_passed, reason = self._parse_evaluation_output(raw_eval)

        return eval_passed, reason

    def _parse_evaluation_output(self, raw_eval: ChatMessage) -> Tuple[bool, Optional[str]]:
        try:
            eval_result = json.loads(raw_eval.content)
            return eval_result["eval_passed"], eval_result.get("reason")
        except (AttributeError, KeyError, json.JSONDecodeError) as err:
            raise LLMGenerationError("Could not parse evaluator output") from err


class LLMBasedEvaluator(_BaseLLMEvaluator):
    def __init__(
        self,
        prompt: str,
        prefix_messages: Optional[Sequence[ChatMessage]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.prefix_messages = prefix_messages or []

    def _format_messages(
        self, model: BaseModel, conversation: Sequence[Dict], meta: Optional[Dict] = None
    ) -> Sequence[ChatMessage]:
        prompt = self.prompt.format(model=model, conversation=format_conversation(conversation))
        return self.prefix_messages + [ChatMessage(role="user", content=prompt)]

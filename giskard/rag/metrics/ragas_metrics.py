from typing import Optional, Sequence

import logging

import ragas

from ...llm.client import ChatMessage, LLMClient, get_default_client
from ...llm.embeddings import BaseEmbedding, get_default_embedding
from ..base import AgentAnswer
from .base import Metric
from ragas.dataset_schema import SingleTurnSample
from langchain_core.outputs import ChatGeneration
import asyncio
logger = logging.getLogger(__name__)

try:
    from langchain_core.outputs import LLMResult
    from langchain_core.outputs.generation import Generation
    from langchain_core.prompt_values import PromptValue
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import BaseRagasLLM
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.metrics.base import Metric as BaseRagasMetric
    from ragas.run_config import RunConfig

except ImportError as err:
    raise ImportError(
        f"Package {err.name} is missing, it is required for the computation of RAGAS metrics. You can install it with `pip install {err.name}`."
    ) from err


try:
    # RAGAS has async functions, we try to apply nest_asyncio to avoid issues in notebooks
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass


class RagasLLMWrapper(BaseRagasLLM):
    def __init__(self, llm_client, context_window_length: int):
        self.llm_client = llm_client
        self.context_window_length = context_window_length

    def trim_prompt(self, prompt: str) -> str:
        return prompt[: int(self.context_window_length * 3.5)]

    def generate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 1e-8, stop=None, callbacks=[]):
        out = self.llm_client.complete([ChatMessage(role="user", content=self.trim_prompt(prompt.to_string()))])
        return LLMResult(generations=[[Generation(text=out.content)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[Sequence[str]] = None,
        callbacks=[],
    ):
        return self.generate_text(prompt, n, temperature, stop, callbacks)
    
    def is_finished(self, response: LLMResult) -> bool:
        """
        Return True if all generations ended normally.
        Checks OpenAI/Vertex-style finish reasons ('stop', 'end_turn', 'MAX_TOKENS')
        and falls back to True when metadata is missing.
        """
        flags: list[bool] = []
        for gens in response.generations:
            if not gens:
                flags.append(True)
                continue
            resp = gens[0]
            fin = None

            if getattr(resp, "generation_info", None):
                fin = resp.generation_info.get("finish_reason")

            if fin is None and isinstance(resp, ChatGeneration):
                meta = resp.message.response_metadata or {}
                fin = meta.get("finish_reason") or meta.get("stop_reason")

            flags.append(fin in ["stop", "STOP", "end_turn", "MAX_TOKENS", None])

        return all(flags)


class RagasEmbeddingsWrapper(BaseRagasEmbeddings):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_query(self, text: str) -> Sequence[float]:
        return self.embedding_model.embed([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self.embedding_model.embed(texts)

    async def aembed_query(self, text: str) -> Sequence[float]:
        return await self.embedding_model.aembed([text])[0]

    async def aembed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await self.embedding_model.aembed(texts)


class RagasMetric(Metric):
    def __init__(
        self,
        name: str,
        metric: BaseRagasMetric,
        context_window_length: int = 8192,
        llm_client: LLMClient = None,
        embedding_model: BaseEmbedding = None,
        requires_context=False,
    ) -> None:
        self.name = name
        self.metric = metric
        self.context_window_length = context_window_length
        self._llm_client = llm_client
        self._embedding_model = embedding_model
        self.requires_context = requires_context
        self.ragas_llm = None
        self.ragas_embeddings = None

    def __call__(self, question_sample: dict, answer: AgentAnswer) -> dict:
        llm_client = self._llm_client or get_default_client()
        embedding_model = self._embedding_model or get_default_embedding()
        if self.ragas_llm is None:
            self.ragas_llm = RagasLLMWrapper(llm_client, self.context_window_length)
        if self.ragas_embeddings is None:
            self.ragas_embeddings = RagasEmbeddingsWrapper(embedding_model)

        run_config = RunConfig()

        if hasattr(self.metric, "llm"):
            self.metric.llm = self.ragas_llm
        if hasattr(self.metric, "embeddings"):
            self.metric.embeddings = self.ragas_embeddings

        self.metric.init(run_config)
        if self.requires_context and answer.documents is None:
            logger.warning(
                f"No retrieved documents are passed to the evaluation function, computation of {self.name} cannot be done without it."
                "Make sure that the 'answer_fn' return documents alongside the answer, wrapped by the 'AgentAnswer' class."
            )
            return {self.name: 0}

        ragas_sample = self.prepare_ragas_sample(question_sample, answer)
                
        #return {self.name: self.metric.score(ragas_sample)}
        
        sample = SingleTurnSample(
            user_input=ragas_sample.get("user_input") or ragas_sample.get("question"),
            response=ragas_sample.get("response") or ragas_sample.get("answer"),
            retrieved_contexts=ragas_sample.get("retrieved_contexts") or ragas_sample.get("contexts"),
            reference=ragas_sample.get("reference") or ragas_sample.get("ground_truth"),
        )

        async def _compute_score():
            if hasattr(self.metric, "single_turn_ascore"):
                return await self.metric.single_turn_ascore(sample)
            elif hasattr(self.metric, "multi_turn_ascore"):
                return await self.metric.multi_turn_ascore(sample)
            else:
                raise AttributeError(
                    f"{self.metric} has neither single_turn_ascore nor multi_turn_ascore "
                    "— check ragas version or metric type."
                )
                
        loop = asyncio.get_event_loop()
        val = loop.run_until_complete(_compute_score())
        return {self.name: val}
    
    @staticmethod
    def prepare_ragas_sample(question_sample: dict, answer: AgentAnswer) -> dict:
        if ragas.__version__.startswith("0.1"):
            logger.warning(
                f"{DeprecationWarning.__name__}: You are using an older version (v0.1) of `ragas` package. "
                "Support for v0.1 is deprecated and may be removed in future versions. "
                "Please consider updating `ragas` to a later version."
            )
            return {
                "question": question_sample["question"],
                "answer": answer.message,
                "contexts": answer.documents,
                "ground_truth": question_sample["reference_answer"],
            }

        return {
            "user_input": question_sample["question"],
            "response": answer.message,
            "retrieved_contexts": answer.documents,
            "reference": question_sample["reference_answer"],
        }


ragas_context_precision = RagasMetric(name="RAGAS Context Precision", metric=context_precision, requires_context=True)
ragas_faithfulness = RagasMetric(name="RAGAS Faithfulness", metric=faithfulness, requires_context=True)
ragas_answer_relevancy = RagasMetric(name="RAGAS Answer Relevancy", metric=answer_relevancy, requires_context=True)
ragas_context_recall = RagasMetric(name="RAGAS Context Recall", metric=context_recall, requires_context=True)

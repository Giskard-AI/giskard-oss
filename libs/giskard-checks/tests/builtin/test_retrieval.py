from typing import Any

import pytest
from giskard.checks import (
    MRR,
    AveragePrecision,
    Check,
    CheckStatus,
    HitRateAtK,
    Interaction,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
    Trace,
)
from giskard.checks.core.extraction import NoMatch

RELEVANT_IDS_KEY = "trace.interactions[-1].inputs.relevant_ids"
RETRIEVED_IDS_KEY = "trace.interactions[-1].outputs.retrieved_ids"


async def _trace(relevant_ids: list[str], retrieved_ids: list[str]):
    return await Trace.from_interactions(
        Interaction(
            inputs={"relevant_ids": relevant_ids},
            outputs={"retrieved_ids": retrieved_ids},
        )
    )


def _check_kwargs(threshold: float = 1.0) -> dict[str, Any]:
    return {
        "relevant_ids_key": RELEVANT_IDS_KEY,
        "retrieved_ids_key": RETRIEVED_IDS_KEY,
        "threshold": threshold,
    }


async def test_recall_at_k_perfect_retrieval_passes():
    trace = await _trace(["doc-1", "doc-2"], ["doc-1", "doc-2", "doc-3"])
    check = RecallAtK(k=2, **_check_kwargs())

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == 1.0


async def test_recall_at_k_partial_overlap_fails_threshold():
    trace = await _trace(["doc-1", "doc-2", "doc-3"], ["doc-1", "doc-4", "doc-5"])
    check = RecallAtK(k=3, **_check_kwargs(threshold=0.5))

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.details["score"] == pytest.approx(1 / 3)


async def test_precision_at_k_partial_overlap_passes_threshold():
    trace = await _trace(["doc-1", "doc-2", "doc-3"], ["doc-1", "doc-4", "doc-2"])
    check = PrecisionAtK(k=3, **_check_kwargs(threshold=0.5))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == pytest.approx(2 / 3)


async def test_precision_at_k_uses_k_as_denominator_for_short_result_lists():
    trace = await _trace(["doc-1", "doc-2"], ["doc-1"])
    check = PrecisionAtK(k=3, **_check_kwargs(threshold=0.3))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == pytest.approx(1 / 3)


async def test_duplicate_retrieved_ids_do_not_inflate_recall():
    trace = await _trace(["doc-1", "doc-2"], ["doc-1", "doc-1", "doc-3"])
    check = RecallAtK(k=3, **_check_kwargs(threshold=0.5))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == pytest.approx(0.5)


async def test_hit_rate_at_k_empty_retrieved_ids_fails():
    trace = await _trace(["doc-1"], [])
    check = HitRateAtK(k=3, **_check_kwargs(threshold=1.0))

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert result.details["score"] == 0.0


async def test_hit_rate_at_k_empty_relevant_ids_passes_zero_threshold():
    trace = await _trace([], ["doc-1"])
    check = HitRateAtK(k=3, **_check_kwargs(threshold=0.0))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == 0.0


async def test_none_retrieved_ids_are_treated_as_empty():
    trace = await Trace.from_interactions(
        Interaction(inputs={"relevant_ids": ["doc-1"]}, outputs={"retrieved_ids": None})
    )
    check = HitRateAtK(k=3, **_check_kwargs(threshold=0.0))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == 0.0


async def test_mrr_is_ranking_sensitive():
    trace = await _trace(["doc-3"], ["doc-1", "doc-2", "doc-3"])
    check = MRR(**_check_kwargs(threshold=0.3))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == pytest.approx(1 / 3)


async def test_ndcg_at_k_is_ranking_sensitive():
    trace = await _trace(["doc-2", "doc-3"], ["doc-1", "doc-2", "doc-3"])
    check = NDCGAtK(k=3, **_check_kwargs(threshold=0.69))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == pytest.approx(0.693426, rel=1e-5)


async def test_average_precision_interleaved_relevant_docs():
    trace = await _trace(["doc-2", "doc-4"], ["doc-1", "doc-2", "doc-3", "doc-4"])
    check = AveragePrecision(**_check_kwargs(threshold=0.5))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == pytest.approx(0.5)


async def test_average_precision_normalizes_by_relevant_count():
    trace = await _trace(
        ["doc-2", "doc-4", "doc-5"], ["doc-1", "doc-2", "doc-3", "doc-4"]
    )
    check = AveragePrecision(**_check_kwargs(threshold=0.3))

    result = await check.run(trace)

    assert result.status == CheckStatus.PASS
    assert result.details["score"] == pytest.approx(1 / 3)


async def test_retrieval_check_reports_missing_relevant_ids_key():
    trace = await Trace.from_interactions(
        Interaction(inputs={}, outputs={"retrieved_ids": ["doc-1"]})
    )
    check = RecallAtK(k=1, **_check_kwargs())

    result = await check.run(trace)

    assert result.status == CheckStatus.FAIL
    assert isinstance(result.details["relevant_ids"], NoMatch)
    assert result.message is not None
    assert "No value found for relevant IDs key" in result.message


def test_retrieval_checks_are_registered():
    check = Check.model_validate(
        {
            "kind": "recall_at_k",
            "k": 3,
            "relevant_ids_key": RELEVANT_IDS_KEY,
            "retrieved_ids_key": RETRIEVED_IDS_KEY,
        }
    )

    assert isinstance(check, RecallAtK)

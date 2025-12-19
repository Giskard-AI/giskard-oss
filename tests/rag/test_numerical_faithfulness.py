import pytest
from unittest.mock import MagicMock
from giskard.rag.metrics.numerical_faithfulness import NumericalFaithfulnessMetric
from giskard.rag.base import AgentAnswer
from giskard.llm.client import ChatMessage

# Mock LLM Client
class MockLLMClient:
    def complete(self, messages, temperature=0, format=None, **kwargs):
        content = '{"numbers": []}'
        user_msg = messages[-1].content
        
        if "five million" in user_msg:
             content = '{"numbers": [5000000.0]}'
        elif "50%" in user_msg:
             content = '{"numbers": [50.0]}' 
        elif "0.5" in user_msg:
             content = '{"numbers": [0.5]}'
             
        return ChatMessage(role="assistant", content=content)

@pytest.fixture
def mock_llm_client():
    return MockLLMClient()

def test_numerical_faithfulness_regex_simple():
    metric = NumericalFaithfulnessMetric(use_llm=False)
    
    # Exact match
    context = "Revenue is 100."
    answer = "Revenue: 100."
    res = metric({"reference_context": context}, AgentAnswer(message=answer, documents=[context]))
    assert res["numerical_faithfulness"] == 1.0

    # Mismatch
    context = "Revenue is 100."
    answer = "Revenue: 90."
    res = metric({"reference_context": context}, AgentAnswer(message=answer, documents=[context]))
    assert res["numerical_faithfulness"] == 0.0

def test_numerical_faithfulness_hybrid_text_numbers(mock_llm_client):
    metric = NumericalFaithfulnessMetric(use_llm=True, llm_client=mock_llm_client)
    
    # "five million" vs "5,000,000"
    # Regex finds 5,000,000 in answer (5000000.0). 
    # Regex finds nothing in context ("five million").
    # LLM finds 5000000.0 in context.
    # -> Match
    context = "Revenue is five million."
    answer = "Revenue: 5,000,000."
    
    res = metric({"reference_context": context}, AgentAnswer(message=answer, documents=[context]))
    assert res["numerical_faithfulness"] == 1.0

def test_numerical_faithfulness_percent_decimal_equivalence(mock_llm_client):
    metric = NumericalFaithfulnessMetric(use_llm=True, llm_client=mock_llm_client)
    
    # 50% vs 0.5
    # Context: 50% -> Mock LLM extracts 50.0
    # Answer: 0.5 -> Mock LLM extracts 0.5
    # Metric logic should reconcile 50.0 and 0.5 using factor 100 check
    context = "Growth: 50%."
    answer = "Growth: 0.5."
    
    res = metric({"reference_context": context}, AgentAnswer(message=answer, documents=[context]))
    assert res["numerical_faithfulness"] == 1.0

def test_numerical_faithfulness_fail_case():
    metric = NumericalFaithfulnessMetric(use_llm=False)
    
    context = "Revenue is 100."
    answer = "Revenue is 100 and Profit is 20." # 20 Not in context
    
    res = metric({"reference_context": context}, AgentAnswer(message=answer, documents=[context]))
    assert res["numerical_faithfulness"] == 0.5 # 1 out of 2 found (100 found, 20 missed) -> 1 - 1/2 = 0.5

def test_numerical_faithfulness_no_numbers():
    metric = NumericalFaithfulnessMetric(use_llm=False)
    context = "No numbers here."
    answer = "Just text."
    res = metric({"reference_context": context}, AgentAnswer(message=answer, documents=[context]))
    assert res["numerical_faithfulness"] == 1.0

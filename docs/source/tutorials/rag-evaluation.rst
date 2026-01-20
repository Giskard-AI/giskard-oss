==============
RAG Evaluation
==============

This tutorial shows how to build a comprehensive test suite for a Retrieval-Augmented Generation (RAG) system.

Overview
--------

We'll test a RAG system that answers questions by:

1. Retrieving relevant context from a knowledge base
2. Generating an answer grounded in that context
3. Handling out-of-scope questions appropriately

Our test suite will validate:

* **Retrieval quality**: Are the retrieved documents relevant?
* **Groundedness**: Is the answer based on the retrieved context?
* **Answer quality**: Is the answer accurate and complete?
* **Handling edge cases**: Out-of-scope questions, empty queries, etc.


Building the RAG System
------------------------

First, let's create a simple RAG system to test:

.. code-block:: python

   from typing import List
   from pydantic import BaseModel

   class Document(BaseModel):
       content: str
       metadata: dict

   class RAGResponse(BaseModel):
       question: str
       answer: str
       retrieved_docs: List[Document]
       confidence: float

   class SimpleRAG:
       def __init__(self, documents: List[Document]):
           self.documents = documents

       def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
           """Retrieve relevant documents (simplified similarity)."""
           # In practice, use embeddings and vector search
           query_lower = query.lower()
           scored_docs = []

           for doc in self.documents:
               score = sum(
                   word in doc.content.lower()
                   for word in query_lower.split()
               )
               if score > 0:
                   scored_docs.append((score, doc))

           scored_docs.sort(reverse=True, key=lambda x: x[0])
           return [doc for _, doc in scored_docs[:top_k]]

       def generate_answer(
           self,
           question: str,
           context_docs: List[Document]
       ) -> str:
           """Generate answer from context (in practice, use LLM)."""
           if not context_docs:
               return "I don't have enough information to answer that question."

           # Simplified: just return relevant content
           # In practice, use an LLM to synthesize an answer
           context_text = "\n".join(doc.content for doc in context_docs)
           return f"Based on the available information: {context_text[:200]}..."

       def answer(self, question: str) -> RAGResponse:
           """Main RAG pipeline."""
           if not question.strip():
               return RAGResponse(
                   question=question,
                   answer="Please provide a valid question.",
                   retrieved_docs=[],
                   confidence=0.0
               )

           # Retrieve
           docs = self.retrieve(question)

           # Generate
           answer = self.generate_answer(question, docs)

           # Estimate confidence based on retrieval quality
           confidence = min(1.0, len(docs) / 3.0)

           return RAGResponse(
               question=question,
               answer=answer,
               retrieved_docs=docs,
               confidence=confidence
           )


Setting Up Test Data
---------------------

Create a knowledge base for testing:

.. code-block:: python

   knowledge_base = [
       Document(
           content="Paris is the capital and largest city of France. It is known for the Eiffel Tower.",
           metadata={"source": "geography", "topic": "France"}
       ),
       Document(
           content="The Eiffel Tower is a wrought-iron lattice tower in Paris. It was completed in 1889.",
           metadata={"source": "landmarks", "topic": "Eiffel Tower"}
       ),
       Document(
           content="France is a country in Western Europe. It has a population of about 67 million.",
           metadata={"source": "geography", "topic": "France"}
       ),
       Document(
           content="Python is a high-level programming language. It was created by Guido van Rossum.",
           metadata={"source": "technology", "topic": "Python"}
       ),
       Document(
           content="Machine learning is a subset of artificial intelligence focused on data-driven learning.",
           metadata={"source": "technology", "topic": "AI"}
       ),
   ]

   rag = SimpleRAG(documents=knowledge_base)


Test 1: Basic Question Answering
---------------------------------

Test that the system answers questions correctly:

.. code-block:: python

   from giskard.agents.generators import Generator
   from giskard.checks import (
       scenario,
       StringMatchingCheck,
       from_fn,
       set_default_generator
   )

   # Configure LLM for checks
   set_default_generator(Generator(model="openai/gpt-4o-mini"))

   async def run_basic_qa_example():
       result = await (
           scenario("basic_qa_france_capital")
           .interact(
               "What is the capital of France?",
               lambda inputs: rag.answer(inputs)
           )
           .check(StringMatchingCheck(
               name="mentions_paris",
               content="Paris",
               key="interactions[-1].outputs.answer"
           ))
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.retrieved_docs) > 0,
               name="retrieved_documents",
               success_message="Retrieved relevant documents",
               failure_message="No documents retrieved"
           ))
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs.confidence > 0.5,
               name="confident_answer",
               success_message="High confidence answer",
               failure_message="Low confidence answer"
           ))
           .run()
       )

       print(f"Test passed: {result.passed}")
       for check_result in result.results:
           print(f"  {check_result.name}: {check_result.status.value}")

   # Run the example
   import asyncio
   asyncio.run(run_basic_qa_example())


Test 2: Groundedness Check
---------------------------

Verify that answers are grounded in retrieved context:

.. code-block:: python

   from giskard.checks import scenario, Groundedness, StringMatchingCheck

   async def run_groundedness_example():
       result = await (
           scenario("groundedness_eiffel_tower")
           .interact(
               "When was the Eiffel Tower completed?",
               lambda inputs: rag.answer(inputs)
           )
           .check(Groundedness(
               name="answer_grounded",
               description="Answer should be based on retrieved documents"
           ))
           .check(StringMatchingCheck(
               name="mentions_year",
               content="1889",
               key="interactions[-1].outputs.answer"
           ))
           .run()
       )
       assert result.passed
       print(f"Groundedness check passed: {result.passed}")

   import asyncio
   asyncio.run(run_groundedness_example())


Test 3: Retrieval Quality
--------------------------

Test that the right documents are retrieved:

.. code-block:: python

   from giskard.checks import scenario, from_fn

   def check_retrieved_topics(trace) -> bool:
       """Verify retrieved docs are about the right topic."""
       docs = trace.interactions[-1].outputs.retrieved_docs
       topics = [doc.metadata.get("topic") for doc in docs]
       return "Eiffel Tower" in topics or "France" in topics

   async def run_retrieval_quality_example():
       result = await (
           scenario("retrieval_quality")
           .interact(
               "Tell me about the Eiffel Tower",
               lambda inputs: rag.answer(inputs)
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.retrieved_docs) >= 2,
               name="sufficient_context",
               success_message="Retrieved multiple documents",
               failure_message="Not enough documents retrieved"
           ))
           .check(from_fn(
               check_retrieved_topics,
               name="relevant_topics",
               success_message="Retrieved documents are topically relevant",
               failure_message="Retrieved documents are off-topic"
           ))
           .run()
       )
       print(f"Retrieval quality check passed: {result.passed}")

   import asyncio
   asyncio.run(run_retrieval_quality_example())


Test 4: Out-of-Scope Questions
-------------------------------

Test how the system handles questions it can't answer:

.. code-block:: python

   from giskard.checks import scenario, LLMJudge, from_fn
   import asyncio

   async def run_out_of_scope_example():
       result = await (
           scenario("out_of_scope_handling")
           .interact(
               "What is the weather in Tokyo today?",
               lambda inputs: rag.answer(inputs)
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.retrieved_docs) == 0,
               name="no_irrelevant_docs",
               success_message="Correctly retrieved no documents",
               failure_message="Retrieved documents for out-of-scope question"
           ))
           .check(LLMJudge(
               name="appropriate_fallback",
               prompt="""
               Evaluate if the system appropriately indicates it cannot answer.

               Question: {{ inputs }}
               Answer: {{ outputs.answer }}

               The answer should politely indicate insufficient information.
               Return 'passed: true' if appropriate, 'passed: false' if it makes up an answer.
               """
           ))
           .run()
       )
       print(f"Out-of-scope handling check passed: {result.passed}")

   asyncio.run(run_out_of_scope_example())


Test 5: Answer Quality with LLM Judge
--------------------------------------

Use an LLM to evaluate answer quality comprehensively:

.. code-block:: python

   from giskard.checks import scenario, LLMJudge
   import asyncio

   async def run_comprehensive_quality_example():
       result = await (
           scenario("comprehensive_quality_check")
           .interact(
               "What is machine learning?",
               lambda inputs: rag.answer(inputs)
           )
           .check(LLMJudge(
               name="answer_quality",
               prompt="""
               Evaluate the answer quality based on these criteria:

               Question: {{ inputs }}
               Answer: {{ outputs.answer }}
               Retrieved Context: {{ outputs.retrieved_docs }}

               Criteria:
               1. Accuracy: Is the answer factually correct?
               2. Completeness: Does it fully address the question?
               3. Clarity: Is it well-written and understandable?
               4. Relevance: Does it stay on topic?

               Return 'passed: true' if the answer meets all criteria.
               Provide brief reasoning.
               """
           ))
           .run()
       )
       print(f"Comprehensive quality check passed: {result.passed}")

   asyncio.run(run_comprehensive_quality_example())


Test 6: Multi-Turn Conversational RAG
--------------------------------------

Test a conversational RAG that handles follow-up questions:

.. code-block:: python

   from giskard.checks import (
       scenario,
       Groundedness,
       StringMatchingCheck,
       from_fn,
       LLMJudge
   )

   class ConversationalRAG(SimpleRAG):
       def __init__(self, documents):
           super().__init__(documents)
           self.conversation_history = []

       def answer(self, question: str) -> RAGResponse:
           # Resolve references using conversation history
           resolved_question = self._resolve_references(
               question,
               self.conversation_history
           )

           response = super().answer(resolved_question)

           self.conversation_history.append({
               "question": question,
               "resolved_question": resolved_question,
               "answer": response.answer
           })

           return response

       def _resolve_references(self, question: str, history: list) -> str:
           """Resolve pronouns and references in follow-up questions."""
           # Simplified: in practice, use LLM for coreference resolution
           if history and ("it" in question.lower() or "its" in question.lower()):
               # Get the topic from previous question
               prev_question = history[-1]["resolved_question"]
               return f"{question} (referring to: {prev_question})"
           return question

   conv_rag = ConversationalRAG(documents=knowledge_base)

   async def run_conversational_rag_example():
       result = await (
           scenario("conversational_rag_flow")
           # First question
           .interact(
               "What is the capital of France?",
               lambda inputs: conv_rag.answer(inputs)
           )
           .check(Groundedness(name="first_answer_grounded"))
           .check(StringMatchingCheck(
               name="first_mentions_paris",
               content="Paris",
               key="interactions[-1].outputs.answer"
           ))
           # Follow-up question with reference
           .interact(
               "What is it known for?",
               lambda inputs: conv_rag.answer(inputs)
           )
           .check(Groundedness(name="followup_grounded"))
           .check(LLMJudge(
               name="resolves_reference",
               prompt="""
               Check if the answer appropriately addresses the follow-up question
               in the context of the conversation.

               First Q: {{ interactions[0].inputs }}
               First A: {{ interactions[0].outputs.answer }}
               Follow-up Q: {{ interactions[1].inputs }}
               Follow-up A: {{ interactions[1].outputs.answer }}

               The follow-up should discuss what Paris is known for.
               Return 'passed: true' if the context was maintained correctly.
               """
           ))
           .run()
       )
       print(f"Conversational RAG example passed: {result.passed}")

   import asyncio
   asyncio.run(run_conversational_rag_example())


Complete Test Suite
-------------------

Combine all tests into a comprehensive suite:

.. code-block:: python

   import asyncio
   from typing import List
   from giskard.checks import scenario, Scenario

   class RAGTestSuite:
       def __init__(self, rag_system: SimpleRAG):
           self.rag = rag_system
           self.scenarios = []
           self._build_scenarios()

       def _build_scenarios(self):
           """Build all scenarios."""
           # Add basic QA tests
           self.scenarios.extend(self._create_qa_tests())

           # Add groundedness tests
           self.scenarios.extend(self._create_groundedness_tests())

           # Add edge case tests
           self.scenarios.extend(self._create_edge_case_tests())

       def _create_qa_tests(self) -> List[Scenario]:
           """Create basic QA test scenarios."""
           test_data = [
               ("What is the capital of France?", "Paris"),
               ("When was the Eiffel Tower completed?", "1889"),
               ("What is Python?", "programming_language"),
           ]

           scenarios = []
           for question, expected_content in test_data:
               test_scenario = (
                   scenario(f"qa_{expected_content.replace(' ', '_')}")
                   .interact(question, lambda inputs: self.rag.answer(inputs))
                   .check(StringMatchingCheck(
                       name=f"contains_{expected_content}",
                       content=expected_content,
                       key="interactions[-1].outputs.answer"
                   ))
                   .check(from_fn(
                       lambda trace: len(trace.interactions[-1].outputs.retrieved_docs) > 0,
                       name="has_context"
                   ))
                   .build()
               )
               scenarios.append(test_scenario)

           return scenarios

       def _create_groundedness_tests(self) -> List[Scenario]:
           """Create groundedness test scenarios."""
           questions = [
               "What is the capital of France?",
               "Tell me about the Eiffel Tower",
               "What is machine learning?",
           ]

           scenarios = []
           for question in questions:
               test_scenario = (
                   scenario(f"groundedness_{question[:20]}")
                   .interact(question, lambda inputs: self.rag.answer(inputs))
                   .check(Groundedness(name="grounded"))
                   .build()
               )
               scenarios.append(test_scenario)

           return scenarios

       def _create_edge_case_tests(self) -> List[Scenario]:
           """Create edge case test scenarios."""
           edge_cases = [
               ("", "empty_query"),
               ("   ", "whitespace_query"),
               ("What is the weather in Tokyo?", "out_of_scope"),
               ("askdjhaksjdhaksjdh", "gibberish"),
           ]

           scenarios = []
           for question, case_name in edge_cases:
               test_scenario = (
                   scenario(f"edge_case_{case_name}")
                   .interact(question, lambda inputs: self.rag.answer(inputs))
                   .check(from_fn(
                       lambda trace: trace.interactions[-1].outputs.answer,
                       name="provides_response",
                       success_message="System provided a response"
                   ))
                   .build()
               )
               scenarios.append(test_scenario)

           return scenarios

       async def run_all(self):
           """Run all scenarios and report results."""
           results = []

           for test_scenario in self.scenarios:
               result = await test_scenario.run()
               results.append((test_scenario.name, result))

           # Summary
           passed = sum(1 for _, r in results if r.passed)
           total = len(results)

           print(f"\nTest Suite Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
           print("\nDetailed Results:")

           for name, result in results:
               status = "✓" if result.passed else "✗"
               print(f"  {status} {name}")
               if not result.passed:
                   for check_result in result.results:
                       if not check_result.passed:
                           print(f"      - {check_result.name}: {check_result.message}")

           return results

   # Run the complete suite
   async def main():
       suite = RAGTestSuite(rag)
       await suite.run_all()

   asyncio.run(main())


Best Practices for RAG Testing
-------------------------------

**1. Test Retrieval Separately**

Validate retrieval quality before testing end-to-end:

.. code-block:: python

   def test_retrieval_precision():
       docs = rag.retrieve("Eiffel Tower")
       relevant_topics = ["Eiffel Tower", "France", "Paris"]
       assert all(
           any(topic in doc.metadata.get("topic", "") for topic in relevant_topics)
           for doc in docs
       )

**2. Use Representative Test Data**

Include diverse question types:

- Factual questions
- Definitional questions
- Comparison questions
- Out-of-scope questions
- Ambiguous questions

**3. Monitor Confidence Scores**

Track confidence metrics to identify problematic queries:

.. code-block:: python

   checks = [
       from_fn(
           lambda trace: trace.interactions[-1].outputs.confidence,
           name="track_confidence",
           success_message=lambda trace: f"Confidence: {trace.interactions[-1].outputs.confidence}"
       ),
   ]

**4. Test with Real User Queries**

Collect and test with actual user questions from logs.


Next Steps
----------

* See :doc:`testing-agents` for agent-specific testing patterns
* Explore :doc:`chatbot-testing` for conversational testing
* Review :doc:`../ai-testing/multi-turn` for advanced scenarios

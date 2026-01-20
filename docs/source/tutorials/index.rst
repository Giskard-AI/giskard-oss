=========
Tutorials
=========

Practical, end-to-end examples of testing AI applications with Giskard Checks.

.. toctree::
   :maxdepth: 2

   rag-evaluation
   testing-agents
   chatbot-testing
   content-moderation


Overview
--------

These tutorials provide complete, working examples that you can adapt for your own use cases. Each tutorial includes:

* Full working code
* Explanation of key concepts
* Common pitfalls and how to avoid them
* Extensions and variations


Available Tutorials
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔍 RAG Evaluation
      :link: rag-evaluation
      :link-type: doc

      Test retrieval quality, groundedness, and answer relevance in RAG systems

   .. grid-item-card:: 🤖 Testing Agents
      :link: testing-agents
      :link-type: doc

      Validate multi-step agent workflows with tool usage and reasoning

   .. grid-item-card:: 💬 Chatbot Testing
      :link: chatbot-testing
      :link-type: doc

      Test conversational flows, context handling, and response quality

   .. grid-item-card:: 🛡️ Content Moderation
      :link: content-moderation
      :link-type: doc

      Implement safety checks and content filtering


What You'll Learn
-----------------

Through these tutorials, you'll learn how to:

* Design effective test suites for different AI application types
* Combine built-in and custom checks for comprehensive validation
* Handle both single-turn and multi-turn scenarios
* Use LLM-as-a-judge for nuanced evaluation
* Track metrics and analyze test results
* Integrate checks into CI/CD pipelines


Prerequisites
-------------

Before starting these tutorials, you should:

* Have completed the :doc:`../getting-started/installation` guide
* Be familiar with the :doc:`../ai-testing/core-concepts`
* Have basic Python and async/await knowledge
* Have access to an LLM API (OpenAI, Anthropic, or compatible)


Getting Help
------------

If you run into issues with these tutorials:

1. Check the :doc:`../ai-testing/core-concepts` for concept clarifications
2. Review the :doc:`../ai-testing/custom-checks` guide for check creation patterns
3. Look at the API reference for detailed documentation
4. Open an issue on GitHub if you find bugs or have suggestions

========================
What are Giskard Checks?
========================

Giskard Checks is a lightweight Python library for testing and evaluating non-deterministic applications such as LLM-based systems.


Overview
--------

**Giskard Checks** provides a flexible and powerful framework for testing AI applications including RAG systems, agents, summarization models, and more. Whether you're building chatbots, question-answering systems, or complex multi-step workflows, Giskard Checks helps you ensure quality and reliability.

Key Features
~~~~~~~~~~~~

* **Built-in Check Library**: Ready-to-use checks including LLM-as-a-judge evaluations, string matching, equality assertions, and more
* **Flexible Testing Framework**: Support for both single-turn and multi-turn scenarios with stateful trace management
* **Type-Safe & Modern**: Built on Pydantic for full type safety and validation
* **Async-First**: Native async/await support for efficient concurrent testing
* **Highly Customizable**: Easy extension points for custom checks and interaction patterns
* **Serializable Results**: Immutable, JSON-serializable results for easy storage and analysis


Quick Links
-----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Quickstart
      :link: ai-testing/quickstart
      :link-type: doc

      Installation, configuration, and your first test

   .. grid-item-card:: 📚 AI Testing Guide
      :link: ai-testing/quickstart
      :link-type: doc

      Learn core concepts, single-turn and multi-turn testing

   .. grid-item-card:: 💡 Tutorials
      :link: tutorials/index
      :link-type: doc

      Practical examples for RAG, agents, and more

   .. grid-item-card:: 🔧 API Reference
      :link: api-reference/index
      :link-type: doc

      Complete API documentation


Use Cases
---------

Giskard Checks is designed for:

* **RAG Evaluation**: Test groundedness, relevance, and context usage in retrieval-augmented generation systems
* **Agent Testing**: Validate multi-step agent workflows with tool calls and complex reasoning
* **Quality Assurance**: Ensure consistent output quality across model updates and deployments
* **LLM Guardrails**: Implement safety checks, content moderation, and compliance validation
* **Regression Testing**: Track model behavior changes over time with reproducible test suites


.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   self
   getting-started/installation


.. toctree::
   :caption: Testing AI Agents
   :maxdepth: 2
   :hidden:

   ai-testing/quickstart
   ai-testing/core-concepts
   ai-testing/single-turn
   ai-testing/multi-turn
   ai-testing/custom-checks

.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   :hidden:
   tutorials/index

.. toctree::
   :caption: API Reference
   :maxdepth: 2
   :hidden:
   api-reference/index

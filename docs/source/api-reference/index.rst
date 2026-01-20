=============
API Reference
=============

Complete API documentation for Giskard Checks.

.. toctree::
   :maxdepth: 2

   core
   builtin-checks
   scenarios
   testing


Overview
--------

Giskard Checks provides a comprehensive API for testing and evaluating AI applications. The library is organized into several key modules:

* **Core**: Fundamental types and base classes (Check, Trace, Interaction)
* **Built-in Checks**: Ready-to-use checks for common scenarios
* **Scenarios**: Multi-step workflow testing (Scenario, TestCase)
* **Testing**: Utilities for testing and debugging


Quick Reference
---------------

**Most Common Imports**

.. code-block:: python

   from giskard.checks import (
       # Core types
       Check,
       CheckResult,
       CheckStatus,
       Interaction,
       Trace,

       # Interaction specs
       InteractionSpec,
       BaseInteractionSpec,

       # Scenarios
       Scenario,
       TestCase,

       # Built-in checks
       from_fn,
       StringMatchingCheck,
       EqualityCheck,
       Groundedness,
       Conformity,
       LLMJudge,

       # Configuration
       set_default_generator,
       get_default_generator,
   )


Package Structure
-----------------

.. code-block:: text

   giskard.checks/
   ├── core/
   │   ├── check.py          # Check base class
   │   ├── trace.py          # Trace and Interaction
   │   ├── interaction.py    # InteractionSpec
   │   ├── result.py         # CheckResult, CheckStatus
   │   ├── scenario.py       # Scenario
   │   ├── testcase.py       # TestCase
   │   └── extraction.py     # Extractors
   │
   ├── builtin/
   │   ├── fn.py             # from_fn
   │   ├── string_matching.py
   │   ├── equality.py
   │   ├── groundedness.py
   │   ├── conformity.py
   │   ├── judge.py          # LLMJudge
   │   └── extraction_check.py
   │
   ├── scenarios/
   │   └── runner.py         # ScenarioRunner
   │
   └── testing/
       ├── runner.py         # TestCaseRunner
       └── spy.py            # WithSpy

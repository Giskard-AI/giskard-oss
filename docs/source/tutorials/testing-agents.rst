==============
Testing Agents
==============

This tutorial demonstrates how to test AI agents that use tools, perform multi-step reasoning, and maintain state across interactions.

Overview
--------

We'll build and test an agent that can:

* **Use multiple tools** (search, calculator, database)
* **Plan multi-step actions** to accomplish goals
* **Maintain state** across interactions
* **Handle failures** and retry with different strategies

Our tests will validate:

* Tool selection logic
* Reasoning quality
* Task completion
* Error handling
* State management


Building a Simple Agent
------------------------

First, let's create an agent to test:

.. code-block:: python

   from typing import Literal, Optional
   from pydantic import BaseModel

   class Tool(BaseModel):
       name: str
       description: str

   class AgentStep(BaseModel):
       thought: str
       tool: str
       tool_input: str
       observation: str

   class AgentResponse(BaseModel):
       steps: list[AgentStep]
       final_answer: str
       success: bool

   class SimpleAgent:
       def __init__(self):
           self.tools = {
               "search": Tool(
                   name="search",
                   description="Search the internet for information"
               ),
               "calculator": Tool(
                   name="calculator",
                   description="Perform mathematical calculations"
               ),
               "database": Tool(
                   name="database",
                   description="Query a database for structured data"
               ),
           }
           self.max_steps = 5

       def _use_tool(self, tool_name: str, tool_input: str) -> str:
           """Execute a tool (simplified for testing)."""
           if tool_name == "search":
               return f"Search results for '{tool_input}': [Relevant information...]"
           elif tool_name == "calculator":
               try:
                   result = eval(tool_input)  # Don't do this in production!
                   return str(result)
               except Exception as e:
                   return f"Error: {e}"
           elif tool_name == "database":
               return f"Database query result for '{tool_input}': [Records...]"
           return "Unknown tool"

       def run(self, task: str) -> AgentResponse:
           """Run the agent on a task."""
           steps = []

           # Simplified agent logic
           if "calculate" in task.lower() or any(c in task for c in "0123456789+-*/"):
               # Math task
               thought = "I need to use the calculator for this math problem"
               tool = "calculator"
               # Extract the calculation
               import re
               calculation = re.findall(r'[\d+\-*/()]+', task)
               tool_input = calculation[0] if calculation else task

               observation = self._use_tool(tool, tool_input)
               steps.append(AgentStep(
                   thought=thought,
                   tool=tool,
                   tool_input=tool_input,
                   observation=observation
               ))

               final_answer = f"The answer is {observation}"
               success = "Error" not in observation

           elif "search" in task.lower() or "find" in task.lower():
               # Search task
               thought = "I should search for this information"
               tool = "search"
               tool_input = task

               observation = self._use_tool(tool, tool_input)
               steps.append(AgentStep(
                   thought=thought,
                   tool=tool,
                   tool_input=tool_input,
                   observation=observation
               ))

               final_answer = f"Based on my search: {observation}"
               success = True

           else:
               # Default case
               thought = "This task doesn't require tools"
               final_answer = "I can answer this directly: " + task
               success = True

           return AgentResponse(
               steps=steps,
               final_answer=final_answer,
               success=success
           )


Test 1: Tool Selection
-----------------------

Verify that the agent selects appropriate tools:

.. code-block:: python

   from giskard.checks import scenario, from_fn, EqualityCheck

   agent = SimpleAgent()

   async def run_tool_selection_example():
       result = await (
           scenario("tool_selection_calculator")
           .interact(
               "What is 15 * 23?",
               lambda inputs: agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.steps) > 0,
               name="used_tools",
               success_message="Agent used tools",
               failure_message="Agent didn't use any tools"
           ))
           .check(EqualityCheck(
               name="selected_calculator",
               expected="calculator",
               key="interactions[-1].outputs.steps[0].tool"
           ))
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs.success,
               name="task_successful",
               success_message="Agent completed task successfully",
               failure_message="Agent failed to complete task"
           ))
           .run()
       )
       assert result.passed
       print(f"Tool selection example passed: {result.passed}")

   import asyncio
   asyncio.run(run_tool_selection_example())


Test 2: Reasoning Quality
--------------------------

Evaluate the quality of the agent's reasoning:

.. code-block:: python

   from giskard.agents.generators import Generator
   from giskard.checks import scenario, LLMJudge, from_fn, set_default_generator

   set_default_generator(Generator(model="openai/gpt-4o-mini"))

   async def run_reasoning_quality_example():
       result = await (
           scenario("reasoning_quality_test")
           .interact(
               "Find information about quantum computing",
               lambda inputs: agent.run(inputs)
           )
           .check(LLMJudge(
               name="reasoning_quality",
               prompt="""
               Evaluate the agent's reasoning process.

               Task: {{ inputs }}
               Thought: {{ outputs.steps[0].thought if outputs.steps else "No reasoning" }}
               Tool Selected: {{ outputs.steps[0].tool if outputs.steps else "None" }}

               Criteria:
               1. Is the reasoning logical?
               2. Is the tool selection appropriate for the task?
               3. Does the thought explain why the tool was chosen?

               Return 'passed: true' if the reasoning is sound.
               """
           ))
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs.steps[0].tool == "search",
               name="correct_tool_for_research",
               success_message="Selected search for research task",
               failure_message="Wrong tool selected"
           ))
           .run()
       )
       print(f"Reasoning quality example passed: {result.passed}")

   import asyncio
   asyncio.run(run_reasoning_quality_example())


Test 3: Multi-Step Agent Workflow
----------------------------------

Test agents that perform multiple steps:

.. code-block:: python

   class MultiStepAgent(SimpleAgent):
       def run(self, task: str) -> AgentResponse:
           """Run agent with multi-step capability."""
           steps = []

           # Example: Complex task requiring multiple tools
           if "research" in task.lower() and "calculate" in task.lower():
               # Step 1: Search
               steps.append(AgentStep(
                   thought="First, I need to search for the data",
                   tool="search",
                   tool_input=task,
                   observation=self._use_tool("search", task)
               ))

               # Step 2: Calculate
               steps.append(AgentStep(
                   thought="Now I'll calculate based on the data",
                   tool="calculator",
                   tool_input="100 * 2",
                   observation=self._use_tool("calculator", "100 * 2")
               ))

               final_answer = f"Based on my research and calculations: {steps[-1].observation}"
               success = True
           else:
               return super().run(task)

           return AgentResponse(
               steps=steps,
               final_answer=final_answer,
               success=success
           )

   multi_agent = MultiStepAgent()

   from giskard.checks import scenario, from_fn, LLMJudge
   import asyncio

   async def run_multi_step_workflow_example():
       result = await (
           scenario("multi_step_agent_workflow")
           .interact(
               "Research the market size and calculate projected growth",
               lambda inputs: multi_agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.steps) >= 2,
               name="multiple_steps_taken",
               success_message="Agent performed multiple steps",
               failure_message="Agent didn't perform enough steps"
           ))
           .check(from_fn(
               lambda trace: any(
                   step.tool == "search"
                   for step in trace.interactions[-1].outputs.steps
               ),
               name="performed_research",
               success_message="Agent performed research",
               failure_message="Agent skipped research step"
           ))
           .check(from_fn(
               lambda trace: any(
                   step.tool == "calculator"
                   for step in trace.interactions[-1].outputs.steps
               ),
               name="performed_calculation",
               success_message="Agent performed calculation",
               failure_message="Agent skipped calculation step"
           ))
           .check(LLMJudge(
               name="steps_logical_order",
               prompt="""
               Evaluate if the agent's steps are in a logical order.

               Task: {{ interactions[0].inputs }}
               Steps:
               {% for step in interactions[0].outputs.steps %}
               {{ loop.index }}. {{ step.thought }} -> {{ step.tool }}
               {% endfor %}

               Return 'passed: true' if steps are well-ordered.
               """
           ))
           .run()
       )
       print(f"Multi-step workflow example passed: {result.passed}")

   asyncio.run(run_multi_step_workflow_example())


Test 4: Error Handling
-----------------------

Verify that agents handle errors gracefully:

.. code-block:: python

   class RobustAgent(SimpleAgent):
       def run(self, task: str) -> AgentResponse:
           steps = []

           # Try first approach
           thought = "I'll try using the calculator"
           observation = self._use_tool("calculator", task)
           steps.append(AgentStep(
               thought=thought,
               tool="calculator",
               tool_input=task,
               observation=observation
           ))

           if "Error" in observation:
               # Fallback strategy
               thought = "Calculator failed, I'll search instead"
               observation = self._use_tool("search", task)
               steps.append(AgentStep(
                   thought=thought,
                   tool="search",
                   tool_input=task,
                   observation=observation
               ))
               final_answer = f"After trying different approaches: {observation}"
               success = True
           else:
               final_answer = f"Result: {observation}"
               success = True

           return AgentResponse(
               steps=steps,
               final_answer=final_answer,
               success=success
           )

   robust_agent = RobustAgent()

   async def run_error_handling_example():
       result = await (
           scenario("error_handling_test")
           .interact(
               "What is the meaning of life?",  # Not a valid calculation
               lambda inputs: robust_agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.steps) > 1,
               name="tried_fallback",
               success_message="Agent tried fallback strategy",
               failure_message="Agent didn't attempt recovery"
           ))
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs.success,
               name="eventually_succeeded",
               success_message="Agent completed task despite initial failure",
               failure_message="Agent failed to complete task"
           ))
           .check(LLMJudge(
               name="error_recovery_appropriate",
               prompt="""
               Evaluate if the agent's error recovery was appropriate.

               Task: {{ inputs }}
               Steps taken:
               {% for step in outputs.steps %}
               {{ loop.index }}. {{ step.thought }} ({{ step.tool }})
                  Result: {{ step.observation }}
               {% endfor %}

               Return 'passed: true' if the agent handled the error well.
               """
           ))
           .run()
       )
       print(f"Error handling example passed: {result.passed}")

   import asyncio
   asyncio.run(run_error_handling_example())


Test 5: Stateful Agent Interactions
------------------------------------

Test agents that maintain state across turns:

.. code-block:: python

   class StatefulAgent(SimpleAgent):
       def __init__(self):
           super().__init__()
           self.memory = {}
           self.conversation_history = []

       def run(self, task: str) -> AgentResponse:
           # Check memory for context
           if "last" in task.lower() or "previous" in task.lower():
               if self.conversation_history:
                   prev_task = self.conversation_history[-1]["task"]
                   thought = f"Recalling previous task: {prev_task}"
                   observation = f"Previous task was: {prev_task}"
                   final_answer = f"I remember: {observation}"

                   steps = [AgentStep(
                       thought=thought,
                       tool="memory",
                       tool_input="recall",
                       observation=observation
                   )]

                   self.conversation_history.append({
                       "task": task,
                       "response": final_answer
                   })

                   return AgentResponse(
                       steps=steps,
                       final_answer=final_answer,
                       success=True
                   )

           # Handle new task
           response = super().run(task)
           self.conversation_history.append({
               "task": task,
               "response": response.final_answer
           })
           return response

   stateful_agent = StatefulAgent()

   async def run_stateful_agent_example():
       result = await (
           scenario("stateful_agent_memory")
           # First interaction
           .interact(
               "Search for Python tutorials",
               lambda inputs: stateful_agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs.success,
               name="first_task_completed"
           ))
           # Second interaction references first
           .interact(
               "What was my last request?",
               lambda inputs: stateful_agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: "Python tutorials" in trace.interactions[-1].outputs.final_answer,
               name="recalls_previous_task",
               success_message="Agent correctly recalled previous task",
               failure_message="Agent failed to recall previous task"
           ))
           .check(LLMJudge(
               name="context_maintained",
               prompt="""
               Evaluate if the agent maintained context correctly.

               First task: {{ interactions[0].inputs }}
               Second task: {{ interactions[1].inputs }}
               Second response: {{ interactions[1].outputs.final_answer }}

               The second response should reference the first task.
               Return 'passed: true' if context was maintained.
               """
           ))
           .run()
       )
       print(f"Stateful agent example passed: {result.passed}")

   import asyncio
   asyncio.run(run_stateful_agent_example())


Test 6: Task Completion Validation
-----------------------------------

Verify that complex tasks are fully completed:

.. code-block:: python

   from giskard.checks import scenario, LLMJudge, from_fn

   class TaskTrackingAgent(SimpleAgent):
       def __init__(self):
           super().__init__()
           self.pending_tasks = []
           self.completed_tasks = []

       def run(self, task: str) -> AgentResponse:
           if "add task" in task.lower():
               task_desc = task.replace("add task", "").strip()
               self.pending_tasks.append(task_desc)
               return AgentResponse(
                   steps=[],
                   final_answer=f"Added task: {task_desc}. Pending: {len(self.pending_tasks)}",
                   success=True
               )

           elif "complete" in task.lower():
               if self.pending_tasks:
                   completed = self.pending_tasks.pop(0)
                   self.completed_tasks.append(completed)

                   return AgentResponse(
                       steps=[AgentStep(
                           thought=f"Completing task: {completed}",
                           tool="task_manager",
                           tool_input=completed,
                           observation="Task completed successfully"
                       )],
                       final_answer=f"Completed: {completed}",
                       success=True
                   )
               return AgentResponse(
                   steps=[],
                   final_answer="No pending tasks to complete",
                   success=False
               )

           elif "status" in task.lower():
               return AgentResponse(
                   steps=[],
                   final_answer=f"Pending: {len(self.pending_tasks)}, Completed: {len(self.completed_tasks)}",
                   success=True
               )

           return super().run(task)

   task_agent = TaskTrackingAgent()

   async def run_task_completion_example():
       result = await (
           scenario("task_completion_workflow")
           # Add tasks
           .interact(
               "add task: Write documentation",
               lambda inputs: task_agent.run(inputs)
           )
           .interact(
               "add task: Review code",
               lambda inputs: task_agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: len(task_agent.pending_tasks) == 2,
               name="tasks_added"
           ))
           # Complete first task
           .interact(
               "complete next task",
               lambda inputs: task_agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: len(task_agent.completed_tasks) == 1,
               name="task_completed"
           ))
           # Check status
           .interact(
               "what's the status?",
               lambda inputs: task_agent.run(inputs)
           )
           .check(from_fn(
               lambda trace: (
                   "Pending: 1" in trace.interactions[-1].outputs.final_answer and
                   "Completed: 1" in trace.interactions[-1].outputs.final_answer
               ),
               name="status_accurate",
               success_message="Agent tracking state correctly",
               failure_message="Agent state tracking is incorrect"
           ))
           .run()
       )
       print(f"Task completion example passed: {result.passed}")

   import asyncio
   asyncio.run(run_task_completion_example())


Complete Agent Test Suite
--------------------------

Combine all tests into a comprehensive suite:

.. code-block:: python

   import asyncio
   from typing import List
   from giskard.checks import Scenario

   class AgentTestSuite:
       def __init__(self, agent):
           self.agent = agent
           self.scenarios: List[Scenario] = []

       def add_scenario(self, test_scenario: Scenario):
           self.scenarios.append(test_scenario)

       async def run_all(self):
           """Run all scenarios."""
           results = []

           print("Running scenarios...")
           for test_scenario in self.scenarios:
               result = await test_scenario.run()
               results.append(("scenario", test_scenario.name, result))

           # Report
           self._report_results(results)

           return results

       def _report_results(self, results):
           total = len(results)
           passed = sum(1 for _, _, r in results if r.passed)

           print(f"\n{'='*60}")
           print(f"Agent Test Suite Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
           print(f"{'='*60}\n")

           for test_type, name, result in results:
               status = "✓" if result.passed else "✗"
               print(f"  {status} [{test_type}] {name}")

               if not result.passed:
                   if hasattr(result, 'results'):
                       for check_result in result.results:
                           if not check_result.passed:
                               print(f"      ↳ {check_result.name}: {check_result.message}")
                   elif hasattr(result, 'message'):
                       print(f"      ↳ {result.message}")

   # Usage
   async def main():
       agent = SimpleAgent()
       suite = AgentTestSuite(agent)

       # Add tests (from examples above)
       # suite.add_test(...)
       # suite.add_scenario(...)

       await suite.run_all()

   asyncio.run(main())


Best Practices
--------------

**1. Test Tool Selection Logic Independently**

Before testing full workflows, validate tool selection:

.. code-block:: python

   def test_tool_selection_logic():
       test_cases = [
           ("Calculate 5 + 3", "calculator"),
           ("Search for recipes", "search"),
           ("Query user database", "database"),
       ]

       for task, expected_tool in test_cases:
           response = agent.run(task)
           assert response.steps[0].tool == expected_tool

**2. Validate Reasoning at Each Step**

Use LLM judges to evaluate reasoning quality:

.. code-block:: python

   LLMJudge(
       name="step_reasoning",
       prompt="Is this reasoning step logical? {{ outputs.steps[0].thought }}"
   )

**3. Test Error Paths**

Ensure agents handle failures gracefully:

.. code-block:: python

   # Test with invalid tool inputs
   # Test with unavailable tools
   # Test with contradictory instructions

**4. Monitor Resource Usage**

Track token usage, API calls, and execution time:

.. code-block:: python

   checks = [
       from_fn(
           lambda trace: len(trace.interactions[-1].outputs.steps) <= 5,
           name="reasonable_step_count",
           success_message="Used reasonable number of steps"
       ),
   ]


Next Steps
----------

* See :doc:`chatbot-testing` for conversational agent patterns
* Explore :doc:`rag-evaluation` for knowledge-grounded agents
* Review :doc:`../ai-testing/multi-turn` for complex workflows

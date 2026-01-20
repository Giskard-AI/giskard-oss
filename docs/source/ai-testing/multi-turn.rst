===================
Multi-Turn Scenarios
===================

Multi-turn scenarios test conversational flows, stateful interactions, and complex workflows that span multiple exchanges.

Many AI applications involve multiple interactions:

* **Agents** that use tools across multiple steps
* **Chatbots** that maintain conversation context
* **Conversational RAG** where follow-up questions reference earlier context


Using Scenarios
---------------

The ``Scenario`` class executes multiple interaction specs and checks in sequence with a shared trace.

Basic Multi-Turn Flow
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from giskard.checks import scenario, StringMatchingCheck
   import asyncio

   async def run_greeting_example():
       result = await (
           scenario("greeting_conversation")
           # First interaction
           .interact(
               "Hello",
               lambda inputs: "Hi! How can I help you?"
           )
           .check(StringMatchingCheck(
               name="polite_greeting",
               content="help",
               key="interactions[-1].outputs"
           ))
           # Second interaction
           .interact(
               "What's the weather?",
               lambda inputs: "It's sunny and 72°F."
           )
           .check(StringMatchingCheck(
               name="provides_weather",
               content="sunny",
               key="interactions[-1].outputs"
           ))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   asyncio.run(run_greeting_example())

**Key Points:**

* Components execute in sequence
* Checks can reference any interaction via the trace
* Execution stops at the first failing check
* All components share the same trace


Stateful Conversations
----------------------

Test systems that maintain conversation state:

.. code-block:: python

   from giskard.checks import scenario, from_fn, Trace

   class Chatbot:
       def __init__(self):
           self.conversation_history = []

       def chat(self, message: str) -> str:
           self.conversation_history.append({"role": "user", "content": message})

           # Your chatbot logic
           if "my name is" in message.lower():
               name = message.split("my name is")[-1].strip()
               response = f"Nice to meet you, {name}!"
           elif "what is my name" in message.lower():
               # Reference earlier context
               for msg in reversed(self.conversation_history):
                   if "my name is" in msg.get("content", "").lower():
                       name = msg["content"].split("my name is")[-1].strip()
                       response = f"Your name is {name}."
                       break
               else:
                   response = "I don't recall your name."
           else:
               response = "I understand."

           self.conversation_history.append({"role": "assistant", "content": response})
           return response

   bot = Chatbot()

   async def run_name_memory_example():
       result = await (
           scenario("name_memory")
           .interact(
               "Hello, my name is Alice",
               lambda inputs: bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: "Alice" in trace.interactions[-1].outputs,
               name="acknowledges_name"
           ))
           .interact(
               "What is my name?",
               lambda inputs: bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: "Alice" in trace.interactions[-1].outputs,
               name="remembers_name",
               success_message="Correctly recalled the name",
               failure_message="Failed to recall the name"
           ))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_name_memory_example())


Testing Agent Workflows
------------------------

Test multi-step agent workflows with tool usage:

.. code-block:: python

   from giskard.agents.generators import Generator
   from giskard.checks import (
       scenario,
       LLMJudge,
       from_fn,
       set_default_generator
   )

   set_default_generator(Generator(model="openai/gpt-4o-mini"))

   class Agent:
       def __init__(self):
           self.available_tools = ["search", "calculator", "database"]

       def run(self, task: str) -> dict:
           # Your agent logic
           return {
               "thought": "I need to search for information",
               "action": "search",
               "action_input": "Python tutorial",
               "observation": "Found 10 Python tutorials",
               "final_answer": "Here are some Python tutorials..."
           }

   agent = Agent()

   async def run_agent_workflow_example():
       result = await (
           scenario("agent_workflow")
           # Agent receives task
           .interact(
               "Find me a Python tutorial",
               lambda inputs: agent.run(inputs)
           )
           # Check that agent chose appropriate tool
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs["action"] == "search",
               name="correct_tool_choice",
               success_message="Agent selected search tool",
               failure_message="Agent selected wrong tool"
           ))
           # Validate reasoning
           .check(LLMJudge(
               name="reasoning_quality",
               prompt="""
               Evaluate the agent's reasoning.

               Task: {{ interactions[0].inputs }}
               Thought: {{ interactions[0].outputs.thought }}
               Action: {{ interactions[0].outputs.action }}

               Return 'passed: true' if the reasoning is logical and appropriate.
               """
           ))
           # Check final answer quality
           .check(LLMJudge(
               name="answer_quality",
               prompt="""
               Evaluate if the final answer addresses the original task.

               Task: {{ interactions[0].inputs }}
               Answer: {{ interactions[0].outputs.final_answer }}

               Return 'passed: true' if the answer is helpful and relevant.
               """
           ))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_agent_workflow_example())


Dynamic Multi-Turn Interactions
--------------------------------

Generate interactions dynamically based on previous outputs:

.. code-block:: python

   from giskard.checks import scenario, from_fn, Trace

   def chatbot(message: str, context: list = None) -> dict:
       # Your chatbot that tracks context
       return {"response": "...", "context": context or []}

   # Second interaction depends on first response
   async def generate_followup(trace: Trace):
       first_response = trace.interactions[-1].outputs["response"]
       return f"Tell me more about {first_response}"

   async def run_dynamic_conversation_example():
       result = await (
           scenario("dynamic_conversation")
           .interact(
               "What's the capital of France?",
               lambda inputs: chatbot(inputs)
           )
           .check(from_fn(lambda trace: len(trace.interactions) == 1, name="first_complete"))
           .interact(
               generate_followup,
               lambda inputs: chatbot(inputs)
           )
           .check(from_fn(lambda trace: len(trace.interactions) == 2, name="second_complete"))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_dynamic_conversation_example())


Testing Error Recovery
----------------------

Verify that systems handle errors gracefully across turns:

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, from_fn, LLMJudge

   class RobustChatbot:
       def chat(self, message: str) -> dict:
           if not message.strip():
               return {
                   "error": "Empty message",
                   "response": "I didn't receive a message. Could you try again?"
               }
           return {"response": "I understand."}

   bot = RobustChatbot()

   async def run_error_recovery_example():
       result = await (
           scenario("error_recovery")
           # Send invalid input
           .interact(
               "",
               lambda inputs: bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: "error" in trace.interactions[-1].outputs,
               name="detects_error"
           ))
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs["response"],
               name="provides_feedback",
               success_message="Bot provided error feedback"
           ))
           # Send valid follow-up
           .interact(
               "Hello",
               lambda inputs: bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: "error" not in trace.interactions[-1].outputs,
               name="recovers_from_error",
               success_message="System recovered successfully"
           ))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_error_recovery_example())


Conversational RAG
------------------

Test RAG systems with follow-up questions and context references:

.. code-block:: python

   from giskard.checks import scenario, Groundedness, from_fn

   class ConversationalRAG:
       def __init__(self):
           self.conversation_history = []

       def answer(self, question: str) -> dict:
           # Retrieve context considering conversation history
           context = self.retrieve(question, self.conversation_history)
           answer = self.generate(question, context, self.conversation_history)

           self.conversation_history.append({
               "question": question,
               "answer": answer,
               "context": context
           })

           return {"answer": answer, "context": context}

       def retrieve(self, question, history):
           # Your retrieval logic
           return ["context chunk 1", "context chunk 2"]

       def generate(self, question, context, history):
           # Your generation logic
           return "Answer based on context..."

   rag = ConversationalRAG()

   async def run_conversational_rag_example():
       result = await (
           scenario("conversational_rag")
           # Initial question
           .interact(
               "What is machine learning?",
               lambda inputs: rag.answer(inputs)
           )
           .check(Groundedness(name="first_answer_grounded"))
           # Follow-up with pronoun reference
           .interact(
               "What are its main applications?",
               lambda inputs: rag.answer(inputs)
           )
           .check(Groundedness(name="followup_grounded"))
           .check(from_fn(
               lambda trace: len(trace.interactions) == 2,
               name="maintains_context",
               success_message="System handled follow-up correctly"
           ))
           # Another follow-up
           .interact(
               "Can you explain the first one in detail?",
               lambda inputs: rag.answer(inputs)
           )
           .check(Groundedness(name="second_followup_grounded"))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_conversational_rag_example())


Task Completion Tracking
-------------------------

Test that multi-step tasks are completed successfully:

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, from_fn, LLMJudge

   class TaskAgent:
       def __init__(self):
           self.tasks = []
           self.completed = []

       def process(self, instruction: str) -> dict:
           # Parse and execute tasks
           if "add task" in instruction.lower():
               task = instruction.split("add task")[-1].strip()
               self.tasks.append(task)
               return {"status": "added", "tasks": self.tasks.copy()}
           elif "complete" in instruction.lower():
               if self.tasks:
                   completed = self.tasks.pop(0)
                   self.completed.append(completed)
                   return {"status": "completed", "task": completed}
               return {"status": "no_tasks"}
           elif "list tasks" in instruction.lower():
               return {"status": "listed", "pending": self.tasks, "completed": self.completed}
           return {"status": "unknown"}

   agent = TaskAgent()

   async def run_task_management_example():
       result = await (
           scenario("task_management")
           # Add first task
           .interact(
               "Add task: Write documentation",
               lambda inputs: agent.process(inputs)
           )
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs["status"] == "added",
               name="task_added"
           ))
           # Add second task
           .interact(
               "Add task: Review pull request",
               lambda inputs: agent.process(inputs)
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs["tasks"]) == 2,
               name="multiple_tasks"
           ))
           # Complete a task
           .interact(
               "Complete the first task",
               lambda inputs: agent.process(inputs)
           )
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs["status"] == "completed",
               name="task_completed"
           ))
           # List remaining tasks
           .interact(
               "List tasks",
               lambda inputs: agent.process(inputs)
           )
           .check(from_fn(
               lambda trace: (
                   len(trace.interactions[-1].outputs["pending"]) == 1 and
                   len(trace.interactions[-1].outputs["completed"]) == 1
               ),
               name="correct_task_state",
               success_message="Task state tracked correctly",
               failure_message="Task state incorrect"
           ))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_task_management_example())


Best Practices
--------------

**1. Check State at Each Step**

Add checks after each interaction to validate state:

.. code-block:: python

   async def run_my_scenario():
       result = await (
           scenario("my_scenario")
           .interact(...)
           .check(from_fn(lambda trace: validate_state(trace), name="state_check_1"))
           .interact(...)
           .check(from_fn(lambda trace: validate_state(trace), name="state_check_2"))
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_my_scenario())

**2. Use Descriptive Scenario Names**

Name scenarios to describe the user flow:

.. code-block:: python

   async def run_onboarding_example():
       result = await (
           scenario("user_onboarding_collect_preferences_send_confirmation")
           .interact(...)
           .check(...)
           .run()
       )
       print(f"Scenario passed: {result.passed}")

   import asyncio
   asyncio.run(run_onboarding_example())

**3. Test Both Happy and Error Paths**

Create separate scenarios for success and failure cases:

.. code-block:: python

   async def run_booking_examples():
       happy_result = await (
           scenario("booking_success")
           .interact(...)
           .check(...)
           .run()
       )

       error_result = await (
           scenario("booking_invalid_date")
           .interact(...)
           .check(...)
           .run()
       )

       print(f"Happy path passed: {happy_result.passed}")
       print(f"Error path passed: {error_result.passed}")

   import asyncio
   asyncio.run(run_booking_examples())

**4. Leverage the Full Trace**

Checks can inspect any previous interaction:

.. code-block:: python

   from_fn(
       lambda trace: (
           trace.interactions[0].inputs == "initial request" and
           trace.interactions[-1].outputs == "final response"
       ),
       name="validates_full_flow"
   )


Next Steps
----------

* Learn how to write :doc:`custom-checks` for domain-specific validation
* Explore :doc:`../tutorials/index` for complete examples
* See :doc:`single-turn` for single-interaction patterns

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

   from giskard.checks import Scenario, InteractionSpec, StringMatchingCheck

   scenario = Scenario(
       name="greeting_conversation",
       sequence=[
           # First interaction
           InteractionSpec(
               inputs="Hello",
               outputs=lambda inputs: "Hi! How can I help you?"
           ),
           StringMatchingCheck(
               name="polite_greeting",
               content="help",
               key="interactions[-1].outputs"
           ),
           # Second interaction
           InteractionSpec(
               inputs="What's the weather?",
               outputs=lambda inputs: "It's sunny and 72°F."
           ),
           StringMatchingCheck(
               name="provides_weather",
               content="sunny",
               key="interactions[-1].outputs"
           ),
       ]
   )

   result = await scenario.run()
   print(f"Scenario passed: {result.passed}")

**Key Points:**

* Components execute in sequence
* Checks can reference any interaction via the trace
* Execution stops at the first failing check
* All components share the same trace


Stateful Conversations
----------------------

Test systems that maintain conversation state:

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, from_fn, Trace

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

   scenario = Scenario(
       name="name_memory",
       sequence=[
           InteractionSpec(
               inputs="Hello, my name is Alice",
               outputs=lambda inputs: bot.chat(inputs)
           ),
           from_fn(
               lambda trace: "Alice" in trace.interactions[-1].outputs,
               name="acknowledges_name"
           ),
           InteractionSpec(
               inputs="What is my name?",
               outputs=lambda inputs: bot.chat(inputs)
           ),
           from_fn(
               lambda trace: "Alice" in trace.interactions[-1].outputs,
               name="remembers_name",
               success_message="Correctly recalled the name",
               failure_message="Failed to recall the name"
           ),
       ]
   )

   result = await scenario.run()


Testing Agent Workflows
------------------------

Test multi-step agent workflows with tool usage:

.. code-block:: python

   from giskard.agents.generators import Generator
   from giskard.checks import (
       Scenario,
       InteractionSpec,
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

   scenario = Scenario(
       name="agent_workflow",
       sequence=[
           # Agent receives task
           InteractionSpec(
               inputs="Find me a Python tutorial",
               outputs=lambda inputs: agent.run(inputs)
           ),
           # Check that agent chose appropriate tool
           from_fn(
               lambda trace: trace.interactions[-1].outputs["action"] == "search",
               name="correct_tool_choice",
               success_message="Agent selected search tool",
               failure_message="Agent selected wrong tool"
           ),
           # Validate reasoning
           LLMJudge(
               name="reasoning_quality",
               prompt="""
               Evaluate the agent's reasoning.

               Task: {{ interactions[0].inputs }}
               Thought: {{ interactions[0].outputs.thought }}
               Action: {{ interactions[0].outputs.action }}

               Return 'passed: true' if the reasoning is logical and appropriate.
               """
           ),
           # Check final answer quality
           LLMJudge(
               name="answer_quality",
               prompt="""
               Evaluate if the final answer addresses the original task.

               Task: {{ interactions[0].inputs }}
               Answer: {{ interactions[0].outputs.final_answer }}

               Return 'passed: true' if the answer is helpful and relevant.
               """
           ),
       ]
   )


Dynamic Multi-Turn Interactions
--------------------------------

Generate interactions dynamically based on previous outputs:

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, from_fn, Trace

   def chatbot(message: str, context: list = None) -> dict:
       # Your chatbot that tracks context
       return {"response": "...", "context": context or []}

   # First interaction with static inputs
   first_interaction = InteractionSpec(
       inputs="What's the capital of France?",
       outputs=lambda inputs: chatbot(inputs)
   )

   # Second interaction depends on first response
   async def generate_followup(trace: Trace):
       first_response = trace.interactions[-1].outputs["response"]
       return f"Tell me more about {first_response}"

   followup_interaction = InteractionSpec(
       inputs=generate_followup,
       outputs=lambda inputs: chatbot(inputs)
   )

   scenario = Scenario(
       name="dynamic_conversation",
       sequence=[
           first_interaction,
           from_fn(lambda trace: len(trace.interactions) == 1, name="first_complete"),
           followup_interaction,
           from_fn(lambda trace: len(trace.interactions) == 2, name="second_complete"),
       ]
   )


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

   scenario = Scenario(
       name="error_recovery",
       sequence=[
           # Send invalid input
           InteractionSpec(
               inputs="",
               outputs=lambda inputs: bot.chat(inputs)
           ),
           from_fn(
               lambda trace: "error" in trace.interactions[-1].outputs,
               name="detects_error"
           ),
           from_fn(
               lambda trace: trace.interactions[-1].outputs["response"],
               name="provides_feedback",
               success_message="Bot provided error feedback"
           ),
           # Send valid follow-up
           InteractionSpec(
               inputs="Hello",
               outputs=lambda inputs: bot.chat(inputs)
           ),
           from_fn(
               lambda trace: "error" not in trace.interactions[-1].outputs,
               name="recovers_from_error",
               success_message="System recovered successfully"
           ),
       ]
   )


Conversational RAG
------------------

Test RAG systems with follow-up questions and context references:

.. code-block:: python

   from giskard.checks import Scenario, InteractionSpec, Groundedness, from_fn

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

   scenario = Scenario(
       name="conversational_rag",
       sequence=[
           # Initial question
           InteractionSpec(
               inputs="What is machine learning?",
               outputs=lambda inputs: rag.answer(inputs)
           ),
           Groundedness(name="first_answer_grounded"),

           # Follow-up with pronoun reference
           InteractionSpec(
               inputs="What are its main applications?",
               outputs=lambda inputs: rag.answer(inputs)
           ),
           Groundedness(name="followup_grounded"),
           from_fn(
               lambda trace: len(trace.interactions) == 2,
               name="maintains_context",
               success_message="System handled follow-up correctly"
           ),

           # Another follow-up
           InteractionSpec(
               inputs="Can you explain the first one in detail?",
               outputs=lambda inputs: rag.answer(inputs)
           ),
           Groundedness(name="second_followup_grounded"),
       ]
   )


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

   scenario = Scenario(
       name="task_management",
       sequence=[
           # Add first task
           InteractionSpec(
               inputs="Add task: Write documentation",
               outputs=lambda inputs: agent.process(inputs)
           ),
           from_fn(
               lambda trace: trace.interactions[-1].outputs["status"] == "added",
               name="task_added"
           ),

           # Add second task
           InteractionSpec(
               inputs="Add task: Review pull request",
               outputs=lambda inputs: agent.process(inputs)
           ),
           from_fn(
               lambda trace: len(trace.interactions[-1].outputs["tasks"]) == 2,
               name="multiple_tasks"
           ),

           # Complete a task
           InteractionSpec(
               inputs="Complete the first task",
               outputs=lambda inputs: agent.process(inputs)
           ),
           from_fn(
               lambda trace: trace.interactions[-1].outputs["status"] == "completed",
               name="task_completed"
           ),

           # List remaining tasks
           InteractionSpec(
               inputs="List tasks",
               outputs=lambda inputs: agent.process(inputs)
           ),
           from_fn(
               lambda trace: (
                   len(trace.interactions[-1].outputs["pending"]) == 1 and
                   len(trace.interactions[-1].outputs["completed"]) == 1
               ),
               name="correct_task_state",
               success_message="Task state tracked correctly",
               failure_message="Task state incorrect"
           ),
       ]
   )


Best Practices
--------------

**1. Check State at Each Step**

Add checks after each interaction to validate state:

.. code-block:: python

   sequence=[
       InteractionSpec(...),
       from_fn(lambda trace: validate_state(trace), name="state_check_1"),
       InteractionSpec(...),
       from_fn(lambda trace: validate_state(trace), name="state_check_2"),
   ]

**2. Use Descriptive Scenario Names**

Name scenarios to describe the user flow:

.. code-block:: python

   scenario = Scenario(
       name="user_onboarding_collect_preferences_send_confirmation",
       sequence=[...]
   )

**3. Test Both Happy and Error Paths**

Create separate scenarios for success and failure cases:

.. code-block:: python

   happy_path = Scenario(name="booking_success", sequence=[...])
   error_path = Scenario(name="booking_invalid_date", sequence=[...])

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

===============
Chatbot Testing
===============

This tutorial covers testing conversational AI systems, including context handling, tone consistency, and multi-turn dialogue flows.

Overview
--------

We'll test a chatbot that:

* Maintains conversation context
* Handles different conversation types (casual, support, sales)
* Manages user preferences and information
* Provides appropriate responses based on context

Our tests will validate:

* Context retention across turns
* Response quality and tone
* Handling of conversation flow
* Edge cases and error scenarios


Building a Chatbot
-------------------

First, let's create a simple chatbot:

.. code-block:: python

   from typing import Optional, Literal
   from pydantic import BaseModel

   class Message(BaseModel):
       role: Literal["user", "assistant", "system"]
       content: str

   class ConversationContext(BaseModel):
       user_name: Optional[str] = None
       user_email: Optional[str] = None
       conversation_type: str = "casual"
       topic: Optional[str] = None

   class ChatResponse(BaseModel):
       message: str
       context: ConversationContext
       suggested_actions: list[str] = []

   class SimpleChatbot:
       def __init__(self, personality: str = "friendly"):
           self.personality = personality
           self.history: list[Message] = []
           self.context = ConversationContext()

       def chat(self, user_message: str) -> ChatResponse:
           """Process user message and generate response."""
           self.history.append(Message(role="user", content=user_message))

           # Update context based on message
           self._update_context(user_message)

           # Generate response
           response_text = self._generate_response(user_message)

           self.history.append(Message(role="assistant", content=response_text))

           return ChatResponse(
               message=response_text,
               context=self.context.model_copy(),
               suggested_actions=self._suggest_actions()
           )

       def _update_context(self, message: str):
           """Extract and update context information."""
           message_lower = message.lower()

           # Extract name
           if "my name is" in message_lower or "i'm" in message_lower:
               words = message.split()
               for i, word in enumerate(words):
                   if word.lower() in ["is", "i'm", "im"] and i + 1 < len(words):
                       self.context.user_name = words[i + 1].strip(",.!?")
                       break

           # Extract email
           if "@" in message:
               import re
               emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)
               if emails:
                   self.context.user_email = emails[0]

           # Detect conversation type
           if any(word in message_lower for word in ["help", "support", "problem", "issue"]):
               self.context.conversation_type = "support"
           elif any(word in message_lower for word in ["buy", "purchase", "price", "cost"]):
               self.context.conversation_type = "sales"

       def _generate_response(self, message: str) -> str:
           """Generate appropriate response based on context."""
           # Greeting
           if any(greeting in message.lower() for greeting in ["hello", "hi", "hey"]):
               if self.context.user_name:
                   return f"Hello {self.context.user_name}! How can I help you today?"
               return "Hello! How can I help you today?"

           # Name recall
           if "what is my name" in message.lower() or "do you know my name" in message.lower():
               if self.context.user_name:
                   return f"Yes, your name is {self.context.user_name}."
               return "I don't believe you've told me your name yet."

           # Support queries
           if self.context.conversation_type == "support":
               return "I understand you need help. Let me connect you with our support team. Could you describe your issue in detail?"

           # Sales queries
           if self.context.conversation_type == "sales":
               return "I'd be happy to help you find the right product. What are you looking for?"

           # Default
           return "I understand. Could you tell me more about that?"

       def _suggest_actions(self) -> list[str]:
           """Suggest next actions based on context."""
           actions = []

           if not self.context.user_name:
               actions.append("introduce_yourself")

           if self.context.conversation_type == "support" and not self.context.user_email:
               actions.append("provide_email")

           return actions


Test 1: Basic Conversation Flow
--------------------------------

Test a simple greeting and name exchange:

.. code-block:: python

   from giskard.checks import scenario, from_fn, StringMatchingCheck

   bot = SimpleChatbot()

   import asyncio

   async def run_basic_conversation_example():
       result = await (
           scenario("greeting_and_introduction")
           # User greets
           .interact(
               "Hello",
               lambda inputs: bot.chat(inputs)
           )
           .check(StringMatchingCheck(
               name="polite_greeting",
               content="help",
               key="interactions[-1].outputs.message"
           ))
           # User introduces themselves
           .interact(
               "My name is Alice",
               lambda inputs: bot.chat(inputs)
           )
           .check(StringMatchingCheck(
               name="acknowledges_name",
               content="Alice",
               key="interactions[-1].outputs.message"
           ))
           .check(from_fn(
               lambda trace: trace.interactions[-1].outputs.context.user_name == "Alice",
               name="stored_name",
               success_message="Chatbot stored the user's name",
               failure_message="Chatbot failed to store name"
           ))
           # Verify name recall
           .interact(
               "What is my name?",
               lambda inputs: bot.chat(inputs)
           )
           .check(StringMatchingCheck(
               name="recalls_name",
               content="Alice",
               key="interactions[-1].outputs.message"
           ))
           .run()
       )
       assert result.passed
       print("✓ Basic conversation flow example passed")

   asyncio.run(run_basic_conversation_example())


Test 2: Context Switching
--------------------------

Verify the chatbot handles different conversation types:

.. code-block:: python

   from giskard.agents.generators import Generator
   from giskard.checks import (
       scenario,
       LLMJudge,
       EqualityCheck,
       set_default_generator
   )

   set_default_generator(Generator(model="openai/gpt-4o-mini"))

   bot = SimpleChatbot()

   async def run_context_switching_example():
       result = await (
           scenario("context_switching")
           # Start with casual conversation
           .interact(
               "Hi there!",
               lambda inputs: bot.chat(inputs)
           )
           .check(EqualityCheck(
               name="casual_context",
               expected="casual",
               key="interactions[-1].outputs.context.conversation_type"
           ))
           # Switch to support
           .interact(
               "I'm having a problem with my account",
               lambda inputs: bot.chat(inputs)
           )
           .check(EqualityCheck(
               name="support_context",
               expected="support",
               key="interactions[-1].outputs.context.conversation_type"
           ))
           .check(LLMJudge(
               name="support_tone",
               prompt="""
               Evaluate if the response is appropriate for a support inquiry.

               User: {{ interactions[1].inputs }}
               Assistant: {{ interactions[1].outputs.message }}

               The response should be helpful and professional.
               Return 'passed: true' if appropriate.
               """
           ))
           # Switch to sales
           .interact(
               "How much does it cost?",
               lambda inputs: bot.chat(inputs)
           )
           .check(EqualityCheck(
               name="sales_context",
               expected="sales",
               key="interactions[-1].outputs.context.conversation_type"
           ))
           .run()
       )
       print(f"Context switching example passed: {result.passed}")

   import asyncio
   asyncio.run(run_context_switching_example())


Test 3: Response Quality and Tone
----------------------------------

Evaluate response quality using LLM-as-a-judge:

.. code-block:: python

   from giskard.checks import scenario, LLMJudge

   bot = SimpleChatbot(personality="professional")

   async def run_response_quality_example():
       result = await (
           scenario("response_quality_test")
           .interact(
               "I need help understanding your pricing",
               lambda inputs: bot.chat(inputs)
           )
           .check(LLMJudge(
               name="tone_check",
               prompt="""
               Evaluate the tone of this chatbot response.

               User message: {{ inputs }}
               Bot response: {{ outputs.message }}
               Expected personality: professional

               Check:
               1. Is the tone professional?
               2. Is it helpful and clear?
               3. Does it address the user's question?

               Return 'passed: true' if tone is appropriate.
               """
           ))
           .check(LLMJudge(
               name="completeness",
               prompt="""
               Evaluate if the response is complete.

               User: {{ inputs }}
               Bot: {{ outputs.message }}

               Does the response:
               1. Acknowledge the user's question?
               2. Provide next steps or information?
               3. Offer to help further?

               Return 'passed: true' if response is complete.
               """
           ))
           .run()
       )
       print(f"Response quality example passed: {result.passed}")

   import asyncio
   asyncio.run(run_response_quality_example())


Test 4: Information Extraction and Storage
-------------------------------------------

Test the chatbot's ability to extract and remember user information:

.. code-block:: python

   from giskard.checks import scenario, from_fn, EqualityCheck

   bot = SimpleChatbot()

   async def run_information_collection_example():
       result = await (
           scenario("information_collection")
           # Collect name
           .interact(
               "Hi, I'm Bob Johnson",
               lambda inputs: bot.chat(inputs)
           )
           .check(EqualityCheck(
               name="extracted_name",
               expected="Bob",
               key="interactions[-1].outputs.context.user_name"
           ))
           # Collect email
           .interact(
               "My email is bob.johnson@example.com",
               lambda inputs: bot.chat(inputs)
           )
           .check(EqualityCheck(
               name="extracted_email",
               expected="bob.johnson@example.com",
               key="interactions[-1].outputs.context.user_email"
           ))
           # Verify information persists
           .interact(
               "Can you remind me what information you have about me?",
               lambda inputs: bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: (
                   trace.interactions[-1].outputs.context.user_name == "Bob" and
                   trace.interactions[-1].outputs.context.user_email == "bob.johnson@example.com"
               ),
               name="information_persisted",
               success_message="Chatbot retained user information",
               failure_message="Chatbot lost user information"
           ))
           .run()
       )
       print(f"Information collection example passed: {result.passed}")

   import asyncio
   asyncio.run(run_information_collection_example())


Test 5: Edge Cases and Error Handling
--------------------------------------

Test how the chatbot handles unusual inputs:

.. code-block:: python

   from giskard.checks import scenario, from_fn, LLMJudge

   bot = SimpleChatbot()

   async def run_edge_case_examples():
       # Test empty input
       result_empty = await (
           scenario("empty_input_handling")
           .interact(
               "",
               lambda inputs: bot.chat(inputs) if inputs else ChatResponse(
                   message="I didn't receive a message. Could you try again?",
                   context=bot.context
               )
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.message) > 0,
               name="provides_response",
               success_message="Bot provided a response to empty input"
           ))
           .run()
       )

       # Test very long input
       result_long = await (
           scenario("long_input_handling")
           .interact(
               "Hello " * 1000,
               lambda inputs: bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: len(trace.interactions[-1].outputs.message) > 0,
               name="handles_long_input",
               success_message="Bot handled long input"
           ))
           .run()
       )

       # Test gibberish
       result_gibberish = await (
           scenario("gibberish_handling")
           .interact(
               "asdfghjkl qwertyuiop zxcvbnm",
               lambda inputs: bot.chat(inputs)
           )
           .check(LLMJudge(
               name="graceful_response",
               prompt="""
               Evaluate if the bot handles gibberish gracefully.

               User input: {{ inputs }}
               Bot response: {{ outputs.message }}

               The bot should:
               1. Not error out
               2. Provide a polite response
               3. Maybe ask for clarification

               Return 'passed: true' if handled well.
               """
           ))
           .run()
       )

       print(f"Empty input handling: {result_empty.passed}")
       print(f"Long input handling: {result_long.passed}")
       print(f"Gibberish handling: {result_gibberish.passed}")

   import asyncio
   asyncio.run(run_edge_case_examples())


Test 6: Conversation State Management
--------------------------------------

Test complex stateful interactions:

.. code-block:: python

   from giskard.checks import scenario, from_fn, LLMJudge, StringMatchingCheck

   class StatefulChatbot(SimpleChatbot):
       def __init__(self):
           super().__init__()
           self.awaiting_confirmation = False
           self.pending_action = None

       def chat(self, user_message: str) -> ChatResponse:
           # Handle confirmations
           if self.awaiting_confirmation:
               if user_message.lower() in ["yes", "confirm", "ok", "sure"]:
                   response_text = f"Great! I'll proceed with {self.pending_action}."
                   self.awaiting_confirmation = False
                   self.pending_action = None
               elif user_message.lower() in ["no", "cancel", "nevermind"]:
                   response_text = "Okay, I won't do that. What else can I help with?"
                   self.awaiting_confirmation = False
                   self.pending_action = None
               else:
                   response_text = "I'm waiting for your confirmation. Please say yes or no."

               self.history.append(Message(role="assistant", content=response_text))
               return ChatResponse(message=response_text, context=self.context)

           # Check for actions requiring confirmation
           if "delete" in user_message.lower() or "cancel" in user_message.lower():
               self.awaiting_confirmation = True
               self.pending_action = "deletion"
               response_text = "Are you sure you want to proceed? Please confirm."

               self.history.append(Message(role="assistant", content=response_text))
               return ChatResponse(message=response_text, context=self.context)

           return super().chat(user_message)

   stateful_bot = StatefulChatbot()

   async def run_confirmation_flow_example():
       result = await (
           scenario("confirmation_flow")
           # Request action requiring confirmation
           .interact(
               "I want to delete my account",
               lambda inputs: stateful_bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: stateful_bot.awaiting_confirmation,
               name="requested_confirmation",
               success_message="Bot requested confirmation"
           ))
           .check(StringMatchingCheck(
               name="asks_confirmation",
               content="confirm",
               key="interactions[-1].outputs.message"
           ))
           # User cancels
           .interact(
               "No, nevermind",
               lambda inputs: stateful_bot.chat(inputs)
           )
           .check(from_fn(
               lambda trace: not stateful_bot.awaiting_confirmation,
               name="cleared_confirmation_state",
               success_message="Bot cleared confirmation state"
           ))
           .check(LLMJudge(
               name="acknowledged_cancellation",
               prompt="""
               Check if the bot acknowledged the cancellation appropriately.

               User: {{ interactions[1].inputs }}
               Bot: {{ interactions[1].outputs.message }}

               Return 'passed: true' if the bot handled cancellation well.
               """
           ))
           .run()
       )
       print(f"Confirmation flow example passed: {result.passed}")

   import asyncio
   asyncio.run(run_confirmation_flow_example())


Complete Chatbot Test Suite
----------------------------

Combine all tests into a comprehensive suite:

.. code-block:: python

   import asyncio
   from giskard.checks import Scenario

   class ChatbotTestSuite:
       def __init__(self, chatbot):
           self.chatbot = chatbot
           self.scenarios = []

       def add_scenario(self, test_scenario: Scenario):
           self.scenarios.append(test_scenario)

       async def run_all(self):
           """Run all scenarios and report results."""
           print("🤖 Running Chatbot Test Suite\n")

           results = []

           # Run scenarios
           for test_scenario in self.scenarios:
               print(f"  Running scenario: {test_scenario.name}")
               result = await test_scenario.run()
               results.append(("scenario", test_scenario.name, result))

           # Report
           self._print_report(results)

           return results

       def _print_report(self, results):
           total = len(results)
           passed = sum(1 for _, _, r in results if r.passed)

           print(f"\n{'='*70}")
           print(f"Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
           print(f"{'='*70}\n")

           for test_type, name, result in results:
               status = "✓" if result.passed else "✗"
               print(f"{status} [{test_type}] {name}")

               if not result.passed:
                   if hasattr(result, 'results'):
                       for check_result in result.results:
                           if not check_result.passed:
                               print(f"    → {check_result.name}: {check_result.message}")

   # Usage
   async def main():
       bot = SimpleChatbot()
       suite = ChatbotTestSuite(bot)

       # Add all scenarios
       # suite.add_scenario(...)

       await suite.run_all()

   asyncio.run(main())


Best Practices
--------------

**1. Test Conversation Flows Holistically**

Don't just test individual responses—test complete conversation flows:

.. code-block:: python

   async def run_complete_support_flow():
       result = await (
           scenario("complete_support_flow")
           # Greeting -> Problem statement -> Information collection -> Resolution
           .interact(...)
           .check(...)
           .run()
       )
       print(f"Complete support flow passed: {result.passed}")

   import asyncio
   asyncio.run(run_complete_support_flow())

**2. Validate Context Retention**

Ensure the chatbot remembers important information:

.. code-block:: python

   from_fn(
       lambda trace: (
           trace.interactions[-1].outputs.context.user_name and
           trace.interactions[-1].outputs.context.user_email
       ),
       name="retains_user_info"
   )

**3. Test Tone Consistency**

Use LLM judges to verify tone remains consistent:

.. code-block:: python

   LLMJudge(
       name="consistent_tone",
       prompt="""
       Evaluate tone consistency across responses.

       {% for interaction in interactions %}
       Response {{ loop.index }}: {{ interaction.outputs.message }}
       {% endfor %}

       Return 'passed: true' if tone is consistent.
       """
   )

**4. Handle Edge Cases**

Test with unusual inputs:

- Empty messages
- Very long messages
- Special characters
- Rapid topic changes
- Contradictory statements


Next Steps
----------

* See :doc:`content-moderation` for safety and filtering
* Explore :doc:`testing-agents` for tool-using chatbots
* Review :doc:`../ai-testing/multi-turn` for complex flows

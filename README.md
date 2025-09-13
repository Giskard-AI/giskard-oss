<p align="center">
  <img alt="giskardlogo" src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/giskard_logo.png#gh-light-mode-only">
  <img alt="giskardlogo" src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/giskard_logo_green.png#gh-dark-mode-only">
</p>
<h1 align="center" weight='300' >The Evaluation & Testing framework for AI systems</h1>
<h3 align="center" weight='300' >Control risks of performance, bias and security issues in AI systems</h3>
<div align="center">

  [![GitHub release](https://img.shields.io/github/v/release/Giskard-AI/giskard)](https://github.com/Giskard-AI/giskard/releases)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Giskard-AI/giskard/blob/main/LICENSE)
  [![Downloads](https://static.pepy.tech/badge/giskard/month)](https://pepy.tech/project/giskard)
  [![CI](https://github.com/Giskard-AI/giskard/actions/workflows/build-python.yml/badge.svg?branch=main)](https://github.com/Giskard-AI/giskard/actions/workflows/build-python.yml?query=branch%3Amain)
  [![Giskard on Discord](https://img.shields.io/discord/939190303397666868?label=Discord)](https://gisk.ar/discord)

  <a rel="me" href="https://fosstodon.org/@Giskard"></a>

</div>
<h3 align="center">
   <a href="https://docs.giskard.ai/en/stable/getting_started/index.html"><b>Docs</b></a> &bull;
  <a href="https://www.giskard.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readmeblog"><b>Website</b></a> &bull;
  <a href="https://gisk.ar/discord"><b>Community</b></a>
 </h3>
<br />

## Install Giskard 🐢
Install the latest version of Giskard from PyPi using pip:
```sh
pip install "giskard[llm]" -U
```
We officially support Python 3.9, 3.10 and 3.11.
## Try in Colab 📙
[Open Colab notebook](https://colab.research.google.com/github/giskard-ai/giskard/blob/main/docs/getting_started/quickstart/quickstart_llm.ipynb)

______________________________________________________________________

Giskard is an open-source Python library that **automatically detects performance, bias & security issues in AI applications**. The library covers LLM-based applications such as RAG agents, all the way to traditional ML models for tabular data.

## Scan: Automatically assess your LLM-based agents for performance, bias & security issues ⤵️

Issues detected include:
- Hallucinations
- Harmful content generation
- Prompt injection
- Robustness issues
- Sensitive information disclosure
- Stereotypes & discrimination
- many more...

<p align="center">
  <img src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/scan_updates.gif" alt="Scan Example" width="800">
</p>

## RAG Evaluation Toolkit (RAGET): Automatically generate evaluation datasets & evaluate RAG application answers ⤵️

If you're testing a RAG application, you can get an even more in-depth assessment using **RAGET**, Giskard's RAG Evaluation Toolkit.

- **RAGET** can generate automatically a list of `question`, `reference_answer` and `reference_context` from the knowledge base of the RAG. You can then use this generated test set to evaluate your RAG agent.
- **RAGET** computes scores *for each component of the RAG agent*. The scores are computed by aggregating the correctness of the agent’s answers on different question types.

  - Here is the list of components evaluated with **RAGET**:
    - `Generator`: the LLM used inside the RAG to generate the answers
    - `Retriever`: fetch relevant documents from the knowledge base according to a user query
    - `Rewriter`: rewrite the user query to make it more relevant to the knowledge base or to account for chat history
    - `Router`: filter the query of the user based on his intentions
    - `Knowledge Base`: the set of documents given to the RAG to generate the answers

<p align="center">
  <img src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/RAGET_updated.gif" alt="Test Suite Example" width="800">
</p>


Giskard works with any model, in any environment and integrates seamlessly with your favorite tools ⤵️ <br/>

<p align="center">
  <img width='600' src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/tools_updated.png">
</p>
<br/>

Looking for solutions to evaluate computer vision models? Check out [giskard-vision](https://github.com/Giskard-AI/giskard-vision), a library dedicated for computer vision tasks.

# Contents

- 🤸‍♀️ **[Quickstart](#quickstart)**
    - **1**. 🏗️ [Build a LLM agent](#build-a-llm-agent)
    - **2**. 🔎 [Scan your model for issues](#scan-your-model-for-issues)
    - **3**. 🪄 [Automatically generate an evaluation dataset for your RAG applications](#automatically-generate-an-evaluation-dataset-for-your-rag-applications)
- 👋 **[Community](#community)**

<h1 id="quickstart">🤸‍♀️ Quickstart</h1>

<h2 id="build-a-llm-agent">1. 🏗️ Build a LLM agent</h2>

Let's build an agent that answers questions about climate change, based on the 2023 Climate Change Synthesis Report by the IPCC.

Before starting let's install the required libraries:
```sh
pip install langchain langchain-community langchain-openai tiktoken "pypdf<=3.17.0"
```


```python
from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prepare vector store (FAISS) with IPPC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")
db = FAISS.from_documents(loader.load_and_split(text_splitter), OpenAIEmbeddings())

# Prepare QA chain
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)
```

<h2 id="scan-your-model-for-issues">2. 🔎 Scan your model for issues</h2>

Next, wrap your agent to prepare it for Giskard's scan:

```python
import giskard
import pandas as pd

def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [climate_qa_chain.run({"query": question}) for question in df["question"]]

# Don’t forget to fill the `name` and `description`: they are used by Giskard
# to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Climate Change Question Answering",
    description="This model answers any question about climate change based on IPCC reports",
    feature_names=["question"],
)
```

✨✨✨Then run Giskard's magical scan✨✨✨
```python
scan_results = giskard.scan(giskard_model)
```
Once the scan completes, you can display the results directly in your notebook:

```python
display(scan_results)

# Or save it to a file
scan_results.to_html("scan_results.html")
```

*If you're facing issues, check out our [docs](https://docs.giskard.ai/en/stable/open_source/scan/scan_llm/index.html) for more information.*

<h2 id="automatically-generate-an-evaluation-dataset-for-your-rag-applications">3. 🪄 Automatically generate an evaluation dataset for your RAG applications</h2>

If the scan found issues in your model, you can automatically extract an evaluation dataset based on the issues found:

```python
test_suite = scan_results.generate_test_suite("My first test suite")
```

By default, RAGET automatically generates 6 different question types (these can be selected if needed, see advanced question generation). The total number of questions is divided equally between each question type. To make the question generation more relevant and accurate, you can also provide a description of your agent.

```python

from giskard.rag import generate_testset, KnowledgeBase

# Load your data and initialize the KnowledgeBase
df = pd.read_csv("path/to/your/knowledge_base.csv")

knowledge_base = KnowledgeBase.from_pandas(df, columns=["column_1", "column_2"])

# Generate a testset with 10 questions & answers for each question types (this will take a while)
testset = generate_testset(
    knowledge_base,
    num_questions=60,
    language='en',  # optional, we'll auto detect if not provided
    agent_description="A customer support chatbot for company X", # helps generating better questions
)
```

Depending on how many questions you generate, this can take a while. Once you’re done, you can save this generated test set for future use:

```python
# Save the generated testset
testset.save("my_testset.jsonl")
```
You can easily load it back

```python
from giskard.rag import QATestset

loaded_testset = QATestset.load("my_testset.jsonl")

# Convert it to a pandas dataframe
df = loaded_testset.to_pandas()
```

Here’s an example of a generated question:

| question                               | reference_context                                                                                                                                                     | reference_answer                                             | metadata                                               |
|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------|
| For which countries can I track my shipping? | Document 1: We offer free shipping on all orders over $50. For orders below $50, we charge a flat rate of $5.99. We offer shipping services to customers residing in all 50 states of the US, in addition to providing delivery options to Canada and Mexico. Document 2: Once your purchase has been successfully confirmed and shipped, you will receive a confirmation email containing your tracking number. You can simply click on the link provided in the email or visit our website’s order tracking page. | We ship to all 50 states in the US, as well as to Canada and Mexico. We offer tracking for all our shippings. | `{"question_type": "simple", "seed_document_id": 1, "topic": "Shipping policy"}` |

Each row of the test set contains 5 columns:

- `question`: the generated question
- `reference_context`: the context that can be used to answer the question
- `reference_answer`: the answer to the question (generated with GPT-4)
- `conversation_history`: not shown in the table above, contain the history of the conversation with the agent as a list, only relevant for conversational question, otherwise it contains an empty list.
- `metadata`: a dictionary with various metadata about the question, this includes the question_type, seed_document_id the id of the document used to generate the question and the topic of the question

<h1 id="community">👋 Community</h1>

We welcome contributions from the AI community! Read this [guide](./CONTRIBUTING.md) to get started, and join our thriving community on [Discord](https://gisk.ar/discord).

🌟 [Leave us a star](https://github.com/Giskard-AI/giskard), it helps the project to get discovered by others and keeps us motivated to build awesome open-source tools! 🌟

❤️ If you find our work useful, please consider [sponsoring us](https://github.com/sponsors/Giskard-AI) on GitHub. With a monthly sponsoring, you can get a sponsor badge, display your company in this readme, and get your bug reports prioritized. We also offer one-time sponsoring if you want us to get involved in a consulting project, run a workshop, or give a talk at your company.

<h2 id="sponsors">💚 Current sponsors</h1>

We thank the following companies which are sponsoring our project with monthly donations:

**[Lunary](https://lunary.ai/)**

<img src="https://lunary.ai/logo-blue-bg.svg" alt="Lunary logo" width="100"/>

**[Biolevate](https://www.biolevate.com/)**

<img src="https://awsmp-logos.s3.amazonaws.com/seller-wgamx5z6umune/2d10badd2ccac49699096ea7fb986b98.png" alt="Biolevate logo" width="400"/>


## Contribution
This is my first contribution by @sohamgupta779-art 🚀
## Contribution
This is my first contribution by @sohamgupta779-art 🚀
xyz

## ✨ Features

### 1. 📄 AI Resume Analyzer
* **Workflow:** Automated Application Tracking System (ATS) matching tool.
* **Mechanism:** Accepts an extracted resume text block and evaluates it against an uploaded target Job Description (JD). 
* **NLP Technique:** Zero-shot descriptive prompting with systemic output layout profiling.
* **Output:** Generates a structured breakdown with a percentage fit score, top strengths, technical skill gaps, and keyword optimizations.

### 2. 💬 Chat with PDF using RAG (Retrieval-Augmented Generation)
* **Workflow:** Context-aware document conversational engine.
* **Mechanism:** 1. Parses uploaded PDFs into standard unicode text strings via `pdfplumber`.
  2. Splits text sections into accurate semantic layers using `RecursiveCharacterTextSplitter` (700 token chunk size, 100 token overlap).
  3. Vectors are derived using the `nomic-embed-text` embedding engine.
  4. Stores semantic records locally in a high-speed memory-mapped database (`ChromaDB`).
* **NLP Technique:** Document retrieval QA pipelines utilizing LangChain orchestration layers to mitigate LLM hallucinations.

### 3. 🔬 Research Paper Summarizer
* **Workflow:** Abstractive text compression system optimized for scientific literature.
* **Mechanism:** Ingests long, structural multi-page text blocks and applies context-limit safeguards (max 15,000 characters to protect system memory allocations).
* **NLP Technique:** Conditional prompt steering allowing selection of multiple distillation levels (*Executive Summary*, *Detailed Analysis*, or *TL;DR Bullet Points*).

---

## 🛠️ System Requirements & Prerequisites

* **Operating System:** macOS (Optimized for Apple Silicon M1/M2/M3/M4 chips via Metal hardware acceleration).
* **Python Runtime:** Python `3.9` to `3.11` recommended.
* **Local Inference Backend:** Ollama desktop client setup.

---

## 📥 Setup & Local Installation Step-by-Step

Follow these commands to deploy the workspace on your local Mac environment:

### Step 1: Install Ollama & Pull the Weights
1. Download and open the package application from [Ollama's Official Website](https://ollama.com).
2. Start the core engine. Then launch your Mac Terminal application and download the necessary weights:
   ```bash
   # Pull the lightweight, high-performance general language model
   ollama pull llama3.2

   # Pull the localized open-source text embedding engine
   ollama pull nomic-embed-text
### Step 2: Clone and Navigate to the Directory
Create your project environment folder on your machine:
``
mkdir local_nlp_dashboard
cd local_nlp_dashboard
``
### Step 3: Configure your Virtual Environment
Construct and activate an isolated development container:

``
python3 -m venv venv
source venv/bin/activate
``
### Step 4: Write Dependency Structure
Create a requirements.txt file inside your root directory and paste the following structural package modules:

``
streamlit==1.32.0
langchain==0.1.12
langchain-community==0.0.28
chromadb==0.4.24
pypdf==4.1.0
pdfplumber==0.11.0
pydantic==2.6.4
``

Install the package requirements via pip:

``
pip install -r requirements.txt
``

### Step 5: Application Deployment Source Code
Create a file named app.py in your working directory and add the fully integrated application source script.
🚀 Execution Guide
Run the Streamlit application runtime server locally:

``
streamlit run app.py
``
The build script will compile and automatically deploy a web-view interface in your default system browser at: http://localhost:8501

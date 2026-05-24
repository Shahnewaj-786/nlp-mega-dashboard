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

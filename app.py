import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- UI Layout Design ---
st.set_page_config(page_title="Advanced NLP Hub", layout="wide", page_icon="🚀")
st.title("🤖 Advanced Local NLP Project Workspace")

# Global instances shared safely across sections
llm = ChatOllama(model="llama3", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- App Switcher Dashboard Module ---
with st.sidebar:
    st.header("🎯 Feature Modules")
    app_mode = st.radio(
        "Choose Your System Tool:",
        ["AI Resume Analyzer", "Chat with PDF (RAG)", "Research Paper Summarizer"]
    )
    st.markdown("---")
    st.info("⚡ Framework status: 100% Local (Mac Metal Pipeline Active)")


# --- Utility Functions for RAG and Text Extraction ---
def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return " ".join([doc.page_content for doc in docs]), docs


# =====================================================================
# MODULE 1: AI RESUME ANALYZER
# =====================================================================
if app_mode == "AI Resume Analyzer":
    st.header("📋 Automated Resume Scanning & Gap Analysis")
    st.write("Upload a Candidate CV against a target Job Description to run comparative semantic parsing.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf", key="resume_uploader")
    with col2:
        job_description = st.text_area("Target Job Specification requirements:", height=150)

    if st.button("Execute Compliance Matching Analysis") and uploaded_resume and job_description:
        with st.spinner("Extracting parameters..."):
            with open("temp_resume.pdf", "wb") as f:
                f.write(uploaded_resume.getbuffer())

            resume_text, _ = extract_text_from_pdf("temp_resume.pdf")

            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an expert HR Data Scientist parsing application tracking systems. Assess cross-compatibility accurately."),
                ("human",
                 f"Compare this Candidate Profile Text:\n{resume_text}\n\nAgainst this Targeted Role Specification:\n{job_description}\n\nProvide a structured summary containing:\n1. Missing technical skills keywords.\n2. Overall matching percentage score.\n3. Strategic modification recommendations.")
            ])

            eval_chain = analysis_prompt | llm
            report = eval_chain.invoke({})
            st.subheader("📊 Evaluation Metrics Report")
            st.write(report.content)

# =====================================================================
# MODULE 2: CHAT WITH PDF USING RAG
# =====================================================================
elif app_mode == "Chat with PDF (RAG)":
    st.header("💬 Knowledge Retrieval Bot Engine")

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    uploaded_rag_file = st.sidebar.file_uploader("Upload Knowledge Base PDF", type="pdf", key="rag_uploader")
    if uploaded_rag_file:
        with open("temp_rag.pdf", "wb") as f:
            f.write(uploaded_rag_file.getbuffer())
        if st.sidebar.button("Index Vector Database"):
            with st.spinner("Chunking tokens into embeddings database..."):
                _, docs = extract_text_from_pdf("temp_rag.pdf")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = splitter.split_documents(docs)
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.session_state.retriever = vectorstore.as_retriever()
                st.sidebar.success("Database Loaded Successfully!")

    # Display thread history
    for msg in st.session_state.rag_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Query indexed context window parameters"):
        if "retriever" not in st.session_state:
            st.error("Engine missing vector maps. Please upload and index a file via the sidebar module.")
        else:
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Resolve queries utilizing retrieved source contexts strictly. If ambiguous context occurs, flag it.\n\n{context}"),
                ("human", "{input}"),
            ])

            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)

            with st.chat_message("assistant"):
                response = rag_chain.invoke({"input": prompt})
                st.write(response["answer"])
                st.session_state.rag_messages.append({"role": "assistant", "content": response["answer"]})

# =====================================================================
# MODULE 3: RESEARCH PAPER SUMMARIZER
# =====================================================================
elif app_mode == "Research Paper Summarizer":
    st.header("🔬 Academic Document Text Compactor")
    st.write("Distill foundational abstracts, experimental methods, and terminal figures from deep publications.")

    uploaded_paper = st.file_uploader("Upload Manuscript Target File (PDF)", type="pdf", key="paper_uploader")
    if uploaded_paper and st.button("Generate Context Summarization Structuring"):
        with st.spinner("Extracting structural data targets..."):
            with open("temp_paper.pdf", "wb") as f:
                f.write(uploaded_paper.getbuffer())

            paper_text, _ = extract_text_from_pdf("temp_paper.pdf")

            summary_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an Elite Academic Peer Reviewer. Synthesize raw manuscript data into precise domain tracking summaries."),
                ("human",
                 f"Parse this academic document payload:\n{paper_text}\n\nGenerate structured technical tracking insights outlining:\n- **Core Novelty / Value Proposition**\n- **Methodology & Architecture Overview**\n- **Primary Quantitative Empirical Discoveries**")
            ])

            summary_chain = summary_prompt | llm
            summary_output = summary_chain.invoke({})
            st.subheader("📝 Executive Structural Synthesis Summary")
            st.markdown(summary_output.content)
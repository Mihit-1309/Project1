import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------- Initialize Models ----------
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None
pdf_text_cache = ""


# ---------- Step 1: Load and Process PDF ----------
def build_vectorstore(file_path: str):
    """Load PDF and create embeddings for retrieval."""
    global vectorstore, pdf_text_cache

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    pdf_text_cache = "\n".join([c.page_content for c in chunks])

    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)


# ---------- Step 2: Main Intelligent Tool ----------
def process_prompt(user_prompt: str) -> str:
    """Intelligently decide whether to summarize, generate questions, or answer."""
    global pdf_text_cache, vectorstore

    if not pdf_text_cache or vectorstore is None:
        return "⚠️ Please upload a PDF first."

    # Detect user intent using LLM itself
    intent_prompt = PromptTemplate(
        input_variables=["prompt"],
        template=(
            "Classify the intent of the following user message as one of: "
            "'summarize', 'question_generation', or 'qa'. "
            "Just return one word.\n\nUser message: {prompt}"
        )
    )
    intent_chain = LLMChain(llm=llm, prompt=intent_prompt)
    intent = intent_chain.run(prompt=user_prompt).strip().lower()

    # ---------- RAG QA ----------
    if "qa" in intent or "answer" in intent or "rag" in intent:
        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa.invoke({"query": user_prompt})
        return f"🧠 **Answer:** {result['result']}"

    # ---------- Summarization ----------
    elif "summarize" in intent:
        summarize_prompt = PromptTemplate(
            input_variables=["context"],
            template="Summarize the following document clearly:\n\n{context}"
        )
        summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
        summary = summarize_chain.run(context=pdf_text_cache)
        return f"📝 **Summary:**\n{summary}"

    # ---------- Question Generation ----------
    elif "question" in intent:
        question_prompt = PromptTemplate(
            input_variables=["context"],
            template=(
                "Generate 5 relevant questions from this document:\n\n{context}"
            )
        )
        question_chain = LLMChain(llm=llm, prompt=question_prompt)
        questions = question_chain.run(context=pdf_text_cache)
        return f"❓ **Generated Questions:**\n{questions}"

    # ---------- Fallback ----------
    else:
        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa.invoke({"query": user_prompt})
        return f"💬 **Answer:** {result['result']}"

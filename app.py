# import os
# from flask import Flask, request, render_template
# from werkzeug.utils import secure_filename
# import os
# from dotenv import load_dotenv
# # ---------------- LangGraph imports ----------------
# from langgraph.graph import StateGraph, START, END
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from typing import List, TypedDict

# # ---------------- Config ----------------
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# flask_app = Flask(__name__)
# flask_app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # ---------------- Embeddings + LLM ----------------
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     api_key=groq_api_key
# )

# # ---------------- Globals ----------------
# store = {}
# vectorstore = None
# retriever = None

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# # ---------------- LangGraph ----------------
# class State(TypedDict):
#     input: str
#     answer: str
#     chat_history: List[str]
#     mode: str

# def build_vectorstore(file_path):
#     """Load PDF/Excel and build vectorstore retriever"""
#     global vectorstore, retriever

#     if file_path.endswith(".pdf"):
#         loader = PyPDFLoader(file_path)
#     elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
#         loader = UnstructuredExcelLoader(file_path)
#     else:
#         raise ValueError("Unsupported file type")

#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)

#     vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
#     retriever = vectorstore.as_retriever()

# def rag_node(state: State):
#     global retriever
#     if retriever is None:
#         state["answer"] = "⚠️ No documents available. Please upload a PDF/Excel file first."
#         return state

#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "Rewrite the question in standalone form if needed."),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#     system_prompt = (
#         "You are an assistant for question-answering tasks. "
#         "Use the retrieved context to answer. "
#         "If unknown, say you don't know.\n\n{context}"
#     )

#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     conversational_rag_chain = RunnableWithMessageHistory(
#         rag_chain,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="answer",
#     )

#     response = conversational_rag_chain.invoke(
#         {"input": state["input"]},
#         config={"configurable": {"session_id": "default"}}
#     )

#     state["answer"] = response["answer"]
#     state["chat_history"].append(f"Assistant: {response['answer']}")
#     return state

# # ---------------- Summarizer ----------------
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# summary_prompt = PromptTemplate(
#     input_variables=["context"],
#     template="Summarize the following text:\n\n{context}"
# )
# summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# def summarize_node(state: State):
#     context_text = state.get("input", "")
#     if not context_text.strip():
#         state["answer"] = "⚠️ Please provide some text to summarize."
#         return state

#     response = summary_chain.run(context=context_text)
#     state["answer"] = response
#     state["chat_history"].append(f"Assistant (Summary): {response}")
#     return state

# # ---------------- LangGraph Build ----------------
# graph = StateGraph(State)
# graph.add_node("rag", rag_node)
# graph.add_node("summarize", summarize_node)

# def router(state: State):
#     return "rag" if state["mode"] == "rag" else "summarize"

# graph.add_conditional_edges(START, router, {"rag": "rag", "summarize": "summarize"})
# graph.add_edge("rag", END)
# graph.add_edge("summarize", END)
# app_graph = graph.compile()

# # ---------------- Flask Routes ----------------
# chat_history: List[str] = []

# @flask_app.route("/", methods=["GET", "POST"])
# def index():
#     global chat_history
#     answer = None

#     if request.method == "POST":
#         # If file uploaded
#         if "file" in request.files and request.files["file"].filename != "":
#             file = request.files["file"]
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(flask_app.config["UPLOAD_FOLDER"], filename)
#             file.save(file_path)
#             build_vectorstore(file_path)  # rebuild retriever
#             answer = f"✅ File {filename} uploaded and processed."

#         # If user asks a question
#         elif "input" in request.form:
#             user_input = request.form["input"]
#             mode = request.form["mode"]

#             result = app_graph.invoke({
#                 "input": user_input,
#                 "answer": "",
#                 "chat_history": chat_history,
#                 "mode": mode
#             })

#             answer = result["answer"]
#             chat_history = result["chat_history"]

#     return render_template("index.html", answer=answer, chat_history=chat_history)

# if __name__ == "__main__":
#     flask_app.run(debug=True)
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tools.pdf_tool import build_vectorstore, process_prompt

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

response = None


@app.route("/", methods=["GET", "POST"])
def index():
    global response
    if request.method == "POST":
        # 1️⃣ Upload PDF
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            build_vectorstore(file_path)
            response = f"✅ '{filename}' uploaded and processed successfully!"

        # 2️⃣ Process Prompt
        elif "prompt" in request.form:
            user_prompt = request.form["prompt"]
            response = process_prompt(user_prompt)

    return render_template("index.html", response=response)


if __name__ == "__main__":
    app.run(debug=True)

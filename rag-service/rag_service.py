"""
service for answering questions about me
"""

import os
import re
from typing import List
from fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA = "/data/about_me.txt"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o-mini"
MCP_PORT = 8001
SCORE_THRESHOLD = 0.2
K = 6

app = FastMCP("rag-service")

def detect_language(sentence: str) -> str:
    """
    detect language of the user input
    """

    return "uk" if re.compile(r"[а-щыьэюяіїєґА-ЩЫЬЭЮЯІЇЄҐ]").search(sentence or "") else "en"

def pick_response_language(english: str, ukrainian: str, real: str) -> str:
    """
    pick response language
    """

    return ukrainian if real == "uk" else english

def to_documents_from_file(path: str) -> List[Document]:
    """
    chunk the user data
    """

    with open(path, "r", encoding="utf-8") as file:
        list_documents = []
        for line in file.readlines():
            line = line.strip()
            document = Document(page_content=line, metadata={"src": os.path.basename(path)})
            list_documents.append(document)
    return list_documents

def build_chain():
    """
    build chain
    """

    embeddings = OpenAIEmbeddings(model = EMBEDDING_MODEL, api_key = OPENAI_API_KEY)
    vector_database = FAISS.from_documents(to_documents_from_file(DATA), embeddings)

    retriever = vector_database.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {"k": K, "score_threshold": SCORE_THRESHOLD}
    )

    llm = ChatOpenAI(
        model = LLM_MODEL, temperature = 0,
        api_key = OPENAI_API_KEY, timeout = 20,
        max_retries = 2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a RAG assistant. Answer strictly and only from the provided CONTEXT.\n"
        "If the answer is not present in the context, reply exactly: {no_data}.\n"
        "Always respond in {target_lang}. Do not use any other language."),
        ("human", "{input}"),
        ("system", "CONTEXT:\n{context}")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return retriever, rag_chain

RETRIEVER, RAG_CHAIN = build_chain()

@app.tool()
def about_me_search(question: str) -> str:
    """
    Answers questions about the user by querying a small local RAG index.
    Input can be English or Ukrainian. Output mirrors the input language.
    If no relevant facts are found, returns a brief 'No data available' message.
    """

    language = detect_language(question)
    target_lang = "Ukrainian" if language == "uk" else "English"
    no_data = "Немає даних" if language == "uk" else "No data available."
    result = RAG_CHAIN.invoke({"input": question, "target_lang": target_lang, "no_data": no_data})
    answer = (result or {}).get("answer", "").strip()
    context = (result or {}).get("context", "")

    no_context = (
        not context or (isinstance(context, str) and
        not context.strip()) or (isinstance(context, list) and
        len(context) == 0)
    )

    if no_context or not answer:
        return no_data

    return answer

@app.tool()
def rag_reindex() -> str:
    """
    rebuild embeddings & index from DATA file.
    """

    global RETRIEVER, RAG_CHAIN

    RETRIEVER, RAG_CHAIN = build_chain()

    return "ok"

if __name__ == "__main__":
    app.run("http", host = "0.0.0.0", port = MCP_PORT)

import streamlit as st
import os
import json
import re
import requests
from urllib.parse import quote
from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import SerpAPIWrapper

# Google Gemini SDK
import google.generativeai as genai


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

GOOGLE_API_KEY="" #""
GOOGLE_CSE_ID=""

# SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_KEY=""

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY environment variable!")
    st.stop()

if not SERPAPI_KEY:
    st.error("Missing SERPAPI_API_KEY environment variable!")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB1_PATH = os.path.join(BASE_DIR, "small_db_using_HF_baai_bge")
DB2_PATH = os.path.join(BASE_DIR, "large_embeddings_baai_bge")


# ============================================================
# EMBEDDINGS (HF BGE-SMALL)
# ============================================================

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ============================================================
# LOAD VECTOR DBs
# ============================================================

db1 = Chroma(persist_directory=DB1_PATH, embedding_function=embeddings)
retriever1 = db1.as_retriever(search_kwargs={"k": 6})

db2 = Chroma(persist_directory=DB2_PATH, embedding_function=embeddings)
retriever2 = db2.as_retriever(search_kwargs={"k": 6})


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_query(q: str) -> str:
    """this function removes the additional whitespaces 
    to ensure we create a clean query and inturn create
    stable embeddings and therefore good semantic retrieval."""
    return re.sub(r"\s+", " ", q.strip())


def gemini_answer(prompt: str) -> str:
    """Generate answer using Gemini"""
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.strip() if resp.text else ""
    except Exception as e:
        return f"(Gemini error: {e})"


# ============================================================
# GREETING DETECTION
# ============================================================

GREETINGS = {
    "hi", "hello", "hey", "hey!", "hi!", "hello!", "hey there",
    "good morning", "good afternoon", "good evening", "greetings",
    "howdy"
}

def is_greeting(text: str) -> bool:
    text = text.lower().strip()
    return any(text == g or text.startswith(g) for g in GREETINGS)

def greeting_response(text: str) -> str:
    t = text.lower()
    if "morning" in t:
        return "Good morning! 😊"
    if "afternoon" in t:
        return "Good afternoon! ☀️"
    if "evening" in t:
        return "Good evening! 🌙"
    return "Hello! 👋 How can I help you today?"


# ============================================================
# EXTRACTIVE QA
# ============================================================

def extractive_answer(query: str, docs: List[Any]) -> str:
    """Extractive QA: This function prevents hallucination by asking the llm 
    to generate the answer only from the retrieved context.
    
    This function acts a function evaluate the generation layer 
    i.e. even if we get Wrong docs from the retriever, it returns 
    "NOINFO" so that next node takes charge."""
    context_text = "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:5])
    )

    prompt = f"""
                    Answer the question ONLY using the provided CONTEXT.
                    Cite sources using [1], [2], etc.
                    If the answer is NOT found in context, return "NOINFO".

                    Question: {query}

                    CONTEXT:
                    {context_text}
                    """

    ans = gemini_answer(prompt)

    if ans.upper().startswith("NOINFO") or len(ans) < 20:
        return ""

    return ans


# ============================================================
# CITATION HELPERS
# ============================================================

def scholarly_lookup(query: str, limit: int = 3) -> List[str]:
    """Get citation strings using CrossRef → Semantic Scholar fallback."""
    citations: List[str] = []

    # ---- CROSSREF ----
    try:
        r = requests.get(
            f"https://api.crossref.org/works?rows={limit}&query={quote(query)}",
            timeout=8,
        ).json()

        for item in r.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0]
            authors = item.get("author", [])
            auth = ", ".join(a.get("family", "") for a in authors[:2]) or "Unknown"
            if len(authors) > 2:
                auth += " et al."
            year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else item.get("URL", "")
            citations.append(f"{auth} ({year}). *{title}*. {link}")
    except Exception:
        pass

    if citations:
        return citations

    # ---- SEMANTIC SCHOLAR FALLBACK ----
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/search?"
            f"query={quote(query)}&limit={limit}&fields=title,authors,year,url",
            timeout=8,
        ).json()

        for item in r.get("data", []):
            title = item.get("title", "Untitled")
            authors = item.get("authors", [])
            auth = ", ".join(a.get("name") for a in authors[:2]) or "Unknown"
            if len(authors) > 2:
                auth += " et al."
            year = item.get("year", "n.d.")
            url = item.get("url", "")
            citations.append(f"{auth} ({year}). *{title}*. {url}")
    except Exception:
        pass

    return citations or ["(No scholarly reference found)"]


def format_clickable_citations(citations: List[str]) -> str:
    """Turn plain citation strings into clickable markdown list."""
    out: List[str] = []
    for idx, c in enumerate(citations, start=1):
        m = re.search(r"(https?://[^\s)]+)", c)
        if not m:
            out.append(f"[{idx}] {c}")
        else:
            url = m.group(1)
            title_m = re.search(r"\*([^*]+)\*", c)
            title = title_m.group(1) if title_m else f"Citation {idx}"
            out.append(f"[{idx}] [{title}]({url})")
    return "\n".join(out)


def is_vague_for_scholarly(q: str) -> bool:
    """
    Heuristic: skip scholarly_lookup for very short / pronoun-y queries
    like 'What are those types?', because they produce random papers.
    """
    text = q.lower()
    words = text.split()
    vague_tokens = {"those", "these", "them", "it", "they", "that", "this"}
    if len(words) <= 3:
        return True
    if any(tok in vague_tokens for tok in words):
        return True
    return False


from sentence_transformers import SentenceTransformer, util

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_citations_from_answer(answer: str, limit=3):
    """
    Generate research citations that match the semantic meaning
    of the chatbot's full answer, not the original query.
    """

    # Step 1 — Generate 4 semantic reformulations of the answer
    prompt = f"""
            Rewrite the following text into:
            1. A concise scientific abstract
            2. A technical research framing
            3. A list of 5 keywords
            4. A practical problem statement

            Return them separated by '---'.

            TEXT:
            {answer}
            """
    try:
        out = gemini_answer(prompt)
        parts = out.split("---")
        queries = [p.strip() for p in parts if len(p.strip()) > 3]
    except:
        queries = [answer]

    # Step 2 — Run scholarly lookup for each reformulation
    raw_citations = []
    for q in queries:
        raw_citations.extend(scholarly_lookup(q))

    # Remove duplicates
    raw_citations = list(dict.fromkeys(raw_citations))

    if not raw_citations:
        return []

    # Step 3 — Embed the answer
    ans_emb = semantic_model.encode(answer, convert_to_tensor=True)

    # Step 4 — Rank citations by semantic similarity
    scored = []
    for cit in raw_citations:
        match = re.search(r"\*([^*]+)\*", cit)   # extract title
        title = match.group(1) if match else cit
        emb = semantic_model.encode(title, convert_to_tensor=True)
        score = util.cos_sim(ans_emb, emb).item()
        scored.append((score, cit))

    # Step 5 — Return best N citations
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [c for _, c in scored[:limit]]

    return top



def looks_like_dont_know(text: str) -> bool:
    text = text.lower()
    patterns = [
        "i don't know",
        "i do not know",
        "cannot be determined",
        "can't be determined",
        "cannot determine",
        "can't determine",
        "not enough information",
        "no information available",
        "insufficient information",
        "i am not sure",
        "i'm not sure",
    ]
    return any(p in text for p in patterns)


# ============================================================
# GRAPH STATE
# ============================================================

class GraphState(TypedDict):
    query: str
    answer: str
    context: str
    citations: List[str]


# ============================================================
# External Search Tools
# ============================================================

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

google_tool = SerpAPIWrapper(
    serpapi_api_key=SERPAPI_KEY
)


# ============================================================
# GRAPH NODES
# ============================================================

def db1_node(state: GraphState) -> GraphState:
    """First local DB (FAQ-style) → NO citations."""
    q = clean_query(state["query"])
    docs = retriever1.invoke(q)

    if not docs:
        return {**state, "context": "no_db1", "citations": []}

    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db1", "citations": []}

    # DB1 → NO citations
    refs = []

    return {**state, "answer": ans, "context": "db1", "citations": refs}


def db2_node(state: GraphState):
    q = clean_query(state["query"])
    docs = retriever2.invoke(q)

    if not docs:
        return {**state, "context": "no_db2", "citations": []}

    ans = extractive_answer(q, docs)
    if not ans:
        return {**state, "context": "no_db2", "citations": []}

    # SEMANTIC CITATIONS BASED ON THE FULL ANSWER
    refs = semantic_citations_from_answer(ans)

    return {
        **state,
        "answer": ans,
        "context": "db2",
        "citations": refs
    }




def google_node(state: GraphState) -> GraphState:
    """Fallback to web search via SerpAPI → Google search link citation."""
    q = clean_query(state["query"])

    try:
        results = google_tool.run(q)
        if not results:
            return {**state, "context": "no_google", "citations": []}
        # print("Google Results:\n",results)
        prompt = f"""
                    You are a helpful expert.
                    Use the following Google search results to answer the question.

                    Question: {q}

                    Search results:
                    {results}

                    Always provide your BEST POSSIBLE answer, even if the information is incomplete.
                    DO NOT say that you cannot determine the answer.
                    DO NOT say that there is not enough information.
                    DO NOT say you don't know.
                    """
        ans = gemini_answer(prompt)

        refs = [f"[Google Search](https://www.google.com/search?q={quote(q)})"]

        return {
            **state,
            "answer": ans,
            "context": "google",
            "citations": refs,
        }

    except Exception as e:
        print("Google error:", e)
        return {**state, "context": "no_google", "citations": []}


def wiki_node(state: GraphState) -> GraphState:
    """Final external fallback → Wikipedia summary + Wikipedia search link."""
    q = clean_query(state["query"])

    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}"
        blob = requests.get(url, timeout=8).json().get("extract", "")

        if not blob:
            return {**state, "context": "no_wiki", "citations": []}

        prompt = f"""
                    You are a helpful expert.
                    Use the following Wikipedia extract to answer the question.

                    Question: {q}

                    Wikipedia extract:
                    {blob}

                    Always provide your BEST POSSIBLE answer.
                    DO NOT say that you cannot determine the answer.
                    DO NOT say that there is not enough information.
                    DO NOT say you don't know.
                    """
        ans = gemini_answer(prompt)

        refs = [
            f"[Wikipedia Search](https://en.wikipedia.org/wiki/Special:Search?search={quote(q)})"
        ]

        return {**state, "answer": ans, "context": "wiki", "citations": refs}

    except Exception:
        return {**state, "context": "no_wiki", "citations": []}


def final_node(state: GraphState) -> GraphState:
    """
    Final cleanup summary — produce ONE best direct answer,
    then append nicely formatted citations if available.
    Also ensures we never end with “I don't know / cannot be determined”.
    """
    q = clean_query(state["query"])
    base_ans = state["answer"]
    cites = state.get("citations", [])

    # If upstream answer is basically "I don't know", regenerate from scratch
    if looks_like_dont_know(base_ans):
        fallback_prompt = f"""
                            You are a knowledgeable assistant.
                            Using your general world knowledge, answer the user's question as helpfully
                            and accurately as you can.

                            Do NOT say you don't know.
                            Do NOT say the answer cannot be determined.
                            If information is uncertain, still give your best reasonable answer.

                            Question: {q}
                            """
        base_ans = gemini_answer(fallback_prompt)

    summary_prompt = f"""
                        Rewrite the following answer into ONE single, clear, direct answer.
                        Do NOT give multiple options.
                        Do NOT give choices, lists, or variations.
                        Give only the best final answer in 2–4 sentences max.

                        Question: {q}
                        Answer: {base_ans}
                        """

    summary = gemini_answer(summary_prompt)

    if cites:
        summary += "\n\n📚 Citations:\n" + format_clickable_citations(cites)

    return {**state, "answer": summary}


# ============================================================
# GRAPH PIPELINE
# ============================================================

workflow = StateGraph(GraphState)

workflow.add_node("db1", db1_node)
workflow.add_node("db2", db2_node)
workflow.add_node("wiki", wiki_node)
workflow.add_node("google", google_node)
workflow.add_node("final", final_node)

workflow.add_edge(START, "db1")

workflow.add_conditional_edges(
    "db1",
    lambda state: state["context"],
    {
        "db1": "final",
        "no_db1": "db2",
    },
)

workflow.add_conditional_edges(
    "db2",
    lambda state: state["context"],
    {
        "db2": "final",
        "no_db2": "wiki",
    },
)

workflow.add_conditional_edges(
    "wiki",
    lambda state: state["context"],
    {
        "wiki": "final",
        "no_wiki": "google",
    },
)

# Google is the final fallback, so always proceed to final.
workflow.add_edge("google", "final")

workflow.add_edge("final", END)

graph = workflow.compile()

# print(graph.get_graph())

# START
#   ↓
# DB1
#  ├─ success → Final
#  └─ no_db1 → DB2
#                 ├─ success → Final
#                 └─ no_db2 → Wiki
#                                ├─ success → Final
#                                └─ no_wiki → Google
#                                                ↓
#                                              Final
#                                                ↓
#                                               END

# import os

# png_graph = graph.get_graph().draw_mermaid_png()
# with open("my_graph.png", "wb") as f:
#     f.write(png_graph)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("Hybrid RAG Chatbot")


# Initialize conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Display conversation
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User input
user_query = st.chat_input("Ask something...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    # 1. FIRST HANDLE GREETINGS
    if is_greeting(user_query):
        reply = greeting_response(user_query)

        with st.chat_message("assistant"):
            st.write(reply)
            st.caption("Source: greeting")

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.stop()

    # 2. RUN HYBRID RAG PIPELINE
    result = graph.invoke({
        "query": user_query,
        "context": "",
        "answer": "",
        "citations": [],
    })

    answer = result["answer"]
    context = result["context"]

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Source: {context}")

    st.session_state["messages"].append({"role": "assistant", "content": answer})


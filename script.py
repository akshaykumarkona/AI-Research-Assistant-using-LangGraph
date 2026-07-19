import streamlit as st
import os
import html
import re
import requests
from textwrap import dedent
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

GOOGLE_API_KEY=""
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
gemini = genai.GenerativeModel("gemini-3.1-flash-lite")


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


# ============================================================
# PREMIUM STREAMLIT UI — BEEMACHINE AI
# ============================================================

import html
import time
from typing import Generator

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BeeMachine AI",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>
        /* ==================================================
           GLOBAL
        ================================================== */

        :root {
            --bg-main: #f8fafc;
            --bg-card: rgba(255, 255, 255, 0.88);
            --bg-card-solid: #ffffff;
            --text-main: #172033;
            --text-secondary: #64748b;
            --border: rgba(226, 232, 240, 0.92);
            --amber: #f59e0b;
            --amber-dark: #b45309;
            --amber-soft: #fff7df;
            --green: #16a34a;
            --blue: #2563eb;
            --purple: #7c3aed;
            --red: #dc2626;
            --shadow-sm: 0 7px 24px rgba(15, 23, 42, 0.055);
            --shadow-lg: 0 22px 60px rgba(15, 23, 42, 0.10);
        }

        html,
        body,
        [class*="css"] {
            font-family:
                Inter,
                ui-sans-serif,
                system-ui,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        .stApp {
            color: var(--text-main);
            background:
                radial-gradient(
                    circle at 90% 3%,
                    rgba(245, 158, 11, 0.13),
                    transparent 25%
                ),
                radial-gradient(
                    circle at 18% 75%,
                    rgba(59, 130, 246, 0.05),
                    transparent 28%
                ),
                linear-gradient(
                    180deg,
                    #fffdf8 0%,
                    #ffffff 35%,
                    #f8fafc 100%
                );
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.45rem;
            padding-bottom: 8rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* ==================================================
           TOP NAV
        ================================================== */

        .top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
            padding: 0.65rem 0.1rem;
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-icon {
            display: flex;
            width: 2.65rem;
            height: 2.65rem;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(245, 158, 11, 0.26);
            border-radius: 14px;
            background: linear-gradient(145deg, #fff8df, #ffffff);
            box-shadow: var(--shadow-sm);
            font-size: 1.35rem;
        }

        .brand-title {
            color: var(--text-main);
            font-size: 1rem;
            font-weight: 850;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            margin-top: 0.05rem;
            color: var(--text-secondary);
            font-size: 0.72rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.42rem 0.7rem;
            border: 1px solid #bbf7d0;
            border-radius: 999px;
            background: rgba(240, 253, 244, 0.88);
            color: #15803d;
            font-size: 0.74rem;
            font-weight: 750;
        }

        .status-dot {
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.11);
        }

        /* ==================================================
           HERO
        ================================================== */

        .hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.25rem;
            padding: 2.35rem 2.5rem;
            border: 1px solid var(--border);
            border-radius: 28px;
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 255, 255, 0.97),
                    rgba(255, 249, 229, 0.95)
                );
            box-shadow: var(--shadow-lg);
            backdrop-filter: blur(20px);
        }

        .hero::before {
            content: "";
            position: absolute;
            width: 330px;
            height: 330px;
            right: -115px;
            top: -145px;
            border-radius: 50%;
            background: rgba(245, 158, 11, 0.13);
        }

        .hero::after {
            content: "🐝";
            position: absolute;
            right: 2.8rem;
            top: 50%;
            transform: translateY(-50%) rotate(-9deg);
            font-size: 7rem;
            opacity: 0.10;
        }

        .hero-content {
            position: relative;
            z-index: 2;
            max-width: 850px;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 1rem;
            padding: 0.42rem 0.78rem;
            border: 1px solid rgba(245, 158, 11, 0.28);
            border-radius: 999px;
            background: rgba(254, 243, 199, 0.72);
            color: #92400e;
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .hero-title {
            max-width: 800px;
            margin: 0;
            color: var(--text-main);
            font-size: clamp(2.25rem, 5vw, 4rem);
            font-weight: 900;
            line-height: 1.02;
            letter-spacing: -0.052em;
        }

        .hero-title span {
            color: var(--amber-dark);
        }

        .hero-subtitle {
            max-width: 790px;
            margin: 1rem 0 0;
            color: #5e6b7c;
            font-size: 1.03rem;
            line-height: 1.72;
        }

        .hero-capabilities {
            display: flex;
            flex-wrap: wrap;
            gap: 0.56rem;
            margin-top: 1.3rem;
        }

        .capability-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.4rem 0.72rem;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.92);
            color: #475569;
            font-size: 0.75rem;
            font-weight: 720;
            box-shadow: 0 3px 10px rgba(15, 23, 42, 0.035);
        }

        /* ==================================================
           METRIC CARDS
        ================================================== */

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.2rem;
        }

        .metric-card {
            padding: 0.9rem 1rem;
            border: 1px solid var(--border);
            border-radius: 17px;
            background: rgba(255, 255, 255, 0.80);
            box-shadow: var(--shadow-sm);
            backdrop-filter: blur(18px);
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 0.26rem;
            color: #263247;
            font-size: 0.9rem;
            font-weight: 820;
        }

        /* ==================================================
           SECTION HEADERS
        ================================================== */

        .section-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 1.35rem 0 0.72rem;
        }

        .section-title {
            color: #475569;
            font-size: 0.76rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .section-caption {
            color: #94a3b8;
            font-size: 0.72rem;
        }

        /* ==================================================
           QUICK ACTIONS
        ================================================== */

        .stButton > button {
            min-height: 3rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.94);
            color: #334155;
            font-weight: 720;
            box-shadow: 0 5px 16px rgba(15, 23, 42, 0.04);
            transition:
                transform 0.2s ease,
                border-color 0.2s ease,
                box-shadow 0.2s ease,
                color 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            border-color: #f59e0b;
            color: #b45309;
            box-shadow: 0 10px 22px rgba(245, 158, 11, 0.13);
        }

        /* ==================================================
           EMPTY STATE
        ================================================== */

        .empty-panel {
            padding: 2rem 1.5rem;
            border: 1px dashed #d5dce7;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.68);
            text-align: center;
            backdrop-filter: blur(16px);
        }

        .empty-icon {
            margin-bottom: 0.6rem;
            font-size: 2.6rem;
        }

        .empty-title {
            color: #29354a;
            font-size: 1.12rem;
            font-weight: 820;
        }

        .empty-text {
            max-width: 600px;
            margin: 0.38rem auto 0;
            color: #718096;
            font-size: 0.89rem;
            line-height: 1.65;
        }

        /* ==================================================
           CHAT
        ================================================== */

        [data-testid="stChatMessage"] {
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            border: 1px solid var(--border);
            border-radius: 19px;
            background: rgba(255, 255, 255, 0.93);
            box-shadow: var(--shadow-sm);
            backdrop-filter: blur(18px);
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]
        ) {
            margin-left: 9%;
            border-color: rgba(245, 158, 11, 0.22);
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 251, 235, 0.98),
                    rgba(255, 255, 255, 0.98)
                );
        }

        [data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-assistant"]
        ) {
            margin-right: 4%;
        }

        [data-testid="stChatMessageContent"] {
            color: #29354a;
            font-size: 0.98rem;
            line-height: 1.73;
        }

        [data-testid="stChatMessageContent"] p {
            margin-bottom: 0.5rem;
        }

        /* ==================================================
           ANSWER METADATA
        ================================================== */

        .answer-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.8rem;
            padding-top: 0.75rem;
            border-top: 1px solid #edf0f4;
        }

        .meta-item {
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            padding: 0.3rem 0.58rem;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            background: #f8fafc;
            color: #64748b;
            font-size: 0.69rem;
            font-weight: 720;
        }

        .source-small-db {
            border-color: #bfdbfe;
            background: #eff6ff;
            color: #1d4ed8;
        }

        .source-large-db {
            border-color: #ddd6fe;
            background: #f5f3ff;
            color: #6d28d9;
        }

        .source-wiki {
            border-color: #cbd5e1;
            background: #f8fafc;
            color: #334155;
        }

        .source-google {
            border-color: #fecaca;
            background: #fef2f2;
            color: #b91c1c;
        }

        .source-greeting {
            border-color: #bbf7d0;
            background: #f0fdf4;
            color: #15803d;
        }

        .source-error {
            border-color: #fecaca;
            background: #fef2f2;
            color: #b91c1c;
        }

        /* ==================================================
           RETRIEVAL PIPELINE
        ================================================== */

        .pipeline-card {
            margin-top: 0.75rem;
            padding: 0.85rem;
            border: 1px solid #e7ebf0;
            border-radius: 15px;
            background: rgba(248, 250, 252, 0.82);
        }

        .pipeline-heading {
            margin-bottom: 0.6rem;
            color: #64748b;
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.065em;
            text-transform: uppercase;
        }

        .pipeline-track {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.35rem;
        }

        .pipeline-node {
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            padding: 0.31rem 0.57rem;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            background: white;
            color: #94a3b8;
            font-size: 0.68rem;
            font-weight: 720;
        }

        .pipeline-node.active {
            border-color: #fcd34d;
            background: #fffbeb;
            color: #a16207;
        }

        .pipeline-node.success {
            border-color: #bbf7d0;
            background: #f0fdf4;
            color: #15803d;
        }

        .pipeline-arrow {
            color: #cbd5e1;
            font-size: 0.7rem;
        }

        /* ==================================================
           CHAT INPUT
        ================================================== */

        [data-testid="stChatInput"] {
            border: 1px solid #d7dee8;
            border-radius: 19px;
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 14px 42px rgba(15, 23, 42, 0.14);
        }

        [data-testid="stChatInput"] textarea {
            min-height: 3.5rem;
            color: #273244;
            font-size: 0.96rem;
        }

        [data-testid="stChatInput"] button {
            border-radius: 13px;
            background: #f59e0b;
            color: white;
        }

        /* ==================================================
           SIDEBAR
        ================================================== */

        [data-testid="stSidebar"] {
            border-right: 1px solid #edf0f4;
            background:
                linear-gradient(
                    180deg,
                    #fff9eb 0%,
                    #ffffff 38%,
                    #f8fafc 100%
                );
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }

        .sidebar-profile {
            margin-bottom: 1.1rem;
            padding: 1rem;
            border: 1px solid #fde7b0;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.91);
            box-shadow: var(--shadow-sm);
        }

        .sidebar-profile-title {
            color: #1e293b;
            font-size: 0.98rem;
            font-weight: 850;
        }

        .sidebar-profile-text {
            margin-top: 0.38rem;
            color: #64748b;
            font-size: 0.76rem;
            line-height: 1.55;
        }

        .sidebar-section-title {
            margin: 1rem 0 0.55rem;
            color: #64748b;
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .sidebar-source {
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            margin-bottom: 0.43rem;
            padding: 0.63rem;
            border: 1px solid #edf0f4;
            border-radius: 13px;
            background: rgba(255, 255, 255, 0.87);
        }

        .sidebar-source-icon {
            display: flex;
            min-width: 1.7rem;
            height: 1.7rem;
            align-items: center;
            justify-content: center;
            border-radius: 9px;
            background: #fff7df;
            font-size: 0.83rem;
        }

        .sidebar-source-title {
            color: #334155;
            font-size: 0.76rem;
            font-weight: 800;
        }

        .sidebar-source-text {
            margin-top: 0.1rem;
            color: #8491a3;
            font-size: 0.66rem;
            line-height: 1.35;
        }

        /* ==================================================
           RESPONSIVE
        ================================================== */

        @media (max-width: 900px) {
            .metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 0.8rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .hero {
                padding: 1.55rem;
                border-radius: 21px;
            }

            .hero::after {
                display: none;
            }

            .metric-grid {
                grid-template-columns: 1fr;
            }

            [data-testid="stChatMessage"] {
                margin-left: 0 !important;
                margin-right: 0 !important;
            }

            .status-pill {
                display: none;
            }
        }
    </style>
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def stream_text(text: str) -> Generator[str, None, None]:
    """
    Stream a response in small word groups for a ChatGPT-like effect.
    """

    words = text.split()

    for index, word in enumerate(words):
        suffix = " " if index < len(words) - 1 else ""
        yield word + suffix
        time.sleep(0.012)


def get_source_details(source: str) -> tuple[str, str, str]:
    """
    Convert graph context into a readable source label.
    """

    normalized = str(source or "").strip().lower()

    if normalized == "greeting":
        return "Greeting Handler", "👋", "source-greeting"

    if (
        normalized == "db1"
        or "small_db" in normalized
        or "small db" in normalized
        or "lab knowledge" in normalized
    ):
        return "Lab Knowledge Base", "🏛️", "source-small-db"

    if (
        normalized == "db2"
        or "large_embeddings" in normalized
        or "large db" in normalized
        or "entomology knowledge" in normalized
    ):
        return "Entomology Knowledge Base", "📚", "source-large-db"

    if "wiki" in normalized:
        return "Wikipedia", "🌐", "source-wiki"

    if "google" in normalized or "serp" in normalized:
        return "Google Search", "🔎", "source-google"

    if "error" in normalized:
        return "System Error", "⚠️", "source-error"

    return str(source or "Hybrid RAG"), "🐝", ""


def get_pipeline_states(source: str) -> dict[str, str]:
    """
    Infer the visible retrieval path from the final source.

    This does not claim that every node returned content. It shows the
    sequence evaluated before the selected source produced the answer.
    """

    normalized = str(source or "").strip().lower()

    states = {
        "db1": "",
        "db2": "",
        "wiki": "",
        "google": "",
        "gemini": "success",
    }

    if normalized == "greeting":
        return states

    if normalized == "db1" or "small" in normalized or "lab knowledge" in normalized:
        states["db1"] = "success"

    elif normalized == "db2" or "large" in normalized or "entomology" in normalized:
        states["db1"] = "active"
        states["db2"] = "success"

    elif "wiki" in normalized:
        states["db1"] = "active"
        states["db2"] = "active"
        states["wiki"] = "success"

    elif "google" in normalized or "serp" in normalized:
        states["db1"] = "active"
        states["db2"] = "active"
        states["wiki"] = "active"
        states["google"] = "success"

    return states


def render_answer_metadata(
    source: str,
    elapsed_seconds: float | None = None,
) -> None:
    """
    Render source, model, and response-time metadata.
    """

    label, icon, css_class = get_source_details(source)

    safe_label = html.escape(label)
    safe_icon = html.escape(icon)

    response_time_html = ""

    if elapsed_seconds is not None:
        response_time_html = (
            f'<span class="meta-item">⏱️ {elapsed_seconds:.2f} sec</span>'
        )

    st.html(
        f"""
        <div class="answer-meta">
            <span class="meta-item {css_class}">
                {safe_icon} {safe_label}
            </span>
            <span class="meta-item">🧠 Gemini</span>
            {response_time_html}
        </div>
        """
    )


def render_pipeline(source: str) -> None:
    """
    Display the retrieval path associated with the final answer source.
    """

    if str(source).lower() == "greeting":
        return

    states = get_pipeline_states(source)

    nodes = [
        ("db1", "🏛️ Lab KB"),
        ("db2", "📚 Scientific KB"),
        ("wiki", "🌐 Wikipedia"),
        ("google", "🔎 Google"),
        ("gemini", "🧠 Gemini"),
    ]

    rendered_nodes = []

    for index, (key, label) in enumerate(nodes):
        css_state = states.get(key, "")

        rendered_nodes.append(
            f'<span class="pipeline-node {css_state}">{label}</span>'
        )

        if index < len(nodes) - 1:
            rendered_nodes.append(
                '<span class="pipeline-arrow">→</span>'
            )

    st.html(
        f"""
        <div class="pipeline-card">
            <div class="pipeline-heading">
                Retrieval path
            </div>
            <div class="pipeline-track">
                {''.join(rendered_nodes)}
            </div>
        </div>
        """
    )


def add_user_message(content: str) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": content,
        }
    )


def add_assistant_message(
    content: str,
    source: str,
    elapsed_seconds: float | None = None,
) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "source": source,
            "elapsed_seconds": elapsed_seconds,
        }
    )


def clear_conversation() -> None:
    st.session_state.messages = []


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.html(
        """
        <div class="sidebar-profile">
            <div class="sidebar-profile-title">
                🐝 BeeMachine AI
            </div>
            <div class="sidebar-profile-text">
                A hybrid RAG assistant for laboratory information,
                bee research, pollinator science, entomology, and
                agricultural knowledge.
            </div>
        </div>
        """
    )

    st.html(
        '<div class="sidebar-section-title">Knowledge network</div>'
    )

    sidebar_sources = [
        (
            "🏛️",
            "Lab Knowledge Base",
            "People, projects, laboratory information, and BeeMachine.",
        ),
        (
            "📚",
            "Scientific Knowledge Base",
            "Entomology and agriculture books and websites.",
        ),
        (
            "🌐",
            "Wikipedia",
            "General reference fallback.",
        ),
        (
            "🔎",
            "Google Search",
            "External web fallback through SerpAPI.",
        ),
        (
            "🧠",
            "Gemini",
            "Synthesizes the final answer from retrieved context.",
        ),
    ]

    for icon, title, description in sidebar_sources:
        st.html(
            f"""
            <div class="sidebar-source">
                <div class="sidebar-source-icon">
                    {html.escape(icon)}
                </div>
                <div>
                    <div class="sidebar-source-title">
                        {html.escape(title)}
                    </div>
                    <div class="sidebar-source-text">
                        {html.escape(description)}
                    </div>
                </div>
            </div>
            """
        )

    st.divider()

    st.html(
        '<div class="sidebar-section-title">Conversation</div>'
    )

    st.caption(
        f"{len(st.session_state.messages)} message(s) in this session"
    )

    if st.button(
        "✨ New chat",
        use_container_width=True,
    ):
        clear_conversation()
        st.rerun()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        clear_conversation()
        st.rerun()

    st.divider()

    st.caption(
        "Responses are generated from retrieved context. "
        "Verify important scientific conclusions using primary sources."
    )


# ============================================================
# TOP NAVIGATION
# ============================================================

st.html(
    """
    <div class="top-nav">
        <div class="brand-wrap">
            <div class="brand-icon">🐝</div>
            <div>
                <div class="brand-title">BeeMachine AI</div>
                <div class="brand-subtitle">
                    Scientific research assistant
                </div>
            </div>
        </div>

        <div class="status-pill">
            <span class="status-dot"></span>
            Knowledge network online
        </div>
    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-content">
            <div class="hero-kicker">
                🐝 AI Research Assistant
            </div>

            <h1 class="hero-title">
                Explore bee research with
                <span>intelligent retrieval</span>
            </h1>

            <p class="hero-subtitle">
                Ask questions about BeeMachine, the laboratory,
                researchers, ongoing projects, bees, pollinators,
                entomology, and agricultural science. The assistant
                searches trusted domain knowledge first and uses
                external sources only when necessary.
            </p>

            <div class="hero-capabilities">
                <span class="capability-chip">⚡ LangGraph</span>
                <span class="capability-chip">🧩 Hybrid RAG</span>
                <span class="capability-chip">🗄️ ChromaDB</span>
                <span class="capability-chip">🧠 Gemini</span>
                <span class="capability-chip">🌐 Multi-source retrieval</span>
            </div>
        </div>
    </div>
    """
)


# ============================================================
# SYSTEM METRICS
# ============================================================

st.html(
    """
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Primary source</div>
            <div class="metric-value">Lab Knowledge Base</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Scientific corpus</div>
            <div class="metric-value">300K+ text lines</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Orchestration</div>
            <div class="metric-value">LangGraph Workflow</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Generation model</div>
            <div class="metric-value">Gemini</div>
        </div>
    </div>
    """
)


# ============================================================
# QUICK ACTIONS
# ============================================================

selected_suggestion = None

if not st.session_state.messages:
    st.html(
        """
        <div class="empty-panel">
            <div class="empty-icon">🍯</div>
            <div class="empty-title">
                Begin exploring the BeeMachine knowledge network
            </div>
            <div class="empty-text">
                Choose a suggested topic below or enter your own
                research question in the chat input.
            </div>
        </div>
        """
    )

    st.html(
        """
        <div class="section-row">
            <div class="section-title">Quick actions</div>
            <div class="section-caption">
                Select a topic to begin
            </div>
        </div>
        """
    )

    quick_actions = [
        ("🐝 What is BeeMachine?", "What is BeeMachine and how does it work?"),
        (
            "👩‍🔬 Meet the lab",
            "Who are the researchers and students in the Brain Spiesman's lab, and what are their research areas?",
        ),
        (
            "🔬 Research projects",
            "What research projects are currently being conducted in the Brain Spiesman's lab?",
        ),
        (
            "🌼 Bee science",
            "Explain the ecological importance of bees and pollinators.",
        ),
        (
            "📚 Scientific resources",
            "What scientific topics are covered by the entomology knowledge base?",
        ),
        (
            "🌎 Bee conservation",
            "What are the major threats to bees and how can they be protected?",
        ),
    ]

    for row_start in range(0, len(quick_actions), 3):
        columns = st.columns(3)

        for column, (button_label, query) in zip(
            columns,
            quick_actions[row_start:row_start + 3],
        ):
            with column:
                if st.button(
                    button_label,
                    use_container_width=True,
                    key=f"quick_{row_start}_{button_label}",
                ):
                    selected_suggestion = query


# ============================================================
# CONVERSATION HEADER
# ============================================================

st.html(
    """
    <div class="section-row">
        <div class="section-title">Conversation</div>
        <div class="section-caption">
            Context-grounded responses
        </div>
    </div>
    """
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    source = message.get("source")
    elapsed_seconds = message.get("elapsed_seconds")

    avatar = "🧑‍🔬" if role == "user" else "🐝"

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

        if role == "assistant" and source:
            render_answer_metadata(
                source=source,
                elapsed_seconds=elapsed_seconds,
            )

            render_pipeline(source)


# ============================================================
# CHAT INPUT
# ============================================================

typed_query = st.chat_input(
    "Ask about BeeMachine, the lab, researchers, bees, or entomology..."
)

user_query = typed_query or selected_suggestion


# ============================================================
# PROCESS QUERY
# ============================================================

if user_query:
    add_user_message(user_query)

    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(user_query)

    # --------------------------------------------------------
    # GREETING HANDLER
    # --------------------------------------------------------

    if is_greeting(user_query):
        start_time = time.perf_counter()

        reply = greeting_response(user_query)
        source = "greeting"

        elapsed_seconds = time.perf_counter() - start_time

        with st.chat_message("assistant", avatar="🐝"):
            st.write_stream(stream_text(reply))

            render_answer_metadata(
                source=source,
                elapsed_seconds=elapsed_seconds,
            )

        add_assistant_message(
            content=reply,
            source=source,
            elapsed_seconds=elapsed_seconds,
        )

    # --------------------------------------------------------
    # HYBRID RAG PIPELINE
    # --------------------------------------------------------

    else:
        start_time = time.perf_counter()

        with st.chat_message("assistant", avatar="🐝"):
            try:
                with st.status(
                    "Searching the BeeMachine knowledge network...",
                    expanded=True,
                ) as status:
                    st.write(
                        "🏛️ Evaluating the laboratory knowledge base"
                    )
                    st.write(
                        "📚 Evaluating scientific entomology resources"
                    )
                    st.write(
                        "🌐 Preparing fallback retrieval when required"
                    )
                    st.write(
                        "🧠 Synthesizing a context-grounded answer"
                    )

                    result = graph.invoke(
                        {
                            "query": user_query,
                            "context": "",
                            "answer": "",
                            "citations": [],
                        }
                    )

                    answer = result.get(
                        "answer",
                        "I could not generate a response.",
                    )

                    source = result.get(
                        "context",
                        "Hybrid RAG",
                    )

                    elapsed_seconds = (
                        time.perf_counter() - start_time
                    )

                    status.update(
                        label="Response ready",
                        state="complete",
                        expanded=False,
                    )

                st.write_stream(stream_text(answer))

                render_answer_metadata(
                    source=source,
                    elapsed_seconds=elapsed_seconds,
                )

                render_pipeline(source)

            except Exception as exc:
                elapsed_seconds = (
                    time.perf_counter() - start_time
                )

                answer = (
                    "I encountered an error while searching the "
                    "knowledge network. Please try again."
                )

                source = "System error"

                st.error(answer)

                # Keep during development. Remove in production.
                st.caption(
                    f"Technical details: {exc}"
                )

                render_answer_metadata(
                    source=source,
                    elapsed_seconds=elapsed_seconds,
                )

        add_assistant_message(
            content=answer,
            source=source,
            elapsed_seconds=elapsed_seconds,
        )

    st.rerun()
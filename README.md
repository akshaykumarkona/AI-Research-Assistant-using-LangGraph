# AI Research Assistant using LangGraph

A Hybrid Retrieval-Augmented Generation (RAG) chatbot built using **LangGraph**, **LangChain**, **ChromaDB**, **Google Gemini**, **Wikipedia**, and **Google Search**. The chatbot was developed to provide researchers, students, and the general public with an intelligent assistant capable of answering questions about a research laboratory, its members, ongoing research, and bee-related topics.

---
## 🚀 Live Application

🌐 **Try the deployed application:**  
**https://ai-research-chatbot-using-langgraph.streamlit.app/**

---


## 🎥 Demo

<p align="center">
  <img src="./Demo_1.gif" alt="BeeMachine AI Demo" width="900"/>
</p>

  ## ▶️ Full Demo 👉 [Watch the complete AI Assistant Demo](./Demo.mp4)


## 📸 Screenshot

<p align="center">

<img src="./Application Screenshot.png" width="850"/>

</p>

---


# Project Background

This project was developed as part of my **Graduate Research Assistantship (GRA)** at **Kansas State University**.

My research supervisor maintains a laboratory website and a mobile application. The objective of this project was to develop an AI-powered chatbot that can be integrated into both the laboratory website and the application.

The chatbot enables users to:

- Learn about the research laboratory
- Explore ongoing research projects
- Know more about faculty, researchers, and students
- Ask questions about bees and pollinator biology
- Retrieve information from scientific entomology resources
- Obtain answers even when information is unavailable within the laboratory database through external knowledge sources

The overall goal is to make scientific information more accessible to researchers, students, educators, and the general public.

---

# System Architecture

The chatbot follows a hierarchical Hybrid RAG pipeline.

```
                        User Query
                            │
                            ▼
          Small Laboratory Knowledge Base
         (Lab Website, People & Projects)
                            │
                 Information Found?
                 ├──── Yes ───────────────┐
                 ▼ No                     │
      Large Entomology Knowledge Base     │
   (Books, Agriculture & Bee Resources)   │
                            │             │
                 Information Found?       │
                 ├──── Yes ───────────────┤
                 ▼ No                     │
                  Wikipedia Search        │
                            │             │
                 Information Found?       │
                 ├──── Yes ───────────────┤
                 ▼ No                     │
              Google Search (SerpAPI)     │
                            │             │
                            └─────────────┘
                                  │
                                  ▼
                 Retrieved Context + User Query
                                  │
                                  ▼
                      Google Gemini LLM
         (Generates a grounded response using the
              retrieved context and user query)
                                  │
                                  ▼
                           Final Response
```

The workflow is implemented using **LangGraph**, allowing conditional execution between multiple retrieval nodes.

---

# Knowledge Sources

## 1. Small Knowledge Base

**Directory**

```
small_db_using_HF_baai_bge/
```

This Chroma vector database contains embeddings generated from information scraped from the laboratory website, including:

- Laboratory information
- About Faculty, Graduate students
- Research staff
- Research projects
- Publications from lab
- Lab resources

This serves as the chatbot's **primary knowledge base**, ensuring laboratory-specific questions are answered before consulting external sources.

---

## 2. Large Knowledge Base

**Directory**

```
large_embeddings_baai_bge/
```

This Chroma vector database contains embeddings generated from approximately **300,000 lines of text** collected from:

- Entomology textbooks
- Agriculture textbooks
- Bee biology resources
- Pollinator research articles
- Scientific websites
- Educational resources

This serves as the chatbot's **secondary knowledge base**, providing scientific context beyond the laboratory's own content.

---

## 3. Wikipedia

If information cannot be found within either vector database, the chatbot searches Wikipedia for additional context using WikipediaQueryRun Tool calling.

---

## 4. Google Search

As a final fallback, the chatbot performs a Google search using SerpAPI to retrieve relevant information from the web.

---

# Tech Stack

- Python
- LangGraph
- LangChain
- ChromaDB
- HuggingFace Embeddings (BAAI/bge-large-en-v1.5)
- Google Gemini
- Wikipedia API
- SerpAPI
- Streamlit

---

# Project Structure

```
AI_Research_Assistant/
│
├── script.py
├── requirements.txt
├── README.md
├── my_graph.png
│
├── small_db_using_HF_baai_bge/
│      └── Laboratory website embeddings
│
└── large_embeddings_baai_bge/
       └── Entomology & Agriculture knowledge base
```

---

# Retrieval Pipeline

The chatbot follows the retrieval order below:

1. Small Laboratory Knowledge Base
2. Large Entomology & Agriculture Knowledge Base
3. Wikipedia
4. Google Search
5. Gemini generates the final response

This approach minimizes unnecessary web searches while prioritizing trusted laboratory information.

---

# Features

- Hybrid Retrieval-Augmented Generation (Hybrid RAG)
- Multi-stage retrieval pipeline
- Two Chroma vector databases
- LangGraph workflow orchestration
- HuggingFace semantic embeddings
- Google Gemini response generation
- Wikipedia integration
- Google Search fallback
- Streamlit web interface

---

# Future Improvements

- Memory-enabled conversations
- Image-based bee identification
- PDF ingestion
- Automatic document updates


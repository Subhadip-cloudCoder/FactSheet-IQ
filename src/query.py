import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# DYNAMIC PATH RESOLUTION
# Ensures the script can find the database

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CHROMA_PATH = os.path.join(BASE_DIR, "vectorstore", "chroma_db")

EMBEDDING_MODEL = "mxbai-embed-large"
CHAT_MODEL = "llama3:8b"

def get_unique_sources(db):
    """Extracts a list of unique PDF filenames currently inside the database. """
    collection = db.get()
    metadatas = collection.get("metadatas", [])
    sources = set()
    for meta in metadatas:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return list(sources)

def main():
    print("Initializing FactSheet-IQ Query Engine...")

    # LOADING THE EXISTING VECTOR DATABASE

    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Vector database not found at {CHROMA_PATH}")
        return

    embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # CONFIGURE THE DATABASE

    retriver = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 7,
            "fetch_k": 25,
            "lambda_unit" : 0.65
        }
    )

    # INITIALIZE THE LOCAL CHAT MODEL

    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0.0,
        num_ctx=8192
    )

    # DEFINE THE SYSTEM PROMPT

    system_prompt = ("""
You are an elite financial analyst specializing in global ETFs, Stocks, and Mutual Funds. Your goal is to provide 100% accurate, audit-ready financial insights based strictly on the provided context.

Each chunk in the context below is tagged with its original file name via a [Source: filename.pdf] prefix.

CRITICAL DOCUMENT TRACKING RULES:
1. Multi-Document Clarity: Always explicitly state WHICH fund or document you are referencing in your analysis based on the [Source] tags present in the context.
2. Source Isolation: Never mix up or blend financial metrics (like expense ratios, AUM, or asset allocations) between different source files.

CRITICAL CURRENCY & LOCALIZATION RULES:
1. Asset Identification: Identify the country or region of the asset mentioned in the query or context (e.g., Nifty, BEES, CPSE, Tata = India; S&P 500, Apple = USA; Nikkei = Japan).
2. Local Currency Enforcement: Dynamically apply the correct local currency symbol for ALL figures, financial calculations, and metrics:
   - India/Indian Assets: Use Indian Rupees (₹ or INR) exclusively. Never default to '$' for Indian markets.
   - USA/Global Default: Use US Dollars ($ or USD).
   - Japan: Use Japanese Yen (¥ or JPY).
3. Rule of Thumb Adaptation: If providing a general trading rule, adapt the numerical examples natively to the target local currency (e.g., "for every ₹1,000 risked, target a ₹3,000 profit").

CRITICAL EXECUTION & GUARDRAIL RULES:
1. No Code Outputs: Do NOT write programming code or output any markdown code blocks (e.g., do not generate Python, JavaScript, or bash scripts). Respond entirely in human financial analysis.
2. Professional Tone: Answer using clean, concise, structured, and professional financial English.
3. Strict Grounding (Zero Tolerance for Hallucinations): Base your insights strictly on the provided context text and tables. If the exact data required to answer the question is missing or structurally insufficient to calculate a definitive answer, reply with this exact phrase:
   "I cannot find sufficient data in the provided financial sheets."

Context:
{context}
"""
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}")
        ]
    )

    # FOMATING THE CONTEXT HELPER FUNC

    def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            clean_content = "\n".join([line.strip() for line in doc.page_content.split("\n") if line.strip()])
            formatted_chunks.append(f"[Source: {source}] \n {clean_content}")
        return "\n\n---\n\n".join(formatted_chunks)
    
    # CONSTRUCT THE LCEL

    rag_chain = (
        {"context" : retriver | format_docs, "question" : RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print(f"Successfully loaded {CHAT_MODEL}. Ready for queries!\n")
    print("-" * 50)

    # INTERATIVE TERMINAL LOOP FOR TESTING

    while True:
        USER_QUERY = input("\n Ask your financhial sheet a question (or type 'exit'): ")
        if USER_QUERY.strip().lower() == 'exit':
            print("Shutting down Query Engine...")
            break

        if not USER_QUERY.strip():
            continue

        print("\n Analyzing documents and calculating mertices...")

        try:
            # STREAM THE RESPONSE TOKEN IN REAL TIME IN THE TERMINAL

            response = rag_chain.stream(USER_QUERY)
            print("\n[Financial Insights]", end="", flush=True)

            for chunk in response:
                print(chunk, end="", flush=True)

            print("\n" + "-" * 50)
        
        except Exception as e:
            print(f"\n An error ocurred during the chain execution: {e}")
    
if __name__ == "__main__":
    main()